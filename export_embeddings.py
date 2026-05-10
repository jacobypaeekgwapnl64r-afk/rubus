import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FlatImageDataset, default_val_transform
from model import build_model


def get_feature_extractor(model: nn.Module):
    """
    将分类模型改造成特征提取器：
    去掉最后分类层，输出倒数第二层特征
    """
    if hasattr(model, "reset_classifier"):
        model.reset_classifier(0)
        return model
    raise RuntimeError("当前模型不支持 reset_classifier()，请检查 timm 模型。")


def extract_embeddings(model, dataloader, device):
    model.eval()
    all_features = []
    all_paths = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting"):
            if len(batch) == 3:
                imgs, labels, paths = batch
                all_labels.extend(labels.numpy().tolist())
            elif len(batch) == 2:
                imgs, paths = batch
                labels = None
            else:
                raise RuntimeError("未知 batch 格式。")

            imgs = imgs.to(device, non_blocking=True)
            feats = model(imgs)
            feats = feats.cpu().numpy()

            all_features.append(feats)
            all_paths.extend(paths)

    all_features = np.concatenate(all_features, axis=0)
    return all_features, all_labels, all_paths


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--modern_dir", type=str, required=True)
    parser.add_argument("--arch_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取模型
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    model_name = ckpt["model_name"]
    img_size = ckpt["img_size"]

    model = build_model(model_name, num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = get_feature_extractor(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 数据集
    modern_ds = FlatImageDataset(
        args.modern_dir,
        transform=default_val_transform(img_size),
        labeled=True,
        class_to_idx=class_to_idx,
        return_path=True
    )

    arch_ds = FlatImageDataset(
        args.arch_dir,
        transform=default_val_transform(img_size),
        labeled=False,
        return_path=True
    )

    num_workers = 2

    modern_loader = DataLoader(
        modern_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    arch_loader = DataLoader(
        arch_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 提取特征
    modern_feats, modern_labels, modern_paths = extract_embeddings(model, modern_loader, device)
    arch_feats, _, arch_paths = extract_embeddings(model, arch_loader, device)

    # 归一化，便于余弦相似度
    modern_feats = modern_feats / np.linalg.norm(modern_feats, axis=1, keepdims=True)
    arch_feats = arch_feats / np.linalg.norm(arch_feats, axis=1, keepdims=True)

    # 为每张考古图找最相似现代图
    sim_matrix = cosine_similarity(arch_feats, modern_feats)

    retrieval_csv = output_dir / "nearest_modern_matches.csv"
    with open(retrieval_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "arch_file",
            "rank",
            "modern_file",
            "modern_label",
            "cosine_similarity"
        ])

        for i, arch_path in enumerate(arch_paths):
            sims = sim_matrix[i]
            topk_idx = np.argsort(-sims)[:5]  # 取最相似5张

            for rank, j in enumerate(topk_idx, start=1):
                writer.writerow([
                    Path(arch_path).name,
                    rank,
                    Path(modern_paths[j]).name,
                    idx_to_class[modern_labels[j]],
                    f"{sims[j]:.6f}"
                ])

    print(f"已保存最近邻检索结果: {retrieval_csv}")

    # PCA 可视化数据
    all_feats = np.vstack([modern_feats, arch_feats])
    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_feats)

    modern_coords = coords[:len(modern_feats)]
    arch_coords = coords[len(modern_feats):]

    pca_csv = output_dir / "pca_coordinates.csv"
    with open(pca_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "file", "label", "pc1", "pc2"])

        for path, label, xy in zip(modern_paths, modern_labels, modern_coords):
            writer.writerow([
                "modern",
                Path(path).name,
                idx_to_class[label],
                f"{xy[0]:.6f}",
                f"{xy[1]:.6f}"
            ])

        for path, xy in zip(arch_paths, arch_coords):
            writer.writerow([
                "archaeological",
                Path(path).name,
                "",
                f"{xy[0]:.6f}",
                f"{xy[1]:.6f}"
            ])

    print(f"已保存 PCA 坐标: {pca_csv}")


if __name__ == "__main__":
    main()
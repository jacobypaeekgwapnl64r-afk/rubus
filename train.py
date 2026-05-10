import os
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from dataset import FlatImageDataset, default_train_transform, default_val_transform
from model import build_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="现代样本平铺目录")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="tf_efficientnetv2_s.in21k_ft_in1k")
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 先建立标签映射
    full_ds_for_classes = FlatImageDataset(
        args.data_dir,
        transform=None,
        labeled=True,
        return_path=False
    )

    n = len(full_ds_for_classes)
    indices = list(range(n))
    labels = full_ds_for_classes.targets

    # 统计每个类别样本数
    label_counts = {}
    for i, y in enumerate(labels):
        label_counts[y] = label_counts.get(y, 0) + 1

    # 仅保留样本数 >= 2 的类别，否则 stratify 会报错
    valid_indices = [i for i, y in enumerate(labels) if label_counts[y] >= 2]
    dropped_indices = [i for i, y in enumerate(labels) if label_counts[y] < 2]

    if dropped_indices:
        idx_to_class = {v: k for k, v in full_ds_for_classes.class_to_idx.items()}
        dropped_classes = sorted(set(idx_to_class[labels[i]] for i in dropped_indices))
        print("以下类别因样本数少于 2 被跳过：")
        for c in dropped_classes:
            print("  -", c)

    if len(valid_indices) == 0:
        raise RuntimeError("没有可用于训练的类别：所有类别样本数都少于 2。")

    filtered_labels = [labels[i] for i in valid_indices]

    train_idx, val_idx = train_test_split(
        valid_indices,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=filtered_labels
    )

    train_ds = FlatImageDataset(
        args.data_dir,
        transform=default_train_transform(args.img_size),
        labeled=True,
        class_to_idx=full_ds_for_classes.class_to_idx,
        return_path=False
    )
    val_ds = FlatImageDataset(
        args.data_dir,
        transform=default_val_transform(args.img_size),
        labeled=True,
        class_to_idx=full_ds_for_classes.class_to_idx,
        return_path=False
    )

    train_ds = Subset(train_ds, train_idx)
    val_ds = Subset(val_ds, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    num_classes = len(full_ds_for_classes.class_to_idx)
    model = build_model(args.model, num_classes=num_classes, pretrained=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 类别权重，可缓解类别不平衡
    class_counts = np.bincount(labels, minlength=num_classes)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.mean()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = 0.0
    best_path = Path(args.output_dir) / "best.pt"

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for imgs, targets in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += imgs.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                logits = model(imgs)
                loss = criterion(logits, targets)

                val_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)

                val_correct += (preds == targets).sum().item()
                val_total += imgs.size(0)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(targets.cpu().numpy().tolist())

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": full_ds_for_classes.class_to_idx,
                "model_name": args.model,
                "img_size": args.img_size,
            }, best_path)

    idx_to_class = {v: k for k, v in full_ds_for_classes.class_to_idx.items()}
    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    report = classification_report(
        all_targets,
        all_preds,
        target_names=target_names,
        digits=4
    )
    cm = confusion_matrix(all_targets, all_preds)

    with open(Path(args.output_dir) / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    np.savetxt(Path(args.output_dir) / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    with open(Path(args.output_dir) / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(full_ds_for_classes.class_to_idx, f, ensure_ascii=False, indent=2)

    print(f"Best val acc: {best_acc:.4f}")
    print(f"Saved to: {best_path}")
    print(f"总样本数: {len(full_ds_for_classes)}")
    print(f"总类别数: {len(full_ds_for_classes.class_to_idx)}")

if __name__ == "__main__":
    main()
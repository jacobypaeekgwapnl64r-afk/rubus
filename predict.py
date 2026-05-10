import csv
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from dataset import is_image_file
from model import build_model


def default_transform(img_size: int = 384):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True, help="考古图片目录")
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.60, help="低于此置信度则标记为unknown_or_low_confidence")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    model_name = ckpt["model_name"]
    img_size = ckpt["img_size"]

    model = build_model(model_name, num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tfm = default_transform(img_size)

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"目录不存在: {args.input_dir}")

    image_paths = [p for p in sorted(input_dir.iterdir()) if p.is_file() and is_image_file(p)]
    if not image_paths:
        raise RuntimeError(f"目录下没有找到图片: {args.input_dir}")

    with open(args.output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file",
            "predicted_label",
            "confidence",
            "top3_labels",
            "top3_confidences",
            "status"
        ])

        with torch.no_grad():
            for p in image_paths:
                img = Image.open(p).convert("RGB")
                x = tfm(img).unsqueeze(0).to(device)

                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0]

                conf, pred = torch.max(probs, dim=0)
                conf = float(conf.item())
                pred_idx = int(pred.item())
                pred_name = idx_to_class[pred_idx]

                topk = min(3, len(idx_to_class))
                top_vals, top_idxs = torch.topk(probs, k=topk)
                top_labels = [idx_to_class[int(i.item())] for i in top_idxs]
                top_confs = [f"{float(v.item()):.4f}" for v in top_vals]

                status = "accepted" if conf >= args.threshold else "unknown_or_low_confidence"

                writer.writerow([
                    p.name,
                    pred_name,
                    f"{conf:.4f}",
                    " | ".join(top_labels),
                    " | ".join(top_confs),
                    status
                ])

    print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
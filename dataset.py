import re
from pathlib import Path
from typing import Optional, List, Tuple

from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms


IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def parse_species_from_filename(filename: str) -> str:
    """
    支持以下命名格式：
      R_parvifolius1.jpg
      R_parvifolius-1.jpg
      R_parvifolius(1).jpg
      Rubus parvifolius12.tif
      Rubus parvifolius-12.tiff
      Rubus parvifolius(12).png

    返回：
      R_parvifolius
      Rubus parvifolius
    """
    stem = Path(filename).stem.strip()

    patterns = [
        r"^(.*)\((\d+)\)$",   # xxx(1)
        r"^(.*)-(\d+)$",      # xxx-1
        r"^(.*?)(\d+)$",      # xxx1
    ]

    for pattern in patterns:
        m = re.match(pattern, stem)
        if m:
            species = m.group(1).strip()
            species = re.sub(r"\s+", " ", species)
            species = species.rstrip("_- ")
            if len(species) == 0:
                break
            return species

    raise ValueError(
        f"文件名不符合规则: {filename}\n"
        f"支持示例: 物种名1.jpg / 物种名-1.jpg / 物种名(1).jpg"
    )


def default_train_transform(img_size: int = 384):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def default_val_transform(img_size: int = 384):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class FlatImageDataset(Dataset):
    """
    单层目录数据集。
    支持两种模式：

    1) labeled=True
       从文件名解析类别，用于训练/验证
       目录示例：
         modern/
           R_parvifolius1.jpg
           R_parvifolius2.tif
           R_hirsutus-1.jpg
           R_rosifolius(3).png

    2) labeled=False
       不解析文件名标签，用于考古未知样本预测
       目录示例：
         archaeological/
           BS001.jpg
           sample_01.tif
           unknown_a.png
    """
    def __init__(
        self,
        root_dir: str,
        transform=None,
        labeled: bool = True,
        class_to_idx: Optional[dict] = None,
        return_path: bool = False
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.labeled = labeled
        self.return_path = return_path

        if not self.root_dir.exists():
            raise FileNotFoundError(f"目录不存在: {root_dir}")

        files = [p for p in sorted(self.root_dir.iterdir()) if p.is_file() and is_image_file(p)]
        if not files:
            raise RuntimeError(f"目录下没有找到图片: {root_dir}")

        self.image_paths: List[str] = []
        self.label_names: List[str] = []

        for p in files:
            if labeled:
                try:
                    cls = parse_species_from_filename(p.name)
                    self.image_paths.append(str(p))
                    self.label_names.append(cls)
                except ValueError as e:
                    print(f"[跳过] {e}")
            else:
                self.image_paths.append(str(p))

        if len(self.image_paths) == 0:
            raise RuntimeError("没有可用样本，请检查图片和命名规则。")

        if labeled:
            if class_to_idx is None:
                unique_classes = sorted(set(self.label_names))
                self.class_to_idx = {c: i for i, c in enumerate(unique_classes)}
            else:
                self.class_to_idx = class_to_idx

            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

            self.targets: List[int] = []
            valid_image_paths: List[str] = []

            for path, cls_name in zip(self.image_paths, self.label_names):
                if cls_name in self.class_to_idx:
                    valid_image_paths.append(path)
                    self.targets.append(self.class_to_idx[cls_name])

            self.image_paths = valid_image_paths

            if len(self.image_paths) == 0:
                raise RuntimeError("映射类别后没有有效样本。")
        else:
            self.class_to_idx = class_to_idx
            self.idx_to_class = None
            self.targets = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except UnidentifiedImageError:
            raise RuntimeError(f"无法读取图片: {path}")

        if self.transform is not None:
            img = self.transform(img)

        if self.labeled:
            label = self.targets[idx]
            if self.return_path:
                return img, label, path
            return img, label
        else:
            if self.return_path:
                return img, path
            return img
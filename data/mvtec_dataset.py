"""
data/mvtec_dataset.py
PyTorch Dataset cho MVTec AD và MVTec-Geo.
"""

import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------------
# Normalization (ImageNet stats — dùng cho WideResNet/MobileViT)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(image_size: int = 224, is_train: bool = False) -> transforms.Compose:
    ops = []
    if is_train:
        ops += [transforms.ColorJitter(brightness=0.1, contrast=0.1)]
    ops += [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def mask_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MVTecDataset(Dataset):
    """
    Hỗ trợ cả MVTec AD gốc (L0) và MVTec-Geo (L1-L3).

    Parameters
    ----------
    root        : str   Path đến mvtec_<severity>/<category>/
    split       : str   'train' hoặc 'test'
    image_size  : int   Kích thước ảnh sau resize
    use_dist_mask: bool  Nếu True, load ground_truth_dist thay vì ground_truth
                         (dùng khi evaluate baseline trực tiếp trên ảnh distorted)
    """

    def __init__(
        self,
        root:           str,
        split:          str = "train",
        image_size:     int = 224,
        use_dist_mask:  bool = False,
    ) -> None:
        super().__init__()
        self.root          = Path(root)
        self.split         = split
        self.image_size    = image_size
        self.use_dist_mask = use_dist_mask

        self.img_tf  = get_transforms(image_size, is_train=(split == "train"))
        self.mask_tf = mask_transform(image_size)

        self.samples = self._collect_samples()

    # ── public ──────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        img  = Image.open(sample["image_path"]).convert("RGB")
        img  = self.img_tf(img)          # [3, H, W]

        # Mask (binary, 0=normal 1=defect)
        if sample["mask_path"] is not None:
            mask = Image.open(sample["mask_path"]).convert("L")
            mask = self.mask_tf(mask)    # [1, H, W]
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros(1, self.image_size, self.image_size)

        # Warp params path (cho Phase 1 training)
        warp_params = None
        if sample.get("warp_params_path") is not None:
            with open(sample["warp_params_path"]) as f:
                warp_params = json.load(f)

        return {
            "image":       img,          # [3, H, W]  float32
            "mask":        mask,         # [1, H, W]  float32  {0,1}
            "label":       sample["label"],      # int  0=normal 1=defect
            "category":    sample["category"],
            "defect_type": sample["defect_type"],
            "image_path":  str(sample["image_path"]),
            "warp_params": warp_params if warp_params is not None else {},rams,  # dict | None
        }

    # ── private ─────────────────────────────────────────────────────────────
    def _collect_samples(self):
        split_dir = self.root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        category = self.root.name
        samples  = []
        mask_root_key = "ground_truth_dist" if self.use_dist_mask else "ground_truth"

        for defect_dir in sorted(split_dir.iterdir()):
            if not defect_dir.is_dir():
                continue
            defect_type = defect_dir.name
            is_normal   = (defect_type == "good")

            for img_path in sorted(defect_dir.glob("*")):
                if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue

                # Ground truth mask
                mask_path = None
                if not is_normal:
                    # Cố gắng tìm mask với nhiều naming convention
                    stem = img_path.stem
                    gt_dir = self.root / mask_root_key / defect_type
                    for candidate in [
                        gt_dir / f"{stem}_mask.png",
                        gt_dir / f"{stem}.png",
                    ]:
                        if candidate.exists():
                            mask_path = candidate
                            break

                # Warp params (chỉ có trong MVTec-Geo)
                warp_params_path = None
                wp = self.root / "warp_params" / self.split / defect_type / f"{img_path.stem}.json"
                if wp.exists():
                    warp_params_path = wp

                samples.append({
                    "image_path":      img_path,
                    "mask_path":       mask_path,
                    "label":           0 if is_normal else 1,
                    "category":        category,
                    "defect_type":     defect_type,
                    "warp_params_path": warp_params_path,
                })

        if len(samples) == 0:
            raise ValueError(f"No samples found in {split_dir}")

        return samples


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------
def make_loader(
    root:        str,
    split:       str = "train",
    image_size:  int = 224,
    batch_size:  int = 8,
    num_workers: int = 4,
    use_dist_mask: bool = False,
    shuffle:     bool = None,
) -> torch.utils.data.DataLoader:
    if shuffle is None:
        shuffle = (split == "train")

    dataset = MVTecDataset(root, split, image_size, use_dist_mask)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

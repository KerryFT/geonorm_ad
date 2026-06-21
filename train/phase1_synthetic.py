"""
train/phase1_synthetic.py
Phase 1: Supervised pre-training của GLN với synthetic warps.

Quá trình:
  1. Load normal images từ MVTec-Geo training set (đã có warp_params JSON).
  2. GT control points = canonical_grid - displacement_at_control_pts
     (approximation đúng cho mild/moderate; xem note bên dưới).
  3. Train GLN để minimize L_sup + L_ent.

Note về GT: make_geo_dataset dùng remap: I_d[y,x] = I[y+dy, x+dx].
  Nên điểm canonical (cx,cy) xuất hiện trong I_d tại ≈ (cx-dx_ctrl, cy-dy_ctrl).
  Đây là approximation tốt cho scale ≤ 0.10; với scale=0.20 cần invert chính xác hơn.
"""

import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Optional

from models.gln    import GLN, canonical_grid
from models.cw_tps import ConfidenceWeightedTPS
from models.geonorm_ad import GeoNormAD

# ---------------------------------------------------------------------------
# Dataset for Phase 1
# ---------------------------------------------------------------------------
class SyntheticWarpDataset(Dataset):
    """
    Load ảnh distorted + warp params JSON từ MVTec-Geo.
    Tính GT control point positions.
    """

    def __init__(
        self,
        root:         str,
        category:     str,
        severity:     str = "moderate",
        K:            int = 16,
        image_size:   int = 224,
    ) -> None:
        self.K          = K
        self.image_size = image_size
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        geo_root = Path(root) / f"mvtec_{severity}" / category
        img_dir  = geo_root / "train" / "good"
        wp_dir   = geo_root / "warp_params" / "train" / "good"

        self.samples = []
        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            wp_path = wp_dir / img_path.with_suffix(".json").name
            if wp_path.exists():
                self.samples.append((img_path, wp_path))

        if len(self.samples) == 0:
            raise ValueError(
                f"Không tìm thấy dữ liệu tại {img_dir}\n"
                "  Chạy data/make_geo_dataset.py trước."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, wp_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.tf(img)    # [3, H, W]

        with open(wp_path) as f:
            wp = json.load(f)

        gt_pts = self._compute_gt_ctrl_pts(wp)   # [K, 2] normalized ∈ [0,1]

        return img, gt_pts

    def _compute_gt_ctrl_pts(self, wp: dict) -> torch.Tensor:
        """
        Tính GT positions của canonical control points trong I_d.

        Từ warp params: I_d[y,x] = I_clean[y + dy_dense, x + dx_dense]
        → canonical pt (cx, cy) xuất hiện ở ≈ (cx/W - dx_ctrl/W, cy/H - dy_ctrl/H) normalized.
        """
        H, W = wp["H"], wp["W"]
        K_side = int(self.K ** 0.5)

        # Canonical grid ∈ [0.1, 0.9] (normalized)
        lin_x = np.linspace(0.1, 0.9, K_side)
        lin_y = np.linspace(0.1, 0.9, K_side)
        gx, gy = np.meshgrid(lin_x, lin_y)   # [K_side, K_side]

        # Displacement tại control points (normalized)
        dx = np.array(wp["dx"])   # [K_ctrl, K_ctrl]  pixel displacement
        dy = np.array(wp["dy"])

        # Resample displacement grid từ K_ctrl=4 về K_side (đã khớp nếu K=16)
        if dx.shape[0] != K_side:
            from scipy.interpolate import RegularGridInterpolator
            ctrl_lin = np.linspace(0.1, 0.9, dx.shape[0])
            interp_x = RegularGridInterpolator((ctrl_lin, ctrl_lin), dx)
            interp_y = RegularGridInterpolator((ctrl_lin, ctrl_lin), dy)
            pts_query = np.stack([gy.ravel(), gx.ravel()], axis=1)
            dx = interp_x(pts_query).reshape(K_side, K_side)
            dy = interp_y(pts_query).reshape(K_side, K_side)

        # Approximate inverse: canonical (cx,cy) → distorted (cx - dx/W, cy - dy/H)
        dist_x = gx - dx / W   # normalized
        dist_y = gy - dy / H

        pts = np.stack([dist_x.ravel(), dist_y.ravel()], axis=1)   # [K, 2]
        pts = np.clip(pts, 0.0, 1.0)   # clamp vào valid range
        return torch.tensor(pts, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def loss_sup(pred_pts: torch.Tensor, gt_pts: torch.Tensor, confs: torch.Tensor) -> torch.Tensor:
    """
    L_sup = (1/K) · Σ_i cᵢ · ‖p̂ᵢ - pᵢᵍᵗ‖² + (1/K) · Σ_i (cᵢ - 1)²

    Phần đầu: weighted coord loss (confident → higher penalty).
    Phần sau: push confidence lên 1.0 khi gt có sẵn.
    """
    coord_err = (pred_pts - gt_pts).pow(2).sum(-1)    # [B, K]
    l_coord   = (confs * coord_err).mean()
    l_conf    = (confs - 1.0).pow(2).mean()
    return l_coord + 0.5 * l_conf


def loss_entropy(confs: torch.Tensor) -> torch.Tensor:
    """
    L_ent = −(1/K) Σ [c·log(c) + (1−c)·log(1−c)]
    Prevent degenerate all-zero confidence.
    """
    eps = 1e-6
    c   = confs.clamp(eps, 1 - eps)
    return -(c * c.log() + (1 - c) * (1 - c).log()).mean()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_phase1(
    model:        GeoNormAD,
    root:         str,
    category:     str,
    severity:     str   = "moderate",
    epochs:       int   = 50,
    lr:           float = 1e-4,
    batch_size:   int   = 16,
    image_size:   int   = 224,
    weight_ent:   float = 0.05,
    device:       str   = "cuda",
    save_path:    Optional[str] = None,
) -> None:
    """
    Train GLN với supervised signal từ warp params.

    Parameters
    ----------
    model     : GeoNormAD instance (GLN sẽ được train)
    root      : path đến output_base của make_geo_dataset
    save_path : nếu set, lưu checkpoint tốt nhất
    """
    model = model.to(device)

    dataset = SyntheticWarpDataset(root, category, severity, model.K, image_size)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=4, pin_memory=True)

    # Chỉ optimize GLN params
    optimizer = optim.AdamW(model.gln.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")

    print(f"\n  Phase 1 training: {category} / {severity}")
    print(f"  Dataset size: {len(dataset)} images")
    print(f"  Epochs: {epochs}  |  LR: {lr}  |  Batch: {batch_size}\n")

    for epoch in range(epochs):
        model.train()
        total_loss = total_lsup = total_lent = 0.0

        for imgs, gt_pts in loader:
            imgs   = imgs.to(device)           # [B, 3, H, W]
            gt_pts = gt_pts.to(device)         # [B, K, 2]

            pred_pts, confs = model.gln(imgs)

            l_sup = loss_sup(pred_pts, gt_pts, confs)
            l_ent = loss_entropy(confs)
            loss  = l_sup + weight_ent * l_ent

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.gln.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_lsup += l_sup.item()
            total_lent += l_ent.item()

        scheduler.step()
        n = len(loader)
        avg = total_loss / n

        if (epoch + 1) % 5 == 0:
            print(
                f"  Epoch [{epoch+1:3d}/{epochs}]  "
                f"Loss: {avg:.5f}  "
                f"L_sup: {total_lsup/n:.5f}  "
                f"L_ent: {total_lent/n:.5f}  "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

        if avg < best_loss and save_path is not None:
            best_loss = avg
            torch.save({
                "epoch":      epoch + 1,
                "gln_state":  model.gln.state_dict(),
                "loss":       avg,
                "K":          model.K,
                "category":   category,
                "severity":   severity,
            }, save_path)
            print(f"    ✔ Checkpoint saved (loss={avg:.5f})")

    print(f"\n  Phase 1 done. Best loss: {best_loss:.5f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",      required=True)
    parser.add_argument("--category",  required=True)
    parser.add_argument("--severity",  default="moderate")
    parser.add_argument("--epochs",    type=int,   default=50)
    parser.add_argument("--lr",        type=float, default=1e-4)
    parser.add_argument("--batch",     type=int,   default=16)
    parser.add_argument("--K",         type=int,   default=16)
    parser.add_argument("--device",    default="cuda")
    parser.add_argument("--save",      default=None)
    args = parser.parse_args()

    model = GeoNormAD(K=args.K, pretrained=True)
    train_phase1(
        model=model,
        root=args.root,
        category=args.category,
        severity=args.severity,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        device=args.device,
        save_path=args.save,
    )

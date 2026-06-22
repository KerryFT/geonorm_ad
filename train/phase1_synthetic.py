"""
train/phase1_synthetic.py
Phase 1: Supervised pre-training của GLN với synthetic warps.

Fix so với phiên bản cũ:
  - save_path=None → luôn có save_dir mặc định
  - Lưu best + last + backup mỗi 10 epoch
  - Không mất checkpoint khi quên truyền --save
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

from models.gln     import GLN, canonical_grid
from models.cw_tps  import ConfidenceWeightedTPS
from models.geonorm_ad import GeoNormAD


# ---------------------------------------------------------------------------
# Dataset for Phase 1
# ---------------------------------------------------------------------------
class SyntheticWarpDataset(Dataset):
    def __init__(self, root, category, severity="moderate", K=16, image_size=224):
        self.K          = K
        self.image_size = image_size
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        geo_root = Path(root) / f"mvtec_{severity}" / category
        img_dir  = geo_root / "train" / "good"
        wp_dir = (
            Path(root)
            / f"mvtec_{severity}"
            / "warp_params"
            / category
            / "train"
            / "good"
        )

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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, wp_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.tf(img)
        with open(wp_path) as f:
            wp = json.load(f)
        gt_pts = self._compute_gt_ctrl_pts(wp)
        return img, gt_pts

    def _compute_gt_ctrl_pts(self, wp):
        H, W   = wp["H"], wp["W"]
        K_side = int(self.K ** 0.5)
        lin_x  = np.linspace(0.1, 0.9, K_side)
        lin_y  = np.linspace(0.1, 0.9, K_side)
        gx, gy = np.meshgrid(lin_x, lin_y)
        dx     = np.array(wp["dx"])
        dy     = np.array(wp["dy"])

        if dx.shape[0] != K_side:
            from scipy.interpolate import RegularGridInterpolator
            ctrl_lin = np.linspace(0.1, 0.9, dx.shape[0])
            interp_x = RegularGridInterpolator((ctrl_lin, ctrl_lin), dx)
            interp_y = RegularGridInterpolator((ctrl_lin, ctrl_lin), dy)
            pts_q    = np.stack([gy.ravel(), gx.ravel()], axis=1)
            dx       = interp_x(pts_q).reshape(K_side, K_side)
            dy       = interp_y(pts_q).reshape(K_side, K_side)

        dist_x = gx - dx / W
        dist_y = gy - dy / H
        pts    = np.stack([dist_x.ravel(), dist_y.ravel()], axis=1)
        return torch.tensor(np.clip(pts, 0.0, 1.0), dtype=torch.float32)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def loss_sup(pred_pts, gt_pts, confs):
    coord_err = (pred_pts - gt_pts).pow(2).sum(-1)
    l_coord   = (confs * coord_err).mean()
    l_conf    = (confs - 1.0).pow(2).mean()
    return l_coord + 0.5 * l_conf


def loss_entropy(confs):
    eps = 1e-6
    c   = confs.clamp(eps, 1 - eps)
    return -(c * c.log() + (1 - c) * (1 - c).log()).mean()


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------
def _save_ckpt(path, model, epoch, loss, category, severity):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":     epoch,
        "gln_state": model.gln.state_dict(),
        "loss":      loss,
        "K":         model.K,
        "category":  category,
        "severity":  severity,
    }, path)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_phase1(
    model,
    root,
    category,
    severity    = "moderate",
    epochs      = 50,
    lr          = 1e-4,
    batch_size  = 16,
    image_size  = 224,
    weight_ent  = 0.05,
    device      = "cuda",
    save_dir    = "/kaggle/working/checkpoints",
):
    """
    Train GLN với supervised signal từ warp params.

    Lưu 3 loại checkpoint tự động vào save_dir:
      best_<category>.pt      — loss thấp nhất → dùng để evaluate
      last_<category>.pt      — epoch cuối     → resume nếu bị interrupt
      epoch<N>_<category>.pt  — backup mỗi 10 epoch
    """
    model = model.to(device)

    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    best_path = save_dir_path / f"best_{category}.pt"
    last_path = save_dir_path / f"last_{category}.pt"

    dataset = SyntheticWarpDataset(root, category, severity, model.K, image_size)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=4, pin_memory=True)

    optimizer = optim.AdamW(model.gln.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")

    print(f"\n  Phase 1: {category} / {severity}")
    print(f"  Samples : {len(dataset)}")
    print(f"  Epochs  : {epochs}  |  LR: {lr}  |  Batch: {batch_size}")
    print(f"  Best ckpt → {best_path}")
    print(f"  Last ckpt → {last_path}\n")

    for epoch in range(epochs):
        model.train()
        total_loss = total_lsup = total_lent = 0.0

        for imgs, gt_pts in loader:
            imgs   = imgs.to(device)
            gt_pts = gt_pts.to(device)

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
        n   = len(loader)
        avg = total_loss / n

        # Log mỗi 5 epoch
        if (epoch + 1) % 5 == 0:
            print(
                f"  Epoch [{epoch+1:3d}/{epochs}]  "
                f"Loss={avg:.5f}  "
                f"L_sup={total_lsup/n:.5f}  "
                f"L_ent={total_lent/n:.5f}  "
                f"LR={scheduler.get_last_lr()[0]:.2e}"
            )

        # Luôn lưu last checkpoint
        _save_ckpt(last_path, model, epoch + 1, avg, category, severity)

        # Lưu best checkpoint nếu loss giảm
        if avg < best_loss:
            best_loss = avg
            _save_ckpt(best_path, model, epoch + 1, avg, category, severity)
            print(f"    ✔ Best saved  epoch={epoch+1}  loss={avg:.5f}  → {best_path.name}")

        # Backup mỗi 10 epoch
        if (epoch + 1) % 10 == 0:
            backup = save_dir_path / f"epoch{epoch+1:03d}_{category}.pt"
            _save_ckpt(backup, model, epoch + 1, avg, category, severity)
            print(f"    💾 Backup     epoch={epoch+1}  → {backup.name}")

    print(f"\n  ✅ Phase 1 done.  Best loss={best_loss:.5f}")
    print(f"  Use for evaluation: --checkpoint {best_path}")
    return str(best_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",      required=True,  help="MVTec-Geo root")
    parser.add_argument("--category",  required=True)
    parser.add_argument("--severity",  default="moderate")
    parser.add_argument("--epochs",    type=int,   default=50)
    parser.add_argument("--lr",        type=float, default=1e-4)
    parser.add_argument("--batch",     type=int,   default=16)
    parser.add_argument("--K",         type=int,   default=16)
    parser.add_argument("--device",    default="cuda")
    parser.add_argument("--save_dir",  default="/kaggle/working/checkpoints",
                        help="Thư mục lưu checkpoints (tự tạo nếu chưa có)")
    args = parser.parse_args()

    model     = GeoNormAD(K=args.K, pretrained=True)
    best_ckpt = train_phase1(
        model      = model,
        root       = args.root,
        category   = args.category,
        severity   = args.severity,
        epochs     = args.epochs,
        lr         = args.lr,
        batch_size = args.batch,
        device     = args.device,
        save_dir   = args.save_dir,
    )
    print(f"\n  Chạy evaluation với:")
    print(f"  python eval/evaluate.py \\")
    print(f"      --data_root  <geo_root> \\")
    print(f"      --category   {args.category} \\")
    print(f"      --checkpoint {best_ckpt}")

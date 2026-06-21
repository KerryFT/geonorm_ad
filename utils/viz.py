"""
utils/viz.py
Visualization utilities — debug + paper figures.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _denorm(t: torch.Tensor) -> np.ndarray:
    """[3,H,W] normalized tensor → [H,W,3] uint8 numpy."""
    img = t.cpu().float() * IMAGENET_STD + IMAGENET_MEAN
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 1. Rectification comparison
# ---------------------------------------------------------------------------
def visualize_rectification(
    x_dist:  torch.Tensor,   # [3, H, W]  distorted image
    x_rect:  torch.Tensor,   # [3, H, W]  rectified image
    x_orig:  Optional[torch.Tensor] = None,  # [3, H, W]  original (GT)
    save_path: Optional[str] = None,
) -> None:
    """3-panel: Distorted | Rectified | Original (GT)."""
    n_cols = 3 if x_orig is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    axes[0].imshow(_denorm(x_dist));  axes[0].set_title("Distorted (input)")
    axes[1].imshow(_denorm(x_rect));  axes[1].set_title("Rectified (GeoNorm)")
    if x_orig is not None:
        axes[2].imshow(_denorm(x_orig)); axes[2].set_title("Original (GT)")

    for ax in axes: ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# 2. Control points + confidence
# ---------------------------------------------------------------------------
def visualize_control_points(
    x_dist:  torch.Tensor,   # [3, H, W]
    coords:  torch.Tensor,   # [K, 2]    predicted distorted positions ∈ [0,1]
    confs:   torch.Tensor,   # [K]       confidence scores ∈ [0,1]
    canonical_pts: Optional[torch.Tensor] = None,  # [K, 2]  canonical positions
    save_path: Optional[str] = None,
) -> None:
    """
    Vẽ control points lên ảnh distorted.
    - Màu điểm: confidence (xanh = cao, đỏ = thấp).
    - Mũi tên: canonical → predicted (độ lớn warp).
    """
    H, W = x_dist.shape[-2:]
    img  = _denorm(x_dist)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)

    coords_px = coords.cpu().numpy()
    coords_px[:, 0] *= W
    coords_px[:, 1] *= H

    c_vals = confs.cpu().numpy()

    sc = ax.scatter(
        coords_px[:, 0], coords_px[:, 1],
        c=c_vals, cmap="RdYlGn", vmin=0, vmax=1,
        s=80, zorder=3, edgecolors="white", linewidths=0.5,
    )
    plt.colorbar(sc, ax=ax, label="Confidence")

    if canonical_pts is not None:
        can_px = canonical_pts.cpu().numpy().copy()
        can_px[:, 0] *= W
        can_px[:, 1] *= H
        ax.scatter(can_px[:, 0], can_px[:, 1],
                   marker="+", s=60, c="white", zorder=2)
        for i in range(len(coords_px)):
            ax.annotate(
                "", xy=(coords_px[i, 0], coords_px[i, 1]),
                xytext=(can_px[i, 0], can_px[i, 1]),
                arrowprops=dict(arrowstyle="->", color="yellow", lw=1.0),
            )

    ax.set_title(f"Control Points (K={len(coords_px)}, mean conf={c_vals.mean():.2f})")
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 3. Anomaly map overlay
# ---------------------------------------------------------------------------
def visualize_anomaly_map(
    x_orig:    torch.Tensor,          # [3, H, W]  original (or distorted) image
    score_map: torch.Tensor,          # [H, W]     anomaly score
    gt_mask:   Optional[torch.Tensor] = None,  # [H, W]  binary
    threshold: Optional[float]        = None,
    title:     str = "Anomaly Map",
    save_path: Optional[str] = None,
) -> None:
    """
    Hiển thị: image | score heatmap | overlay.
    """
    img   = _denorm(x_orig)
    score = score_map.cpu().numpy()

    # Normalize score map về [0, 1] để hiển thị
    s_min, s_max = score.min(), score.max()
    score_norm = (score - s_min) / (s_max - s_min + 1e-8)

    n_cols = 4 if gt_mask is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    axes[0].imshow(img); axes[0].set_title("Image")

    axes[1].imshow(score_norm, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Anomaly Score")

    # Overlay
    axes[2].imshow(img)
    axes[2].imshow(score_norm, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    if threshold is not None:
        pred_mask = (score_norm > threshold).astype(np.uint8)
        axes[2].contour(pred_mask, colors="white", linewidths=0.8)
    axes[2].set_title("Overlay")

    if gt_mask is not None:
        axes[3].imshow(gt_mask.cpu().numpy(), cmap="gray")
        axes[3].set_title("GT Mask")

    for ax in axes: ax.axis("off")
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 4. Ablation comparison grid
# ---------------------------------------------------------------------------
def compare_methods(
    x_dist:   torch.Tensor,          # [3, H, W]
    results:  dict,                   # {"Method Name": score_map [H,W]}
    gt_mask:  Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Side-by-side comparison của nhiều methods.
    Dùng cho ablation figure trong paper.
    """
    methods = list(results.keys())
    n_cols  = len(methods) + 2  # image + methods + (optional mask)
    if gt_mask is not None:
        n_cols += 1

    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4.5))
    img = _denorm(x_dist)

    axes[0].imshow(img); axes[0].set_title("Input", fontsize=10)

    for i, (name, smap) in enumerate(results.items(), start=1):
        s  = smap.cpu().numpy()
        sn = (s - s.min()) / (s.max() - s.min() + 1e-8)
        axes[i].imshow(img)
        axes[i].imshow(sn, cmap="jet", alpha=0.5)
        axes[i].set_title(name, fontsize=9)

    if gt_mask is not None:
        axes[-1].imshow(gt_mask.cpu().numpy(), cmap="gray")
        axes[-1].set_title("GT Mask", fontsize=10)

    for ax in axes: ax.axis("off")
    plt.suptitle("Method Comparison", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 5. Quick sanity check (dùng trong notebook)
# ---------------------------------------------------------------------------
def sanity_check_geonorm(model, x_d: torch.Tensor, device: str = "cuda") -> None:
    """
    In ra thống kê nhanh để verify GeoNorm đang làm gì.
    Dùng sau khi gặp kết quả bất thường.
    """
    model.eval()
    with torch.no_grad():
        x_rect, flow, confs = model.rectify(x_d.to(device))

    diff = (x_rect - x_d.to(device)).abs().mean().item()
    print(f"  Mean pixel diff (rectified vs distorted): {diff:.6f}")
    print(f"  ↳ 0.0 → identity transform (GLN chưa học)")
    print(f"  ↳ > 0.01 → GeoNorm đang warp ảnh (bình thường sau training)")
    print(f"  Mean confidence: {confs.mean().item():.4f} (target: > 0.7)")
    print(f"  Std  confidence: {confs.std().item():.4f}  (spread = diverse)")
    print(f"  Flow range: [{flow.min().item():.3f}, {flow.max().item():.3f}]  (expect ≈ [-1, 1])")

    visualize_rectification(x_d[0].cpu(), x_rect[0].cpu())

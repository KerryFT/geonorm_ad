"""
eval/metrics.py
Metrics cho GeoNorm-AD evaluation.

Metrics chính:
  - Image-AUROC  (iAUROC): standard AD metric
  - Pixel-AUROC  (pAUROC): segmentation quality
  - Geo-Robustness Score  : AUROC(L3) / AUROC(L0)  — novel metric
  - Rectification Error   : ‖I' - I_canonical‖_F
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Core AUROC
# ---------------------------------------------------------------------------
def compute_image_auroc(labels: List[int], scores: List[float]) -> float:
    """Image-level AUROC. labels: 0=normal, 1=defect."""
    if len(set(labels)) < 2:
        return float("nan")   # Không có đủ cả 2 class
    return roc_auc_score(labels, scores)


def compute_pixel_auroc(
    masks:  List[np.ndarray],   # list of [H, W] binary masks {0, 1}
    amaps:  List[np.ndarray],   # list of [H, W] anomaly score maps
) -> float:
    """Pixel-level AUROC (chỉ tính trên ảnh có defect)."""
    # Lọc ra ảnh có mask (defect images)
    valid_masks = [m for m in masks  if m.max() > 0]
    valid_amaps = [a for a, m in zip(amaps, masks) if m.max() > 0]

    if len(valid_masks) == 0:
        return float("nan")

    flat_masks = np.concatenate([m.flatten() for m in valid_masks])
    flat_amaps = np.concatenate([a.flatten() for a in valid_amaps])

    return roc_auc_score(flat_masks.astype(int), flat_amaps)


# ---------------------------------------------------------------------------
# Geo-Robustness Score
# ---------------------------------------------------------------------------
def geo_robustness_score(auroc_l0: float, auroc_lx: float) -> float:
    """
    GRS = AUROC(Lx) / AUROC(L0)
    Càng gần 1.0 → method càng robust với geometric distortion.
    """
    if auroc_l0 == 0:
        return float("nan")
    return auroc_lx / auroc_l0


# ---------------------------------------------------------------------------
# Rectification Error
# ---------------------------------------------------------------------------
def rectification_error(
    x_rect:    torch.Tensor,   # [B, 3, H, W]  rectified image
    x_canonical: torch.Tensor, # [B, 3, H, W]  ground truth canonical image
) -> float:
    """Mean per-pixel L2 error sau rectification (lower = better)."""
    with torch.no_grad():
        err = (x_rect - x_canonical).pow(2).sum(dim=1).sqrt()   # [B, H, W]
        return err.mean().item()


# ---------------------------------------------------------------------------
# Results aggregation
# ---------------------------------------------------------------------------
class ResultsAccumulator:
    """Tích luỹ results qua nhiều batches."""

    def __init__(self) -> None:
        self.img_labels:  List[int]        = []
        self.img_scores:  List[float]      = []
        self.pix_masks:   List[np.ndarray] = []
        self.pix_amaps:   List[np.ndarray] = []

    def update(
        self,
        labels:    torch.Tensor,   # [B]     int {0,1}
        scores:    torch.Tensor,   # [B]     float (image-level)
        masks:     torch.Tensor,   # [B,1,H,W]  binary
        score_maps: torch.Tensor,  # [B,H,W]    pixel-level
    ) -> None:
        self.img_labels.extend(labels.cpu().numpy().tolist())
        self.img_scores.extend(scores.cpu().numpy().tolist())

        for m, a in zip(masks.squeeze(1).cpu().numpy(),
                        score_maps.cpu().numpy()):
            self.pix_masks.append(m)
            self.pix_amaps.append(a)

    def compute(self) -> Dict[str, float]:
        iauroc = compute_image_auroc(self.img_labels, self.img_scores)
        pauroc = compute_pixel_auroc(self.pix_masks, self.pix_amaps)
        return {"iAUROC": iauroc, "pAUROC": pauroc}


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------
def print_results_table(
    results: Dict[str, Dict[str, float]],   # {method: {metric: value}}
    title:   str = "Evaluation Results",
) -> None:
    """
    Ví dụ input:
    {
      "Baseline (PatchCore)":      {"iAUROC": 72.46, "pAUROC": 68.1},
      "GeoNorm-AD (PatchCore)":   {"iAUROC": 78.32, "pAUROC": 74.5},
    }
    """
    metrics = list(next(iter(results.values())).keys())
    col_w   = max(len(k) for k in results) + 2

    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")

    header = f"  {'Method':<{col_w}}" + "".join(f"{m:>12}" for m in metrics)
    print(header)
    print(f"  {'─'*68}")

    for method, vals in results.items():
        row = f"  {method:<{col_w}}"
        for m in metrics:
            v = vals.get(m, float("nan"))
            row += f"{v:>11.2f}%"
        print(row)

    print(f"{'═'*70}\n")


def print_geo_robustness_table(
    per_severity: Dict[str, float],   # {"l0": 72.46, "mild": 70.1, ...}
    method_name:  str = "GeoNorm-AD",
) -> None:
    """In bảng iAUROC theo từng severity level."""
    print(f"\n  Geo-Robustness: {method_name}")
    print(f"  {'─'*40}")
    l0 = per_severity.get("l0", per_severity.get("L0", None))
    for sev, auroc in per_severity.items():
        grs = geo_robustness_score(l0, auroc) if l0 else float("nan")
        print(f"  {sev:<12} iAUROC: {auroc:6.2f}%   GRS: {grs:.4f}")
    print()

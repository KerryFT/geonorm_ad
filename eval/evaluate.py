"""
eval/evaluate.py
Full evaluation pipeline cho GeoNorm-AD.

Quy trình:
  1. Build memory bank từ TRAINING set (ảnh đi qua GeoNorm trước).
  2. Score TEST set (ảnh cũng đi qua GeoNorm trước).
  3. Tính iAUROC và pAUROC.

Đây là fix cho Bug Memory Bank Mismatch từ kết quả 72.33% trước đó.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from models.geonorm_ad import GeoNormAD
from ad_heads.patchcore import PatchCore
from data.mvtec_dataset import make_loader
from eval.metrics import ResultsAccumulator, print_results_table, print_geo_robustness_table


# ---------------------------------------------------------------------------
# Memory bank builder (FIX: dùng rectified images, không phải original)
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_memory_bank(
    model:        GeoNormAD,
    patchcore:    PatchCore,
    train_loader: torch.utils.data.DataLoader,
    device:       str = "cuda",
) -> None:
    """
    Build PatchCore memory bank từ RECTIFIED training images.

    QUAN TRỌNG: Phải dùng cùng pipeline (GeoNorm → features) như khi inference.
    Nếu memory bank từ original images nhưng test features từ rectified → mismatch
    → AUROC thấp (đây là bug trong kết quả 72.33% trước).
    """
    model.eval()
    patchcore.eval()
    all_feats = []

    for batch in tqdm(train_loader, desc="  Building memory bank (rectified)"):
        imgs = batch["image"].to(device)    # [B, 3, H, W]

        # ── GeoNorm rectification (KHÔNG có AD head) ──
        x_rect, _, _ = model.rectify(imgs)  # [B, 3, H, W]

        # ── Extract features từ RECTIFIED image ──
        feats = patchcore._extract_patch_features(x_rect)   # [B, N_patches, C]
        all_feats.append(feats.reshape(-1, feats.shape[-1]).cpu())

    all_feats = torch.cat(all_feats, dim=0)    # [N_total, C]
    patchcore.memory_bank = patchcore._coreset_sample(all_feats)
    print(f"  Memory bank: {patchcore.memory_bank.shape[0]:,} vectors")


# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_category(
    model:        GeoNormAD,
    patchcore:    PatchCore,
    test_loader:  torch.utils.data.DataLoader,
    device:       str = "cuda",
    use_back_proj: bool = True,
) -> dict:
    """
    Evaluate trên test set, trả về iAUROC và pAUROC.

    use_back_proj: Nếu True, dùng inverse TPS để map score về original coords.
    """
    model.eval()
    patchcore.eval()
    model.set_ad_head(patchcore)

    acc = ResultsAccumulator()

    for batch in tqdm(test_loader, desc="  Evaluating"):
        imgs   = batch["image"].to(device)        # [B, 3, H, W]
        masks  = batch["mask"].to(device)          # [B, 1, H, W]
        labels = batch["label"].to(device)         # [B]

        # Full pipeline
        out = model(imgs)

        if use_back_proj:
            score_map = out["score_orig"]   # [B, H, W]  original coords
        else:
            score_map = out["score_rect"]   # [B, H, W]  rectified coords

        # Image-level score = max anomaly score trên ảnh
        img_scores = score_map.flatten(1).max(dim=1).values   # [B]

        acc.update(labels, img_scores, masks, score_map)

    return acc.compute()


# ---------------------------------------------------------------------------
# Full multi-severity evaluation
# ---------------------------------------------------------------------------
def _resolve_cat_root(
    severity:    str,
    data_root:   str,
    category:    str,
    l0_root:     Optional[str] = None,
) -> Optional[Path]:
    """
    Trả về path đến {category}/ cho một severity cụ thể.

    L0 có thể trỏ thẳng vào Kaggle input (không cần copy):
        l0_root = "/kaggle/input/mvtec-anomaly-detection"
        → cat_root = /kaggle/input/mvtec-anomaly-detection/{category}/

    L1-L3 dùng data_root như bình thường:
        → cat_root = {data_root}/mvtec_{severity}/{category}/
    """
    if severity == "l0":
        root = Path(l0_root) if l0_root else Path(data_root) / "mvtec_l0"
        return root / category
    else:
        return Path(data_root) / f"mvtec_{severity}" / category


def run_full_evaluation(
    model:        GeoNormAD,
    patchcore:    PatchCore,
    data_root:    str,
    category:     str,
    image_size:   int = 224,
    batch_size:   int = 4,
    num_workers:  int = 4,
    device:       str = "cuda",
    severities:   tuple = ("l0", "mild", "moderate"),
    checkpoint:   Optional[str] = None,
    l0_root:      Optional[str] = None,
) -> None:
    """
    Chạy evaluation theo các severity levels.

    Parameters
    ----------
    severities : tuple
        Mặc định ("l0", "mild", "moderate") — bỏ "severe" để tiết kiệm disk.
        Thêm "severe" sau khi confirm approach works.
    l0_root : str | None
        Path đến MVTec AD gốc. Nếu set, L0 dùng thẳng nguồn này
        thay vì tìm mvtec_l0/ trong data_root.
        Trên Kaggle: "/kaggle/input/mvtec-anomaly-detection"
    """
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location=device)
        model.gln.load_state_dict(ckpt["gln_state"])
        print(f"  Loaded GLN checkpoint: {checkpoint}")

    model = model.to(device)
    patchcore = patchcore.to(device)

    results_iaur   = {}
    results_paulr  = {}

    print(f"\n  Category: {category}")

    for severity in severities:
        cat_root = _resolve_cat_root(severity, data_root, category, l0_root)

        if not cat_root.exists():
            print(f"  ⚠ {cat_root} không tồn tại, bỏ qua.")
            continue

        print(f"\n  {'─'*50}")
        print(f"  Severity: {severity.upper()}")

        # ── Train loader (build memory bank) ──
        train_loader = make_loader(
            str(cat_root), split="train",
            image_size=image_size, batch_size=batch_size,
            num_workers=num_workers, shuffle=False,
        )
        build_memory_bank(model, patchcore, train_loader, device)

        # ── Test loader ──
        # Với L0: dùng original GT mask
        # Với L1-L3: dùng original GT mask (đánh giá sau back-projection)
        test_loader = make_loader(
            str(cat_root), split="test",
            image_size=image_size, batch_size=batch_size,
            num_workers=num_workers, shuffle=False,
            use_dist_mask=False,   # mask gốc — cho GeoNorm evaluation
        )
        metrics = evaluate_category(model, patchcore, test_loader, device)

        results_iaur[severity]  = metrics["iAUROC"] * 100
        results_paulr[severity] = metrics["pAUROC"] * 100

        print(f"  iAUROC: {metrics['iAUROC']*100:.2f}%   pAUROC: {metrics['pAUROC']*100:.2f}%")

    # In bảng tổng kết
    combined = {
        sev: {"iAUROC": results_iaur.get(sev, float("nan")),
              "pAUROC": results_paulr.get(sev, float("nan"))}
        for sev in severities
    }
    print_results_table(combined, title=f"GeoNorm-AD vs Baseline — {category}")
    print_geo_robustness_table(results_iaur, method_name=f"GeoNorm-AD ({category})")


# ---------------------------------------------------------------------------
# Baseline evaluation (không dùng GeoNorm — để so sánh)
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_baseline_evaluation(
    patchcore:   PatchCore,
    data_root:   str,
    category:    str,
    image_size:  int = 224,
    batch_size:  int = 4,
    device:      str = "cuda",
    severities:  tuple = ("l0", "mild", "moderate", "severe"),
) -> None:
    """
    PatchCore thuần (không GeoNorm) để làm baseline.
    Memory bank build từ ORIGINAL train images.
    """
    patchcore = patchcore.to(device)
    patchcore.eval()

    print(f"\n  [BASELINE] PatchCore — no GeoNorm — {category}")

    for severity in severities:
        cat_root = Path(data_root) / f"mvtec_{severity}" / category
        if not cat_root.exists():
            continue

        print(f"\n  Severity: {severity.upper()}")

        # Memory bank từ ORIGINAL (không qua GeoNorm)
        train_loader = make_loader(str(cat_root), "train", image_size, batch_size)
        patchcore.fit(train_loader, device)

        # Test với distorted GT mask (đánh giá trực tiếp trên ảnh distorted)
        test_loader = make_loader(
            str(cat_root), "test", image_size, batch_size,
            use_dist_mask=True,   # dùng distorted mask cho baseline eval
        )

        acc = ResultsAccumulator()
        for batch in tqdm(test_loader, desc="  Baseline scoring"):
            imgs   = batch["image"].to(device)
            masks  = batch["mask"].to(device)
            labels = batch["label"].to(device)

            score_map  = patchcore(imgs)                          # [B, H, W]
            img_scores = score_map.flatten(1).max(dim=1).values  # [B]
            acc.update(labels, img_scores, masks, score_map)

        m = acc.compute()
        print(f"  iAUROC: {m['iAUROC']*100:.2f}%   pAUROC: {m['pAUROC']*100:.2f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  required=True)
    parser.add_argument("--category",   required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--baseline_only", action="store_true")
    args = parser.parse_args()

    patchcore = PatchCore(coreset_ratio=0.1, k_nearest=5)

    if args.baseline_only:
        run_baseline_evaluation(
            patchcore, args.data_root, args.category,
            args.image_size, args.batch_size, args.device,
        )
    else:
        model = GeoNormAD(K=16, pretrained=True)
        run_full_evaluation(
            model, patchcore,
            args.data_root, args.category,
            args.image_size, args.batch_size,
            device=args.device,
            checkpoint=args.checkpoint,
        )

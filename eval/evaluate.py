"""
eval/evaluate.py  — fixed version

Fix so với phiên bản cũ:
  - L0 KHÔNG đi qua GeoNorm (pure PatchCore baseline)
  - L1-L3 đi qua GeoNorm (memory bank + scoring đều qua GeoNorm)
  - _resolve_cat_root hỗ trợ l0_root trỏ thẳng vào Kaggle input
  - --no_severe để bỏ qua severe khi hết disk
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple

from models.geonorm_ad import GeoNormAD
from ad_heads.patchcore import PatchCore
from data.mvtec_dataset import make_loader
from eval.metrics import ResultsAccumulator, print_results_table, print_geo_robustness_table


# ---------------------------------------------------------------------------
# Path resolver
# ---------------------------------------------------------------------------
def _resolve_cat_root(severity, data_root, category, l0_root=None):
    if severity == "l0":
        root = Path(l0_root) if l0_root else Path(data_root) / "mvtec_l0"
        return root / category
    return Path(data_root) / f"mvtec_{severity}" / category


# ---------------------------------------------------------------------------
# Memory bank builders
# ---------------------------------------------------------------------------
@torch.no_grad()
def _build_bank_plain(patchcore, train_loader, device):
    """Build memory bank KHÔNG qua GeoNorm — dùng cho L0."""
    patchcore.eval()
    all_feats = []
    for batch in tqdm(train_loader, desc="  Memory bank (plain)"):
        imgs = batch["image"].to(device)
        f    = patchcore._extract(imgs)
        all_feats.append(f.cpu())
    all_feats = torch.cat(all_feats, dim=0)
    patchcore.memory_bank = patchcore._coreset(all_feats)
    print(f"  Memory bank: {patchcore.memory_bank.shape[0]:,} vectors")


@torch.no_grad()
def _build_bank_geonorm(model, patchcore, train_loader, device):
    """Build memory bank QUA GeoNorm — dùng cho L1-L3."""
    model.eval(); patchcore.eval()
    all_feats = []
    for batch in tqdm(train_loader, desc="  Memory bank (GeoNorm)"):
        imgs = batch["image"].to(device)
        x_rect, _, _ = model.rectify(imgs)   # rectify trước
        f = patchcore._extract(x_rect)
        all_feats.append(f.cpu())
    all_feats = torch.cat(all_feats, dim=0)
    patchcore.memory_bank = patchcore._coreset(all_feats)
    print(f"  Memory bank: {patchcore.memory_bank.shape[0]:,} vectors")


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------
@torch.no_grad()
def _score_plain(patchcore, test_loader, device):
    """Score KHÔNG qua GeoNorm."""
    acc = ResultsAccumulator()
    for batch in tqdm(test_loader, desc="  Scoring (plain)"):
        imgs   = batch["image"].to(device)
        masks  = batch["mask"].to(device)
        labels = batch["label"].to(device)
        smap   = patchcore(imgs)
        acc.update(labels, smap.flatten(1).max(1).values, masks, smap)
    return acc.compute()


@torch.no_grad()
def _score_geonorm(model, patchcore, test_loader, device):
    """Score QUA GeoNorm + inverse back-projection."""
    model.set_ad_head(patchcore)
    acc = ResultsAccumulator()
    for batch in tqdm(test_loader, desc="  Scoring (GeoNorm)"):
        imgs   = batch["image"].to(device)
        masks  = batch["mask"].to(device)
        labels = batch["label"].to(device)
        out    = model(imgs)
        smap   = out["score_orig"]    # back-projected về original coords
        acc.update(labels, smap.flatten(1).max(1).values, masks, smap)
    return acc.compute()


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def run_full_evaluation(
    model,
    patchcore,
    data_root,
    category,
    image_size   = 224,
    batch_size   = 4,
    num_workers  = 4,
    device       = "cuda",
    severities   = ("l0", "mild", "moderate"),
    checkpoint   = None,
    l0_root      = None,
):
    """
    L0  → PatchCore thuần (không GeoNorm) — đây là upper bound / baseline
    L1+ → GeoNorm + PatchCore

    l0_root : trỏ thẳng vào /kaggle/input/mvtec-anomaly-detection
              để không cần copy L0 về working dir (tiết kiệm disk)
    """
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location=device)
        model.gln.load_state_dict(ckpt["gln_state"])
        print(f"  ✓ GLN checkpoint: epoch={ckpt['epoch']}, loss={ckpt['loss']:.5f}")
    else:
        print("  ⚠ Không có checkpoint → GLN dùng random weights!")
        print("    Truyền --checkpoint <path> để dùng GLN đã trained.\n")

    model     = model.to(device)
    patchcore = patchcore.to(device)

    results_i = {}
    results_p = {}

    print(f"\n  Category: {category}\n")

    for severity in severities:
        cat_root = _resolve_cat_root(severity, data_root, category, l0_root)
        if not cat_root.exists():
            print(f"  ⚠ Bỏ qua {severity}: {cat_root} không tồn tại")
            continue

        print(f"  {'─'*52}")
        print(f"  Severity: {severity.upper()}")

        train_ldr = make_loader(str(cat_root), "train", image_size,
                                batch_size, num_workers, shuffle=False)
        test_ldr  = make_loader(str(cat_root), "test",  image_size,
                                batch_size, num_workers, shuffle=False,
                                use_dist_mask=False)

        if severity == "l0":
            # ── L0: PatchCore thuần, KHÔNG qua GeoNorm ──────────────────
            _build_bank_plain(patchcore, train_ldr, device)
            m = _score_plain(patchcore, test_ldr, device)
        else:
            # ── L1-L3: GeoNorm + PatchCore ───────────────────────────────
            _build_bank_geonorm(model, patchcore, train_ldr, device)
            m = _score_geonorm(model, patchcore, test_ldr, device)

        results_i[severity] = m["iAUROC"] * 100
        results_p[severity] = m["pAUROC"] * 100
        print(f"  iAUROC={m['iAUROC']*100:.2f}%   pAUROC={m['pAUROC']*100:.2f}%")

    # Tổng kết
    combined = {s: {"iAUROC": results_i.get(s, float("nan")),
                    "pAUROC": results_p.get(s, float("nan"))}
                for s in severities if s in results_i}
    print_results_table(combined, f"GeoNorm-AD vs Baseline — {category}")
    print_geo_robustness_table(results_i, f"GeoNorm-AD ({category})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   required=True)
    parser.add_argument("--category",    required=True)
    parser.add_argument("--checkpoint",  default=None)
    parser.add_argument("--image_size",  type=int,   default=224)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--l0_root",     default=None,
                        help="Path đến MVTec AD gốc. "
                             "Kaggle: /kaggle/input/mvtec-anomaly-detection")
    parser.add_argument("--no_severe",   action="store_true")
    parser.add_argument("--coreset",     type=float, default=0.01)
    args = parser.parse_args()

    sevs = ["l0", "mild", "moderate"]
    if not args.no_severe:
        sevs.append("severe")

    model     = GeoNormAD(K=16, pretrained=True)
    patchcore = PatchCore(coreset_ratio=args.coreset)

    run_full_evaluation(
        model, patchcore,
        args.data_root, args.category,
        args.image_size, args.batch_size, args.num_workers,
        args.device,
        severities=tuple(sevs),
        checkpoint=args.checkpoint,
        l0_root=args.l0_root,
    )

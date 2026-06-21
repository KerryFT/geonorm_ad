"""
data/make_geo_dataset.py
Sinh bộ dữ liệu MVTec-Geo từ MVTec AD gốc.

Fixes so với phiên bản cũ:
  #1  Unique seed cho từng ảnh (SHA-256 hash của filepath + severity).
  #2  Control grid lùi vào 10% biên (0.1W..0.9W) để tránh spline extrapolation.
  #3  Lưu warp params (dx, dy, grid, seed) dạng JSON — cần cho Phase 1 training.
  #4  Tạo cả distorted GT mask — cần cho baseline evaluation trực tiếp trên ảnh distorted.
  #5  Clamp map_x/map_y về phạm vi hợp lệ.
  #6  Copy L0 (original) để có điểm so sánh.
"""

import os
import cv2
import json
import hashlib
import shutil
import numpy as np
from scipy.interpolate import RectBivariateSpline
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional, List

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEVERITY_SCALE = {
    "mild":     0.05,
    "moderate": 0.10,
    "severe":   0.20,
}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


# ---------------------------------------------------------------------------
# Core distortion function
# ---------------------------------------------------------------------------
def generate_tps_distortion(
    image: np.ndarray,
    severity: str = "moderate",
    seed: int = 42,
) -> Tuple[np.ndarray, dict]:
    """
    Áp dụng TPS distortion có kiểm soát lên image.

    Returns
    -------
    distorted : np.ndarray   Ảnh sau biến dạng (cùng shape với input).
    params    : dict         Warp parameters để dùng làm GT trong Phase 1.
    """
    rng = np.random.RandomState(seed)   # Tách biệt với global random state
    H, W = image.shape[:2]
    scale = SEVERITY_SCALE[severity]

    K = 4
    # FIX #2: inset control grid 10% từ biên → tránh extrapolation artifacts
    grid_x = np.linspace(0.1 * W, 0.9 * W, K)
    grid_y = np.linspace(0.1 * H, 0.9 * H, K)

    dx = rng.randn(K, K) * W * scale
    dy = rng.randn(K, K) * H * scale

    # Giảm biên độ tại 4 cạnh biên để hạn chế border warp
    dx[[0, -1], :] *= 0.25
    dx[:, [0, -1]] *= 0.25
    dy[[0, -1], :] *= 0.25
    dy[:, [0, -1]] *= 0.25

    spline_x = RectBivariateSpline(grid_y, grid_x, dx)
    spline_y = RectBivariateSpline(grid_y, grid_x, dy)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx_dense = spline_x(yy.ravel(), xx.ravel(), grid=False).reshape(H, W).astype(np.float32)
    dy_dense = spline_y(yy.ravel(), xx.ravel(), grid=False).reshape(H, W).astype(np.float32)

    map_x = xx + dx_dense
    map_y = yy + dy_dense

    # FIX #5: clamp về phạm vi hợp lệ — tránh black borders
    map_x = np.clip(map_x, 0, W - 1)
    map_y = np.clip(map_y, 0, H - 1)

    distorted = cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    # FIX #3: lưu đủ thông tin để tính GT inverse warp trong Phase 1
    params = {
        "seed":     seed,
        "severity": severity,
        "scale":    float(scale),
        "H": H, "W": W,
        "K": K,
        "grid_x":   grid_x.tolist(),
        "grid_y":   grid_y.tolist(),
        "dx":       dx.tolist(),   # K×K control displacements
        "dy":       dy.tolist(),
    }
    return distorted, params


# ---------------------------------------------------------------------------
# Seed helper
# ---------------------------------------------------------------------------
def _file_seed(rel_path: Path, severity: str) -> int:
    """Deterministic, unique seed từ filepath + severity."""
    key = str(rel_path) + "|" + severity
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2 ** 31 - 1)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def build_mvtec_geo(
    mvtec_dir:       str,
    output_base_dir: str,
    severities:      List[str] = ("mild", "moderate", "severe"),
    include_l0:      bool = True,
    save_warp_params: bool = True,
) -> None:
    """
    Dựng MVTec-Geo từ MVTec AD gốc.

    Cấu trúc output:
        output_base/
          mvtec_l0/           ← bản sao MVTec AD gốc (nếu include_l0=True)
          mvtec_mild/
          mvtec_moderate/
          mvtec_severe/
            {category}/
              train/good/
              test/{defect}/
              ground_truth/{defect}/        ← mask gốc (cho GeoNorm eval)
              ground_truth_dist/{defect}/   ← mask distorted (cho baseline eval)
              warp_params/…/*.json          ← GT warp cho Phase 1 training
    """
    mvtec_path  = Path(mvtec_dir)
    output_base = Path(output_base_dir)

    if not mvtec_path.exists():
        raise FileNotFoundError(f"MVTec root không tồn tại: {mvtec_dir}")

    categories = sorted(d.name for d in mvtec_path.iterdir() if d.is_dir())
    print(f"  Tìm thấy {len(categories)} categories: {categories}\n")

    # ── L0: bản gốc ────────────────────────────────────────────────────────
    if include_l0:
        _copy_l0(mvtec_path, output_base / "mvtec_l0", categories)

    # ── L1-L3: distorted severities ────────────────────────────────────────
    for severity in severities:
        out_sev = output_base / f"mvtec_{severity}"
        print(f"\n{'─'*56}")
        print(f"  Severity: {severity.upper()}")
        print(f"{'─'*56}")

        for category in tqdm(categories, desc=f"[{severity}]"):
            src_cat = mvtec_path / category

            all_files = [
                p for p in src_cat.rglob("*")
                if p.is_file() and p.suffix.lower() in IMG_EXTS
            ]

            for src_path in all_files:
                rel = src_path.relative_to(mvtec_path)
                is_mask = "ground_truth" in rel.parts

                img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue

                if is_mask:
                    _process_mask(img, rel, src_path, out_sev, severity)
                else:
                    _process_image(img, rel, out_sev, severity, save_warp_params)

    _print_summary(output_base, list(severities))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _process_image(
    img:            np.ndarray,
    rel:            Path,
    out_sev:        Path,
    severity:       str,
    save_warp_params: bool,
) -> None:
    # FIX #1: unique seed per (file, severity)
    seed = _file_seed(rel, severity)
    distorted, params = generate_tps_distortion(img, severity=severity, seed=seed)

    dst = out_sev / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), distorted)

    # FIX #3: lưu warp params
    if save_warp_params:
        p = out_sev / "warp_params" / rel.with_suffix(".json")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(params, f)


def _process_mask(
    img:      np.ndarray,
    rel:      Path,
    src_path: Path,
    out_sev:  Path,
    severity: str,
) -> None:
    # Mask gốc (cho evaluation sau khi GeoNorm back-project)
    dst_orig = out_sev / rel
    dst_orig.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_orig), img)

    # FIX #4: mask distorted với SAME seed như ảnh test tương ứng
    # Quy ước MVTec: ground_truth/crack/000_mask.png ↔ test/crack/000.png
    test_rel_str = str(rel).replace("ground_truth", "test")
    # Bỏ hậu tố _mask nếu có
    for suffix in ("_mask.png", "_mask.PNG"):
        if test_rel_str.endswith(suffix):
            test_rel_str = test_rel_str[: -len(suffix)] + ".png"
    test_rel = Path(test_rel_str)

    seed = _file_seed(test_rel, severity)
    distorted_mask, _ = generate_tps_distortion(img, severity=severity, seed=seed)

    dst_dist = out_sev / Path(str(rel).replace("ground_truth", "ground_truth_dist"))
    dst_dist.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_dist), distorted_mask)


def _copy_l0(mvtec_path: Path, dst_root: Path, categories: list) -> None:
    print("  Copying L0 (original MVTec AD)…")
    for cat in tqdm(categories, desc="[l0]"):
        dst = dst_root / cat
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(mvtec_path / cat, dst)


def _print_summary(output_base: Path, severities: list) -> None:
    print("\n  Dataset summary:")
    for tag in ["l0"] + severities:
        d = output_base / f"mvtec_{tag}"
        if d.exists():
            n = sum(1 for _ in d.rglob("*.png"))
            print(f"    mvtec_{tag}/  →  {n:,} PNG files")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build MVTec-Geo dataset")
    parser.add_argument("--mvtec_dir",   required=True, help="Path to MVTec AD root")
    parser.add_argument("--output_dir",  required=True, help="Output directory")
    parser.add_argument("--severities",  nargs="+", default=["mild", "moderate", "severe"])
    parser.add_argument("--no_l0",           action="store_true")
    parser.add_argument("--no_warp_params",  action="store_true")
    args = parser.parse_args()

    build_mvtec_geo(
        mvtec_dir=args.mvtec_dir,
        output_base_dir=args.output_dir,
        severities=args.severities,
        include_l0=not args.no_l0,
        save_warp_params=not args.no_warp_params,
    )

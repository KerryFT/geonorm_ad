"""
models/geonorm_ad.py
Full GeoNorm-AD pipeline: GLN → CW-TPS → Grid Sampler → [AD Head] → InvTPS.

Design goal: AD head là plug-and-play — không biết về geometric normalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

from .gln    import GLN, canonical_grid
from .cw_tps import ConfidenceWeightedTPS
from .inv_tps import InverseTPS


class GeoNormAD(nn.Module):
    """
    End-to-end GeoNorm-AD pipeline.

    Modes
    -----
    training   : chỉ chạy GLN + CW-TPS + Grid Sampler, trả về rectified image.
                 AD head được train riêng (phase 3).
    inference  : full pipeline bao gồm AD head + InvTPS back-projection.

    Parameters
    ----------
    K        : số control points (phải là số chính phương, e.g. 16=4²)
    backbone : tên backbone cho GLN
    lam_tps  : λ trong CW-TPS regularization
    inv_iters: số iterations cho inverse TPS
    """

    def __init__(
        self,
        K:         int   = 16,
        backbone:  str   = "mobilevit_xs",
        lam_tps:   float = 0.1,
        inv_iters: int   = 10,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.K = K

        self.gln    = GLN(K=K, backbone=backbone, pretrained=pretrained)
        self.cw_tps = ConfidenceWeightedTPS(lam=lam_tps)
        self.inv_tps = InverseTPS(iters=inv_iters)

        # AD head được gắn sau (plug-and-play)
        self.ad_head: Optional[nn.Module] = None

    # ── public ──────────────────────────────────────────────────────────────
    def set_ad_head(self, ad_head: nn.Module) -> None:
        """Gắn AD head (e.g. PatchCore) vào pipeline."""
        self.ad_head = ad_head

    def rectify(self, x_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Chỉ chạy geometric normalization (dùng cho Phase 1+2 training và
        build memory bank).

        Parameters
        ----------
        x_d : [B, 3, H, W]  distorted image (ImageNet normalized)

        Returns
        -------
        x_rect  : [B, 3, H, W]  rectified image
        flow    : [B, H, W, 2]  forward flow field (canonical → distorted)
        confs   : [B, K]        per-point confidence scores
        """
        B, _, H, W = x_d.shape
        device = x_d.device

        # 1. Predict distorted positions of canonical control points
        pred_dist_pts, confs = self.gln(x_d)        # [B,K,2], [B,K]

        # 2. Fixed canonical grid (same for all images)
        c_grid = canonical_grid(self.K, device)      # [K, 2]
        c_grid = c_grid.unsqueeze(0).expand(B, -1, -1)  # [B, K, 2]

        # 3. CW-TPS: solve canonical → distorted flow
        flow = self.cw_tps(
            from_pts=c_grid,
            to_pts=pred_dist_pts,
            confs=confs,
            out_size=(H, W),
        )                                            # [B, H, W, 2]

        # 4. Grid sample: rectify I_d
        x_rect = F.grid_sample(
            x_d, flow,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        return x_rect, flow, confs

    def forward(self, x_d: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full inference pipeline (cần ad_head đã được set).

        Returns dict với:
            'score_orig'   : [B, H, W]    anomaly score map in original coords
            'score_rect'   : [B, H, W]    anomaly score map in rectified coords
            'x_rect'       : [B, 3, H, W] rectified image
            'flow'         : [B, H, W, 2] forward flow
            'confs'        : [B, K]       control point confidence
        """
        if self.ad_head is None:
            raise RuntimeError("AD head chưa được set. Gọi model.set_ad_head(ad_head) trước.")

        x_rect, flow, confs = self.rectify(x_d)

        # AD head trên rectified image
        score_rect = self.ad_head(x_rect)             # [B, H, W]

        # Inverse TPS: back-project về original coords
        score_orig = self.inv_tps(
            score_rect.unsqueeze(1), flow
        ).squeeze(1)                                   # [B, H, W]

        return {
            "score_orig": score_orig,
            "score_rect": score_rect,
            "x_rect":     x_rect,
            "flow":       flow,
            "confs":      confs,
        }

    def get_gln_params(self):
        """Trả về parameters của GLN (cho Phase 1 optimizer)."""
        return self.gln.parameters()

    def freeze_gln(self) -> None:
        """Đóng băng GLN + CW-TPS sau Phase 1+2 (để train AD head)."""
        for p in self.gln.parameters():
            p.requires_grad_(False)

    def unfreeze_gln(self) -> None:
        for p in self.gln.parameters():
            p.requires_grad_(True)

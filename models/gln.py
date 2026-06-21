"""
models/gln.py
Geometric Localisation Network (GLN).
Dự đoán K control points và confidence scores từ ảnh distorted.

Architecture: MobileViT-XS (hybrid CNN + Transformer) từ timm.
- CNN layers: local edge/texture features
- Transformer blocks: long-range geometric dependencies
→ Đây là lý do justify "Transformer" trong tên dự án.
"""

import torch
import torch.nn as nn
from typing import Tuple

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False


class GLN(nn.Module):
    """
    Geometric Localisation Network.

    Input  : I_d  [B, 3, H, W]  ảnh distorted (normalized ImageNet)
    Output :
        coords  [B, K, 2]  toạ độ K control points trong [0, 1]²
        confs   [B, K]     confidence score ∈ (0, 1)  (sigmoid)
    """

    BACKBONE_FEAT_DIM = {
        "mobilevit_xs":   384,
        "mobilevit_s":    640,
        "efficientnet_b0": 1280,
        "resnet18":       512,
    }

    def __init__(
        self,
        K:        int   = 16,
        backbone: str   = "mobilevit_xs",
        pretrained: bool = True,
        dropout:  float = 0.1,
    ) -> None:
        super().__init__()
        self.K        = K
        self.backbone_name = backbone

        feat_dim = self._build_backbone(backbone, pretrained)

        # Head chung
        self.neck = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Control point coordinates (normalized [0, 1]²)
        self.coord_head = nn.Sequential(
            nn.Linear(256, K * 2),
            nn.Sigmoid(),           # clamp tự động về [0,1]
        )

        # Per-point confidence (không dùng Sigmoid ở đây — dùng khi call)
        self.conf_head = nn.Sequential(
            nn.Linear(256, K),
            nn.Sigmoid(),
        )

        self._init_weights()

    # ── public ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        coords : [B, K, 2]  — predicted distorted positions ∈ [0, 1]²
        confs  : [B, K]     — confidence scores ∈ (0, 1)
        """
        feat   = self._extract(x)          # [B, feat_dim]
        neck   = self.neck(feat)            # [B, 256]
        coords = self.coord_head(neck).view(-1, self.K, 2)   # [B, K, 2]
        confs  = self.conf_head(neck)                         # [B, K]
        return coords, confs

    # ── private ─────────────────────────────────────────────────────────────
    def _build_backbone(self, name: str, pretrained: bool) -> int:
        if not _TIMM_AVAILABLE:
            raise ImportError("timm is required: pip install timm")

        # Tạo backbone, bỏ classification head
        self.backbone = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,      # bỏ classifier
            global_pool="avg",  # global average pool → 1D vector
        )
        # Lấy feat dim từ bảng hoặc infer tự động
        feat_dim = self.BACKBONE_FEAT_DIM.get(name)
        if feat_dim is None:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                out   = self.backbone(dummy)
            feat_dim = out.shape[-1]
        return feat_dim

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        """Global feature vector [B, feat_dim]."""
        return self.backbone(x)

    def _init_weights(self) -> None:
        """Khởi tạo heads gần identity transform."""
        # coord_head: init về 0.5 (canonical center)
        nn.init.zeros_(self.coord_head[0].weight)
        # bias = 0 → sigmoid(0) = 0.5 → control pts bắt đầu ở giữa
        nn.init.zeros_(self.coord_head[0].bias)

        # conf_head: init về moderate confidence
        nn.init.zeros_(self.conf_head[0].weight)
        # sigmoid(1.0) ≈ 0.73 — bắt đầu với confidence khá cao
        nn.init.constant_(self.conf_head[0].bias, 1.0)


# ---------------------------------------------------------------------------
# Canonical grid helper (dùng chung toàn dự án)
# ---------------------------------------------------------------------------
def canonical_grid(K: int = 16, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Fixed K-point canonical control grid trong [0.1, 0.9]² (inset từ biên).

    Returns
    -------
    pts : [K, 2]  — (x, y) normalized coordinates
    """
    K_side = int(K ** 0.5)
    assert K_side * K_side == K, f"K phải là số chính phương, nhận được K={K}"
    lin    = torch.linspace(0.1, 0.9, K_side, device=device)
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")
    return torch.stack([gx.flatten(), gy.flatten()], dim=1)  # [K, 2]

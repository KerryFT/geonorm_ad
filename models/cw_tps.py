"""
models/cw_tps.py
Confidence-Weighted TPS Layer — đóng góp toán học chính của GeoNorm-AD.

Standard TPS:
    E(f) = Σᵢ ‖f(pᵢ) - qᵢ‖² + λ·J(f)
    Giải: W = (K + λI)⁻¹ · Y

Confidence-Weighted TPS (novel):
    E_c(f) = Σᵢ cᵢ·‖f(pᵢ) - qᵢ‖² + λ·J(f)
    Giải: W = (C·K + λ·I_c)⁻¹ · (C·Y)

    với:
        C   = diag(c₁,...,c_K)      — confidence diagonal
        I_c = diag(1-c₁,...,1-c_K) — extra regularization tại uncertain pts

    Khi cᵢ → 1: hành xử như standard TPS.
    Khi cᵢ → 0: W_i → 0, point i effectively bị bỏ qua.
    Gradient flows qua cᵢ → mạng tự học khi nào nên uncertain.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConfidenceWeightedTPS(nn.Module):
    """
    Differentiable Confidence-Weighted TPS layer.

    Pipeline:
        (from_pts, to_pts, confs) → flow field [B, H, W, 2]  ([-1,1] range)

    from_pts : canonical control point positions   [B, K, 2]  ∈ [0,1]
    to_pts   : predicted distorted positions       [B, K, 2]  ∈ [0,1]
    confs    : per-point confidence scores         [B, K]     ∈ (0,1)

    TPS maps from_pts → to_pts.
    Flow is evaluated at every pixel of the dense canonical grid.
    Output flow được dùng với F.grid_sample(I_d, flow) để rectify ảnh.
    """

    def __init__(self, lam: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        self.lam = lam
        self.eps = eps

    # ── public ──────────────────────────────────────────────────────────────
    def forward(
        self,
        from_pts:  torch.Tensor,   # [B, K, 2]  canonical positions
        to_pts:    torch.Tensor,   # [B, K, 2]  distorted positions (GLN output)
        confs:     torch.Tensor,   # [B, K]
        out_size:  Tuple[int, int],  # (H, W)
    ) -> torch.Tensor:             # [B, H, W, 2]  flow in [-1,1]
        """
        Returns
        -------
        flow : [B, H, W, 2]  flow field cho F.grid_sample ([-1,1] range).
        """
        B, K, _ = from_pts.shape
        device   = from_pts.device

        # 1. TPS kernel matrix K_mat [B, K, K]
        K_mat = self._tps_kernel(from_pts)

        # 2. Confidence matrices
        C   = torch.diag_embed(confs)           # [B, K, K]
        I_c = torch.diag_embed(1.0 - confs)    # [B, K, K]

        # 3. System: A·W = b
        #    A = C·K_mat + λ·I_c  (confidence-weighted kernel)
        #    b = C·to_pts
        eye = torch.eye(K, device=device).unsqueeze(0)          # [1, K, K]
        A   = C @ K_mat + self.lam * I_c + self.eps * eye       # [B, K, K]
        b   = C @ to_pts                                          # [B, K, 2]

        # 4. Solve W = A⁻¹·b  (gradient flows qua đây bằng implicit diff)
        W = torch.linalg.solve(A, b)   # [B, K, 2]

        # 5. Evaluate TPS tại dense canonical grid
        H, W_out = out_size
        flow = self._evaluate_dense(from_pts, W, H, W_out)   # [B, H, W, 2]

        return flow

    # ── private ─────────────────────────────────────────────────────────────
    @staticmethod
    def _tps_kernel(pts: torch.Tensor) -> torch.Tensor:
        """
        Tính TPS kernel matrix.
        U(r) = r² · log(r² + ε)  — TPS radial basis function.

        pts : [B, K, 2]
        Returns K_mat : [B, K, K]
        """
        diff = pts.unsqueeze(2) - pts.unsqueeze(1)   # [B, K, K, 2]
        r2   = (diff ** 2).sum(-1)                    # [B, K, K]
        return r2 * torch.log(r2 + 1e-6)

    def _evaluate_dense(
        self,
        ctrl_pts: torch.Tensor,   # [B, K, 2]  — canonical control pts
        W:        torch.Tensor,   # [B, K, 2]  — TPS weights
        H: int,
        W_out: int,
    ) -> torch.Tensor:            # [B, H, W_out, 2]  in [-1,1]
        """
        Evaluate f(q) = Σ_k W_k · U(‖q - p_k‖)  tại mọi pixel canonical.
        """
        B, K, _ = ctrl_pts.shape
        device   = ctrl_pts.device

        # Dense canonical grid ∈ [0,1]²
        gy = torch.linspace(0.0, 1.0, H,     device=device)
        gx = torch.linspace(0.0, 1.0, W_out, device=device)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
        # [H, W, 2] → [B, N, 2]  với N = H×W
        grid = torch.stack([grid_x, grid_y], dim=-1)
        pts_flat = grid.reshape(1, H * W_out, 2).expand(B, -1, -1)  # [B, N, 2]

        # Kernel evaluation tại N điểm, K control points
        diff = pts_flat.unsqueeze(2) - ctrl_pts.unsqueeze(1)  # [B, N, K, 2]
        r2   = (diff ** 2).sum(-1)                              # [B, N, K]
        kern = r2 * torch.log(r2 + 1e-6)                       # [B, N, K]

        # Deformation delta = kern · W  [B, N, 2]
        delta = kern @ W    # bmm: [B, N, K] × [B, K, 2] = [B, N, 2]

        # Flow = canonical position + deformation  (→ distorted sampling coords)
        flow_01 = (pts_flat + delta).reshape(B, H, W_out, 2)   # [B, H, W, 2] ∈ [0,1] approx

        # Chuyển [0,1] → [-1,1] cho F.grid_sample
        flow = flow_01 * 2.0 - 1.0
        return flow

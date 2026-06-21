"""
models/inv_tps.py
Inverse TPS Back-Projection — map anomaly scores từ rectified space về original.

Sau khi detect anomaly trên I' (rectified), anomaly map A' ở coordinate space
của I'. Cần project ngược về I_d để deployment (highlight defect trên ảnh gốc).

Method: fixed-point iteration để invert flow field.
    Φ⁻¹(y) ≈ fixed-point của  g(x) = y - Φ(x) + x
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InverseTPS(nn.Module):
    """
    Invert a dense flow field bằng fixed-point iteration.

    forward_flow : [B, H, W, 2]  flow từ CW-TPS (canonical → distorted), [-1,1]
    score_map    : [B, 1, H, W]  anomaly score map từ AD head (trong rectified space)

    Returns      : [B, 1, H, W]  anomaly scores trong original distorted space
    """

    def __init__(self, iters: int = 10) -> None:
        super().__init__()
        self.iters = iters

    def forward(
        self,
        score_map:    torch.Tensor,   # [B, 1, H, W]
        forward_flow: torch.Tensor,   # [B, H, W, 2]  in [-1,1]
    ) -> torch.Tensor:                # [B, 1, H, W]
        inv_flow = self._invert_flow(forward_flow)
        return F.grid_sample(
            score_map, inv_flow,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    # ── private ─────────────────────────────────────────────────────────────
    def _invert_flow(self, flow: torch.Tensor) -> torch.Tensor:
        """
        Numerical inversion qua fixed-point iteration.

        Bài toán: tìm Φ⁻¹ sao cho Φ(Φ⁻¹(y)) = y.
        Update:  x_{t+1} = y - Φ(x_t) + x_t

        Hội tụ tốt khi max displacement < ~25% image size.
        """
        B, H, W, _ = flow.shape
        device      = flow.device

        # Identity grid trong [-1,1]
        identity = self._identity_grid(B, H, W, device)   # [B, H, W, 2]

        # Bắt đầu từ identity (assumption: Φ ≈ identity khi không distort nhiều)
        inv = identity.clone()

        for _ in range(self.iters):
            # Φ(current estimate)
            flow_perm = flow.permute(0, 3, 1, 2)          # [B, 2, H, W]
            composed  = F.grid_sample(
                flow_perm, inv,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ).permute(0, 2, 3, 1)                          # [B, H, W, 2]

            # Update: x_{t+1} = identity - (Φ(x_t) - identity)
            inv = 2 * identity - composed

        return inv

    @staticmethod
    def _identity_grid(B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Identity sampling grid trong [-1,1]."""
        gy = torch.linspace(-1, 1, H, device=device)
        gx = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)       # [H, W, 2]
        return grid.unsqueeze(0).expand(B, -1, -1, -1)     # [B, H, W, 2]

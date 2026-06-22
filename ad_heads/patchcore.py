"""
ad_heads/patchcore.py
PatchCore (Roth et al., CVPR 2022) — fixed version.

Fix so với phiên bản cũ:
  Bug 1: Thiếu L2 normalization → distances không có nghĩa
  Bug 2: Dùng mean(k-nearest) thay vì min(1-nearest) → AUROC thấp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from typing import Optional
from tqdm import tqdm


class PatchCore(nn.Module):
    """
    PatchCore anomaly scoring — plug-and-play AD head.

    Usage
    -----
    pc = PatchCore().cuda()
    pc.fit(train_loader)          # ảnh trong loader đã qua GeoNorm nếu cần
    score_map = pc(test_image)    # [B, H, W]
    """

    def __init__(
        self,
        coreset_ratio: float = 0.1,
        output_size:   int   = 224,
    ) -> None:
        super().__init__()
        self.coreset_ratio = coreset_ratio
        self.output_size   = output_size
        self._build_extractor()
        self.memory_bank: Optional[torch.Tensor] = None  # [N, C] L2-normalized

    # ── public ──────────────────────────────────────────────────────────────
    def fit(self, train_loader, device: str = "cuda") -> None:
        """Build memory bank từ training images."""
        self.eval()
        all_feats = []
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="  Building memory bank"):
                imgs = batch["image"].to(device) if isinstance(batch, dict) else batch[0].to(device)
                f    = self._extract(imgs)              # [B*N_p, C] normalized
                all_feats.append(f.cpu())

        all_feats = torch.cat(all_feats, dim=0)         # [N_total, C]
        self.memory_bank = self._coreset(all_feats)
        print(f"  Memory bank: {self.memory_bank.shape[0]:,} × {self.memory_bank.shape[1]} (L2-normalized)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, 3, H, W]
        Returns score_map : [B, H, W]  (higher = more anomalous)
        """
        if self.memory_bank is None:
            raise RuntimeError("Chưa build memory bank. Gọi fit() trước.")

        B      = x.shape[0]
        device = x.device
        mem    = self.memory_bank.to(device)             # [M, C]

        with torch.no_grad():
            patch_feats = self._extract(x)               # [B*N_p, C] normalized

        H_p = W_p = self._patch_hw
        N_p = H_p * W_p

        # Anomaly score = min L2 distance tới memory bank (1-NN)
        scores_flat = self._min_dist(patch_feats, mem)   # [B*N_p]
        scores = scores_flat.reshape(B, 1, H_p, W_p)    # [B, 1, Hp, Wp]

        # Upsample về output_size
        return F.interpolate(
            scores, size=(self.output_size, self.output_size),
            mode="bilinear", align_corners=False,
        ).squeeze(1)                                      # [B, H, W]

    # ── private ─────────────────────────────────────────────────────────────
    def _build_extractor(self) -> None:
        bb = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        bb.eval()
        for p in bb.parameters():
            p.requires_grad_(False)

        # Chỉ giữ lại các block cần thiết
        self.stem   = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1
        self.layer2 = bb.layer2   # → [B, 512,  28, 28] cho input 224px
        self.layer3 = bb.layer3   # → [B, 1024, 14, 14]
        self._patch_hw = None     # set sau lần extract đầu tiên

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward qua backbone, lấy layer2 + layer3, upsample + concat + L2-norm.

        Returns : [B*N_patches, 1536]  L2-normalized
        """
        with torch.no_grad():
            s  = self.stem(x)
            l1 = self.layer1(s)
            l2 = self.layer2(l1)                        # [B, 512,  h, w]
            l3 = self.layer3(l2)                        # [B, 1024, h/2, w/2]

        # Upsample layer3 → cùng spatial với layer2
        l3_up = F.interpolate(l3, l2.shape[-2:], mode="bilinear", align_corners=False)

        # Concat → [B, 1536, h, w]
        feat = torch.cat([l2, l3_up], dim=1)

        # ── Bug 1 fix: L2 normalize theo channel dim ──────────────────────
        feat = F.normalize(feat, dim=1)

        B, C, H_p, W_p = feat.shape
        self._patch_hw = H_p   # lưu để dùng trong forward

        # Reshape → [B*N_patches, C]
        return feat.permute(0, 2, 3, 1).reshape(B * H_p * W_p, C)

    def _min_dist(self, query: torch.Tensor, bank: torch.Tensor,
                  chunk: int = 2048) -> torch.Tensor:
        """
        Minimum L2 distance từ mỗi query patch đến memory bank.

        ── Bug 2 fix: min distance (1-NN), không phải mean(k-NN) ──────────

        Vì features đã L2-normalized, ||q - m||² = 2 - 2·qᵀm
        → có thể dùng dot product để tối ưu tốc độ.
        """
        scores = []
        for i in range(0, len(query), chunk):
            q   = query[i:i + chunk]                    # [chunk, C]
            # ||q-m||² = 2 - 2·qᵀm  (vì ||q||=||m||=1)
            sim = q @ bank.T                             # [chunk, M]  cosine sim
            d2  = 2.0 - 2.0 * sim                       # [chunk, M]  L2² / 2
            scores.append(d2.min(dim=1).values)          # [chunk]  min dist²
        return torch.cat(scores, dim=0).sqrt()           # [N]  min L2 dist

    def _coreset(self, feats: torch.Tensor) -> torch.Tensor:
        """Greedy coreset subsampling."""
        n_keep = max(1, int(len(feats) * self.coreset_ratio))
        if n_keep >= len(feats):
            return feats

        print(f"  Coreset: {len(feats):,} → {n_keep:,} vectors…")
        selected = [torch.randint(len(feats), (1,)).item()]
        sel_t    = feats[selected]                        # [k, C]

        for _ in tqdm(range(n_keep - 1), desc="  Coreset", leave=False):
            # Khoảng cách tới nearest selected
            sim  = feats @ sel_t.T                        # [N, k]
            d2   = (2.0 - 2.0 * sim).min(dim=1).values   # [N]
            next_idx = d2.argmax().item()
            selected.append(next_idx)
            sel_t = feats[selected]

        return feats[selected]

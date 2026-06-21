"""
ad_heads/patchcore.py
PatchCore anomaly detection head (Roth et al., CVPR 2022).

Đây là plug-and-play AD head — không biết về geometric normalization upstream.

Key steps:
  1. WideResNet50-2 → extract layer2 + layer3 features
  2. Adaptive average pool → patch-level descriptors
  3. Greedy coreset subsampling → reduce memory bank size
  4. KNN distance in memory bank → anomaly score map
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from typing import Optional
from tqdm import tqdm
from sklearn.random_projection import SparseRandomProjection

class PatchCore(nn.Module):
    """
    PatchCore anomaly scoring.

    Usage
    -----
    # Build memory bank trên training set
    pc = PatchCore(coreset_ratio=0.1).cuda()
    pc.fit(train_loader)   # ← ảnh trong train_loader phải đã qua GeoNorm

    # Score test image
    score_map = pc(test_image)   # [B, H, W]
    """

    def __init__(
        self,
        coreset_ratio: float = 0.1,    # tỷ lệ giữ lại sau subsampling
        k_nearest:     int   = 5,      # k trong KNN scoring
        patch_size:    int   = 3,      # pooling kernel size
        output_size:   int   = 224,    # upsample score map về image size
    ) -> None:
        super().__init__()
        self.coreset_ratio = coreset_ratio
        self.k_nearest     = k_nearest
        self.patch_size    = patch_size
        self.output_size   = output_size
        self.projector = SparseRandomProjection(
        n_components='auto',
        eps=0.9
        )

        # Feature extractor (frozen)
        self._build_extractor()

        # Memory bank (được set sau khi gọi fit())
        self.memory_bank: Optional[torch.Tensor] = None   # [N, C]

    # ── public ──────────────────────────────────────────────────────────────
    def fit(self, train_loader: torch.utils.data.DataLoader, device: str = "cuda") -> None:
        """Build memory bank từ training images (đã qua GeoNorm nếu dùng GeoNorm-AD)."""
        self.eval()
        all_feats = []

        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Building memory bank"):
                imgs = batch["image"].to(device) if isinstance(batch, dict) else batch[0].to(device)
                feats = self._extract_patch_features(imgs)   # [B, N_patches, C]
                all_feats.append(feats.reshape(-1, feats.shape[-1]))

        selected_idx = self._coreset_sample(all_feats)

        self.memory_bank = all_feats[selected_idx].cpu()
        print(f"  Memory bank: {self.memory_bank.shape[0]:,} vectors × {self.memory_bank.shape[1]} dims")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tính anomaly score map.

        Parameters
        ----------
        x : [B, 3, H, W]  ảnh đã normalized (và đã rectify nếu dùng GeoNorm)

        Returns
        -------
        score_map : [B, H, W]  anomaly score (higher = more anomalous)
        """
        if self.memory_bank is None:
            raise RuntimeError("Chưa build memory bank. Gọi fit() trước.")

        B = x.shape[0]
        device = x.device
        mem = self.memory_bank.to(device)

        with torch.no_grad():
            # [B, N_patches, C]
            patch_feats = self._extract_patch_features(x)

        # Spatial resolution của patch features
        _, H_p, W_p, C = self._last_spatial_shape
        patch_feats_flat = patch_feats.reshape(B * H_p * W_p, C)   # [B*N, C]

        # KNN distances tới memory bank
        # Chia nhỏ để tránh OOM (B*N_patches có thể lớn)
        scores_flat = self._batch_knn(patch_feats_flat, mem)        # [B*N]
        scores = scores_flat.reshape(B, 1, H_p, W_p)               # [B, 1, Hp, Wp]

        # Upsample về output_size
        score_map = F.interpolate(
            scores, size=(self.output_size, self.output_size),
            mode="bilinear", align_corners=False,
        ).squeeze(1)   # [B, H, W]

        return score_map

    # ── private ─────────────────────────────────────────────────────────────
    def _build_extractor(self) -> None:
        """WideResNet50-2 pretrained, frozen. Extract layer2 + layer3."""
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)

        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2   # output: [B, 512, H/8, W/8]
        self.layer3 = backbone.layer3   # output: [B, 1024, H/16, W/16]

        # Adaptive pool để tạo patch descriptors
        self.avg_pool = nn.AvgPool2d(kernel_size=self.patch_size, stride=1, padding=self.patch_size // 2)

        # Channel dims
        self._feat_dim: Optional[int] = None
        self._last_spatial_shape = None

    def _extract_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract multi-scale patch features.

        Returns : [B, H_p*W_p, C]  (C = 512 + 1024 = 1536 sau concat+pool)
        """
        with torch.no_grad():
            x0 = self.layer0(x)
            x1 = self.layer1(x0)
            x2 = self.layer2(x1)   # [B, 512, h2, w2]
            x3 = self.layer3(x2)   # [B, 1024, h3, w3]

        # Upsample layer3 về cùng spatial với layer2
        h2, w2 = x2.shape[-2:]
        x3_up = F.interpolate(x3, size=(h2, w2), mode="bilinear", align_corners=False)

        # Concat và average pool
        concat = torch.cat([x2, x3_up], dim=1)          # [B, 1536, h2, w2]
        feats  = self.avg_pool(concat)                   # [B, 1536, h2, w2]

        B, C, H_p, W_p = feats.shape
        self._last_spatial_shape = (B, H_p, W_p, C)

        # Reshape thành [B, N_patches, C]
        return feats.permute(0, 2, 3, 1).reshape(B, H_p * W_p, C)

    def _coreset_sample(
        self,
        feats: torch.Tensor
    ):

        N = len(feats)

        n_keep = max(
            1,
            int(
                N *
                self.coreset_ratio
            )
        )

        if n_keep >= N:

            return torch.arange(
                N,
                device=feats.device
            )


        print(
            f"  Coreset sampling:"
            f" {N:,}"
            f" → {n_keep:,} vectors..."
        )

        device = feats.device


        # ===============================
        # Random Projection
        # ===============================

        feats_np = feats.cpu().numpy()

        feats_proj = self.projector.fit_transform(
            feats_np
        )

        feats_proj = torch.from_numpy(
            feats_proj
        ).float().to(device)



        # ===============================
        # Approx Greedy Coreset
        # ===============================

        selected = []

        first_idx = torch.randint(
            N,
            (1,),
            device=device
        ).item()

        selected.append(first_idx)


        min_dist = torch.cdist(

            feats_proj,

            feats_proj[
                first_idx:
                first_idx+1
            ]

        ).squeeze(1)



        for _ in tqdm(

            range(n_keep - 1),

            desc="  Coreset",

            leave=False

        ):

            next_idx = min_dist.argmax().item()

            selected.append(next_idx)


            dist_new = torch.cdist(

                feats_proj,

                feats_proj[
                    next_idx:
                    next_idx+1
                ]

            ).squeeze(1)


            min_dist = torch.minimum(

                min_dist,

                dist_new

            )


        return torch.tensor(

            selected,

            device=device,

            dtype=torch.long

        )

    def _batch_knn(
        self,
        query:  torch.Tensor,   # [N, C]
        bank:   torch.Tensor,   # [M, C]
        chunk:  int = 1024,
    ) -> torch.Tensor:          # [N]
        """KNN distances chia nhỏ để tránh OOM."""
        scores = []
        for i in range(0, len(query), chunk):
            q   = query[i : i + chunk]                    # [chunk, C]
            d   = torch.cdist(q, bank)                    # [chunk, M]
            top = d.topk(self.k_nearest, dim=1, largest=False).values  # [chunk, k]
            scores.append(top.mean(dim=1))                # mean distance
        return torch.cat(scores, dim=0)

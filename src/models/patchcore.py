import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.neighbors import NearestNeighbors
import numpy as np

class PatchCore(nn.Module):
    def __init__(self, f_coreset=0.01):
        """
        f_coreset: Tỷ lệ lấy mẫu phụ. 
        Để 0.01 (1%) để tiết kiệm RAM tối đa khi chạy trên Jetson Nano ở Giai đoạn 3.
        """
        super().__init__()
        self.f_coreset = f_coreset
        
        # 1. Tải backbone WideResNet50 (Đã pre-train trên ImageNet)
        weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        self.backbone = models.wide_resnet50_2(weights=weights)
        self.backbone.eval() # Bắt buộc đóng băng
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.layer2_features = []
        self.layer3_features = []
        
        # 2. Gắn Hook để lấy output từ các lớp trung gian
        def hook_layer2(module, input, output):
            self.layer2_features.append(output)
        def hook_layer3(module, input, output):
            self.layer3_features.append(output)
            
        self.backbone.layer2.register_forward_hook(hook_layer2)
        self.backbone.layer3.register_forward_hook(hook_layer3)
        
        self.memory_bank = None
        # Thuật toán tìm kiếm láng giềng gần nhất
        self.knn = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=-1)

    def extract_features(self, x):
        """Trích xuất và gộp đặc trưng từ Layer 2 và Layer 3"""
        self.layer2_features.clear()
        self.layer3_features.clear()
        
        _ = self.backbone(x)
        
        # Làm mượt bằng Average Pooling
        feat2 = F.avg_pool2d(self.layer2_features[0], 3, 1, 1)
        feat3 = F.avg_pool2d(self.layer3_features[0], 3, 1, 1)
        
        # Phóng to feat3 cho bằng kích thước feat2 rồi nối lại
        feat3 = F.interpolate(feat3, size=feat2.shape[-2:], mode='bilinear', align_corners=False)
        features = torch.cat([feat2, feat3], dim=1) # Shape: [B, C, H, W]
        
        return features

    def fit(self, subsampled_features_list):
        """Xây dựng Memory Bank từ các list đã được ép cân (chỉ còn 1%)"""
        # Lúc này list đã rất nhẹ, ghép lại thoải mái không sợ tràn RAM
        features = torch.cat(subsampled_features_list, dim=0) 
        self.memory_bank = features.numpy()
        
        print(f"🧠 Đã nạp {self.memory_bank.shape[0]} vector đặc trưng vào Memory Bank.")
        self.knn.fit(self.memory_bank)

    def predict(self, features):
        """Đo lường khoảng cách để chấm điểm dị thường"""
        B, C, H, W = features.shape
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
        
        # Đo khoảng cách tới điểm gần nhất trong Memory Bank chuẩn
        distances, _ = self.knn.kneighbors(features_flat)
        distances = distances.reshape(B, H, W)
        
        # Điểm dị thường của ảnh là điểm có sai số (khoảng cách) lớn nhất trên bản đồ nhiệt
        image_scores = distances.max(axis=(1, 2))
        return image_scores
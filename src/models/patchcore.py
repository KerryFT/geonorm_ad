import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
class PatchCoreBaseline:
    """ Khung xử lý trích xuất đặc trưng vùng lân cận (Local patch embedding) """
    def __init__(self):
        # Sử dụng backbone chuẩn theo paper gốc
        resnet = models.wide_resnet50_2(weights='DEFAULT')
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:7]) # Lấy đến layer3
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.memory_bank = None

    def extract_features(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
        return features

    def fit_memory_bank(self, train_loader):
        """ Thu thập tất cả embedding của ảnh bình thường (Canonical pose) """
        embeddings = []
        for imgs, _ in train_loader:
            feats = self.extract_features(imgs)
            # Biến đổi cục bộ thông qua phép lấy trung bình trượt (Average Pooling) theo PatchCore tiêu chuẩn
            pooled_feats = F.avg_pool2d(feats, kernel_size=3, stride=1, padding=1)
            flat_feats = pooled_feats.permute(0, 2, 3, 1).reshape(-1, pooled_feats.size(1))
            embeddings.append(flat_feats)
        
        self.memory_bank = torch.cat(embeddings, dim=0)
        # Trong Phase A thực tế, bạn có thể lấy ngẫu nhiên 1% số lượng Vector để mô phỏng Coreset Sampling
        print(f"Khởi tạo Memory Bank thành công với kích thước: {self.memory_bank.shape}")

    def compute_anomaly_map(self, x):
        """ Tính khoảng cách tối thiểu từ pixel đặc trưng tới Memory Bank """
        feats = self.extract_features(x)
        pooled_feats = F.avg_pool2d(feats, kernel_size=3, stride=1, padding=1)
        B, C, H, W = pooled_feats.size()
        flat_feats = pooled_feats.permute(0, 2, 3, 1).reshape(-1, C)
        
        # Tính toán ma trận khoảng cách Euclid (Mô phỏng khoảng cách Anomaly)
        # Dùng toán tử linalg.norm để tối ưu tốc độ tính trên GPU
        distances = torch.cdist(flat_feats.unsqueeze(0), self.memory_bank.unsqueeze(0)).squeeze(0)
        min_distances, _ = torch.min(distances, dim=1)
        
        anomaly_map = min_distances.view(B, 1, H, W)
        # Nội suy song tuyến tính đưa bản đồ lỗi về bằng kích thước ảnh gốc 224x224
        anomaly_map = F.interpolate(anomaly_map, size=(224, 224), mode='bilinear', align_corners=True)
        return anomaly_map
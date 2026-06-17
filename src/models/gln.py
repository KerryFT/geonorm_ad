import torch
import torch.nn as nn
import timm

class GLN(nn.Module):
    """
    Geometric Localisation Network (GLN) sử dụng backbone MobileViT-XS.
    Nhiệm vụ: Trích xuất đặc trưng toàn cục và dự đoán tọa độ lưới cùng độ tự tin.
    """
    def __init__(self, K=16):
        super().__init__()
        self.K = K
        
        # 1. Gọi backbone MobileViT-XS từ thư viện timm
        # Tham số num_classes=0 và global_pool='' giúp lấy trực tiếp feature map thô
        self.backbone = timm.create_model('mobilevit_xs', pretrained=True, num_classes=0, global_pool='')
        
        # Đặc trưng đầu ra của mobilevit_xs thường có 384 channels
        in_features = 384 
        
        # 2. Nhánh 1: Regression dự đoán tọa độ K điểm kiểm soát (x, y)
        # Trong gln.py, chỗ định nghĩa coord_head
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, K * 2),
            nn.Tanh() # Ép buộc mọi output nằm gọn trong [-1, 1]
        )
        
        # 3. Nhánh 2: Classification dự đoán trọng số tự tin c_i
        # Đầu ra đi qua Sigmoid để ép giá trị về khoảng [0, 1]
        self.conf_head = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.GELU(),
            nn.Linear(in_features // 2, K),
            nn.Sigmoid()
        )
        
        # Khởi tạo trọng số an toàn cho nhánh tọa độ (Tránh grid bị xoắn ở epoch đầu)
        self._initialize_weights()

    def _initialize_weights(self):
        """ Ép đầu ra ban đầu của coord_head gần bằng 0 để mạng bắt đầu với biến đổi Identity """
        nn.init.zeros_(self.coord_head[-1].weight)
        nn.init.zeros_(self.coord_head[-1].bias)

    def forward(self, x):
        # x: [B, 3, H, W]
        feat = self.backbone(x)  # Trả về feature map kích thước [B, 384, H', W']
        
        # Global Average Pooling để thu về vector đặc trưng toàn cục
        feat = feat.mean(dim=[2, 3])  # [B, 384]
        
        # Đưa qua các nhánh phân loại
        coords = self.coord_head(feat).view(-1, self.K, 2)  # Output: [B, K, 2]
        confs = self.conf_head(feat)                        # Output: [B, K]
        
        return coords, confs

# --- Unit Test Nhanh ---
if __name__ == "__main__":
    print("Khởi tạo mạng GLN với K=16...")
    model = GLN(K=16)
    
    # Tạo một tensor ảnh ngẫu nhiên kích thước 224x224, batch_size=2
    dummy_input = torch.randn(2, 3, 224, 224)
    
    print("Tiến hành chạy thử (Forward Pass)...")
    coords, confs = model(dummy_input)
    
    print(f"Kích thước Tọa độ (Coords)  : {coords.shape}  | Kỳ vọng: [2, 16, 2]")
    print(f"Kích thước Tự tin (Confs)   : {confs.shape}     | Kỳ vọng: [2, 16]")
    
    # Kiểm tra tính hợp lệ của hàm Sigmoid
    assert (confs >= 0).all() and (confs <= 1).all(), "Lỗi: Confidence score vượt quá biên [0, 1]"
    print("✅ Unit Test cục bộ: Hoàn hảo!")
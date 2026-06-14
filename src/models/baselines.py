import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class AffineSTN(nn.Module):
    """ Config B: Dự đoán ma trận Affine 2x3 tiêu chuẩn để khử xoay/dịch """
    def __init__(self):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 52 * 52, 32), # Giả định kích thước đầu vào ảnh đã resize về 224x224
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Khởi tạo ma trận Identity ban đầu
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x_rectified = F.grid_sample(x, grid, align_corners=True)
        return x_rectified

class StandardTPSSTN(nn.Module):
    """ Config C: TPS-STN truyền thống của Jaderberg, không có trọng số tự tin c_i """
    def __init__(self, K=16):
        super().__init__()
        self.K = K  # Ví dụ lưới 4x4
        self.localization = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 53 * 53, 128),
            nn.ReLU(True),
            nn.Linear(128, K * 2)
        )
        # Điểm kiểm soát đích cố định (Target Canonical Grid)
        grid_size = int(np.sqrt(K))
        r = np.linspace(-1, 1, grid_size)
        xs, ys = np.meshgrid(r, r)
        tgt_pts = np.stack([xs.flatten(), ys.flatten()], axis=1)
        self.register_buffer('tgt_pts', torch.tensor(tgt_pts, dtype=torch.float32))

    def _tps_kernel(self, p1, p2):
        """
        Tính khoảng cách radial basis: r^2 * log(r^2)
        Xử lý an toàn kích thước Batch bằng torch.cdist
        """
        # Đảm bảo p1, p2 luôn có 3 chiều [Batch, Số điểm, 2]
        if p1.dim() == 2:
            p1 = p1.unsqueeze(0)
        if p2.dim() == 2:
            p2 = p2.unsqueeze(0)
            
        # torch.cdist tính khoảng cách Euclid (r) giữa p1 và p2 một cách an toàn
        r = torch.cdist(p1, p2)
        r2 = r ** 2
        
        return r2 * torch.log(r2 + 1e-6)

    def forward(self, x):
        B, C, H, W = x.size()
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)
        src_pts = self.fc_loc(xs).view(-1, self.K, 2) 
        
        # 1. Tính K_mat: _tps_kernel giờ đã an toàn trả về [1, K, K]
        K_mat = self._tps_kernel(self.tgt_pts, self.tgt_pts)
        
        # SỬA Ở ĐÂY: Xóa unsqueeze(0), chỉ cần lặp ma trận cho khớp Batch Size
        K_mat = K_mat.repeat(B, 1, 1) # Kết quả: [B, K, K]
        
        # 2. Giải hệ phương trình tuyến tính
        A = K_mat + 0.1 * torch.eye(self.K, device=x.device)
        W_coef = torch.linalg.solve(A, src_pts) 
        
        # 3. Tạo lưới biến đổi chi tiết cho toàn bộ pixel
        identity_grid = F.affine_grid(
            torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float, device=x.device).repeat(B, 1, 1), 
            x.size(), 
            align_corners=True
        )
        flat_grid = identity_grid.view(B, -1, 2)
        
        # 4. Đánh giá toán tử nội suy và áp dụng biến đổi
        K_eval = self._tps_kernel(flat_grid, self.tgt_pts) # [B, H*W, K]
        transformed_grid = torch.bmm(K_eval, W_coef).view(B, H, W, 2)
        
        return F.grid_sample(x, transformed_grid, align_corners=True)
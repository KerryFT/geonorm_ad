import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConfidenceWeightedTPS(nn.Module):
    """
    Confidence-Weighted Thin-Plate Spline (CW-TPS).
    Đóng góp toán học chính: Đưa ma trận tự tin C vào toán tử giải hệ phương trình.
    """
    def __init__(self, K=16, lam=0.1):
        super().__init__()
        self.K = K
        self.lam = lam  # Hệ số điều hòa (Regularization factor)
        
        # 1. Khởi tạo lưới điểm chuẩn (Canonical Target Points)
        grid_size = int(math.sqrt(K))
        r = torch.linspace(-1, 1, grid_size)
        ys, xs = torch.meshgrid(r, r, indexing='ij')
        tgt_pts = torch.stack([xs.flatten(), ys.flatten()], dim=1) # [K, 2]
        
        # Đăng ký làm buffer để tự động chuyển theo thiết bị (CPU/GPU) cùng model
        self.register_buffer('tgt_pts', tgt_pts)

    def _tps_kernel(self, p1, p2):
        """ Tính khoảng cách radial basis: U(r) = r^2 * log(r^2) """
        if p1.dim() == 2: p1 = p1.unsqueeze(0)
        if p2.dim() == 2: p2 = p2.unsqueeze(0)
        
        r = torch.cdist(p1, p2)
        r2 = r ** 2
        return r2 * torch.log(r2 + 1e-6)

    def forward(self, src_pts, confs, out_size):
        """
        src_pts : [B, K, 2] (Tọa độ bị biến dạng do GLN dự đoán)
        confs   : [B, K]    (Trọng số tự tin do GLN dự đoán)
        out_size: Kích thước ảnh đầu ra (H, W)
        """
        B = src_pts.size(0)
        H, W = out_size
        
        # 2. Xây dựng ma trận Kernel dựa trên lưới chuẩn cố định
        tgt_pts_b = self.tgt_pts.unsqueeze(0).expand(B, -1, -1)
        K_mat = self._tps_kernel(tgt_pts_b, tgt_pts_b) # [B, K, K]
        
        # 3. Tạo các ma trận trọng số (Key Novelty)
        C = torch.diag_embed(confs)                  # [B, K, K]
        I_c = torch.diag_embed(1.0 - confs)          # [B, K, K]
        
        # 4. Thiết lập hệ phương trình: (C @ K_mat + lambda * I_c) * W = C @ src_pts
        A = torch.bmm(C, K_mat) + self.lam * I_c     # [B, K, K]
        b = torch.bmm(C, src_pts)                    # [B, K, 2]
        
        # 5. Giải hệ phương trình tuyến tính (Gradient tự động chảy qua đây)
        W_coef = torch.linalg.solve(A, b)            # [B, K, 2]
        
        # 6. Nội suy ra trường tọa độ dày đặc (Dense Flow Field) cho toàn bộ pixel
        identity_grid = F.affine_grid(
            torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float, device=src_pts.device).expand(B, -1, -1),
            [B, 3, H, W], align_corners=True
        )
        flat_grid = identity_grid.view(B, -1, 2)
        
        K_eval = self._tps_kernel(flat_grid, tgt_pts_b) # [B, H*W, K]
        flow_grid = torch.bmm(K_eval, W_coef).view(B, H, W, 2) # [B, H, W, 2]
        
        return flow_grid

# --- Unit Test Cục Bộ ---
if __name__ == "__main__":
    print("Khởi tạo lớp CW-TPS với K=16...")
    cw_tps = ConfidenceWeightedTPS(K=16, lam=0.1)
    
    # Giả lập đầu ra từ mạng GLN
    B = 2
    dummy_src_pts = torch.randn(B, 16, 2, requires_grad=True)
    dummy_confs = torch.rand(B, 16, requires_grad=True) # Giá trị ngẫu nhiên từ 0 đến 1
    out_size = (224, 224)
    
    print("Tiến hành chạy thử (Forward Pass)...")
    flow_grid = cw_tps(dummy_src_pts, dummy_confs, out_size)
    
    print(f"Kích thước lưới xuất ra: {flow_grid.shape} | Kỳ vọng: [{B}, 224, 224, 2]")
    
    # Kiểm tra tính Differentiable (CỰC KỲ QUAN TRỌNG)
    print("Kiểm tra luồng đạo hàm (Backward Pass)...")
    loss = flow_grid.sum()
    loss.backward()
    
    if dummy_confs.grad is not None and dummy_src_pts.grad is not None:
        print("✅ Unit Test: Luồng đạo hàm truyền ngược hoàn hảo qua linalg.solve!")
    else:
        print("❌ Lỗi: Mất Gradient! Hãy kiểm tra lại các phép toán tách rời (detach).")
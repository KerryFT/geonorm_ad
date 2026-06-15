import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.gln import GLN
from src.models.cw_tps import ConfidenceWeightedTPS

class GeoNorm(nn.Module):
    
    Module tiền xử lý End-to-End hoàn chỉnh.
    Nhận ảnh đầu vào bị biến dạng - Dự đoán lưới - Nắn chỉnh - Trả về ảnh đã nắn.
    
    def __init__(self, K=16, lam=0.1)
        super().__init__()
        self.gln = GLN(K=K)
        self.cw_tps = ConfidenceWeightedTPS(K=K, lam=lam)
    
    def forward(self, x)
        # 1. Trích xuất tọa độ biến dạng và độ tự tin
        src_pts, confs = self.gln(x)
        
        # 2. Sinh trường dòng chảy (Flow Field) qua CW-TPS
        flow_grid = self.cw_tps(src_pts, confs, out_size=x.shape[-2])
        
        # 3. Differentiable Grid Sampler Nắn thẳng ảnh
        x_rectified = F.grid_sample(
            x, flow_grid, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True
        )
        
        return x_rectified, src_pts, confs, flow_grid
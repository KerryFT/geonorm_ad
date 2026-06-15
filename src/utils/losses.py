import torch
import torch.nn.functional as F
def supervised_warp_loss(pred_pts, gt_pts, confs):
    """
    Tính toán hàm L_sup cho Phase 1.
    pred_pts: [B, K, 2] - Tọa độ do model dự đoán
    gt_pts:   [B, K, 2] - Tọa độ ground-truth từ lúc sinh dữ liệu
    confs:    [B, K]    - Trọng số tự tin do model dự đoán
    """
    # 1. Tính khoảng cách bình phương (MSE) giữa dự đoán và Ground Truth
    dist_sq = torch.sum((pred_pts - gt_pts) ** 2, dim=-1) # [B, K]
    
    # 2. Cân bằng MSE bằng trọng số tự tin (c_i * MSE)
    weighted_loc_loss = torch.mean(confs * dist_sq)
    
    # 3. Ép tự tin về 1 (Vì dữ liệu Phase 1 là dữ liệu nhân tạo sạch, không có occlusion)
    conf_loss = torch.mean((confs - 1.0) ** 2)
    
    return weighted_loc_loss + conf_loss


# ... (Hàm supervised_warp_loss của Phase 1 giữ nguyên ở trên) ...

def unsupervised_warp_loss(x_original, x_rectified, flow_grid, confs):
    """
    Tính toán hàm L_unsup cho Phase 2 (Không cần nhãn tọa độ).
    """
    # 1. Reconstruction Loss (L_rec): Ép ảnh nắn ra phải giống hệt ảnh gốc về mặt chi tiết pixel
    l_rec = F.l1_loss(x_rectified, x_original)
    
    # 2. Grid Regularization Loss (L_reg): Total Variation
    # Ngăn chặn việc mạng "gấp nếp" hay "xoắn" lưới tọa độ lung tung
    dx = flow_grid[:, :, 1:, :] - flow_grid[:, :, :-1, :]
    dy = flow_grid[:, 1:, :, :] - flow_grid[:, :-1, :, :]
    l_reg = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))
    
    # 3. Confidence Entropy (L_conf): Khuyến khích mô hình duy trì độ tự tin
    l_conf = torch.mean((confs - 1.0) ** 2)
    
    # Tổng hợp Loss với các hệ số cân bằng (có thể tinh chỉnh sau)
    total_loss = l_rec + 0.1 * l_reg + 0.01 * l_conf
    
    return total_loss
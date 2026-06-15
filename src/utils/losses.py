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

def cycle_consistency_loss(x_orig, x_rect, x_cycle, pred_pts, identity_pts, confs, flow_grid):
    """
    Hàm Loss Nhất quán chu trình kép chuẩn nghiên cứu WACV.
    """
    # 1. Forward Reconstruction Loss (L1 + SSIM nếu cần, ở đây dùng L1 làm gốc)
    l_rec = F.l1_loss(x_rect, x_orig)
    
    # 2. Cycle Image Consistency Loss
    l_cycle_img = F.l1_loss(x_cycle, x_rect)
    
    # 3. Identity Grid Regularization (Ép không nắn lệch ảnh chuẩn)
    # pred_pts: [B, 16, 2], identity_pts: [16, 2]
    l_identity = F.mse_loss(pred_pts, identity_pts.unsqueeze(0).expand(pred_pts.shape[0], -1, -1))
    
    # 4. Total Variation Loss (Giữ lưới mượt mà, không chồng chéo hình học)
    dx = flow_grid[:, :, 1:, :] - flow_grid[:, :, :-1, :]
    dy = flow_grid[:, 1:, :, :] - flow_grid[:, :-1, :, :]
    l_tv = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))
    
    # 5. Entropy Confidence Loss (Ép trọng số tự tin đạt tối ưu)
    l_conf = torch.mean((confs - 1.0) ** 2)
    
    # Tổng hợp trọng số các thành phần Loss theo thiết kế hệ thống
    total_loss = l_rec + 1.0 * l_cycle_img + 0.5 * l_identity + 0.1 * l_tv + 0.01 * l_conf
    
    return total_loss
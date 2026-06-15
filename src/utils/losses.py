import torch

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
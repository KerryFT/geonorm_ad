import numpy as np
from sklearn.metrics import roc_auc_score

def compute_auroc(ground_truths: np.ndarray, predictions: np.ndarray) -> float:
    """
    Tính toán chỉ số AUROC tiêu chuẩn.
    ground_truths: Mảng nhị phân [N] hoặc [N, H, W] (0: bình thường, 1: bất thường)
    predictions: Mảng điểm số bất thường [N] hoặc [N, H, W]
    """
    y_true = ground_truths.flatten()
    y_pred = predictions.flatten()
    
    if len(np.unique(y_true)) < 2:
        # Tránh lỗi nếu tập test chỉ toàn ảnh sạch hoặc toàn ảnh lỗi
        return 0.5
        
    return float(roc_auc_score(y_true, y_pred))

def compute_geo_robustness(auroc_l3: float, auroc_l0: float) -> float:
    """
    Geo-Robustness Score = AUROC(L3) / AUROC(L0)
    Đo lường mức độ giữ phong độ của mô hình khi bị biến dạng nặng.
    """
    if auroc_l0 == 0:
        return 0.0
    return float(auroc_l3 / auroc_l0)
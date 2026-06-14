import cv2
import numpy as np
from pathlib import Path

def save_rectification_pair(distorted_img: np.ndarray, rectified_img: np.ndarray, save_path: str):
    """
    Ghép đôi ảnh bị biến dạng và ảnh đã được nắn thẳng để kiểm tra trực quan.
    """
    # Chuyển đổi tensor về định dạng ảnh OpenCV nếu cần
    if distorted_img.shape[0] == 3:  # Nếu là định dạng [C, H, W]
        distorted_img = distorted_img.transpose(1, 2, 0) * 255
        distorted_img = distorted_img.astype(np.uint8)
    if rectified_img.shape[0] == 3:
        rectified_img = rectified_img.transpose(1, 2, 0) * 255
        rectified_img = rectified_img.astype(np.uint8)

    # Ghép hai ảnh nằm cạnh nhau (Horizontal Concatenation)
    vis_grid = np.hstack((distorted_img, rectified_img))
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), vis_grid)
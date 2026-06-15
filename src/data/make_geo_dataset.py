import os
import cv2
import numpy as np
from scipy.interpolate import RectBivariateSpline
from pathlib import Path
from tqdm import tqdm

def generate_tps_distortion(image: np.ndarray, severity: str = 'moderate', seed: int = 42) -> np.ndarray:
    """
    Áp dụng biến dạng TPS ngẫu nhiên nhưng có kiểm soát lên ảnh đầu vào.
    """
    np.random.seed(seed)
    H, W = image.shape[:2]
    
    # Biên độ biến dạng theo tài liệu thiết kế
    displacement_scale = {'mild': 0.05, 'moderate': 0.10, 'severe': 0.20}[severity]
    
    # Tạo lưới điểm kiểm soát cố định 4x4 (K=4)
    K = 4
    grid_x = np.linspace(0, W, K)
    grid_y = np.linspace(0, H, K)
    
    # Sinh độ dời ngẫu nhiên cho các điểm kiểm soát
    dx = np.random.randn(K, K) * W * displacement_scale
    dy = np.random.randn(K, K) * H * displacement_scale
    
    # Nội suy ra dense flow field qua Spline
    spline_x = RectBivariateSpline(grid_y, grid_x, dx)
    spline_y = RectBivariateSpline(grid_y, grid_x, dy)
    
    yy, xx = np.mgrid[0:H, 0:W]
    
    map_x = (xx + spline_x(yy.ravel(), xx.ravel(), grid=False).reshape(H, W)).astype(np.float32)
    map_y = (yy + spline_y(yy.ravel(), xx.ravel(), grid=False).reshape(H, W)).astype(np.float32)
    
    # Áp dụng remap chống vùng đen ở biên
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def build_mvtec_geo(mvtec_dir: str, output_base_dir: str):
    """ Dựng toàn bộ cấu trúc thư mục MVTec-Geo """
    mvtec_path = Path(mvtec_dir)
    output_base = Path(output_base_dir)
    severities = ['mild', 'moderate', 'severe']
    
    if not mvtec_path.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục MVTec gốc tại: {mvtec_dir}")
        
    categories = [d.name for d in mvtec_path.iterdir() if d.is_dir()]
    print(f"👉 Tìm thấy {len(categories)} đối tượng công nghiệp. Bắt đầu xử lý...")
    
    for category in categories:
        for sev in severities:
            src_cat_dir = mvtec_path / category
            # Giữ nguyên cấu trúc train/test của MVTec
            for root, dirs, files in os.walk(src_cat_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        src_file_path = Path(root) / file
                        rel_path = src_file_path.relative_to(mvtec_path)
                        
                        # Chỉ bóp méo ảnh đầu vào (image), KHÔNG bóp méo nhãn lỗi hình học trực tiếp
                        dest_file_path = output_base / f"mvtec_{sev}" / rel_path
                        dest_file_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        img = cv2.imread(str(src_file_path))
                        if "ground_truth" in str(rel_path):
                            # Giữ nguyên mặt nạ lỗi gốc (Ground Truth) để kiểm tra tính hiệu quả của toán tử nghịch đảo sau này
                            cv2.imwrite(str(dest_file_path), img)
                        else:
                            distorted = generate_tps_distortion(img, severity=sev)
                            cv2.imwrite(str(dest_file_path), distorted)
    print("✅ Đã hoàn thành khởi tạo bộ dữ liệu MVTec-Geo!")
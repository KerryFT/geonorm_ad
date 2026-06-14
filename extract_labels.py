import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path

def extract_labels_fast(mvtec_geo_dir):
    print("⏳ Đang trích xuất nhãn tọa độ siêu tốc...")
    scales = {'mild': 0.05, 'moderate': 0.10, 'severe': 0.20}
    labels_dict = {}
    base_path = Path(mvtec_geo_dir)
    
    for severity in ['mild', 'moderate', 'severe']:
        sev_path = base_path / f"mvtec_{severity}"
        if not sev_path.exists(): continue
        
        for root, _, files in os.walk(sev_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg')) and "ground_truth" not in file:
                    img_path = Path(root) / file
                    rel_path = img_path.relative_to(base_path).as_posix()
                    
                    # Chỉ đọc header để lấy kích thước (Cực nhanh)
                    with Image.open(img_path) as img:
                        W, H = img.size
                    
                    # Kích hoạt lại đúng seed đã dùng để bóp méo ảnh
                    np.random.seed(42)
                    displacement_scale = scales[severity]
                    K_grid = 4
                    
                    # Tái tạo lại độ dời dx, dy
                    dx = np.random.randn(K_grid, K_grid) * W * displacement_scale
                    dy = np.random.randn(K_grid, K_grid) * H * displacement_scale
                    
                    # Tính tọa độ chuẩn hóa về không gian [-1, 1] của mô hình GLN
                    grid_x = np.linspace(0, W, K_grid)
                    grid_y = np.linspace(0, H, K_grid)
                    xx, yy = np.meshgrid(grid_x, grid_y)
                    
                    distorted_x = xx + dx
                    distorted_y = yy + dy
                    
                    norm_x = (distorted_x / W) * 2.0 - 1.0
                    norm_y = (distorted_y / H) * 2.0 - 1.0
                    
                    # Gộp thành tensor [16, 2]
                    gt_pts = np.stack([norm_x.flatten(), norm_y.flatten()], axis=1)
                    labels_dict[rel_path] = torch.tensor(gt_pts, dtype=torch.float32)

    output_file = base_path / "geo_labels.pt"
    torch.save(labels_dict, output_file)
    print(f"✅ Đã trích xuất thành công {len(labels_dict)} nhãn tọa độ!")
    print(f"📁 File nhãn được lưu tại: {output_file}")

if __name__ == "__main__":
    extract_labels_fast("data/mvtec_geo")
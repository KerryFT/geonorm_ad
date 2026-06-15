import os
import torch
from torch.utils.data import Dataset
import cv2
import torchvision.transforms as transforms
from pathlib import Path

class GeoNormDataset(Dataset):
    """
    Dataset phục vụ huấn luyện Phase 1 (Supervised Pre-training).
    Tải ảnh biến dạng từ thư mục và nhãn tọa độ (gt_pts) từ file .pt.
    """
    def __init__(self, data_dir, labels_file, img_size=(224, 224)):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        
        # 1. Tải toàn bộ nhãn từ file .pt vào bộ nhớ RAM (dict)
        print(f"Đang tải nhãn từ: {labels_file}...")
        self.labels_dict = torch.load(labels_file)
        
        # 2. Tạo danh sách các đường dẫn ảnh (loại bỏ ảnh ground_truth)
        self.img_paths = []
        for severity in ['mild', 'moderate', 'severe']:
            sev_path = self.data_dir / f"mvtec_{severity}"
            if not sev_path.exists():
                continue
                
            for root, _, files in os.walk(sev_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg')) and "ground_truth" not in file:
                        full_path = Path(root) / file
                        # Lấy đường dẫn tương đối (để khớp với key trong dictionary nhãn)
                        rel_path = full_path.relative_to(self.data_dir).as_posix()
                        
                        # Chỉ thêm vào danh sách nếu file ảnh có nhãn tương ứng
                        if rel_path in self.labels_dict:
                            self.img_paths.append((full_path, rel_path))
                            
        print(f"Đã tải {len(self.img_paths)} mẫu dữ liệu vào bộ nhớ.")

        # 3. Định nghĩa các phép biến đổi ảnh (Transforms)
        # MobileViT thường yêu cầu đầu vào chuẩn hóa theo chuẩn ImageNet
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.img_size, antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        full_path, rel_path = self.img_paths[idx]
        
        # Đọc ảnh bằng OpenCV (BGR -> RGB)
        img = cv2.imread(str(full_path))
        if img is None:
            raise ValueError(f"Không thể đọc ảnh: {full_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Áp dụng Transform
        img_tensor = self.transform(img)
        
        # Lấy nhãn tọa độ lưới
        gt_pts = self.labels_dict[rel_path] # [16, 2]
        
        return img_tensor, gt_pts
    
class MVTecRealDataset(Dataset):
    """
    Dataset phục vụ huấn luyện Phase 2.
    Chỉ nạp các ảnh 'good' (không lỗi, không bóp méo nhân tạo) từ thư mục train.
    """
    def __init__(self, data_dir, img_size=(224, 224)):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.img_paths = []
        
        # Duyệt qua tất cả các danh mục con (bottle, cable, capsule,...)
        for category in os.listdir(self.data_dir):
            train_good_dir = self.data_dir / category / "train" / "good"
            if train_good_dir.exists():
                for file in os.listdir(train_good_dir):
                    if file.lower().endswith(('.png', '.jpg')):
                        self.img_paths.append(train_good_dir / file)
                        
        print(f"Đã nạp {len(self.img_paths)} ảnh thực tế (good) vào bộ nhớ Phase 2.")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.img_size, antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Không thể đọc ảnh: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img)
        
        # Phase 2 KHÔNG CÓ nhãn gt_pts, trả về chính tensor đó làm mỏ neo
        return img_tensor
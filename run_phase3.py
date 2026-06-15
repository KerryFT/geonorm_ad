import torch
import os
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from src.models.geonorm import GeoNorm
from src.models.patchcore import PatchCore

class MVTecEvalDataset(Dataset):
    """Dataset đọc cả ảnh Good và ảnh Defect để đánh giá Phase 3"""
    def __init__(self, data_dir, split="train", img_size=(224, 224)):
        self.data_dir = Path(data_dir)
        self.img_paths = []
        self.labels = [] # 0: Bình thường (Normal), 1: Dị thường (Anomaly)
        
        for category in os.listdir(self.data_dir):
            target_dir = self.data_dir / category / split
            if not target_dir.exists(): continue
                
            for class_name in os.listdir(target_dir):
                class_dir = target_dir / class_name
                is_anomaly = 0 if class_name == "good" else 1
                
                for file in os.listdir(class_dir):
                    if file.lower().endswith(('.png', '.jpg')):
                        self.img_paths.append(class_dir / file)
                        self.labels.append(is_anomaly)
                        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size, antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = str(self.img_paths[idx])
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        return self.transform(img), self.labels[idx]

def run_evaluation(data_dir, geonorm_weights):
    print("🚀 Khởi động Phase 3: Đánh giá hệ thống GeoNorm + PatchCore End-to-End...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Khởi tạo mô hình Nắn hình học (GeoNorm)
    geonorm = GeoNorm(K=16, lam=0.1).to(device)
    geonorm.load_state_dict(torch.load(geonorm_weights, map_location=device, weights_only=True))
    geonorm.eval() # Bắt buộc Eval, không train nữa
    
    # Khởi tạo mô hình Phát hiện dị thường (PatchCore)
    patchcore = PatchCore(f_coreset=0.01).to(device)
    
    # 2. Xây dựng Memory Bank từ tập Train (Chỉ ảnh chuẩn)
    train_dataset = MVTecEvalDataset(data_dir, split="train")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    print("\n[1/2] Đang xây dựng Memory Bank...")
    train_features = []
    with torch.no_grad():
        for imgs, _ in tqdm(train_loader):
            imgs = imgs.to(device)
            # NẮN THẲNG TRƯỚC
            x_rectified, _, _, _ = geonorm(imgs)
            # TRÍCH ĐẶC TRƯNG TỪ ẢNH ĐÃ NẮN
            feats = patchcore.extract_features(x_rectified)
            train_features.append(feats.cpu())
            
    patchcore.fit(train_features)
    
    # 3. Đánh giá trên tập Test (Bao gồm cả ảnh lỗi)
    test_dataset = MVTecEvalDataset(data_dir, split="test")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    print("\n[2/2] Đang kiểm tra ảnh Test và tính toán AUROC...")
    y_true = []
    y_scores = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader):
            imgs = imgs.to(device)
            # Nắn thẳng ảnh Test
            x_rectified, _, _, _ = geonorm(imgs)
            
            # Tính điểm dị thường
            feats = patchcore.extract_features(x_rectified)
            scores = patchcore.predict(feats)
            
            y_true.extend(labels.numpy())
            y_scores.extend(scores)
            
    # 4. Chấm điểm AUROC
    auroc = roc_auc_score(y_true, y_scores)
    print("\n" + "🔥" * 25)
    print(f"🏆 KẾT QUẢ CUỐI CÙNG AUROC: {auroc * 100:.2f}%")
    print("🔥" * 25)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Giữ nguyên đường dẫn máy local
    parser.add_argument("--data_dir", type=str, default="data/mvtec")
    # Trỏ đến file trọng số Cycle-Consistent vừa chạy xong
    parser.add_argument("--geonorm_weights", type=str, default="checkpoints/phase2_cycle_final_weights.pth")
    args = parser.parse_args()
    
    run_evaluation(args.data_dir, args.geonorm_weights)
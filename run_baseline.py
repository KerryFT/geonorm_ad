import torch
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.models.patchcore import PatchCore
from run_phase3 import MVTecEvalDataset # Tái sử dụng DataLoader của Phase 3

def run_baseline(data_dir):
    print("🚀 Khởi động Đánh giá Baseline: PatchCore Gốc (KHÔNG có GeoNorm)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # CHỈ khởi tạo PatchCore (giữ nguyên coreset 1% để so sánh công bằng)
    patchcore = PatchCore(f_coreset=0.01).to(device)
    
    # 1. Xây dựng Memory Bank
    train_dataset = MVTecEvalDataset(data_dir, split="train")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    print("\n[1/2] Đang xây dựng Memory Bank (Baseline)...")
    train_features = []
    with torch.no_grad():
        for imgs, _ in tqdm(train_loader):
            imgs = imgs.to(device)
            # ẢNH ĐI THẲNG VÀO PATCHCORE, KHÔNG QUA GEONORM
            feats = patchcore.extract_features(imgs) 
            
            B, C, H, W = feats.shape
            feats_flat = feats.permute(0, 2, 3, 1).reshape(-1, C)
            num_samples = max(1, int(feats_flat.shape[0] * 0.01))
            indices = torch.randperm(feats_flat.shape[0])[:num_samples]
            train_features.append(feats_flat[indices].cpu())
            
    patchcore.fit(train_features)
    
    # 2. Đánh giá Test
    test_dataset = MVTecEvalDataset(data_dir, split="test")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    print("\n[2/2] Đang kiểm tra ảnh Test và tính toán AUROC...")
    y_true = []
    y_scores = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader):
            imgs = imgs.to(device)
            # ẢNH ĐI THẲNG VÀO PATCHCORE
            feats = patchcore.extract_features(imgs)
            scores = patchcore.predict(feats)
            
            y_true.extend(labels.numpy())
            y_scores.extend(scores)
            
    auroc = roc_auc_score(y_true, y_scores)
    print("\n" + "🧊" * 25)
    print(f"📊 KẾT QUẢ BASELINE AUROC (Không GeoNorm): {auroc * 100:.2f}%")
    print("🧊" * 25)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/mvtec")
    args = parser.parse_args()
    run_baseline(args.data_dir)
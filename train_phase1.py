import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.geonorm import GeoNorm
from src.utils.losses import supervised_warp_loss
from src.data.dataset import GeoNormDataset
from tqdm import tqdm
import os

def train_phase1(data_dir, labels_file, epochs=5, batch_size=32, lr=1e-4):
    print("🚀 Khởi động Phase 1: Supervised Pre-training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị tính toán: {device}")
    
    # 1. Khởi tạo Dataset và DataLoader
    dataset = GeoNormDataset(data_dir, labels_file)
    # num_workers=4 giúp nạp dữ liệu nhanh hơn trên Kaggle (GPU Server)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # 2. Khởi tạo model và optimizer
    model = GeoNorm(K=16, lam=0.1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 3. Vòng lặp huấn luyện chính
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        # Tích hợp tqdm để hiển thị tiến trình của từng epoch
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for imgs, gt_pts in progress_bar:
            imgs = imgs.to(device, non_blocking=True)
            gt_pts = gt_pts.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass: chỉ cần quan tâm đến pred_pts và confs
            _, pred_pts, confs, _ = model(imgs)
            
            # Tính toán Supervised Warp Loss
            loss = supervised_warp_loss(pred_pts, gt_pts, confs)
            
            # Backward pass & Optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(Loss=f"{loss.item():.4f}")
            
        avg_loss = total_loss / len(train_loader)
        print(f"✅ Hoàn thành Epoch {epoch+1}. Average Loss: {avg_loss:.4f}")
        
        # --- Lưu trọng số mô hình ---
        # Tạo thư mục checkpoints nếu chưa có
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/geonorm_phase1_epoch{epoch+1}.pth")

if __name__ == "__main__":
    # Cấu hình đường dẫn. Khi lên Kaggle, bạn sẽ đổi hai đường dẫn này.
    DATA_DIR = "data/mvtec_geo"
    LABELS_FILE = "data/mvtec_geo/geo_labels.pt"
    
    train_phase1(DATA_DIR, LABELS_FILE)
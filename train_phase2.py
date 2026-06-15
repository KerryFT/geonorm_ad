import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.geonorm import GeoNorm
from src.utils.losses import unsupervised_warp_loss
from src.data.dataset import MVTecRealDataset
from tqdm import tqdm
import os

def train_phase2(data_dir, pretrained_weights, epochs=5, batch_size=16, lr=1e-5):
    print("🚀 Khởi động Phase 2: Unsupervised Fine-tuning...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị tính toán: {device}")
    
    # 1. Khởi tạo Dataset thực tế (Dùng ảnh gốc MVTec, KHÔNG CÓ nhãn tọa độ)
    dataset = MVTecRealDataset(data_dir)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # 2. Khởi tạo model và BƠM TRỌNG SỐ TỪ PHASE 1
    model = GeoNorm(K=16, lam=0.1).to(device)
    print(f"Đang nạp kiến thức nền tảng từ: {pretrained_weights}")
    
    # Kích hoạt trí nhớ Phase 1 (Nạp file weights)
    if not os.path.exists(pretrained_weights):
        raise FileNotFoundError(f"❌ Không tìm thấy file trọng số tại: {pretrained_weights}")
    
    model.load_state_dict(torch.load(pretrained_weights, map_location=device, weights_only=True))
    
    # Learning rate ở Phase 2 rất nhỏ (1e-5) để Fine-tune từ từ
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 3. Vòng lặp huấn luyện Không Giám Sát
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for imgs in progress_bar:
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            # Forward pass: Đẩy ảnh gốc qua hệ thống
            x_rectified, _, confs, flow_grid = model(imgs)
            
            # Tính Loss Không Giám Sát
            loss = unsupervised_warp_loss(imgs, x_rectified, flow_grid, confs)
            
            # Backward pass & Optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(L_unsup=f"{loss.item():.4f}")
            
        avg_loss = total_loss / len(train_loader)
        print(f"✅ Hoàn thành Epoch {epoch+1}. Average L_unsup: {avg_loss:.4f}")
        
        # Lưu trọng số mô hình Phase 2
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/geonorm_phase2_epoch{epoch+1}.pth")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Giữ nguyên đường dẫn máy local ở default
    parser.add_argument("--data_dir", type=str, default="data/mvtec")
    parser.add_argument("--pretrained_weights", type=str, default="checkpoints/phase1_best_weights.pth")
    args = parser.parse_args()
    
    train_phase2(args.data_dir, args.pretrained_weights)
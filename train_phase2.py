import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.models.geonorm import GeoNorm
from src.utils.losses import cycle_consistency_loss
from src.data.dataset import MVTecRealDataset
from tqdm import tqdm
import os
import argparse

def generate_smooth_distortion(imgs, scale=0.25):
    """
    Sinh biến dạng hình học ngẫu nhiên, mượt mà trực tiếp trên GPU để phục vụ luồng Cycle.
    """
    B, C, H, W = imgs.shape
    device = imgs.device
    
    # Tạo lưới nhiễu nhỏ (4x4) rồi phóng to bằng nội suy song tuyến tính để tạo độ mượt hình học
    noise = torch.randn(B, 2, 4, 4, device=device) * scale
    flow_field = F.interpolate(noise, size=(H, W), mode='bilinear', align_corners=True)
    
    # Khởi tạo lưới đơn vị (Identity Grid)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device), 
        torch.linspace(-1, 1, W, device=device), 
        indexing='ij'
    )
    identity = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
    
    # Cộng nhiễu vào lưới gốc để tạo ma trận biến dạng
    distorted_grid = identity + flow_field
    distorted_grid = distorted_grid.permute(0, 2, 3, 1) # [B, H, W, 2]
    
    # Thực hiện bóp méo ảnh thông qua Grid Sampler
    return F.grid_sample(imgs, distorted_grid, mode='bilinear', padding_mode='reflection', align_corners=True)

def train_phase2(data_dir, pretrained_weights, epochs=5, batch_size=16, lr=1e-5):
    print("🚀 Khởi động Phase 2: Unsupervised Cycle-Consistent Fine-tuning...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Khởi tạo dữ liệu thực tế
    dataset = MVTecRealDataset(data_dir)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # 2. Khởi tạo mô hình và nạp trọng số nền tảng Phase 1
    model = GeoNorm(K=16, lam=0.1).to(device)
    if not os.path.exists(pretrained_weights):
        raise FileNotFoundError(f"❌ Không tìm thấy trọng số pre-train tại: {pretrained_weights}")
    model.load_state_dict(torch.load(pretrained_weights, map_location=device, weights_only=True))
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Định nghĩa ma trận điểm kiểm soát đơn vị (Tọa độ chuẩn hóa gốc)
    grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, 4), torch.linspace(-1, 1, 4), indexing='ij')
    identity_pts = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1).to(device) # [16, 2]
    
    # 3. Vòng lặp huấn luyện Chu trình
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for imgs in progress_bar:
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            # --- LUỒNG XUÔI (Forward Pass) ---
            x_rectified, pred_pts, confs, flow_grid = model(imgs)
            
            # --- LUỒNG NGHỊCH (Cycle Pass) ---
            # Cố tình bóp méo ảnh vừa nắn thẳng ngay trên GPU
            with torch.no_grad():
                imgs_distorted = generate_smooth_distortion(x_rectified, scale=0.08)
                
            # Ép mô hình nắn thẳng lại ảnh bị bóp méo lần hai
            x_cycle_rectified, _, _, _ = model(imgs_distorted)
            
            # --- TÍNH TOÁN LOSS CHU TRÌNH ---
            loss = cycle_consistency_loss(
                imgs, x_rectified, x_cycle_rectified, 
                pred_pts, identity_pts, confs, flow_grid
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(L_cycle=f"{loss.item():.4f}")
            
        avg_loss = total_loss / len(train_loader)
        print(f"✅ Thiên niên kỷ Epoch {epoch+1} hoàn thành. L_cycle trung bình: {avg_loss:.4f}")
        
        # Lưu vết Checkpoint học sâu
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/geonorm_phase2_cycle_epoch{epoch+1}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/mvtec")
    parser.add_argument("--pretrained_weights", type=str, default="checkpoints/phase1_best_weights.pth")
    args = parser.parse_args()
    
    train_phase2(args.data_dir, args.pretrained_weights)
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.models.geonorm import GeoNorm
from run_phase3 import MVTecEvalDataset # Dùng lại class cũ

def visualize_rectification(data_dir, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model Phase 2
    model = GeoNorm(K=16, lam=0.1).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    
    # Lấy data từ tập Test (có lỗi)
    dataset = MVTecEvalDataset(data_dir, split="test")
    loader = DataLoader(dataset, batch_size=4, shuffle=True) # Chỉ lấy 4 ảnh random
    
    imgs, labels = next(iter(loader))
    imgs = imgs.to(device)
    
    with torch.no_grad():
        x_rectified, _, _, _ = model(imgs)
        
    # Chuyển tensor về numpy để vẽ (Denormalize)
    imgs = imgs.cpu().permute(0, 2, 3, 1).numpy()
    x_rectified = x_rectified.cpu().permute(0, 2, 3, 1).numpy()
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    imgs = (imgs * std + mean).clip(0, 1)
    x_rectified = (x_rectified * std + mean).clip(0, 1)
    
    # Vẽ biểu đồ
    fig, axes = plt.subplots(4, 2, figsize=(8, 12))
    fig.suptitle("Kiểm tra Thực tế: Ảnh Méo vs Ảnh Đã Nắn", fontsize=16)
    
    for i in range(4):
        axes[i, 0].imshow(imgs[i])
        axes[i, 0].set_title("Ảnh Gốc (Bị bóp méo Severe)")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(x_rectified[i])
        axes[i, 1].set_title("Sau khi qua GeoNorm")
        axes[i, 1].axis('off')
        
    plt.tight_layout()
    plt.savefig("debug_vision.png")
    print("📸 Đã lưu ảnh trực quan hóa vào file debug_vision.png")

if __name__ == "__main__":
    # Điền cứng đường dẫn để chạy test nhanh
    data = "/kaggle/input/datasets/trungnguynhongv/geonorm-mvtec-geo/mvtec_geo/mvtec_severe"
    weights = "/kaggle/input/datasets/trungnguynhongv/geonorm-phase1-weights/phase1_best_weights.pth"
    visualize_rectification(data, weights)
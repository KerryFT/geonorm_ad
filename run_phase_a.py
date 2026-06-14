import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.data.make_geo_dataset import build_mvtec_geo
from src.models.baselines import AffineSTN, StandardTPSSTN
from src.models.patchcore import PatchCoreBaseline

if __name__ == "__main__":
    # 1. Chốt chặn kiểm tra dữ liệu
    mvtec_original_path = "data/mvtec"
    mvtec_geo_output_path = "data/mvtec_geo"
    
    # Kiểm tra xem thư mục đầu ra đã có chưa và bên trong có dữ liệu không
    if os.path.exists(mvtec_geo_output_path) and len(os.listdir(mvtec_geo_output_path)) > 0:
        print("✅ Dữ liệu MVTec-Geo đã tồn tại, tự động bỏ qua bước sinh dữ liệu!")
    else:
        print("⏳ Bắt đầu sinh MVTec-Geo...")
        build_mvtec_geo(mvtec_original_path, mvtec_geo_output_path)
    
    print("\n🚀 Đang khởi động tiến trình đánh giá hệ thống thuộc Phase A...")
    
    # Giả định dữ liệu nạp vào kích thước chuẩn: [B, C, H, W]
    dummy_clean_train = torch.randn(5, 3, 224, 224) 
    dummy_distorted_test = torch.randn(2, 3, 224, 224)
    
    train_loader = DataLoader(TensorDataset(dummy_clean_train, torch.zeros(5)), batch_size=2)
    
    # Khởi tạo mô hình nền tảng PatchCore
    patchcore = PatchCoreBaseline()
    patchcore.fit_memory_bank(train_loader)
    
    # Thử nghiệm luồng tiền xử lý nắn chỉnh hình học qua các Baseline
    affine_stn = AffineSTN()
    tps_stn = StandardTPSSTN()
    
    print("\n--- Tiến hành kiểm tra kích thước đầu ra hệ thống ---")
    out_affine = affine_stn(dummy_distorted_test)
    print(f"Kích thước sau tiền xử lý Affine-STN: {out_affine.shape}")
    
    out_tps = tps_stn(dummy_distorted_test)
    print(f"Kích thước sau tiền xử lý TPS-STN: {out_tps.shape}")
    
    anomaly_maps = patchcore.compute_anomaly_map(out_tps)
    print(f"Kích thước Anomaly Map đầu ra cuối cùng: {anomaly_maps.shape}")
    print("\n✅ Luồng kết nối mã nguồn Phase A vận hành chính xác!")
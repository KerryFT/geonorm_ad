import unittest
import torch
import numpy as np
from src.models.baselines import StandardTPSSTN
from src.utils.metrics import compute_auroc

class TestPhaseAPipeline(unittest.TestCase):
    
    def setUp(self):
        # Thiết lập cấu hình giả lập lưới 4x4
        self.tps_stn = StandardTPSSTN(K=16)
        self.dummy_input = torch.randn(2, 3, 224, 224)
        
    def test_stn_output_shape(self):
        """ Kiểm tra xem mô hình hình học có giữ nguyên kích thước ảnh không """
        output = self.tps_stn(self.dummy_input)
        self.assertEqual(output.shape, (2, 3, 224, 224))
        
    def test_metric_calculation(self):
        """ Kiểm tra tính chính xác của hàm tính toán AUROC """
        gt = np.array([0, 0, 1, 1])
        pred = np.array([0.1, 0.2, 0.8, 0.9])
        auroc = compute_auroc(gt, pred)
        self.assertEqual(auroc, 1.0)

if __name__ == '__main__':
    unittest.main()
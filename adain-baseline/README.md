# Dự án AdaIN - Đồ án Deep Learning (23KDL)

Project skeleton cho AdaIN baseline. Triển khai mô hình Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.

## Cấu trúc thư mục
```text
adain-baseline/
├── configs/            # Chứa các file cấu hình (.yaml)
├── data/               # Dữ liệu ảnh gốc và đã qua xử lý
├── debug_data/         # Dữ liệu dùng để smoke test luồng
├── scripts/            # Các file chạy tự động (.sh / .bat)
├── src/                # Mã nguồn chính (models, losses, infer, evaluate...)
├── checkpoints/        # Nơi lưu trọng số mô hình (.pth)
└── outputs/            # Nơi lưu ảnh kết quả và báo cáo đánh giá

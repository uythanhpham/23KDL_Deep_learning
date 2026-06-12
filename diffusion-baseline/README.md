# diffusion-baseline

**Mô tả ngắn:** DDPM image generation baseline theo phong cách adain-baseline

## Cách chạy từng giai đoạn

1. **Prepare data:** - Chuẩn bị, phân chia và tiền xử lý tập dữ liệu hình ảnh.
2. **Train:** - Khởi chạy quá trình huấn luyện mô hình DDPM, nạp config từ thư mục `configs/` và lưu checkpoints vào `checkpoints/`.
3. **Sample:** - Sử dụng checkpoint đã huấn luyện để quá trình reverse diffusion sinh ra ảnh mới. Kết quả lưu tại `outputs/samples/`.
4. **Evaluate:** - Đánh giá chất lượng ảnh sinh ra bằng các metrics như FID hoặc IS.

## Tech stack
- PyTorch
- torchvision
- PyYAML
- tqdm

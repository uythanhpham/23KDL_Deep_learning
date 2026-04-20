# 23KDL_Deep_learning/adain-baseline/src/train.py
"""Điều phối: đọc config, chuẩn bị model, data, optimizer."""
import torch
from torch.optim import Adam
from pathlib import Path

# Import các thành phần từ project
from adain_baseline.src.data.datasets import build_debug_dataset, build_dataloader
from adain_baseline.src.models.adain import AdaINStyleTransfer
from adain_baseline.src.trainer import AdaINTrainer

def main():
    # 1. Cấu hình các tham số (Hyperparameters)
    # : Bạn có thể đưa các tham số này vào file config.yaml ở Giai đoạn 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = "adain_baseline/debug_data"        # folder lấy data gốc để train
    image_size = 256                    
    batch_size = 8
    lr = 1e-4
    lambda_style = 10.0                           # Theo paper
    epochs = 5

    print(f"--- Đang chạy trên thiết bị: {device} ---")

    # 2. Thiết lập Data Pipeline (Theo đúng contract của Role 1)
    # Role 1 trả về dict: {"content": tensor, "style": tensor, ...}
    dataset = build_debug_dataset(
        root_dir=root_dir,
        image_size=image_size,
        pair_mode="cycle"
    )
    dataloader = build_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 # Tăng tốc độ load ảnh nếu chạy trên nhiều core
    )

    # 3. Khởi tạo Mô hình & Optimizer
    model = AdaINStyleTransfer().to(device)
    optimizer = Adam(model.decoder.parameters(), lr=lr)
    # Lưu ý: Chỉ train Decoder vì Encoder VGG đã bị freeze (requires_grad=False)

    # 4. Khởi tạo Trainer (Lớp logic bạn đã viết ở src/trainer.py)
    trainer = AdaINTrainer(
        model=model,
        optimizer=optimizer,
        lambda_style=lambda_style,
        device=device
    )

    # 5. Vòng lặp huấn luyện chính
    print(f"Bắt đầu huấn luyện với {len(dataset)} mẫu ảnh...")

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            # UNPACK: Lấy tensor từ dictionary theo key đã thỏa thuận với Role 1
            content_images = batch["content"].to(device)
            style_images = batch["style"].to(device)

            # Thực hiện một bước train
            loss_dict = trainer.train_step(content_images, style_images)

            epoch_loss += loss_dict["total_loss"]

            # Log tiến độ mỗi 5 batch
            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Step [{batch_idx+1}/{len(dataloader)}] "
                      f"| Loss: {loss_dict['total_loss']:.4f} "
                      f"(C: {loss_dict['content_loss']:.4f}, S: {loss_dict['style_loss']:.4f})")

        avg_loss = epoch_loss / len(dataloader)
        print(f"==> Kết thúc Epoch {epoch+1} - Average Loss: {avg_loss:.4f}\n")

        # 6. (Tùy chọn) Lưu checkpoint đơn giản *** Note: đổi lại check_point nếu cần
        checkpoint_path = Path("adain_baseline/checkpoints")                            # check_point folder -> sau này đổi thành debug và official
        checkpoint_path.mkdir(exist_ok=True)                    
        torch.save(model.state_dict(), checkpoint_path / f"adain_epoch_{epoch+1}.pth")  # Tên model

    print("Hoàn thành quá trình huấn luyện!")

if __name__ == "__main__":
    main()

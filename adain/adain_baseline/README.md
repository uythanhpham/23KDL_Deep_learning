# AdaIN Baseline

Hướng dẫn nhanh để train và inference mô hình AdaIN style transfer trong workspace này.

## Yêu cầu

- Python 3.10+ khuyến nghị
- PyTorch
- torchvision
- Pillow
- scikit-image
- lpips
- numpy

Nếu bạn chưa cài dependencies, hãy cài theo môi trường bạn đang dùng trước khi chạy các lệnh bên dưới.

## Cấu trúc dữ liệu train

Script train dùng `build_dataloaders(...)` từ `src/data/datasets.py`. Dữ liệu đầu vào cần được chuẩn bị theo đúng format mà hàm này mong đợi.

Nếu bạn đang dùng thư mục debug mặc định, `train.py` sẽ đọc từ:

- `adain_baseline/debug_data`

Bạn có thể đổi bằng tham số `--root_dir`.

## Train model

Chạy từ thư mục gốc của project:

```bash
python -m src.train
```

Ví dụ với tham số tùy chỉnh:

```bash
python -m src.train --root_dir adain_baseline/debug_data --checkpoint_dir adain_baseline/checkpoints --epochs 5 --batch_size 8 --lr 1e-4
```

### Tham số chính

- `--root_dir`: thư mục dữ liệu train
- `--image_size`: kích thước ảnh đầu vào
- `--batch_size`: batch size
- `--lr`: learning rate
- `--lambda_style`: trọng số style loss
- `--epochs`: số epoch
- `--val_split`: tỉ lệ validation
- `--num_workers`: số worker cho dataloader
- `--checkpoint_dir`: nơi lưu checkpoint
- `--resume_from`: đường dẫn checkpoint để tiếp tục train

### Kết quả train

Sau khi train, script sẽ lưu:

- `best_model.pth`
- `adain_checkpoint_epoch_*.pth`
- `history.csv`

trong thư mục `--checkpoint_dir`.

## Inference

Chạy inference bằng checkpoint đã train:

```bash
python -m src.infer --checkpoint adain_baseline/checkpoints/best_model.pth --content_dir path/to/content --style_dir path/to/style --output_dir outputs/infer
```

### Tham số chính

- `--checkpoint`: file checkpoint hoặc state dict
- `--content_dir`: thư mục ảnh content
- `--style_dir`: thư mục ảnh style
- `--output_dir`: thư mục lưu ảnh kết quả
- `--alpha`: mức trộn content/style
- `--size`: kích thước ảnh khi inference

### Output

Mỗi cặp content/style sẽ tạo một file ảnh trong `--output_dir` với tên dạng:

- `result_<content>_<style>.jpg`

## Lưu ý khi chạy

- Khuyến nghị chạy bằng `python -m src.train` và `python -m src.infer` từ thư mục gốc project.
- Nếu dùng checkpoint resume, `train.py` sẽ nạp `model_state_dict`, `optimizer_state_dict` và `epoch` từ file checkpoint.
- `infer.py` hiện dùng model `AdaINStyleTransfer` và sẽ load checkpoint theo cả hai dạng: full checkpoint hoặc state dict thuần.

## Gợi ý kiểm tra nhanh

Nếu bạn muốn xác nhận môi trường trước khi train, hãy kiểm tra:

```bash
python -c "import torch; print(torch.__version__)"
```

và đảm bảo các package ở phần Yêu cầu đã sẵn sàng.

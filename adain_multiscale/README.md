# AdaIN Multi-Scale Style Transfer

Project này cài đặt mô hình style transfer dựa trên AdaIN với encoder VGG19 cố định và decoder đa tầng để tiêm style theo nhiều cấp độ feature.

## Cấu trúc thư mục

- `DataSet/DataLoader.py`: xây dựng `Dataset` và `DataLoader`
- `Model/adain_multiscale.py`: encoder, decoder và model chính
- `Loss/loss.py`: hàm loss cho training
- `Train/trainer.py`: logic train/validate theo batch
- `Train/train.py`: pipeline huấn luyện, lưu checkpoint và ảnh mẫu

## Yêu cầu dữ liệu

Cần chuẩn bị dữ liệu theo cấu trúc sau:

```text
dataset/
  content/
    img1.jpg
    img2.jpg
  style/
    style1.jpg
    style2.jpg
```

Ảnh sẽ được resize và normalize theo ImageNet trước khi đưa vào VGG19.

## Cài đặt

Cài các package cần thiết cho môi trường Python:

```bash
pip install torch torchvision pillow
```

Nếu bạn dùng GPU, hãy cài đúng bản `torch`/`torchvision` phù hợp với CUDA trên máy.

## Chạy training

File `Train/train.py` hiện cung cấp hàm `train_pipeline`, nên cách chạy ổn định nhất là tạo một script khởi chạy riêng ở thư mục gốc, ví dụ `run_train.py`.

```python
import torch
from Model.adain_multiscale import AdaINStyleTransfer
from DataSet.DataLoader import build_dataloaders
from Train.train import train_pipeline

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AdaINStyleTransfer().to(device)
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=1e-4)

    train_loader, val_loader = build_dataloaders(
        root_dir="dataset",
        image_size=256,
        pair_mode="cycle",
        batch_size=8,
        val_split=0.2,
        num_workers=4,
    )

    train_pipeline(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir="checkpoints",
        epochs=100,
        scheduler=None,
        lambda_style=10.0,
        lambda_content=1.0,
        patience=15,
        min_delta=1e-4,
        resume_path=None,
    )

if __name__ == "__main__":
    main()
```

Sau đó chạy:

```bash
python run_train.py
```

Kết quả training sẽ được lưu trong thư mục `checkpoints/`:

- `history.csv`: log loss theo epoch
- `epoch_*.pth`: checkpoint từng epoch
- `best_model.pth`: model tốt nhất theo validation loss
- `samples/`: ảnh content/style/result minh họa theo epoch

## Chạy model để sinh ảnh

Model nhận vào 2 tensor ảnh đã normalize theo ImageNet: `content` và `style`, rồi trả về ảnh đã stylize.

Ví dụ inference:

```python
import torch
from PIL import Image
from torchvision import transforms
from Model.adain_multiscale import AdaINStyleTransfer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AdaINStyleTransfer().to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

content = transform(Image.open("content.jpg").convert("RGB")).unsqueeze(0).to(device)
style = transform(Image.open("style.jpg").convert("RGB")).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(content, style, alpha=1.0)
```

Tham số `alpha` dùng để trộn giữa ảnh gốc và ảnh stylize:

- `alpha = 1.0`: ưu tiên style mạnh nhất
- `alpha = 0.0`: giữ nguyên ảnh content
- giá trị ở giữa: cân bằng giữa content và style

## Lưu ý khi chạy

- Hãy chạy lệnh từ thư mục gốc của project để các import kiểu `Train/...`, `Model/...`, `Loss/...` hoạt động đúng.
- Mô hình dùng `torchvision.models.vgg19(weights=...)`, nên lần đầu có thể tải weight pretrained từ Internet.
- Đầu vào của model phải là ảnh RGB đã normalize theo ImageNet.
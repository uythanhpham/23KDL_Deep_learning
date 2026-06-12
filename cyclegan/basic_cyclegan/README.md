# cyclegan-baseline

Project này train **CycleGAN truyền thống** theo đúng kịch bản dataset của bạn:

```text
D:/data/processed/cyclegan/
  monet/
    trainA/  # photo/content
    trainB/  # Monet painting
    testA/
    testB/
  vangogh/
    trainA/
    trainB/
    testA/
    testB/
  ukiyoe/
    trainA/
    trainB/
    testA/
    testB/
  cezanne/
    trainA/
    trainB/
    testA/
    testB/
```

Trong code này:

- `A = photo/content domain`
- `B = style/art domain`
- `G_A2B = photo -> tranh style`
- `G_B2A = style -> photo`
- Train 4 model riêng: `monet`, `vangogh`, `ukiyoe`, `cezanne`
- Không trộn 4 style vào một `trainB` chung.

---

## 1. Cài môi trường

Mở PowerShell/CMD trong folder `cyclegan-baseline`, rồi chạy:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Nếu bạn dùng CUDA, hãy cài PyTorch CUDA theo lệnh phù hợp máy bạn từ trang PyTorch. Sau đó kiểm tra:

```bat
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 2. Kiểm tra dataset

```bat
scripts\00_check_data.bat
```

Hoặc chạy trực tiếp:

```bat
python -m src.inspect_data --root "D:/data/processed/cyclegan"
```

Nếu mọi thứ đúng, bạn sẽ thấy đủ 4 style và 4 split `trainA/trainB/testA/testB`.

---

## 3. Train một style

Ví dụ train Monet:

```bat
scripts\train_monet.bat
```

Hoặc:

```bat
python -m src.train --config configs/train_monet.yaml
```

Các style khác:

```bat
scripts\train_vangogh.bat
scripts\train_ukiyoe.bat
scripts\train_cezanne.bat
```

---

## 4. Train cả 4 style

```bat
scripts\train_all_4styles.bat
```

Lưu ý: file này train tuần tự, không train song song. Nếu team bạn có 3 máy, nên chia:

- Máy 1: `monet`
- Máy 2: `vangogh`
- Máy 3: `ukiyoe`
- Sau đó một máy train tiếp `cezanne`

---

## 5. Output sau khi train

```text
outputs/
  checkpoints/
    monet/latest.pth
    vangogh/latest.pth
    ukiyoe/latest.pth
    cezanne/latest.pth

  samples/
    monet/epoch_001.jpg
    ...

  logs/
    monet/train_log.csv
```

Mỗi ảnh sample có 2 hàng:

```text
real_A -> fake_B -> rec_A
real_B -> fake_A -> rec_B
```

---

## 6. Infer photo -> style

Ví dụ Monet:

```bat
scripts\infer_monet_A2B.bat
```

Hoặc chạy trực tiếp:

```bat
python -m src.infer ^
  --config configs/train_monet.yaml ^
  --checkpoint outputs\checkpoints\monet\latest.pth ^
  --direction A2B ^
  --input_dir "D:\data\processed\cyclegan\monet\testA" ^
  --output_dir "outputs\inference\monet\A2B" ^
  --max_images 50
```

---

## 7. Chỉnh config cho GPU yếu như RTX 3050

Config mặc định đã chọn hướng an toàn:

```yaml
batch_size: 1
ngf: 32
ndf: 32
n_res_blocks: 6
use_amp: true
crop_size: 256
```

Nếu vẫn tràn VRAM:

1. Giữ `batch_size: 1`.
2. Giảm `crop_size: 192`, `image_size: 224`.
3. Giữ `ngf/ndf = 32`.
4. Tắt phần mềm đang chiếm GPU.
5. Chạy từng style, không train song song.

Nếu GPU mạnh hơn và muốn sát CycleGAN official hơn:

```yaml
ngf: 64
ndf: 64
n_res_blocks: 9
epochs: 100
epochs_decay: 100
```

---

## 8. Train debug nhanh

Muốn test pipeline trước, chạy:

```bat
python -m src.train --config configs/train_monet.yaml --max_steps_per_epoch 50 --epochs 1
```

Lệnh này chỉ chạy 1 epoch và 50 step để xem code/dataset có ổn không.

---

## 9. Resume training

```bat
python -m src.train ^
  --config configs/train_monet.yaml ^
  --resume outputs\checkpoints\monet\latest.pth
```

---

## 10. Ghi chú báo cáo

Project này là **CycleGAN gốc/truyền thống**, chưa thêm palette-guided branch. Dữ liệu vẫn giữ đúng cấu trúc:

- Mỗi style là một model riêng.
- `trainA` là photo/content.
- `trainB` là tranh style.
- `testA/testB` không đưa vào train.
- Không trộn Monet/Van Gogh/Ukiyo-e/Cezanne vào cùng một domain.

Sau khi baseline này chạy ổn, mới nên mở rộng sang `palette-guided CycleGAN`.

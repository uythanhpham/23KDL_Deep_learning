# Palette-guided CycleGAN (CycleGAN cải tiến)

Bản **cải tiến** của [basic_cyclegan](../basic_cyclegan/): thay vì mỗi style phải train một model riêng và style bị "nướng cứng" vào trọng số, generator được **điều kiện hóa theo bảng màu (palette)** — cho phép điều hướng tông màu đầu ra bằng một vector palette bất kỳ tại inference.

## Ý tưởng chính

```text
Palette 24-d ──► MappingNetwork (24→256→512→1024→9216) ──► style vector
(6 màu LAB + 6 trọng số)                                       │ cắt thành (scale, shift)
                                                               ▼
Ảnh vào ──► Encoder ──► 9 × ResnetBlock có AdaIN ──► Decoder ──► Ảnh ra
            (InstanceNorm thường)  ▲ 2 AdaIN/block   (InstanceNorm thường)
```

- **MappingNetwork** (`src/models/networks.py`): MLP biến palette 24-d thành style vector 9216-d (= 9 blocks × 2 AdaIN × 2 tham số × 256 kênh), tương tự ý tưởng mapping network của StyleGAN.
- **AdaIN trong res-blocks**: chỉ phần res-blocks dùng AdaIN theo palette; encoder/decoder giữ InstanceNorm thường để bảo toàn cấu trúc.
- Generator nhận `(ảnh, palette)`: `G_A2B(photo, palette_target)` → tranh theo tông màu mong muốn; `G_B2A(tranh, palette_photo)` → ảnh thật.

## Palette Bank

Mỗi ảnh trong dataset được trích trước **6 màu chủ đạo trong không gian CIELAB + 6 trọng số** → vector 24 chiều, lưu file JSONL (cấu hình tại `data.palette_photo` / `data.palette_art` trong config):

```text
train_palettes_photo.jsonl   # palette của từng ảnh photo (domain A)
train_palettes_art.jsonl     # palette của từng tranh (domain B)
```

Mỗi sample train trả về: palette của ảnh A, palette của ảnh B, và **một palette art ngẫu nhiên** làm target điều hướng. Ảnh không có palette tương ứng được gán vector 0.

Sinh palette bank bằng script trong repo (k-means k=6 trong không gian LAB, chạy từ thư mục `extended_cyclegan`):

```bash
python -m src.data.build_palette_bank \
    --input_dir ../../data/archive/trainA \
    --output_jsonl ../../data/palette_bank/train_palettes_photo.jsonl

python -m src.data.build_palette_bank \
    --input_dir ../../data/archive/trainB \
    --output_jsonl ../../data/palette_bank/train_palettes_art.jsonl
```

Config mặc định đã trỏ tới 2 file trên. Tùy chọn: `--k` (số màu), `--resize`, `--max_pixels` (số pixel sample cho k-means), `--max_images` (smoke test).

## Hàm Loss

So với CycleGAN chuẩn (LSGAN + Cycle + Identity), bổ sung **Palette Loss**:

| Loss | Weight | Mục đích |
|------|--------|----------|
| Adversarial (LSGAN) | 1.0 | Đánh lừa Discriminator |
| Cycle Consistency | λ_A = λ_B = 10.0 | Bảo toàn nội dung qua chu trình A→B→A |
| Identity | 0.5·λ | Ổn định màu khi input đã thuộc domain đích |
| **Palette (L1)** | **0.05** | Kéo màu trung bình của `fake_B` về anchor màu của palette target |

## Cách chạy

Yêu cầu môi trường như basic_cyclegan (`pip install -r requirements.txt`).

```bash
# Kiểm tra dataset
python -m src.inspect_data --root <đường dẫn data root>

# Train Van Gogh (sửa đường dẫn data + palette trong config trước)
python -m src.train --config configs/train_vangogh.yaml

# Resume
python -m src.train --config configs/train_vangogh.yaml --resume outputs/checkpoints/vangogh/latest.pth
```

Trên Windows có sẵn các file `.bat` trong `scripts/` (`train_vangogh.bat`, `infer_vangogh_A2B.bat`, ...).

### Inference

```bash
python -m src.infer \
    --config configs/train_vangogh.yaml \
    --checkpoint outputs/checkpoints/vangogh/latest.pth \
    --direction A2B \
    --input_dir ../../data/archive/testA \
    --output_dir outputs/inference/vangogh/A2B
```

Chọn palette điều hướng:
- Mặc định: bốc ngẫu nhiên 1 palette từ bank (A2B → bank art, B2A → bank photo) với `--seed 42` cố định để tái lập
- `--palette_index N`: chọn palette thứ N trong bank (theo thứ tự tên file) — dùng để minh họa "cùng ảnh, khác palette" trong báo cáo
- `--palette_jsonl <file>`: dùng bank khác với config

## Output

```text
outputs/
  checkpoints/vangogh/latest.pth
  samples/vangogh/epoch_XXX.jpg     # real_A→fake_B→rec_A / real_B→fake_A→rec_B
  logs/vangogh/train_log.csv        # gồm cả cột palette_loss
```

## Cấu hình chính (`configs/train_vangogh.yaml`)

ngf=ndf=64, ResNet 9 blocks, InstanceNorm, ảnh 286→crop 256, batch_size=1, Adam lr=2e-4 (β₁=0.5), 100 epochs + 100 epochs decay tuyến tính, image pool 50, λ_palette=0.05.

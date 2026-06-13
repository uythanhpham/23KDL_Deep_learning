# CycleGAN Baseline

CycleGAN **truyền thống** (Zhu et al. 2017) cho bài toán chuyển ảnh thật ↔ tranh Van Gogh, làm baseline so sánh với bản cải tiến [extended_cyclegan](../extended_cyclegan/) (palette-guided).

## Kiến trúc

- **2 Generator** ResNet (9 res-blocks, ngf=64): `G_A2B` (photo → tranh Van Gogh) và `G_B2A` (tranh → photo)
- **2 Discriminator** PatchGAN 70×70 (ndf=64)
- Instance Normalization, không dropout
- **Image Pool** (buffer 50 ảnh fake cũ) để ổn định Discriminator
- Mixed Precision (AMP)

## Hàm Loss

| Loss | Weight | Mục đích |
|------|--------|----------|
| Adversarial (**LSGAN**) | 1.0 | Generator đánh lừa Discriminator |
| Cycle Consistency | λ = 10.0 | A→B→A ≈ A và B→A→B ≈ B (bảo toàn nội dung) |
| Identity | 0.5·λ | G_A2B(B) ≈ B — ổn định màu sắc |

## Dữ liệu

Config trỏ tới cấu trúc `root/<style>/trainA|trainB|testA|testB` với:

- `A` = photo/content (ảnh thật)
- `B` = style/art (tranh Van Gogh)

Config mặc định trỏ `root: ../../data/cyclegan` — chạy `bash scripts/setup_data.sh` ở repo gốc để tạo layout này (symlink `data/cyclegan/vangogh` → `data/archive`).

Tiền xử lý: Resize 286 → RandomCrop 256 → Normalize [-1, 1].

## Cách chạy

```bash
# Cài môi trường
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Kiểm tra dataset
python -m src.inspect_data --root <data root>

# Train
python -m src.train --config configs/train_vangogh.yaml

# Smoke test nhanh (1 epoch, 50 step)
python -m src.train --config configs/train_vangogh.yaml --max_steps_per_epoch 50 --epochs 1

# Resume
python -m src.train --config configs/train_vangogh.yaml --resume outputs/checkpoints/vangogh/latest.pth
```

### Inference (photo → tranh)

```bash
python -m src.infer \
    --config configs/train_vangogh.yaml \
    --checkpoint outputs/checkpoints/vangogh/latest.pth \
    --direction A2B \
    --input_dir <thư mục ảnh photo> \
    --output_dir outputs/inference/vangogh/A2B
```

`--direction B2A` cho chiều ngược lại; `--max_images N` để giới hạn số ảnh (0 = chạy hết).

### Tạo grid ảnh cho báo cáo

```bash
python -m src.make_report_grid --help
```

## Output

```text
outputs/
  checkpoints/vangogh/latest.pth      # + config_used.yaml ghi lại config đã dùng
  samples/vangogh/epoch_XXX.jpg       # 2 hàng: real_A→fake_B→rec_A / real_B→fake_A→rec_B
  logs/vangogh/train_log.csv
```

## Cấu hình huấn luyện (`configs/train_vangogh.yaml`)

| Tham số | Giá trị |
|---|---|
| Ảnh | resize 286 → crop 256 |
| Batch size | 1 (chuẩn CycleGAN) |
| Optimizer | Adam lr=2e-4, β₁=0.5 |
| Lịch lr | 100 epochs cố định + 100 epochs decay tuyến tính về 0 |
| λ_cycle / λ_identity | 10.0 / 0.5 |
| AMP | bật |
| Seed | 42 |

Nếu GPU yếu (ví dụ RTX 3050) và bị tràn VRAM: giảm `ngf`/`ndf` xuống 32, `n_res_blocks` xuống 6, hoặc `crop_size` xuống 192.

> [!NOTE]
> Code data/config hỗ trợ nhiều style (monet, ukiyoe, cezanne...) theo cấu trúc `root/<style>/`, nhưng phạm vi dự án này chỉ train **vangogh** (config duy nhất trong `configs/`).

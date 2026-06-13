# Style Transfer — Chuyển Đổi Phong Cách Tranh Van Gogh

Dự án môn Deep Learning (lớp 23KDL): xây dựng và **so sánh 3 hướng tiếp cận** Deep Learning cho bài toán Style Transfer dạng unpaired — biến ảnh chụp thông thường thành ảnh mang phong cách hội họa Van Gogh.

## Tổng quan 3 hướng tiếp cận — 5 mô hình

| Hướng | Baseline | Cải tiến | Ý tưởng cải tiến |
|---|---|---|---|
| **AdaIN** (feedforward) | [adain/adain_baseline](adain/adain_baseline/) | [adain/adain_multiscale](adain/adain_multiscale/) | Tiêm style tại **4 cấp độ feature** trong decoder thay vì chỉ 1 lần ở bottleneck |
| **CycleGAN** (adversarial) | [cyclegan/basic_cyclegan](cyclegan/basic_cyclegan/) | [cyclegan/extended_cyclegan](cyclegan/extended_cyclegan/) | Điều khiển style bằng **bảng màu (palette)** qua MappingNetwork + AdaIN |
| **Diffusion** (generative) | — | [diffusion-baseline](diffusion-baseline/) | DDPM/DDIM + AdaIN conditioning + Classifier-Free Guidance |

Chi tiết kiến trúc, cách train/infer/evaluate của từng mô hình xem README trong thư mục tương ứng.

## Cấu trúc repo

```text
23KDL_Deep_learning/
├── adain/
│   ├── adain_baseline/      # AdaIN chuẩn (Huang & Belongie 2017)
│   └── adain_multiscale/    # AdaIN cải tiến: tiêm style đa tầng
├── cyclegan/
│   ├── basic_cyclegan/      # CycleGAN truyền thống (ResNet-9, PatchGAN, LSGAN)
│   └── extended_cyclegan/   # Palette-guided CycleGAN
├── diffusion-baseline/      # Style-guided Diffusion (DDPM/DDIM + AdaIN + CFG)
├── scripts/
│   ├── setup_data.sh        # Tạo layout dữ liệu cho cả 5 pipeline (symlink)
│   └── evaluate_all.sh      # Đánh giá thống nhất output của 5 model
├── data/archive/            # Dataset Van Gogh2Photo (gitignore — tải riêng)
├── requirements.txt         # Dependency gộp cho cả 5 model
└── bao_cao_structure.md     # Dàn ý báo cáo
```

## Cài đặt & chuẩn bị dữ liệu

```bash
# 1. Cài PyTorch bản CUDA phù hợp máy (https://pytorch.org), sau đó:
pip install -r requirements.txt

# 2. Giải nén dataset Van Gogh2Photo vào data/archive/ (trainA/trainB/testA/testB)

# 3. Tạo layout dữ liệu cho các pipeline (symlink, chạy lại được nhiều lần)
bash scripts/setup_data.sh

# 4. (Cho Palette-guided CycleGAN) Sinh palette bank — xem cyclegan/extended_cyclegan/README.md
```

## Dataset

Dataset **Van Gogh2Photo** (họ dataset unpaired của CycleGAN), đặt tại `data/archive/`:

| Tập | Nội dung | Vai trò | Số lượng |
|-----|----------|---------|----------|
| trainA | Ảnh thật (Photo) | Content — train | 6.287 |
| trainB | Tranh Van Gogh | Style — train | 400 |
| testA | Ảnh thật (Photo) | Content — eval | 751 |
| testB | Tranh Van Gogh | Style ref — eval | 400 |

Dữ liệu unpaired: không có cặp ảnh 1-1 giữa hai domain. Ảnh được xử lý về 256×256 trong tất cả pipeline.

## Đánh giá thống nhất

Output của cả 5 mô hình được đánh giá bằng chung một bộ metric, lõi tính toán nằm tại [diffusion-baseline/src/eval/metrics.py](diffusion-baseline/src/eval/metrics.py):

| Metric | Đo gì | Hướng tốt |
|--------|-------|-----------|
| LPIPS | Giữ content (perceptual, so với ảnh gốc) | ↓ |
| DINOv2 Cosine | Giữ ngữ nghĩa | ↑ |
| CLIP Style Score | Mức độ "chất Van Gogh" | ↑ |
| FID / **KID** | Khoảng cách phân phối với tập tranh Van Gogh thật | ↓ |

Chạy cho cả 5 model một lượt (sau khi đã infer từng model trên `testA`):

```bash
bash scripts/evaluate_all.sh
# Override đường dẫn output: ADAIN_OUT=... CYCLEGAN_OUT=... bash scripts/evaluate_all.sh
# Thiếu timm/open_clip: EXTRA_FLAGS="--no_dino --no_clip" bash scripts/evaluate_all.sh
```

Ràng buộc ghép cặp: output mỗi model phải **1 ảnh cho 1 ảnh content** và tên file khi sort trùng thứ tự với thư mục content (AdaIN infer dùng `--pair_mode cycle`). Kết quả lưu tại `outputs/eval/<tên model>.json` + `per_image.csv`.

## Tech stack

PyTorch · torchvision · VGG19 (perceptual/encoder) · AMP Mixed Precision · EMA · lpips · torchmetrics (FID/KID) · open_clip · timm (DINOv2)

## Báo cáo

Dàn ý chi tiết của báo cáo: [bao_cao_structure.md](bao_cao_structure.md)

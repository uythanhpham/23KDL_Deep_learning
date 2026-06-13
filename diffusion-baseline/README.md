# Style-guided Diffusion (DDPM/DDIM + AdaIN Conditioning)

Hướng tiếp cận thứ 3 của dự án (bên cạnh [AdaIN](../adain/) và [CycleGAN](../cyclegan/)): diffusion model pixel-space được điều kiện hóa theo ảnh style, sinh ảnh mang phong cách Van Gogh từ ảnh content (img2img) hoặc từ nhiễu thuần (noise2img).

## Kiến trúc

```text
Ảnh Style ──► StyleEncoder (VGG19 frozen → mean/std 4 tầng → MLP) ──► style_emb (512-d)
                                                                          │
Ảnh Content ──► add_noise(x₀, t) ──► x_t ──► UNet(x_t, t, style_emb) ──► ε̂
                                              │  (AdaIN trong mỗi ResBlock)
                                              ▼
                                   DDPM/DDIM reverse ──► ảnh stylized
```

- **StyleEncoder** (`src/models/style_encoder.py`): VGG19 pretrained (đóng băng) trích 4 feature maps → mean+std mỗi tầng → concat 1920-d → MLP (1920→1024→512, SiLU) → `style_emb`. Có `null_style` learnable cho CFG.
- **UNet** (`src/models/unet.py`): `base_channels=64`, `channel_mults=[1,2,4,8]`, 2 res-blocks/cấp, attention tại 16×16 và 8×8. Timestep dùng Sinusoidal Embedding → MLP (256-d).
- **AdaIN conditioning** — điểm khác biệt chính: mỗi ResidualBlock thay InstanceNorm bằng AdaIN, với γ/β chiếu từ `style_emb` (2 lớp AdaIN/block).
- **Scheduler** (`src/diffusion/scheduler.py`): DDPM 1000 timesteps, **cosine schedule** (Nichol & Dhariwal).
- **DDIM Sampler** (`src/diffusion/ddim.py`): sampling nhảy cóc (mặc định 50 steps), tích hợp **Classifier-Free Guidance** (2 lần forward style/null) + Guidance Rescale (Lin et al. 2023). Hai chế độ: `noise_to_stylized` và `content_to_stylized`.

## Hàm Loss (3 thành phần — `src/losses/diffusion_loss.py`)

| Loss | Công thức | Weight | Mục đích |
|------|-----------|--------|----------|
| Noise | MSE(ε̂, ε) | 1.0 | Học khử nhiễu (DDPM core) |
| Style | Σ MSE Gram(x̂₀) vs Gram(style) trên 4 tầng VGG | 500.0 | Ép phong cách Van Gogh |
| Content | MSE feature relu3_1 của x̂₀ vs x₀ | 0.01 | Giữ cấu trúc không gian |

**t-masking**: style/content loss chỉ tính khi `t < T/5` (khi x̂₀ đủ sạch để VGG đo có ý nghĩa).

## Kỹ thuật huấn luyện

- Adam lr=2e-4, CosineAnnealingLR (T_max=30), gradient clipping 1.0
- **AMP** (autocast + GradScaler), **EMA** decay=0.9999
- **Style Dropout** p=0.15 (thay style_emb bằng null → học nhánh CFG)
- Early stopping (patience=15), batch_size=2 (giới hạn VRAM T4 15.6GB), seed=42

## Cách chạy

```bash
# 1. (Tùy chọn) Sinh debug data để smoke test pipeline
bash scripts/01_prepare_data.sh

# 2. Train — configs/train.yaml mặc định trỏ ../data/archive/trainA (content)
#    và ../data/archive/trainB (style); đổi về debug_data/* nếu chỉ smoke test
bash scripts/02_train.sh

# 3. Sinh ảnh theo configs/sample.yaml (mode, strength, ddim_steps, ...)
bash scripts/03_sample.sh

# 4. Đánh giá
bash scripts/04_evaluate.sh
```

### Inference hàng loạt cho đánh giá (`src/infer.py`)

```bash
python -m src.infer \
    --checkpoint checkpoints/best_model.pth \
    --content_dir ../data/archive/testA \
    --style_dir ../data/archive/testB \
    --out_dir outputs/eval_out \
    --mode img2img --strength 0.8 \
    --guidance_scale 2.0 --guidance_rescale 0.7 --ddim_steps 50
```

Tham số đáng chú ý: `--mode` (`img2img` | `noise`), `--strength` (mức nhiễu hóa content, 0–1), `--guidance_scale` (CFG, mức độ ép style), `--num_samples`.

### Đánh giá (`src/evaluate.py` — dùng chung cho mọi model trong dự án)

```bash
python -m src.evaluate \
    --pred_dir outputs/eval_out/output \
    --ref_dir outputs/eval_out/content \
    --style_dir ../data/archive/testB \
    --model_name diffusion \
    --output_file outputs/eval/summary.json
```

Metric: **LPIPS** + **DINOv2 cosine** (giữ content), **CLIP style score** (chất Van Gogh), **FID/KID** (so phân phối với tập tranh thật). Xuất `summary.json` + `per_image.csv`. Flag `--no_dino` / `--no_clip` nếu thiếu timm/open_clip.

## Cấu hình

- `configs/model.yaml` — kiến trúc UNet, StyleEncoder, diffusion (1000 steps, cosine)
- `configs/train.yaml` — data, hyperparameters, loss weights, early stopping
- `configs/sample.yaml` — checkpoint, mode, strength, sampler, ddim_steps

## Checkpoint

Lưu tại `checkpoints/`: gồm `model`, `ema_model`, `style_encoder`, `optimizer`, `loss_weights`. Resume hỗ trợ cả full checkpoint lẫn raw state_dict. Inference nên dùng **EMA weights**.

> [!NOTE]
> `checkpoints/` và `outputs/` trong repo đang rỗng — kết quả train nằm trên máy GPU (T4). Cần tải checkpoint về hoặc train lại trước khi chạy sample/evaluate.

## Tech stack

PyTorch · torchvision · PyYAML · tqdm · lpips · torchmetrics · open_clip_torch · timm

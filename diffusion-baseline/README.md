# Style-guided Diffusion (DDPM/DDIM + AdaIN Conditioning)

Hướng tiếp cận thứ 3 của dự án (bên cạnh [AdaIN](../adain/) và [CycleGAN](../cyclegan/)): diffusion model **pixel-space** được điều kiện hóa theo ảnh style, sinh ảnh mang phong cách Van Gogh từ ảnh content (img2img) hoặc từ nhiễu thuần (noise2img).

## 1. Diffusion model là gì?

Lấy cảm hứng từ nhiệt động học: như giọt mực lan dần trong nước đến khi đồng nhất.

- **Forward process:** thêm dần nhiễu Gauss vào ảnh sạch $x_0$ qua $T$ bước đến khi thành nhiễu trắng. Nhờ tính tái tổ hợp Gauss, lấy mẫu $x_t$ trực tiếp: $x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$.
- **Reverse process:** huấn luyện UNet khử nhiễu *ngược*. Thay vì dự đoán trực tiếp $x_0$ (rất khó), mục tiêu được đơn giản hóa thành **dự đoán lượng nhiễu $\epsilon$** đã thêm — một bài toán hồi quy MSE thuần, **không có cơ chế đối kháng** nên huấn luyện ổn định hơn GAN.

> **"Style-Guided"** = tiêm vector phong cách (`style_emb`) vào UNet qua AdaIN ở **mọi tầng** — khác với việc chỉ nối (concat) style vào một điểm. Đo được: đổi `style_emb` làm output đổi ~44%, chứng tỏ mạng phản ứng mạnh & đồng đều.

## 2. Kiến trúc

```text
Ảnh Style ──► StyleEncoder (VGG19 frozen → mean/std 4 tầng → MLP) ──► style_emb (512-d)
                                                                          │
Ảnh Content ──► add_noise(x₀, t) ──► x_t ──► UNet(x_t, t, style_emb) ──► ε̂
                                              │  (AdaIN trong mỗi ResBlock)
                                              ▼
                                   DDPM/DDIM reverse ──► ảnh stylized
```

- **StyleEncoder** (`src/models/style_encoder.py`): VGG19 pretrained (đóng băng) trích 4 feature maps → mean+std mỗi tầng → concat **1920-d** → MLP (1920→1024→512, SiLU) → `style_emb`. Có `null_style` learnable cho CFG.
- **UNet** (`src/models/unet.py`, ~72.6M params): `base_channels=64`, `channel_mults=[1,2,4,8]`, 2 res-blocks/cấp, **Self-Attention tại 16×16 và 8×8** (chỉ ở phân giải thấp để khả thi về bộ nhớ). Timestep dùng Sinusoidal Embedding → MLP (256-d).
- **AdaIN conditioning** — điểm khác biệt chính: mỗi ResidualBlock thay InstanceNorm bằng AdaIN, với $\gamma/\beta$ chiếu từ `style_emb` (2 lớp AdaIN/block → tổng **20 tầng AdaIN** toàn UNet).
- **Scheduler** (`src/diffusion/scheduler.py`): DDPM 1000 timesteps, **cosine schedule** (giữ tín hiệu sạch lâu hơn ở $t$ nhỏ — nơi quan trọng để học nét cọ vi mô).
- **DDIM Sampler** (`src/diffusion/ddim.py`): sampling nhảy cóc 1000→**50 steps** ($\eta=0$, tất định), tích hợp **CFG** (2 lần forward style/null) + **Guidance Rescale**. Hai chế độ: `noise_to_stylized` và `content_to_stylized` (img2img, dựa trên SDEdit).

> [!IMPORTANT]
> **Bài học lỗi quan trọng nhất của dự án (`no_grad`).** Ban đầu `style_emb` được tính trong `torch.no_grad()` → MLP không nhận gradient → vector style chỉ là phép chiếu ngẫu nhiên cố định → model **"bơ" hoàn toàn** phong cách. *Khắc phục:* bỏ `no_grad`, đưa MLP vào optimizer để gradient từ Noise Loss chảy ngược về MLP. Đây là ranh giới giữa "model không học style" và "model vẽ được Van Gogh".

## 3. Hàm Loss 3 thành phần (`src/losses/diffusion_loss.py`)

| Loss | Công thức | Weight | Mục đích |
|------|-----------|--------|----------|
| Noise | MSE(ε̂, ε) | 1.0 | Học khử nhiễu (DDPM core, đồng thời giữ content gián tiếp) |
| Style | Σ MSE Gram(x̂₀) vs Gram(style) trên 4 tầng VGG | 500.0 | Ép phong cách Van Gogh |
| Content | MSE feature relu3_1 của x̂₀ vs x₀ | 0.01 | Giữ cấu trúc không gian |

- **Tính trên $\hat{x}_0$:** style/content loss được tính trên *ước lượng ảnh sạch* $\hat{x}_0=(x_t-\sqrt{1-\bar\alpha_t}\,\hat\epsilon)/\sqrt{\bar\alpha_t}$ (giữ gradient), KHÔNG trên không gian nhiễu $\epsilon$ (vốn vô nghĩa với VGG/Gram).
- **Vì sao weight style = 500?** Gram Matrix bị chia cho $C\cdot H\cdot W$ nên giá trị rất nhỏ (~$10^{-3}$); weight lớn để cân bằng gradient với Noise Loss.
- **Vì sao weight content = 0.01?** Noise Loss vốn đã ép giữ content (target khử nhiễu chính là ảnh content); content loss chỉ cần nhẹ để không "kéo ngược" style.
- **t-masking:** style/content loss chỉ tính khi `t < T/5` (≈ t<200); ở $t$ cao, $\hat{x}_0$ còn quá nhiễu, VGG trích đặc trưng vô nghĩa → gradient "bẩn".
- Toàn bộ pipeline loss gói trong một `nn.Module` (`StyleDiffusionLoss`) để tương thích `DataParallel`.

## 4. Classifier-Free Guidance (CFG)

"Núm vặn" cường độ phong cách lúc sinh ảnh, không cần train lại:
- **Train:** Style Dropout $p=0.15$ — 15% mẫu thay `style_emb` bằng `null_style` → mạng học cả nhánh có-style lẫn không-style.
- **Inference:** chạy UNet 2 lần rồi ngoại suy $\epsilon_{cfg}=\epsilon_{null}+s\cdot(\epsilon_{style}-\epsilon_{null})$. Khi $s$ cao, **Guidance Rescale** co $\epsilon_{cfg}$ về cùng std với nhánh style để tránh cháy màu (cho phép đẩy $s$ lên 4–7).

Thí nghiệm sweep guidance scale: xem `src/sweep_guidance.py`.

## 5. Kỹ thuật huấn luyện

- Adam lr=2e-4, CosineAnnealingLR (T_max=30), gradient clipping 1.0
- **AMP** (autocast + GradScaler), **EMA** decay=0.9999 (luôn dùng EMA khi sinh ảnh)
- **Style Dropout** p=0.15 (cho CFG)
- Early stopping (patience=15), **batch_size=2** (hạ từ 4 do OOM trên T4 15.6GB), seed=42

> [!WARNING]
> **Huấn luyện gián đoạn nhiều phiên.** Do session timeout của Kaggle, quá trình train bị ngắt giữa chừng và phải **resume thủ công** qua nhiều phiên. Hệ quả: không thể trích một đường cong hội tụ Loss liền mạch; sự hội tụ được giám sát gián tiếp qua giá trị loss ở các checkpoint cuối + quan sát ảnh mẫu. Đây cũng là một phần lý do điểm số định lượng của Diffusion khiêm tốn trong điều kiện đề tài (under-training + pixel-space-from-scratch + dữ liệu style nhỏ).

## 6. Cách chạy

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
    --mode img2img --strength 0.6 \
    --guidance_scale 2.0 --guidance_rescale 0.7 --ddim_steps 50
```

Tham số đáng chú ý: `--mode` (`img2img` | `noise`), `--strength` (mức nhiễu hóa content, 0–1 — thấp giữ content nhiều/ít style, cao vẽ lại nhiều/nhiều style; **khuyến nghị 0.6** để giữ content), `--guidance_scale` (CFG), `--num_samples`.

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

## 7. Cấu hình

- `configs/model.yaml` — kiến trúc UNet, StyleEncoder, diffusion (1000 steps, cosine)
- `configs/train.yaml` — data, hyperparameters, loss weights, early stopping
- `configs/sample.yaml` — checkpoint, mode, strength, sampler, ddim_steps

## 8. Checkpoint

Lưu tại `checkpoints/`: gồm `model`, `ema_model`, `style_encoder`, `optimizer`, `loss_weights`. Resume hỗ trợ cả full checkpoint lẫn raw state_dict. **Inference nên dùng EMA weights.**

> [!NOTE]
> `checkpoints/` và `outputs/` không nằm trong repo (dung lượng lớn). Tải checkpoint + output đã train sẵn tại [**Google Drive của dự án**](https://drive.google.com/drive/u/0/folders/1-iy75YIsNDqAvMT1GMJj6zQtD2LmFHnK), hoặc train lại theo hướng dẫn trên.

## 9. Tech stack

PyTorch · torchvision · PyYAML · tqdm · lpips · torchmetrics[image] · open_clip_torch · timm

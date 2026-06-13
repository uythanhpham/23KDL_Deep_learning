# 📝 Đề Xuất Cấu Trúc Báo Cáo Dự Án

## Thông tin dự án (tổng hợp từ code — cập nhật 06/2026)

| Thành phần | Chi tiết |
|---|---|
| **Đề tài** | Style Transfer — Chuyển đổi phong cách nghệ thuật (Van Gogh) cho ảnh |
| **3 hướng tiếp cận** | (1) AdaIN feedforward, (2) CycleGAN unpaired translation, (3) Style-guided Diffusion |
| **5 mô hình** | AdaIN baseline → AdaIN Multi-scale (cải tiến) · CycleGAN baseline → Palette-guided CycleGAN (cải tiến) · Style-guided Diffusion (DDPM + DDIM + AdaIN conditioning + CFG) |
| **Dataset** | Van Gogh2Photo — trainA (Photo): 6.287 ảnh, trainB (Van Gogh): 400 ảnh, testA (Photo): 751 ảnh, testB (Van Gogh): 400 ảnh |
| **Tech stack** | PyTorch, torchvision, VGG19, Mixed Precision (AMP), EMA, LPIPS/FID/KID/CLIP/DINOv2 |

### Cấu trúc repo hiện tại

```text
23KDL_Deep_learning/
├── adain/
│   ├── adain_baseline/      # AdaIN chuẩn (Huang & Belongie 2017): train + infer + evaluate
│   └── adain_multiscale/    # Cải tiến: tiêm style đa tầng (4 cấp feature) trong decoder
├── cyclegan/
│   ├── basic_cyclegan/      # CycleGAN truyền thống (ResNet-9, PatchGAN, LSGAN)
│   └── extended_cyclegan/   # Cải tiến: Palette-guided (palette LAB 24-d → MappingNetwork → AdaIN)
├── diffusion-baseline/      # Style-guided Diffusion (DDPM/DDIM + AdaIN + CFG)
└── data/archive/            # Van Gogh2Photo: trainA/trainB/testA/testB
```

> [!IMPORTANT]
> So với phiên bản cấu trúc báo cáo trước: AdaIN và CycleGAN **không còn là skeleton** — cả 4 baseline đã hiện thực đầy đủ. Báo cáo trình bày theo khung "3 hướng tiếp cận **có vai trò ngang nhau**, mỗi hướng có baseline → cải tiến" (riêng Diffusion là 1 model); kết quả so sánh công bằng trên cùng dataset và bộ metric.

---

## Cấu Trúc Báo Cáo Đề Xuất

### Chương 1: Tổng Quan Đề Tài
- **1.1 Đặt vấn đề**: Giới thiệu bài toán Style Transfer — chuyển đổi phong cách nghệ thuật của hình ảnh. Tại sao đây là bài toán thú vị và có ứng dụng thực tiễn?
- **1.2 Mục tiêu**: Xây dựng và **so sánh có hệ thống 3 hướng tiếp cận** Deep Learning cho bài toán biến ảnh thường thành ảnh mang phong cách Van Gogh: feedforward (AdaIN), adversarial (CycleGAN), và generative diffusion
- **1.3 Phạm vi**: Style Transfer dạng unpaired (không cần cặp ảnh 1-1), phong cách Van Gogh, ảnh 256×256
- **1.4 Đóng góp chính**:
  - Hiện thực và so sánh 3 họ phương pháp trên cùng một dataset và bộ metric
  - Cải tiến AdaIN: tiêm style **đa tầng** (multi-scale) thay vì chỉ ở bottleneck
  - Cải tiến CycleGAN: điều khiển style bằng **bảng màu (palette)** qua MappingNetwork + AdaIN
  - Đề xuất Style-guided Diffusion: DDPM/DDIM + AdaIN conditioning + loss 3 thành phần + CFG

---

### Chương 2: Cơ Sở Lý Thuyết

> [!IMPORTANT]
> Đây là phần **nền tảng** của báo cáo. Cần giải thích rõ ràng các khái niệm để người đọc hiểu được phần hiện thực. Sắp xếp theo thứ tự tăng dần độ phức tạp: NST → AdaIN → GAN/CycleGAN → Diffusion.

- **2.1 Neural Style Transfer truyền thống**
  - Gatys et al. (2015) — Optimization-based
  - Gram Matrix và ý nghĩa trích xuất phong cách

- **2.2 Adaptive Instance Normalization (AdaIN)**
  - Công thức AdaIN: $\text{AdaIN}(x, y) = \sigma(y) \left(\frac{x - \mu(x)}{\sigma(x)}\right) + \mu(y)$
  - Ưu điểm: real-time, feedforward, arbitrary style
  - *Lưu ý quan trọng cho người đọc*: AdaIN xuất hiện ở **3 nơi** trong dự án với 3 vai trò khác nhau — (1) cơ chế style transfer chính trong adain_baseline/multiscale, (2) lớp điều kiện hóa theo palette trong extended_cyclegan, (3) cơ chế conditioning style trong UNet của diffusion

- **2.3 GAN và CycleGAN**
  - GAN cơ bản: Generator vs Discriminator, adversarial loss
  - LSGAN (least-squares loss) — biến thể dùng trong dự án
  - CycleGAN: kiến trúc 2 Generator + 2 Discriminator, Cycle Consistency Loss, Identity Loss, PatchGAN Discriminator, Image Pool (buffer 50 ảnh)

- **2.4 Biểu diễn màu sắc và Palette** *(nền tảng cho extended_cyclegan)*
  - Không gian màu CIELAB — vì sao phù hợp đo khoảng cách màu tri giác
  - Palette = 6 màu chủ đạo (LAB) + 6 trọng số → vector 24 chiều
  - Ý tưởng mapping network biến vector điều kiện thành tham số AdaIN (liên hệ StyleGAN/StarGAN v2)

- **2.5 Denoising Diffusion Probabilistic Models (DDPM)**
  - Forward Process (thêm nhiễu): $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$
  - Reverse Process (khử nhiễu): huấn luyện UNet dự đoán $\epsilon_\theta(x_t, t)$
  - Noise Schedule: Linear vs **Cosine** (Nichol & Dhariwal 2021 — dùng trong dự án)
  - Hàm Loss chính: $L_{noise} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$

- **2.6 DDIM (Denoising Diffusion Implicit Models)**
  - Tăng tốc sampling bằng cách nhảy cóc timesteps (1000 → 50 steps)
  - Công thức bước lùi DDIM
  - Tham số $\eta$ kiểm soát tính ngẫu nhiên (0 = deterministic, 1 = DDPM)

- **2.7 Classifier-Free Guidance (CFG)**
  - Ý tưởng: huấn luyện cả nhánh có-style lẫn không-style
  - Công thức: $\epsilon_{cfg} = \epsilon_{null} + s \cdot (\epsilon_{style} - \epsilon_{null})$
  - Guidance Rescale (Lin et al. 2023) để giảm artifact

- **2.8 VGG19 Feature Extraction & Perceptual Loss**
  - Trích xuất feature từ các layer [relu1_1, relu2_1, relu3_1, relu4_1] cho Style Loss
  - Layer relu3_1 cho Content Loss
  - Gram Matrix Loss vs Mean/Std Matching Loss (AdaIN-style) — dự án dùng cả hai (diffusion dùng Gram, AdaIN dùng mean/std)

---

### Chương 3: Dữ Liệu (Dataset)

- **3.1 Nguồn dữ liệu**: Dataset Van Gogh2Photo (họ dataset CycleGAN unpaired)
- **3.2 Phân bố dữ liệu** *(⚠ đã sửa so với bản trước — A là Photo, B là tranh)*:

| Tập | Loại | Vai trò | Số lượng |
|-----|------|---------|----------|
| trainA | Ảnh thật (Photo) | Content — train | 6.287 |
| trainB | Tranh Van Gogh | Style — train | 400 |
| testA | Ảnh thật (Photo) | Content — eval | 751 |
| testB | Tranh Van Gogh | Style ref — eval | 400 |

- **3.3 Tiền xử lý dữ liệu** — mỗi pipeline một kiểu, nên trình bày bằng bảng so sánh:
  - **AdaIN (cả 2 bản)**: Resize/Crop 256 → Normalize **ImageNet** (vì VGG19 encoder đóng băng)
  - **CycleGAN (cả 2 bản)**: Resize 286 → RandomCrop 256 → Flip → Normalize **[-1, 1]**
  - **Diffusion**: Content: Resize 1.12× → RandomCrop(256) → Flip → [-1, 1]; Style: Resize(288) → CenterCrop(256) → [-1, 1] (`StyleEncoder._to_vgg_input()` tự chuyển sang ImageNet norm bên trong)
- **3.4 Chiến lược ghép cặp unpaired**: Content load tuần tự, Style chọn ngẫu nhiên (AdaIN/Diffusion); CycleGAN ghép ngẫu nhiên 2 domain mỗi iteration
- **3.5 Palette Bank** *(riêng cho extended_cyclegan)*:
  - Mỗi ảnh được trích 6 màu chủ đạo trong không gian LAB + 6 trọng số → vector 24-d, lưu file JSONL (`train_palettes_photo.jsonl`, `train_palettes_art.jsonl`)
  - Mỗi sample train trả về: palette của A, palette của B, và 1 palette art **ngẫu nhiên** làm target điều hướng
  - *Cần bổ sung vào báo cáo*: quy trình sinh palette (thuật toán phân cụm màu, số cụm = 6) — hiện chỉ có file kết quả, script sinh palette nằm ngoài repo

---

### Chương 4: Các Phương Pháp (⭐ Phần quan trọng nhất)

> [!TIP]
> Trình bày theo 3 hướng **với dung lượng tương đương nhau**, mỗi hướng đi từ baseline → cải tiến. Mỗi phương pháp nên có 1 sơ đồ kiến trúc.

#### 4.1 Hướng 1 — AdaIN Feedforward

- **4.1.1 AdaIN Baseline** (`adain/adain_baseline`)
  - VGG19 encoder (pretrained, đóng băng) → trích feature relu4_1
  - AdaIN một lần tại bottleneck: căn mean/std của content feature theo style feature
  - Decoder đối xứng VGG (ReflectionPad + Conv + Upsample) sinh ảnh
  - Loss: $L = L_{content} + \lambda_{style} L_{style}$ với $\lambda_{style} = 10$; style loss = matching mean/std trên 4 tầng VGG (đúng paper gốc)
  - Inference: tham số `alpha` trộn content/style

- **4.1.2 AdaIN Multi-scale** (`adain/adain_multiscale`) — *cải tiến*
  - Khác biệt chính: decoder **tiêm style tại 4 cấp độ feature** (relu1_1 → relu4_1) thay vì chỉ 1 lần ở bottleneck — mỗi giai đoạn upsample được AdaIN lại với stats của tầng style tương ứng
  - Động cơ: style không chỉ nằm ở feature sâu (bố cục màu) mà cả feature nông (nét cọ, texture) → tiêm đa tầng giữ được chi tiết phong cách tốt hơn
  - Alpha blending ở mức ảnh đầu ra: $out = \alpha \cdot stylized + (1-\alpha) \cdot content$

#### 4.2 Hướng 2 — CycleGAN

- **4.2.1 CycleGAN Baseline** (`cyclegan/basic_cyclegan`)
  - 2 Generator ResNet (9 res-blocks, ngf=64) + 2 PatchGAN Discriminator (ndf=64)
  - LSGAN loss + Cycle Consistency ($\lambda=10$) + Identity Loss ($0.5\lambda$)
  - Image Pool 50 ảnh để ổn định Discriminator, Instance Norm, AMP
  - `G_A2B`: photo → tranh Van Gogh; `G_B2A`: tranh → photo

- **4.2.2 Palette-guided CycleGAN** (`cyclegan/extended_cyclegan`) — *cải tiến*
  - **MappingNetwork**: palette 24-d → MLP (24→256→512→1024→9216) → style vector
  - Style vector cắt thành (scale, shift) cấp cho **2 lớp AdaIN trong mỗi ResNet block** (9 blocks × 2 AdaIN × 2 tham số × 256 kênh = 9216)
  - Generator nhận `(ảnh, palette)`: encoder/decoder giữ Instance Norm thường, chỉ phần res-blocks dùng AdaIN theo palette
  - **Palette Loss**: L1 giữa màu trung bình của `fake_B` và anchor màu của palette target, $\lambda_{palette} = 0.05$
  - Ý nghĩa: một model điều khiển được tông màu đầu ra theo palette bất kỳ (thay vì cố định 1 style/model)

#### 4.3 Hướng 3 — Style-guided Diffusion (`diffusion-baseline`)

- **4.3.1 Tổng quan Pipeline**
  1. Style Encoder mã hóa ảnh Style → vector style_emb (512-d)
  2. UNet nhận (x_t, t, style_emb) → dự đoán noise ε
  3. Scheduler khử nhiễu từng bước → ảnh sắc nét

- **4.3.2 Style Encoder**
  - VGG19 pretrained (frozen, eval mode) → trích xuất 4 feature maps
  - Tính Mean + Std trên (H,W) cho mỗi feature → concat → 1920-d vector
  - MLP (1920 → 1024 → 512) với SiLU activation → style_emb
  - `null_style` parameter cho CFG (learnable zeros vector)

- **4.3.3 UNet Backbone**
  - Cấu hình: `base_channels=64`, `channel_mults=[1,2,4,8]`, `num_res_blocks=2`
  - **Sinusoidal Position Embedding** → MLP → Timestep Embedding (256-d)
  - **ResidualBlock**: Conv → AdaIN(style) → SiLU → +t_proj → Conv → AdaIN(style) → Dropout → Skip
  - **AttentionBlock**: GroupNorm → MultiheadAttention (4 heads) → Residual (chỉ ở resolution 16×16 và 8×8)
  - **DownBlock** → Bottleneck → **UpBlock** (skip connections)
  - Downsample: Conv2d(stride=2), Upsample: ConvTranspose2d(stride=2)

- **4.3.4 AdaIN Conditioning trong UNet** (điểm khác biệt chính)
  - Thay vì InstanceNorm thông thường, mỗi ResidualBlock dùng AdaIN:
    $\text{out} = \gamma(s) \cdot \frac{h - \mu(h)}{\sigma(h)} + \beta(s)$
  - $\gamma, \beta$ được tính từ style_emb qua Linear projection
  - Mỗi block có 2 lớp AdaIN (sau mỗi Conv)
  - *Điểm hay để nhấn trong báo cáo*: cùng một ý tưởng "điều kiện hóa bằng AdaIN" xuyên suốt cả extended_cyclegan và diffusion — thể hiện tính nhất quán của dự án

- **4.3.5 Diffusion Scheduler (DDPMScheduler)**
  - Cosine schedule (Nichol & Dhariwal): tránh ảnh bị phá hủy quá nhanh
  - Forward: `add_noise(x0, t)`
  - Predict x0: $\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta}{\sqrt{\bar{\alpha}_t}}$
  - Reverse step: tính $\mu_\theta$ và $\sigma$ từ $\epsilon_\theta$

- **4.3.6 DDIM Sampler**
  - Nhảy cóc timesteps (mặc định 50 steps thay vì 1000)
  - Hỗ trợ 2 chế độ: `noise_to_stylized` (txt2img) và `content_to_stylized` (img2img)
  - CFG tích hợp: 2 lần forward (style + null) → interpolation
  - Guidance Rescale để giảm nhiễu artifact

---

### Chương 5: Huấn Luyện (Training)

- **5.1 Bảng so sánh cấu hình huấn luyện 5 mô hình** *(nên là bảng tổng hợp đầu chương)*:

| | AdaIN base | AdaIN multi-scale | CycleGAN base | Palette CycleGAN | Diffusion |
|---|---|---|---|---|---|
| Tham số học | Decoder | Decoder | 2G + 2D | 2G (+Mapping) + 2D | UNet + StyleEnc MLP |
| Loss | content + 10·style | content + 10·style | LSGAN + 10·cycle + 0.5·idt | + 0.05·palette (L1) | noise + 500·style + 0.01·content |
| Optimizer | Adam 1e-4 | Adam 1e-4 | Adam 2e-4, β₁=0.5 | Adam 2e-4, β₁=0.5 | Adam 2e-4 |
| Batch size | 8 | 8 | 1 | 1 | 2 |
| Epochs | — (điền thực tế) | — | 100 + 100 decay | 100 + 100 decay | 30 |
| Kỹ thuật | early stop | early stop | AMP, image pool, lr decay tuyến tính | như base | AMP, EMA, CFG dropout, cosine LR, early stop |

- **5.2 Huấn luyện AdaIN (2 bản)**
  - Chỉ decoder được học; VGG19 encoder đóng băng → ít tham số, hội tụ nhanh
  - Loss: $L_{content} + 10 \cdot L_{style}$ — style loss matching mean/std trên 4 tầng VGG (theo paper gốc); **cùng dạng loss cho cả 2 bản** để khác biệt kết quả chỉ đến từ kiến trúc decoder (1 tầng vs 4 tầng tiêm style)
  - Adam lr=1e-4, val_split 0.2, early stopping (patience=15), checkpoint `best_model.pth` theo val loss

- **5.3 Huấn luyện CycleGAN (2 bản)**
  - Train xen kẽ G/D mỗi iteration; LSGAN ổn định hơn vanilla GAN
  - **Image Pool 50 ảnh**: D học trên ảnh fake cũ → giảm dao động
  - Lịch lr 2 pha: 100 epochs cố định + 100 epochs decay tuyến tính về 0 (chuẩn CycleGAN)
  - Bản palette thêm: mỗi batch bốc 1 palette art ngẫu nhiên làm target; palette loss L1 (λ=0.05) kéo màu trung bình fake_B về anchor
  - AMP; batch_size=1 (chuẩn CycleGAN với InstanceNorm)

- **5.4 Huấn luyện Diffusion**
  - **Hàm Loss 3 thành phần**:

| Loss | Công thức | Weight | Mục đích |
|------|-----------|--------|----------|
| Noise Loss | $MSE(\epsilon_{pred}, \epsilon_{true})$ | 1.0 | Học khử nhiễu (DDPM core) |
| Style Loss | $\sum_l MSE(G(\hat{x}_0^l), G(s^l))$ | 500.0 | Ép Gram Matrix → phong cách Van Gogh |
| Content Loss | $MSE(f(\hat{x}_0), f(x_0))$ | 0.01 | Giữ cấu trúc không gian |

  - **t-masking**: Perceptual loss chỉ tính khi $t < T/5$ (khi $\hat{x}_0$ đủ sạch cho VGG)
  - Style loss tính trên dự đoán $\hat{x}_0$ (có gradient) so với ảnh style gốc (no_grad)
  - **StyleDiffusionLoss** là `nn.Module` (không phải hàm) → hỗ trợ DataParallel
  - Optimizer: Adam (lr=0.0002), CosineAnnealingLR (T_max=30, eta_min=1e-6), Gradient Clipping (max_norm=1.0)
  - **Mixed Precision (AMP)** + **EMA** (decay=0.9999) → inference ổn định hơn
  - **Style Dropout** (p=0.15): thay style_emb bằng `null_style` → học nhánh CFG
  - **Early Stopping** (patience=15, min_delta=0.0001)
  - Batch size = 2 (hạ từ 4 do OOM trên T4 15.6GB tại ranh giới epoch — validation + VGG perceptual 256px)

- **5.5 Hạ tầng & checkpoint**
  - GPU: NVIDIA T4 15.6GB (diffusion, Colab/cloud); RTX 3050 (CycleGAN, máy cá nhân — config ngf/ndf=32 dự phòng OOM)
  - Checkpoint diffusion lưu: model, ema_model, style_encoder, optimizer, loss_weights; resume hỗ trợ cả full checkpoint lẫn raw state_dict
  - CycleGAN: `latest.pth` + samples mỗi epoch + `train_log.csv`; AdaIN: `best_model.pth` + `history.csv`

---

### Chương 6: Thí Nghiệm & Đánh Giá

- **6.1 Cấu hình thí nghiệm**: GPU, epochs, image size 256×256, seed 42

- **6.2 Giao thức đánh giá thống nhất** — đã chuẩn hóa bằng `scripts/evaluate_all.sh`
  - Cả 5 model chạy chung bộ metric (lõi: `diffusion-baseline/src/eval/metrics.py`) → 1 bảng so sánh công bằng
  - Quy ước: content = testA (751 photo), style reference cho FID/KID = testB (400 tranh)
  - Ràng buộc: output mỗi model phải 1-ảnh-1-content (AdaIN infer dùng `--pair_mode cycle`); metric phụ LPIPS/SSIM/RMSE của adain_baseline có thể báo cáo thêm để đối chiếu

- **6.3 Các Metric đánh giá**

| Metric | Loại | Ý nghĩa | Hướng tốt |
|--------|------|----------|-----------|
| **LPIPS** | Content (per-image) | Perceptual distance giữa content gốc và output | ↓ thấp hơn = giữ content tốt |
| **DINOv2 Cosine** | Content (per-image) | Semantic similarity (self-supervised) | ↑ cao hơn = giữ ngữ nghĩa |
| **CLIP Style Score** | Style (per-image) | So sánh directional: "Van Gogh painting" vs "photograph" | ↑ cao hơn = style mạnh hơn |
| **FID** | Distribution | Fréchet Inception Distance vs tập Van Gogh | ↓ thấp hơn |
| **KID** ⭐ | Distribution | Kernel Inception Distance (metric chính — ổn định với tập nhỏ 400 ảnh) | ↓ thấp hơn |
| SSIM / RMSE | Content (phụ) | Đối chiếu với kết quả adain_baseline đã đo | ↑ / ↓ |

- **6.4 Kết quả định lượng**
  - **Bảng chính của báo cáo**: 5 model × 5 metric (+ số tham số, thời gian inference / ảnh)
  - Learning curves từng model (đã có sẵn `history.csv` / `train_log.csv`)
  - So sánh tốc độ inference: AdaIN (1 forward) ≪ CycleGAN (1 forward) ≪ Diffusion (50 DDIM steps × 2 CFG)

- **6.5 Kết quả định tính**
  - Grid ảnh chuẩn: Content | AdaIN | AdaIN-MS | CycleGAN | Palette-CG | Diffusion (CycleGAN đã có sẵn `make_report_grid.py`)
  - Riêng Diffusion: so sánh `noise_to_stylized` vs `content_to_stylized`; DDPM vs DDIM; ảnh hưởng của strength (img2img) và CFG scale
  - Riêng Palette-CycleGAN: cùng 1 ảnh + các palette khác nhau → minh họa khả năng điều khiển màu
  - Riêng AdaIN: ảnh hưởng của alpha; baseline vs multi-scale trên cùng cặp ảnh (texture/nét cọ)

- **6.6 Ablation Study** *(nếu có thời gian)*
  - Diffusion: loss weights, style_dropout (CFG), cosine vs linear schedule, số DDIM steps
  - AdaIN: 1 tầng vs 4 tầng tiêm style (chính là baseline vs multiscale — ablation tự nhiên)
  - CycleGAN: có/không palette loss; giá trị $\lambda_{palette}$

---

### Chương 7: Thảo Luận

- **7.1 So sánh 3 hướng tiếp cận** *(bảng + phân tích)*

| Tiêu chí | AdaIN | CycleGAN | Diffusion |
|---|---|---|---|
| Tốc độ inference | Real-time | Nhanh | Chậm (iterative) |
| Chất lượng style | Trung bình | Tốt (theo domain) | Tốt + điều khiển được |
| Arbitrary style | ✅ | ❌ (1 model/style; palette giải quyết một phần) | ✅ (style image bất kỳ) |
| Độ khó huấn luyện | Dễ | Khó (GAN instability) | Trung bình nhưng tốn compute |
| Điều khiển tại inference | alpha | palette (bản extended) | CFG scale, strength, steps |

- **7.2 Ưu điểm từng cải tiến**
  - Multi-scale AdaIN: chi tiết phong cách ở nhiều cấp độ
  - Palette-guided: 1 model — nhiều tông màu, có loss màu tường minh
  - Diffusion + AdaIN conditioning: linh hoạt nhất; DDIM tăng tốc 20×; CFG điều khiển mức style; EMA ổn định output

- **7.3 Hạn chế**
  - Diffusion: batch size nhỏ (2) do VRAM; inference chậm hơn 2 hướng còn lại nhiều lần
  - Style domain nhỏ (400 tranh Van Gogh train) — rủi ro overfit style
  - Palette 24-d chỉ nắm thống kê màu toàn cục (6 màu chủ đạo), chưa mô tả texture/nét cọ
  - Chưa có user study (đánh giá chủ quan)

- **7.4 Trade-off Content vs Style**
  - Giải thích mâu thuẫn: LPIPS thấp (giữ content) vs CLIP style cao (thay đổi nhiều)
  - Loss weights / alpha / CFG scale là các "nút vặn" tương ứng của từng hướng

---

### Chương 8: Kết Luận & Hướng Phát Triển

- **8.1 Kết luận**: Tóm tắt 3 hướng đã hiện thực, kết quả so sánh chính, khuyến nghị phương pháp theo use-case (real-time → AdaIN; chất lượng + điều khiển → Diffusion)
- **8.2 Hướng phát triển**:
  - Thử nghiệm với nhiều phong cách khác (Monet, Ukiyo-e, Cezanne — cấu trúc code CycleGAN đã hỗ trợ sẵn)
  - Nâng độ phân giải (512×512 hoặc cao hơn)
  - Latent Diffusion (Stable Diffusion) thay vì pixel-space diffusion
  - Kết hợp palette conditioning vào diffusion (giao thoa 2 cải tiến của dự án)
  - Distillation/consistency model để giảm số bước sampling

---

## Phụ Lục

- **Phụ lục A**: Kiến trúc chi tiết UNet diffusion (bảng layer-by-layer); kiến trúc decoder AdaIN multi-scale; sơ đồ MappingNetwork palette
- **Phụ lục B**: File cấu hình đầy đủ (model.yaml, train.yaml, sample.yaml, train_vangogh.yaml ×2)
- **Phụ lục C**: Hướng dẫn tái tạo kết quả (reproduce) — gồm cả bước sinh palette bank
- **Phụ lục D**: Bảng phân công công việc nhóm (mỗi thành viên phụ trách hướng nào)

---

## Tài Liệu Tham Khảo (Gợi ý)

**Phương pháp:**
1. Gatys et al. (2015) — *A Neural Algorithm of Artistic Style*
2. Huang & Belongie (2017) — *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization* (AdaIN)
3. Johnson et al. (2016) — *Perceptual Losses for Real-Time Style Transfer*
4. Goodfellow et al. (2014) — *Generative Adversarial Networks*
5. Mao et al. (2017) — *Least Squares Generative Adversarial Networks* (LSGAN)
6. Zhu et al. (2017) — *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks* (CycleGAN)
7. Karras et al. (2019) — *A Style-Based Generator Architecture for GANs* (StyleGAN — mapping network + AdaIN)
8. Ho et al. (2020) — *Denoising Diffusion Probabilistic Models* (DDPM)
9. Song et al. (2021) — *Denoising Diffusion Implicit Models* (DDIM)
10. Nichol & Dhariwal (2021) — *Improved Denoising Diffusion Probabilistic Models* (Cosine schedule)
11. Ho & Salimans (2022) — *Classifier-Free Diffusion Guidance*
12. Lin et al. (2023) — *Common Diffusion Noise Schedules and Sample Steps are Flawed* (Guidance Rescale)

**Metric:**
13. Zhang et al. (2018) — *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric* (LPIPS)
14. Heusel et al. (2017) — *GANs Trained by a Two Time-Scale Update Rule…* (FID)
15. Bińkowski et al. (2018) — *Demystifying MMD GANs* (KID)
16. Radford et al. (2021) — *Learning Transferable Visual Models From Natural Language Supervision* (CLIP)
17. Oquab et al. (2023) — *DINOv2: Learning Robust Visual Features without Supervision*

---

> [!NOTE]
> **Việc còn lại trước khi viết báo cáo** (sau đợt làm sạch repo 12/06/2026):
> 1. **Thu thập kết quả về repo**: checkpoint/ảnh kết quả của cả 5 model đang nằm trên máy train của từng thành viên — cần tập hợp về (hoặc link Drive) để chạy `scripts/evaluate_all.sh` và vẽ hình cho báo cáo
> 2. **Chạy đánh giá thống nhất**: infer mỗi model trên testA (AdaIN dùng `--pair_mode cycle`) rồi chạy `scripts/evaluate_all.sh` → bảng 5 model × 5 metric cho mục 6.4
> 3. Nên vẽ ít nhất 5 sơ đồ: (1) Pipeline so sánh 3 hướng, (2) AdaIN baseline vs multi-scale, (3) Palette-guided generator, (4) Kiến trúc UNet diffusion với AdaIN, (5) Luồng huấn luyện diffusion với 3 loss
> 4. Các công thức toán trong phần lý thuyết đều đã có sẵn trong comment của source code (viết chi tiết bằng tiếng Việt) → có thể trích dẫn trực tiếp
> 5. Khi viết Chương 3, lưu ý domain dataset: trainA = Photo (6.287), trainB = Van Gogh (400) — draft cũ từng ghi ngược
>
> **Đã xử lý trong đợt làm sạch 12/06/2026** (không còn là mâu thuẫn):
> - ✅ Mỗi model có README riêng + README tổng quan ở root
> - ✅ Sửa 2 bug chặn chạy: `adain_baseline` train (ImportError `build_dataloaders`) và `extended_cyclegan` infer (thiếu tham số palette)
> - ✅ Bỏ đường dẫn cứng `D:/...` — config trỏ đường dẫn tương đối + `scripts/setup_data.sh` tạo layout dữ liệu bằng symlink
> - ✅ Palette bank sinh được trong repo: `python -m src.data.build_palette_bank` (k-means LAB, k=6) — đã sinh sẵn 6.287 + 400 palette tại `data/palette_bank/`
> - ✅ Giao thức đánh giá thống nhất: `scripts/evaluate_all.sh` chạy chung bộ metric cho cả 5 model
> - ✅ `adain_multiscale` có entry script chạy được (`run_train.py`, `infer.py`); `requirements.txt` gộp ở root

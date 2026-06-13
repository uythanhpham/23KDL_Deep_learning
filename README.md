<div align="center">

# 🎨 Style Transfer — Chuyển Đổi Phong Cách Tranh Van Gogh

**So sánh có hệ thống 3 hướng tiếp cận Deep Learning** cho bài toán *Unpaired Style Transfer*:
biến ảnh chụp thông thường thành tác phẩm mang phong cách hội họa của Vincent van Gogh.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-enabled-76B900?logo=nvidia&logoColor=white)
![Models](https://img.shields.io/badge/Models-5-success)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

*Đồ án môn **Học sâu cho Khoa học Dữ liệu** — Lớp 23KDL, Khoa Toán–Tin học, ĐH KHTN (ĐHQG-HCM)*

</div>

---

## 📑 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Tổng quan: 3 hướng — 5 mô hình](#-tổng-quan-3-hướng--5-mô-hình)
- [📥 Tải Dataset, Output & Checkpoint](#-tải-dataset-output--checkpoint)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Luồng chạy tổng quan (Pipeline)](#-luồng-chạy-tổng-quan-pipeline)
- [Dataset](#-dataset)
- [Cài đặt & Chuẩn bị dữ liệu](#-cài-đặt--chuẩn-bị-dữ-liệu)
- [Chi tiết kiến trúc 5 mô hình](#-chi-tiết-kiến-trúc-5-mô-hình)
  - [1. AdaIN Baseline](#1️⃣-adain-baseline)
  - [2. AdaIN Multi-Scale](#2️⃣-adain-multi-scale-cải-tiến)
  - [3. CycleGAN Baseline](#3️⃣-cyclegan-baseline)
  - [4. Palette-guided CycleGAN](#4️⃣-palette-guided-cyclegan-cải-tiến)
  - [5. Style-guided Diffusion](#5️⃣-style-guided-diffusion)
- [Bộ Metric đánh giá](#-bộ-metric-đánh-giá)
- [Đánh giá thống nhất cho cả 5 mô hình](#-đánh-giá-thống-nhất-cho-cả-5-mô-hình)
- [Tech stack](#-tech-stack)
- [Thành viên & Credits](#-thành-viên--credits)

---

## 🌟 Giới thiệu

**Style Transfer** là bài toán tái tạo một bức ảnh thực tế (giữ **nội dung** — bố cục, hình
khối, vật thể) dưới lăng kính thị giác của một tác phẩm hội họa (mang **phong cách** — màu
sắc, nét cọ, chất liệu). Thách thức cốt lõi: làm sao để mạng nơ-ron *tách bạch* rồi *tái tổ
hợp* hai thành tố vốn quyện chặt này.

Dự án **không chọn một phương pháp duy nhất**, mà đặt **ba trường phái tiêu biểu** lên cùng
một bàn cân — cùng dataset Van Gogh, cùng hệ thống thang đo, cùng điều kiện phần cứng — để
so sánh công bằng. Trên nền đối chuẩn đó, nhóm đề xuất thêm các **cải tiến** cho từng kiến
trúc.

> [!NOTE]
> Bài toán thuộc dạng **unpaired** (không có cặp ảnh photo↔tranh tương ứng 1-1), mô phỏng
> sát điều kiện thực tế. Toàn bộ ảnh được xử lý ở độ phân giải **256×256**.

---

## 🧭 Tổng quan: 3 hướng — 5 mô hình

| # | Hướng | Mô hình | Thư mục | Mô tả ngắn |
|:-:|---|---|---|---|
| 1 | 🟦 **AdaIN** *(feedforward)* | AdaIN Baseline | [`adain/adain_baseline/`](adain/adain_baseline/) | AdaIN chuẩn (Huang & Belongie 2017); tiêm style **1 lần** tại bottleneck `relu4_1` |
| 2 | 🟦 **AdaIN** *(feedforward)* | AdaIN Multi-Scale ⭐ | [`adain/adain_multiscale/`](adain/adain_multiscale/) | Cải tiến: tiêm style tại **4 cấp độ feature** trong decoder |
| 3 | 🟥 **CycleGAN** *(adversarial)* | CycleGAN Baseline | [`cyclegan/basic_cyclegan/`](cyclegan/basic_cyclegan/) | CycleGAN truyền thống: ResNet-9, PatchGAN 70×70, LSGAN, Cycle + Identity |
| 4 | 🟥 **CycleGAN** *(adversarial)* | Palette-guided CycleGAN ⭐ | [`cyclegan/extended_cyclegan/`](cyclegan/extended_cyclegan/) | Cải tiến: điều khiển màu bằng **palette 24-d** qua MappingNetwork + AdaIN |
| 5 | 🟩 **Diffusion** *(generative)* | Style-guided Diffusion | [`diffusion-baseline/`](diffusion-baseline/) | DDPM/DDIM pixel-space + StyleEncoder + AdaIN conditioning + CFG |

> Ba hướng có **vai trò ngang nhau**; ⭐ = phiên bản cải tiến do nhóm đề xuất.

---

## 📥 Tải Dataset, Output & Checkpoint

> [!IMPORTANT]
> Dataset, ảnh output và checkpoint **không** được đẩy lên repo (dung lượng lớn — đã
> `gitignore`). Tải toàn bộ tại Google Drive:
>
> ### 🔗 [**Dataset + Output + Checkpoint (Google Drive)**](https://drive.google.com/drive/u/0/folders/1-iy75YIsNDqAvMT1GMJj6zQtD2LmFHnK)
>
> - 📦 **Dataset** Van Gogh2Photo → giải nén vào [`data/archive/`](data/)
> - 🖼️ **Output** ảnh stylized của cả 5 mô hình
> - 💾 **Checkpoint** trọng số đã huấn luyện

---

## 📁 Cấu trúc thư mục

<details open>
<summary><b>Cây thư mục đầy đủ (click để thu gọn)</b></summary>

```text
23KDL_Deep_learning/
├── README.md                          # ← FILE NÀY
├── requirements.txt                   # Dependency gộp cho cả 5 mô hình
├── .gitignore
│
├── scripts/
│   ├── setup_data.sh                  # Tạo symlink layout dữ liệu cho mọi pipeline
│   └── evaluate_all.sh                # Chạy evaluate thống nhất cho cả 5 mô hình
│
├── data/
│   ├── archive/                       # Dataset gốc Van Gogh2Photo (gitignore)
│   │   ├── trainA/  (6287)            # ảnh thật (photo)  — content train
│   │   ├── trainB/  (400)             # tranh Van Gogh    — style train
│   │   ├── testA/   (751)             # ảnh thật          — content eval
│   │   └── testB/   (400)             # tranh Van Gogh    — style ref eval
│   ├── adain/                         # Symlink: content→trainA, style→trainB
│   ├── cyclegan/                      # Symlink: vangogh→archive
│   └── palette_bank/                  # JSONL palette cho extended_cyclegan
│       ├── train_palettes_photo.jsonl
│       └── train_palettes_art.jsonl
│
├── adain/
│   ├── adain_baseline/
│   │   ├── README.md
│   │   ├── scripts/                   # .sh + .bat (prepare / infer / evaluate)
│   │   ├── debug_data/                # 10 content + 10 style để smoke test
│   │   └── src/
│   │       ├── train.py · infer.py · evaluate.py · trainer.py
│   │       ├── models/adain.py        # VGG19 (frozen) + AdaIN + Decoder
│   │       ├── losses/perceptual.py   # Content + Style (mean/std matching)
│   │       └── data/{datasets,prepare_data}.py
│   └── adain_multiscale/
│       ├── README.md · run_train.py · infer.py
│       ├── Model/adain_multiscale.py  # Encoder + Multi-scale AdaIN Decoder
│       ├── Loss/loss.py               # StyleTransferLoss
│       ├── Train/{train,trainer}.py
│       └── DataSet/DataLoader.py
│
├── cyclegan/
│   ├── basic_cyclegan/
│   │   ├── README.md · configs/train_vangogh.yaml · requirements.txt
│   │   └── src/
│   │       ├── train.py · infer.py · inspect_data.py · make_report_grid.py
│   │       ├── models/{networks,cyclegan}.py   # ResnetGenerator, PatchGAN, 2G+2D
│   │       ├── data/datasets.py
│   │       └── utils/{image_pool,visualize,config,misc}.py
│   └── extended_cyclegan/
│       ├── README.md · configs/train_vangogh.yaml · requirements.txt · scripts/
│       └── src/
│           ├── train.py · infer.py · inspect_data.py · make_report_grid.py
│           ├── models/{networks,cyclegan}.py   # + MappingNetwork + AdaIN res-blocks
│           ├── data/datasets.py
│           └── data/build_palette_bank.py      # K-means k=6 trong không gian LAB
│
└── diffusion-baseline/
    ├── README.md · evaluate_local.py
    ├── configs/{model,train,sample}.yaml
    ├── scripts/                       # 01_prepare → 02_train → 03_sample → 04_evaluate
    └── src/
        ├── train.py · sample.py · infer.py
        ├── evaluate.py                # ⭐ BỘ EVALUATE CHUNG cho cả 5 mô hình
        ├── sweep_guidance.py          # Thí nghiệm sweep guidance scale
        ├── models/{unet,style_encoder,attention,embeddings}.py
        ├── diffusion/{scheduler,ddim}.py
        ├── losses/diffusion_loss.py   # Noise + Style (Gram) + Content
        ├── trainers/trainer.py
        └── eval/metrics.py            # ⭐ LÕI TÍNH TOÁN: LPIPS, DINOv2, CLIP, FID/KID
```

</details>

**Một số file/thư mục quan trọng:**

| Đường dẫn | Vai trò |
|---|---|
| [`scripts/setup_data.sh`](scripts/setup_data.sh) | Tạo symlink layout dữ liệu cho cả 5 pipeline (chạy lại được nhiều lần) |
| [`scripts/evaluate_all.sh`](scripts/evaluate_all.sh) | Đánh giá thống nhất output 5 mô hình bằng cùng một bộ metric |
| `diffusion-baseline/src/eval/metrics.py` | **Lõi tính toán metric** dùng chung (LPIPS/DINOv2/CLIP/FID/KID) |
| `diffusion-baseline/src/evaluate.py` | CLI gọi metric cho một thư mục output bất kỳ |
| `cyclegan/extended_cyclegan/src/data/build_palette_bank.py` | Sinh palette bank (k-means LAB) cho mô hình Palette |

---

## 🔄 Luồng chạy tổng quan (Pipeline)

```text
┌─ Bước 0 · CÀI ĐẶT ───────────────────────────────────────────────┐
│  Cài PyTorch (bản CUDA phù hợp máy) → pip install -r requirements │
└──────────────────────────────────────────────────────────────────┘
                              │
┌─ Bước 1 · CHUẨN BỊ DỮ LIỆU ──────────────────────────────────────┐
│  Tải Van Gogh2Photo → data/archive/{trainA,trainB,testA,testB}    │
│  bash scripts/setup_data.sh           → symlink cho AdaIN/CycleGAN │
│  (Palette) python -m src.data.build_palette_bank → palette_bank/  │
└──────────────────────────────────────────────────────────────────┘
                              │
┌─ Bước 2 · HUẤN LUYỆN ────────────────────────────────────────────┐
│  Mỗi mô hình có entry riêng (train.py / run_train.py)             │
│  Chi tiết: xem README trong thư mục từng mô hình                  │
└──────────────────────────────────────────────────────────────────┘
                              │
┌─ Bước 3 · INFERENCE ─────────────────────────────────────────────┐
│  Sinh output trên testA (1 ảnh ↔ 1 content, tên file sort khớp)   │
│  → <model>/outputs/infer/  hoặc  outputs/inference/vangogh/A2B    │
└──────────────────────────────────────────────────────────────────┘
                              │
┌─ Bước 4 · ĐÁNH GIÁ THỐNG NHẤT ───────────────────────────────────┐
│  bash scripts/evaluate_all.sh  → outputs/eval/<model>.json (+csv) │
│  Dùng chung diffusion-baseline/src/evaluate.py cho cả 5 mô hình   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Dataset

Bộ **Van Gogh2Photo** (họ dataset *unpaired* công bố cùng CycleGAN), đặt tại `data/archive/`:

| Tập | Nội dung | Vai trò | Số lượng |
|---|---|---|:-:|
| `trainA` | Ảnh thật (Photo — Flickr) | Content — train | **6.287** |
| `trainB` | Tranh Van Gogh (WikiArt) | Style — train | **400** |
| `testA` | Ảnh thật (Photo) | Content — eval | **751** |
| `testB` | Tranh Van Gogh | Style ref — eval | **400** |

> [!WARNING]
> **Unpaired:** không tồn tại cặp ảnh photo↔tranh tương ứng 1-1. Sự chênh lệch lớn giữa
> miền Content (6.287) và Style (400) buộc mô hình phải có năng lực biểu diễn phong cách
> tốt để tránh overfit trên 400 bức tranh.

📥 Tải dataset: [**Google Drive**](https://drive.google.com/drive/u/0/folders/1-iy75YIsNDqAvMT1GMJj6zQtD2LmFHnK)

---

## ⚙️ Cài đặt & Chuẩn bị dữ liệu

```bash
# 1️⃣ Cài PyTorch bản CUDA phù hợp máy (xem https://pytorch.org), rồi cài phần còn lại
pip install -r requirements.txt

# 2️⃣ Tải dataset từ Drive, giải nén vào data/archive/
#    data/archive/{trainA, trainB, testA, testB}

# 3️⃣ Tạo layout dữ liệu (symlink) cho AdaIN & CycleGAN
bash scripts/setup_data.sh

# 4️⃣ (Chỉ cho Palette-guided CycleGAN) sinh palette bank
cd cyclegan/extended_cyclegan
python -m src.data.build_palette_bank \
    --input_dir ../../data/archive/trainA \
    --output_jsonl ../../data/palette_bank/train_palettes_photo.jsonl
python -m src.data.build_palette_bank \
    --input_dir ../../data/archive/trainB \
    --output_jsonl ../../data/palette_bank/train_palettes_art.jsonl
```

> [!TIP]
> Nếu chữ tiếng Việt hoặc font hiển thị lỗi khi build palette / train, kiểm tra terminal
> đang ở UTF-8. Các script `.sh` chạy trên Linux/WSL/macOS; trên Windows dùng các file
> `.bat` tương ứng trong từng thư mục mô hình.

---

## 🏛️ Chi tiết kiến trúc 5 mô hình

> Mỗi mô hình trình bày: **ý tưởng → sơ đồ → hàm loss → siêu tham số → cách chạy**. Sợi chỉ
> đỏ xuyên suốt: cơ chế **AdaIN (dịch chuyển thống kê mean/std)** xuất hiện ở cả ba hướng.

### 1️⃣ AdaIN Baseline

**Ý tưởng.** Thay vì tối ưu lặp như Gatys (vài phút/ảnh), AdaIN chuyển phong cách *bất kỳ*
trong **một lần truyền xuôi**: chỉ cần căn chỉnh thống kê (mean/std) của feature Content
theo feature Style tại tầng sâu của VGG19.

```text
Content ─┐
         ├─► VGG19 (FROZEN, →relu4_1) ─► f_c ─┐
Style  ──┘                              f_s ──┤
                                              ▼
                    AdaIN: σ(f_s)·(f_c−μ(f_c))/σ(f_c) + μ(f_s)
                                              ▼
                    Decoder (đối xứng VGG, HỌC từ đầu) ─► Ảnh stylized
```

**Hàm loss:** &nbsp; `L = L_content + 10 · L_style`
- `L_content` = MSE feature tại `relu4_1`
- `L_style` = MSE **mean/std** trên 4 tầng `relu1_1…relu4_1` (đúng paper gốc)
- Hệ số **10** cân bằng gradient (Style loss nhỏ hơn ~10× do làm phẳng về mean/std)

| Siêu tham số | Giá trị |
|---|---|
| Encoder | VGG19 pretrained, **đóng băng** |
| Optimizer | Adam, lr = 1e-4 |
| Batch size | 8 |
| Dừng | Early Stopping (patience 15) |
| Inference | `alpha ∈ [0,1]` trộn ở feature space |

```bash
cd adain/adain_baseline
python -m src.train  --root_dir ../../data/adain --epochs 20 --batch_size 8
python -m src.infer  --checkpoint checkpoints/best_model.pth \
    --content_dir ../../data/archive/testA --style_dir ../../data/archive/testB \
    --output_dir outputs/infer --pair_mode cycle
```
📖 README chi tiết: [`adain/adain_baseline/README.md`](adain/adain_baseline/README.md)

---

### 2️⃣ AdaIN Multi-Scale *(cải tiến)*

**Motivation.** Baseline chỉ tiêm style ở feature sâu (`relu4_1`) nên giữ được bố cục màu
nhưng **mất nét cọ vi mô** (vốn nằm ở feature nông `relu1_1`, `relu2_1`). Giải pháp: **tiêm
style tại cả 4 cấp độ** trong decoder.

```text
Content ─► VGG19 ─► h1,h2,h3,h4            Style ─► VGG19 ─► (μ,σ) của 4 tầng
                                  │
Decoder: AdaIN(h4,stats4)→conv→AdaIN(·,stats3)→conv→AdaIN(·,stats2)→conv→AdaIN(·,stats1)→out
```

**Điểm cải tiến so với baseline:**
- Tiêm AdaIN **4 lần** (mỗi giai đoạn upsample) thay vì 1 lần → giữ phong cách từ vi mô
  (nét cọ) đến vĩ mô (phối màu).
- **Alpha blending chuyển ra mức ảnh đầu ra** (`out = α·stylized + (1−α)·content`) để tránh
  xung đột phân phối khi tiêm đa tầng.
- Hàm loss **giữ nguyên** dạng baseline → khác biệt chất lượng đến *thuần túy từ kiến trúc*.

```bash
cd adain/adain_multiscale
python run_train.py --root_dir ../../data/adain --epochs 100 --batch_size 8
python infer.py --checkpoint checkpoints/best_model.pth \
    --content_dir ../../data/archive/testA --style_dir ../../data/archive/testB \
    --output_dir outputs/infer --pair_mode cycle
```
📖 README chi tiết: [`adain/adain_multiscale/README.md`](adain/adain_multiscale/README.md)

---

### 3️⃣ CycleGAN Baseline

**Ý tưởng.** Không cần ảnh style mẫu: mạng sinh `G_a2b` học **toàn bộ** phân phối tranh Van
Gogh và nhúng vào trọng số. Để ràng buộc, dùng 4 mạng + 3 loss.

```text
            ┌────────────► D_b (thật/giả tranh)
real_a ─► G_a2b ─► fake_b ─► G_b2a ─► rec_a  ──(L1)── real_a   (Cycle Loss)
real_b ─► G_b2a ─► fake_a ─► G_a2b ─► rec_b  ──(L1)── real_b
            └────────────► D_a (thật/giả ảnh)
G = Encoder(7×7→2 down) → 9×ResBlock(IN+Skip) → Decoder(2 up→7×7+Tanh)   [ngf=64]
D = PatchGAN 5 conv → ma trận 30×30 (chấm điểm từng patch ~70×70)        [ndf=64]
```

**Hàm loss:** &nbsp; `L = L_LSGAN + 10·L_cycle + 5·L_identity`
- **LSGAN** (MSE) thay Cross-Entropy → tránh vanishing gradient
- **Cycle (λ=10):** dịch xuôi rồi ngược phải về ảnh gốc → giữ bố cục
- **Identity (0.5λ=5):** đưa tranh thật vào `G_a2b` không được đổi màu → giữ tư duy màu

| Siêu tham số | Giá trị |
|---|---|
| Optimizer | Adam, lr = 2e-4, β₁ = 0.5 |
| Batch size | 1 (chuẩn CycleGAN, InstanceNorm) |
| Image Pool | 50 ảnh giả (ổn định D) |
| LR schedule | 100 epoch cố định + 100 epoch decay tuyến tính |
| Tiền xử lý | Resize 286 → RandomCrop 256 → Flip → Normalize [-1,1] |

```bash
cd cyclegan/basic_cyclegan
python -m src.train --config configs/train_vangogh.yaml
python -m src.infer --config configs/train_vangogh.yaml \
    --checkpoint outputs/checkpoints/vangogh/latest.pth --direction A2B
```
📖 README chi tiết: [`cyclegan/basic_cyclegan/README.md`](cyclegan/basic_cyclegan/README.md)

---

### 4️⃣ Palette-guided CycleGAN *(cải tiến)*

**Motivation.** CycleGAN baseline **khóa cứng** một tông màu. Giải pháp: cho người dùng cấp
một **bảng màu mục tiêu (palette)** để điều khiển tông màu đầu ra — biến bài toán thành
*Conditional Image Translation*.

```text
Palette 24-d ─► MappingNetwork (24→256→512→1024→9216) ─► (γ,β) cho từng AdaIN
                                       │
Ảnh ─► Encoder(IN) ─► 9×ResBlock dùng AdaIN(γ,β)+Skip ─► Decoder(Upsample+Conv) ─► Ảnh
```

**Điểm cải tiến so với baseline:**
- **MappingNetwork**: MLP biến palette 24-d → vector style **9216-d**
  (= 9 block × 2 AdaIN × 256 kênh × 2 tham số γ/β).
- **AdaIN chỉ ở 9 ResBlock** (Encoder/Decoder giữ InstanceNorm) — đúc kết sau chuỗi thí
  nghiệm sửa lỗi *loang màu → bàn cờ → mờ*.
- **Palette Loss** (L1, weight = 0.05): kéo màu trung bình output về anchor của palette.
- Decoder thay `ConvTranspose2d` bằng **Upsample(nearest)+Conv2d** để khử hiệu ứng bàn cờ.

> Palette = **6 màu chủ đạo** trong không gian **CIELAB** + 6 trọng số (trích bằng k-means).

```bash
cd cyclegan/extended_cyclegan
python -m src.train --config configs/train_vangogh.yaml
python -m src.infer --config configs/train_vangogh.yaml \
    --checkpoint outputs/checkpoints/vangogh/latest.pth --direction A2B \
    --palette_index 0       # chọn palette điều hướng
```
📖 README chi tiết: [`cyclegan/extended_cyclegan/README.md`](cyclegan/extended_cyclegan/README.md)

---

### 5️⃣ Style-guided Diffusion

**Ý tưởng.** Phá hủy ảnh bằng nhiễu Gauss rồi học UNet khử nhiễu ngược; **tiêm phong cách**
trực tiếp vào mọi tầng UNet qua AdaIN.

```text
Style ─► VGG19(frozen) ─► mean/std 4 tầng (1920-d) ─► MLP ─► style_emb (512-d)
                                                              │
x_t (ảnh nhiễu) + t ─► UNet [AdaIN mỗi ResBlock] ─► ε̂ ─► DDIM khử nhiễu ─► ảnh stylized
                                                              ▲
                                              CFG: ε_cfg = ε_null + s·(ε_style − ε_null)
```

**Hàm loss (3 thành phần):** &nbsp; `L = 1.0·L_noise + 500·L_style + 0.01·L_content`
- `L_noise` = MSE(ε̂, ε) — lõi DDPM (đồng thời giữ content gián tiếp)
- `L_style` = Gram Matrix loss trên 4 tầng VGG (weight lớn 500 vì Gram rất nhỏ ~1e-3)
- `L_content` = MSE feature `relu3_1`
- **t-masking:** chỉ tính style/content loss khi `t < T/5` (lúc ảnh ước lượng đủ sạch)

| Siêu tham số | Giá trị |
|---|---|
| UNet | base=64, mults=[1,2,4,8], attention 16×16 + 8×8 (~72.6M params) |
| Scheduler | DDPM 1000 steps, **cosine** schedule |
| Sampler | DDIM 50 steps + CFG + Guidance Rescale |
| Optimizer | Adam, lr = 2e-4, CosineAnnealingLR |
| Batch size | 2 (giới hạn VRAM T4) |
| Kỹ thuật | AMP · EMA 0.9999 · Style Dropout 0.15 · grad clip 1.0 |

```bash
cd diffusion-baseline
bash scripts/02_train.sh          # huấn luyện
python -m src.infer --checkpoint checkpoints/best_model.pth \
    --content_dir ../data/archive/testA --style_dir ../data/archive/testB \
    --out_dir outputs/eval_out --mode img2img --strength 0.6 --guidance_scale 2.0
```
📖 README chi tiết: [`diffusion-baseline/README.md`](diffusion-baseline/README.md)

---

## 📊 Bộ Metric đánh giá

Lõi tính toán dùng chung: `diffusion-baseline/src/eval/metrics.py`. Chia 2 nhóm:

<details>
<summary><b>Nhóm 1 — Content Metrics (per-image): đo mức giữ nội dung</b></summary>

| Metric | Đo gì | Mô hình nền | Hướng tốt |
|---|---|---|:-:|
| **LPIPS** | Khoảng cách tri giác giữa output và content gốc | AlexNet/VGG | ↓ thấp = giữ content |
| **DINOv2 Cosine** | Tương đồng ngữ nghĩa | DINOv2 ViT (self-supervised) | ↑ cao = giữ ngữ nghĩa |
| **CLIP Style Score** | Mức độ "chất Van Gogh" | CLIP ViT-B/32 | ↑ cao = style mạnh |

</details>

<details>
<summary><b>Nhóm 2 — Distribution Metrics: đo mức hợp phong cách toàn tập</b></summary>

| Metric | Đặc điểm | Hướng tốt |
|---|---|:-:|
| **FID** | Fréchet Inception Distance vs tập Van Gogh; nhạy với N nhỏ | ↓ thấp |
| **KID** ⭐ | Kernel Inception Distance; *unbiased*, hợp tập nhỏ (metric chính) | ↓ thấp |

</details>

> [!NOTE]
> **Trade-off cốt lõi:** LPIPS thấp (giữ content) và CLIP Style cao (style mạnh) là hai mục
> tiêu *mâu thuẫn*. Mô hình tốt là mô hình **cân bằng hợp lý**, không tối ưu một bên mà bỏ
> bên kia.

---

## 🧪 Đánh giá thống nhất cho cả 5 mô hình

Sau khi mỗi mô hình đã infer trên `testA` (output 1 ảnh ↔ 1 content):

```bash
bash scripts/evaluate_all.sh
# Override đường dẫn output từng model qua biến môi trường:
#   ADAIN_OUT=... CYCLEGAN_OUT=... DIFFUSION_OUT=... bash scripts/evaluate_all.sh
# Thiếu timm/open_clip:  EXTRA_FLAGS="--no_dino --no_clip" bash scripts/evaluate_all.sh
```

> [!IMPORTANT]
> **Ràng buộc ghép cặp:** output mỗi mô hình phải là **1 ảnh cho 1 ảnh content**, tên file
> khi sort phải trùng thứ tự với thư mục content (AdaIN infer dùng `--pair_mode cycle`).
> Kết quả lưu tại `outputs/eval/<tên_model>.json` + `per_image.csv`.

📤 Xem output & checkpoint đã chạy sẵn: [**Google Drive**](https://drive.google.com/drive/u/0/folders/1-iy75YIsNDqAvMT1GMJj6zQtD2LmFHnK)

---

## 🔧 Tech stack

`PyTorch` · `torchvision` · **VGG19** (perceptual/encoder) · **AMP** Mixed Precision · **EMA**
· `lpips` · `torchmetrics[image]` (FID/KID) · `open_clip_torch` (CLIP) · `timm` (DINOv2) ·
`scikit-image` (LAB/k-means palette) · `PyYAML` · `tqdm` · `Pillow` · `NumPy`

> Môi trường huấn luyện: **NVIDIA T4 (15.6 GB VRAM)**, seed = 42.

---

## 👥 Thành viên & Credits

| Thành viên | MSSV | Phụ trách |
|---|---|---|
| Phạm Thanh Uy | 23280097 | *(Thực hiện CycleGAN)* |
| Đặng Đỉnh Đoàn | 23280046 | *(Thực hiện Diffusion)* |
| Tạ Minh Hoàng Nguyên | 23280073 | *(Thực hiện AdaIN)* |

**Tài liệu nền tảng:** Gatys et al. (2015) · Huang & Belongie (2017, AdaIN) · Zhu et al.
(2017, CycleGAN) · Karras et al. (2019, StyleGAN) · Ho et al. (2020, DDPM) · Song et al.
(2020, DDIM) · Nichol & Dhariwal (2021) · Ho & Salimans (2022, CFG) · Meng et al. (2021,
SDEdit).

<div align="center">

*Đồ án môn Học sâu cho Khoa học Dữ liệu — Lớp 23KDL — Tháng 6, 2026*

</div>

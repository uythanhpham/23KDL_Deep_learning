# CycleGAN Baseline

CycleGAN **truyền thống** (Zhu et al. 2017) cho bài toán chuyển ảnh thật ↔ tranh Van Gogh, làm baseline so sánh với bản cải tiến [extended_cyclegan](../extended_cyclegan/) (palette-guided).

## 1. Vì sao CycleGAN (so với AdaIN)?

AdaIN biến đổi màu rất nhanh nhưng chỉ "dịch chuyển thống kê" — khó tái tạo **nét cọ, chất liệu, tư duy phối màu**. CycleGAN nặng đô hơn: mạng sinh `G_A2B` học **toàn bộ phân phối** tranh Van Gogh và nhúng các quy luật đường nét/hình khối/cách đi cọ vào trọng số. Đổi lại, mỗi model chỉ học được **một** phong cách cố định (hạn chế mà bản [Palette](../extended_cyclegan/) khắc phục).

> **Quy ước:** `a` = ảnh chụp, `b` = tranh; tiền tố `real_` (ảnh thật) / `fake_` (do G sinh). `G_A2B`: ảnh→tranh; `G_B2A`: tranh→ảnh.

## 2. Ba tiêu chuẩn → Ba thành phần Loss

Một bức `fake_b` thành công phải thỏa **đồng thời** 3 tiêu chuẩn, mỗi cái ứng với một loss:

1. **Phong cách (Style)** → **Adversarial Loss**: `fake_b` phải giống miền Van Gogh để đánh lừa $D_b$. *Nếu chỉ có loss này*, G sẽ sinh ảnh chẳng liên quan ảnh gốc (chỉ lo đánh lừa D).
2. **Bối cảnh & vật thể (Content)** → **Cycle Loss**: dịch xuôi rồi ngược phải về ảnh gốc (`real_a → fake_b → rec_a ≈ real_a`).
3. **Tư duy màu sắc** → **Identity Loss**: đưa tranh Van Gogh thật vào `G_A2B` thì không được đổi màu (giữ cách phối màu nóng–lạnh đặc trưng).

## 3. Kiến trúc (4 mạng nơ-ron)

```text
            ┌────────────► D_b (thật/giả tranh)
real_a ─► G_A2B ─► fake_b ─► G_B2A ─► rec_a  ──(L1)── real_a   (Cycle)
real_b ─► G_B2A ─► fake_a ─► G_A2B ─► rec_b  ──(L1)── real_b
            └────────────► D_a (thật/giả ảnh)
```

- **2 Generator** ResNet (9 res-blocks, `ngf=64`): `G_A2B` (photo → tranh) và `G_B2A` (tranh → photo).
  - Encoder: Conv $7\times7$ (stride 1) → 2 lớp downsample $3\times3$ (stride 2).
  - 9 ResBlock (mỗi khối 2 Conv + **Skip Connection**): Skip giúp "bọc lót" — khôi phục phần cấu trúc mà một khối vô tình phá hỏng trong khi giữ phần cải biên tốt.
  - Decoder: 2 lớp ConvTranspose (upsample) → Conv $7\times7$ + **Tanh** (ép về $[-1,1]$).
- **2 Discriminator** PatchGAN (`ndf=64`): xuất **ma trận $30\times30$** (~900 patch, trường thụ cảm ~$70\times70$), chấm điểm thật/giả từng vùng cục bộ → buộc G hoàn thiện kết cấu đồng đều toàn ảnh.
- **Instance Normalization** (chuẩn hóa từng ảnh), không dropout, **Image Pool** 50 ảnh giả cũ để ổn định D, **AMP** tiết kiệm VRAM.

> [!NOTE]
> **D không dùng Sigmoid** ở lớp cuối: 4 lớp đầu (LeakyReLU) cho giá trị dương lớn, gắn Sigmoid sẽ gây vanishing gradient. LSGAN dùng MSE trực tiếp trên điểm số liên tục.

## 4. Hàm Loss

| Loss | Weight | Mục đích |
|------|--------|----------|
| Adversarial (**LSGAN**, MSE) | 1.0 | Generator đánh lừa Discriminator (ổn định hơn Cross-Entropy) |
| Cycle Consistency (L1) | λ = 10.0 | A→B→A ≈ A và B→A→B ≈ B (bảo toàn nội dung) |
| Identity (L1) | 0.5·λ = 5.0 | G_A2B(B) ≈ B — giữ tư duy màu |

> [!IMPORTANT]
> **Backprop hai pha + đóng băng:** Loss GAN chịu tác động của *cả hai* mạng đối kháng (D muốn tăng, G muốn giảm). Mỗi iteration chia 2 pha: **(1)** đóng băng D, cập nhật G; **(2)** mở khóa D, đóng băng G. Nếu cập nhật đồng thời, cả hai sẽ "tự bóp" mình và dậm chân tại chỗ — đóng băng luân phiên là *điều kiện* để trò chơi đối kháng có ý nghĩa.

## 5. Dữ liệu

Config trỏ tới cấu trúc `root/<style>/{trainA,trainB,testA,testB}` với `A` = photo (content), `B` = tranh (style). Config mặc định: `root: ../../data/cyclegan` — chạy `bash scripts/setup_data.sh` ở repo gốc để tạo symlink `data/cyclegan/vangogh` → `data/archive`.

Tiền xử lý: Resize 286 → RandomCrop 256 → Flip → Normalize $[-1,1]$ (Augmentation chống overfit; $[-1,1]$ khớp Tanh ở đầu ra Decoder).

## 6. Cách chạy

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

## 7. Output

```text
outputs/
  checkpoints/vangogh/latest.pth      # + config_used.yaml ghi lại config đã dùng
  samples/vangogh/epoch_XXX.jpg       # 2 hàng: real_A→fake_B→rec_A / real_B→fake_A→rec_B
  logs/vangogh/train_log.csv
```

## 8. Cấu hình huấn luyện (`configs/train_vangogh.yaml`)

| Tham số | Giá trị |
|---|---|
| Ảnh | resize 286 → crop 256 |
| Batch size | **1** (chuẩn CycleGAN) |
| Optimizer | Adam lr=2e-4, β₁=0.5 |
| Lịch lr | 100 epochs cố định + 100 epochs decay tuyến tính về 0 |
| λ_cycle / λ_identity | 10.0 / 0.5 |
| AMP | bật |
| Seed | 42 |

> [!NOTE]
> **Vì sao batch_size = 1?** Đây là chuẩn của CycleGAN do dùng InstanceNorm (chuẩn hóa từng ảnh đơn lẻ); batch lớn dễ tràn VRAM và thực nghiệm cho thấy huấn luyện kém ổn định hơn. **Vì sao không Early Stopping?** GAN Loss dao động cao, không hội tụ đơn điệu — dừng theo "loss không giảm" dễ dừng sớm. Thay vào đó đặt `max_epoch` cố định, lưu checkpoint tốt nhất + theo dõi ảnh mẫu.

Nếu GPU yếu (ví dụ RTX 3050) và bị tràn VRAM: giảm `ngf`/`ndf` xuống 32, `n_res_blocks` xuống 6, hoặc `crop_size` xuống 192.

> Code data/config hỗ trợ nhiều style (monet, ukiyoe, cezanne...) theo cấu trúc `root/<style>/`, nhưng phạm vi dự án chỉ train **vangogh**.

# Palette-guided CycleGAN (CycleGAN cải tiến)

Bản **cải tiến** của [basic_cyclegan](../basic_cyclegan/): thay vì mỗi style phải train một model riêng và style bị "nướng cứng" vào trọng số, generator được **điều kiện hóa theo bảng màu (palette)** — cho phép điều hướng tông màu đầu ra bằng một vector palette bất kỳ tại inference. Đây là một bài toán **Conditional Image Translation**.

## 1. Ý tưởng chính

```text
Palette 24-d ──► MappingNetwork (24→256→512→1024→9216) ──► style vector
(6 màu LAB + 6 trọng số)                                       │ cắt thành (γ, β)
                                                               ▼
Ảnh vào ──► Encoder ──► 9 × ResnetBlock có AdaIN ──► Decoder ──► Ảnh ra
            (InstanceNorm)        ▲ 2 AdaIN/block   (Upsample+Conv)
```

- **MappingNetwork** (`src/models/networks.py`): MLP biến palette 24-d thành style vector **9216-d**, tương tự ý tưởng mapping network của StyleGAN.
- **AdaIN chỉ trong res-blocks**: encoder/decoder giữ InstanceNorm thường để bảo toàn cấu trúc; phong cách màu chỉ được tiêm ở 9 res-block.
- Generator nhận `(ảnh, palette)`: `G_A2B(photo, palette_target)` → tranh theo tông màu mong muốn; `G_B2A(tranh, palette_photo)` → ảnh thật.

> [!NOTE]
> **Giải mã con số 9216.** Vector điều khiển được "thái lát" tuần tự bằng một con trỏ (pointer) để cấp riêng cặp $(\gamma,\beta)$ cho từng tầng AdaIN: 9 block × 2 AdaIN/block × 256 kênh × 2 tham số ($\gamma,\beta$) = **9216**. AdaIN được **loại bỏ hoàn toàn ở các lớp khởi tạo/Downsample/Upsample** để bảo toàn cấu trúc thô; toàn bộ "dung lượng" điều khiển dồn cho 9 res-block xử lý đặc trưng sâu.

## 2. Cơ sở: điều khiển màu bằng AdaIN

Phong cách (tông màu, độ tương phản) được mã hóa bởi mean/std của feature map. AdaIN hai bước: **(1)** InstanceNorm xóa phong cách gốc ($\hat{h}=(h-\mu)/\sigma$); **(2)** áp $(\gamma,\beta)$ trích từ palette: $\text{out}=\gamma\cdot\hat{h}+\beta$. Palette = **6 màu chủ đạo trong không gian CIELAB** + 6 trọng số (CIELAB gần cảm nhận thị giác con người hơn RGB).

## 3. Hành trình khắc phục lỗi (Ablation thực nghiệm)

Đây là phần diễn ra nhiều thử nghiệm nhất — chuỗi lỗi và cách sửa:

| # | Lỗi | Nguyên nhân | Khắc phục |
|:-:|---|---|---|
| 1 | **Loang màu lem nhem** | AdaIN ở *mọi* khối + `λ_palette=1.0` quá lớn lấn át cycle/identity → mạng bỏ nội dung | Hạ `λ_palette` → **0.05** |
| 2 | **Hiệu ứng bàn cờ** | `ConvTranspose2d` chồng lấp không đều, cộng hưởng với việc gỡ InstanceNorm | Thay bằng **Upsample(nearest)+Conv2d** |
| 3 | **Ảnh mờ như sương** | Đặt **AdaIN trong Encoder** làm sai lệch feature đầu vào cho ResBlock | **Loại bỏ AdaIN ở Encoder** |

→ Kiến trúc cuối: AdaIN **chỉ ở 9 ResBlock**, Encoder giữ IN, Decoder dùng Upsample+Conv2d.

## 4. Palette Bank

Mỗi ảnh được trích trước **6 màu chủ đạo trong CIELAB + 6 trọng số** → vector 24 chiều, lưu file JSONL:

```text
train_palettes_photo.jsonl   # palette của từng ảnh photo (domain A)
train_palettes_art.jsonl     # palette của từng tranh (domain B)
```

Mỗi sample train trả về: palette của A, palette của B, và **một palette art ngẫu nhiên** làm target điều hướng (ảnh thiếu palette được gán vector 0).

Sinh palette bank (k-means k=6 trong LAB, chạy từ thư mục `extended_cyclegan`):

```bash
python -m src.data.build_palette_bank \
    --input_dir ../../data/archive/trainA \
    --output_jsonl ../../data/palette_bank/train_palettes_photo.jsonl
python -m src.data.build_palette_bank \
    --input_dir ../../data/archive/trainB \
    --output_jsonl ../../data/palette_bank/train_palettes_art.jsonl
```

Config mặc định đã trỏ tới 2 file trên. Tùy chọn: `--k`, `--resize`, `--max_pixels`, `--max_images` (smoke test).

## 5. Hàm Loss

So với CycleGAN chuẩn (LSGAN + Cycle + Identity), bổ sung **Palette Loss**:

| Loss | Weight | Mục đích |
|------|--------|----------|
| Adversarial (LSGAN) | 1.0 | Đánh lừa Discriminator |
| Cycle Consistency | λ_A = λ_B = 10.0 | Bảo toàn nội dung qua chu trình A→B→A |
| Identity | 0.5·λ | Ổn định màu khi input đã thuộc domain đích |
| **Palette (L1)** | **0.05** | Kéo màu trung bình `fake_B` về anchor màu của palette target |

> `λ_palette = 0.05` là "điểm ngọt": đủ kéo tông màu về palette mong muốn mà không phá vỡ cân bằng đối kháng hay mâu thuẫn với Cycle Loss.

## 6. Cách chạy

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
- Mặc định: bốc ngẫu nhiên 1 palette từ bank (A2B → bank art, B2A → bank photo) với `--seed 42` cố định để tái lập.
- `--palette_index N`: chọn palette thứ N trong bank — dùng minh họa **"cùng ảnh, khác palette"** trong báo cáo (thế mạnh chính của bản cải tiến).
- `--palette_jsonl <file>`: dùng bank khác với config.

## 7. Output

```text
outputs/
  checkpoints/vangogh/latest.pth
  samples/vangogh/epoch_XXX.jpg     # real_A→fake_B→rec_A / real_B→fake_A→rec_B
  logs/vangogh/train_log.csv        # gồm cả cột palette_loss
```

## 8. Cấu hình chính (`configs/train_vangogh.yaml`)

ngf=ndf=64, ResNet 9 blocks, ảnh 286→crop 256, batch_size=1, Adam lr=2e-4 (β₁=0.5), 100 epochs + 100 epochs decay tuyến tính, image pool 50, **λ_palette=0.05**. AdaIN ở res-block (điều khiển bởi MappingNetwork), Decoder dùng Upsample+Conv2d.

> [!TIP]
> **Ý nghĩa cải tiến:** một model duy nhất giờ có thể sinh vô hạn phiên bản tranh Van Gogh với các sắc độ khác nhau (chiều tà, đêm tối, đồng xanh...) chỉ bằng cách đổi vector palette 24-d ở inference — bước tiến lớn về kiểm soát dòng sinh thành (Generative Control).

# AdaIN Baseline

Hiện thực AdaIN style transfer **chuẩn theo paper gốc** (Huang & Belongie 2017 — *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization*), làm baseline so sánh với bản cải tiến [adain_multiscale](../adain_multiscale/).

## 1. Bối cảnh & động lực

Trước AdaIN, các phương pháp style transfer hoặc **chạy rất chậm** (Gatys 2015 — tối ưu lặp trực tiếp trên điểm ảnh, mất vài phút/ảnh), hoặc **chỉ học được một vài phong cách cố định** (mạng feed-forward một-phong-cách). AdaIN phá vỡ thế lưỡng nan này: chuyển đổi phong cách **bất kỳ (arbitrary)** chỉ trong **một lần truyền xuôi (single forward pass)**.

Phát hiện cốt lõi: **phong cách của một bức ảnh được mã hóa bởi thống kê (mean/std) của các feature map VGG**. Vì vậy, để "áp" phong cách của ảnh Style lên ảnh Content, ta chỉ cần *căn chỉnh lại* mean/std của feature Content cho khớp với feature Style — một phép biến đổi affine cực rẻ thay cho tối ưu lặp.

## 2. Kiến trúc

```text
Content ──► VGG19 Encoder (frozen, đến relu4_1) ──► f(c) ─┐
Style   ──► VGG19 Encoder (frozen, đến relu4_1) ──► f(s) ─┤
                                                          ▼
                          AdaIN: σ(f(s)) · (f(c) − μ(f(c)))/σ(f(c)) + μ(f(s))
                                                          ▼
                          Decoder (đối xứng VGG, học từ đầu) ──► ảnh stylized
```

- **Encoder = VGG19 pretrained ImageNet, đóng băng hoàn toàn** (`requires_grad=False`). VGG đóng vai trò "thước đo tri giác" ổn định; chỉ **decoder được học**. Việc đóng băng giúp giảm mạnh số tham số huấn luyện và tốc độ hội tụ nhanh.
- **Vì sao trích tại `relu4_1`?** Càng vào sâu, VGG càng nắm cấu trúc/bố cục tổng quát (mất chi tiết cục bộ). `relu4_1` đủ sâu để mô tả nội dung mà chưa bị "lọc" quá mức như các tầng `relu_2/_3/_4` (vốn hợp cho phân loại hơn tái tạo).
- **AdaIN một lần** tại bottleneck: xóa phong cách cũ của Content (đưa về mean 0, std 1) rồi áp mean/std của Style. Đây chính là *phép dịch chuyển thống kê* — bản chất của toàn bộ phương pháp.
- **Decoder** thiết kế đối xứng ngược VGG nhưng thay MaxPool bằng `Upsample` và dùng `ReflectionPad2d(1)` trước mỗi Conv để tránh lỗi viền đen.

## 3. Hàm Loss (`src/losses/perceptual.py`)

$$L = L_{content} + \lambda_{style} \cdot L_{style}, \qquad \lambda_{style} = 10$$

- **Content loss**: MSE giữa feature của ảnh output (đưa ngược lại VGG) và feature lý tưởng $t$ sau AdaIN, tại `relu4_1`.
- **Style loss**: MSE giữa **mean/std** của output và của Style, cộng dồn trên 4 tầng `relu1_1 → relu4_1` (đúng paper gốc — khác với Gram Matrix của Gatys).

> [!NOTE]
> **Vì sao $\lambda_{style}=10$? (Scale Mismatch).** Content loss tính MSE trực tiếp trên ma trận feature lớn (biên độ activation lớn) nên giá trị cao; Style loss thu gọn cả không gian $H\times W$ về 2 con số mean/std nên giá trị nhỏ hơn ~10 lần. Đặt $\lambda=10$ để **cân bằng gradient** — kéo hai thành phần về cùng bậc độ lớn, tránh để Content "thống trị" khiến ảnh ra nhạt nhòa, không học được phong cách.

## 4. Yêu cầu

Python 3.10+; cài dependency bằng `requirements.txt` ở repo gốc: `pip install -r ../../requirements.txt`

Dữ liệu: thư mục gốc chứa 2 thư mục con `content/` và `style/` — chạy `bash scripts/setup_data.sh` ở repo gốc để tạo sẵn tại `data/adain/`. Ảnh được resize + normalize theo **chuẩn ImageNet** (vì VGG19 huấn luyện trước trên ImageNet).

## 5. Huấn luyện

Chạy từ thư mục `adain_baseline/`:

```bash
# Train (mặc định đọc debug_data — đổi --root_dir sang dữ liệu thật)
python -m src.train \
    --root_dir ../../data/adain \
    --checkpoint_dir checkpoints \
    --epochs 20 --batch_size 8 --lr 1e-4 --lambda_style 10.0
```

Tham số chính: `--image_size` (mặc định 256), `--val_split` (0.2), `--patience`/`--min_delta` (early stopping), `--resume_from` (tiếp tục từ checkpoint).

Kết quả lưu trong `--checkpoint_dir`: `best_model.pth`, `adain_checkpoint_epoch_*.pth`, `history.csv`. Tập huấn luyện được chia 80:20 (train/val); chỉ decoder được cập nhật bằng Adam.

## 6. Inference

```bash
python -m src.infer \
    --checkpoint checkpoints/best_model.pth \
    --content_dir ../../data/archive/testA \
    --style_dir ../../data/archive/testB \
    --output_dir outputs/infer \
    --alpha 1.0 --size 256 --pair_mode cycle
```

- **`--alpha` ∈ [0,1]**: trộn ở *feature space* — `alpha=1` style mạnh nhất, `alpha=0` giữ nguyên content, giá trị giữa cân bằng hai bên.
- `--pair_mode cycle`: mỗi content ghép 1 style → output 1-1 (`<content>_stylized.jpg`), **dùng được ngay cho `scripts/evaluate_all.sh`**.
- `--pair_mode all` (mặc định): tích chéo content×style, mỗi cặp tạo `result_<content>_<style>.jpg` — chỉ nên dùng với số ảnh nhỏ.
- Checkpoint load được cả dạng full checkpoint lẫn state dict thuần.

## 7. Evaluate

```bash
python -m src.evaluate \
    --pred_dir outputs/infer \
    --ref_dir <thư mục ảnh content gốc> \
    --output_file outputs/eval/metrics.json
```

Đo **LPIPS / SSIM / RMSE** so với ảnh content gốc (mức độ giữ nội dung) — đây là bộ metric *nội bộ* của AdaIN.

> [!NOTE]
> Để **so sánh công bằng** với các model khác trong dự án (FID/KID/CLIP style), chạy thêm bộ evaluate chung tại `diffusion-baseline/src/evaluate.py` trên cùng thư mục output — xem [README gốc](../../README.md).

## 8. Phân tích kết quả (đặc trưng baseline)

Baseline **giữ bố cục và hình khối rất tốt** nhưng thường **thiếu nét cọ xoáy/impasto đặc trưng của Van Gogh**: bề mặt giống một lớp vân áp lên hơn là nét cọ thật. Nguyên nhân: phong cách bị tiêm **chỉ một lần** ở tầng sâu nên đánh mất các đặc trưng phong cách vi mô ở tầng nông — đây chính là động lực cho bản cải tiến [Multi-Scale](../adain_multiscale/).

> Trên Windows có sẵn script `.bat` trong `scripts/` (`01_prepare_data.bat`, `03_infer.bat`, `04_evaluate.bat`).

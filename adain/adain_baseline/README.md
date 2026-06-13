# AdaIN Baseline

Hiện thực AdaIN style transfer **chuẩn theo paper gốc** (Huang & Belongie 2017 — *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization*), làm baseline so sánh với bản cải tiến [adain_multiscale](../adain_multiscale/).

## Kiến trúc

```text
Content ──► VGG19 Encoder (frozen, đến relu4_1) ──► f(c) ─┐
Style   ──► VGG19 Encoder (frozen, đến relu4_1) ──► f(s) ─┤
                                                          ▼
                          AdaIN: σ(f(s)) · (f(c) − μ(f(c)))/σ(f(c)) + μ(f(s))
                                                          ▼
                          Decoder (đối xứng VGG, học từ đầu) ──► ảnh stylized
```

- **Encoder**: VGG19 pretrained ImageNet, đóng băng — chỉ **decoder được học**.
- **AdaIN một lần** tại bottleneck (relu4_1): căn mean/std của content feature theo style feature.
- Inference có tham số `alpha` ∈ [0, 1] trộn feature trước/sau AdaIN để điều chỉnh mức độ style.

## Hàm Loss (`src/losses/perceptual.py`)

$L = L_{content} + \lambda_{style} \cdot L_{style}$ với $\lambda_{style} = 10$ (theo paper):

- **Content loss**: MSE giữa feature của output và feature sau AdaIN (target).
- **Style loss**: MSE giữa mean/std của output và style trên 4 tầng VGG (relu1_1 → relu4_1).

## Yêu cầu

Python 3.10+; cài dependency bằng `requirements.txt` ở repo gốc: `pip install -r ../../requirements.txt`

Dữ liệu: thư mục gốc chứa 2 thư mục con `content/` và `style/` — chạy `bash scripts/setup_data.sh` ở repo gốc để tạo sẵn tại `data/adain/`.

## Cách chạy

Chạy từ thư mục `adain_baseline/`:

```bash
# Train (mặc định đọc debug_data — đổi --root_dir sang dữ liệu thật)
python -m src.train \
    --root_dir ../../data/adain \
    --checkpoint_dir checkpoints \
    --epochs 20 --batch_size 8 --lr 1e-4 --lambda_style 10.0
```

Tham số chính: `--image_size` (mặc định 256), `--val_split` (0.2), `--patience`/`--min_delta` (early stopping), `--resume_from` (tiếp tục từ checkpoint).

Kết quả train lưu trong `--checkpoint_dir`: `best_model.pth`, `adain_checkpoint_epoch_*.pth`, `history.csv`.

### Inference

```bash
python -m src.infer \
    --checkpoint checkpoints/best_model.pth \
    --content_dir ../../data/archive/testA \
    --style_dir ../../data/archive/testB \
    --output_dir outputs/infer \
    --alpha 1.0 --size 256 --pair_mode cycle
```

- `--pair_mode cycle`: mỗi content ghép 1 style → output 1-1 (`<content>_stylized.jpg`), dùng được ngay cho `scripts/evaluate_all.sh`
- `--pair_mode all` (mặc định): tích chéo content×style, mỗi cặp tạo file `result_<content>_<style>.jpg` — chỉ dùng với số ảnh nhỏ
- Checkpoint load được cả dạng full checkpoint lẫn state dict thuần.

### Evaluate

```bash
python -m src.evaluate \
    --pred_dir outputs/infer \
    --ref_dir <thư mục ảnh content gốc> \
    --output_file outputs/eval/metrics.json
```

Đo **LPIPS / SSIM / RMSE** so với ảnh content gốc (mức độ giữ nội dung).

> [!NOTE]
> Để so sánh công bằng với các model khác trong dự án (FID/KID/CLIP style), chạy thêm bộ evaluate chung tại `diffusion-baseline/src/evaluate.py` trên cùng thư mục output — xem [README gốc](../../README.md).

Trên Windows có sẵn script `.bat` trong `scripts/` (`01_prepare_data.bat`, `03_infer.bat`, `04_evaluate.bat`).

# AdaIN Multi-Scale (AdaIN cải tiến)

Bản **cải tiến** của [adain_baseline](../adain_baseline/): thay vì tiêm style một lần duy nhất tại bottleneck, decoder **tiêm style tại 4 cấp độ feature** (multi-scale injection).

## Ý tưởng cải tiến

Phong cách hội họa không chỉ nằm ở feature sâu (bố cục màu tổng thể) mà cả ở feature nông (nét cọ, texture). Baseline chỉ AdaIN tại relu4_1 nên dễ mất chi tiết phong cách tinh. Bản multi-scale căn lại mean/std theo style ở **mỗi giai đoạn upsample** của decoder:

```text
Content ──► VGG19 Encoder (frozen) ──► h1, h2, h3, h4   (relu1_1 → relu4_1)
Style   ──► VGG19 Encoder (frozen) ──► (μ, σ) của 4 tầng style

Decoder:  AdaIN(h4, stats₄) → conv → AdaIN(·, stats₃) → conv×4
          → AdaIN(·, stats₂) → conv×2 → AdaIN(·, stats₁) → conv → ảnh ra
```

- Encoder VGG19 đóng băng — chỉ **decoder được học** (như baseline).
- Alpha blending ở **mức ảnh đầu ra**: `out = α · stylized + (1−α) · content`.

## Hàm Loss (`Loss/loss.py`)

`StyleTransferLoss`: $L = L_{content} + \lambda_{style} \cdot L_{style}$ với $\lambda_{style} = 10$ — cùng dạng với baseline (style loss = matching mean/std đa tầng) để so sánh công bằng: khác biệt kết quả đến từ **kiến trúc decoder**, không phải từ loss.

## Cấu trúc thư mục

- `DataSet/DataLoader.py`: xây dựng `Dataset` và `DataLoader`
- `Model/adain_multiscale.py`: encoder, decoder và model chính
- `Loss/loss.py`: hàm loss cho training
- `Train/trainer.py`: logic train/validate theo batch
- `Train/train.py`: pipeline huấn luyện, lưu checkpoint và ảnh mẫu
- `run_train.py`: entry script huấn luyện (CLI)
- `infer.py`: entry script inference (CLI)

## Yêu cầu dữ liệu

Thư mục gốc chứa 2 thư mục con `content/` (ảnh thật) và `style/` (tranh Van Gogh). Chạy `bash scripts/setup_data.sh` ở repo gốc để tạo sẵn layout này tại `data/adain/` (symlink tới dataset). Ảnh được resize và normalize theo ImageNet trước khi đưa vào VGG19.

## Cài đặt

Dùng `requirements.txt` ở repo gốc: `pip install -r ../../requirements.txt`

## Chạy training

Chạy từ thư mục `adain_multiscale`:

```bash
python run_train.py --root_dir ../../data/adain --epochs 100 --batch_size 8
```

Tham số chính: `--lr` (1e-4), `--lambda_style` (10.0), `--lambda_content` (1.0), `--val_split` (0.2), `--pair_mode` (`cycle`/`random`), `--patience`/`--min_delta` (early stopping), `--resume` (train tiếp từ checkpoint).

Kết quả lưu trong `checkpoints/`: `history.csv`, `epoch_*.pth`, `best_model.pth`, `samples/` (ảnh content/style/result theo epoch).

## Inference

```bash
python infer.py --checkpoint checkpoints/best_model.pth \
    --content_dir ../../data/archive/testA \
    --style_dir ../../data/archive/testB \
    --output_dir outputs/infer --alpha 1.0
```

- Mặc định `--pair_mode cycle`: mỗi content ghép 1 style (output 1-1 với content → dùng được ngay cho `scripts/evaluate_all.sh`); `--pair_mode all` để chạy tích chéo content×style.
- `alpha = 1.0`: style mạnh nhất; `alpha = 0.0`: giữ nguyên content; giá trị giữa: cân bằng.

## Lưu ý

- Chạy lệnh từ thư mục gốc project để các import `Train/...`, `Model/...`, `Loss/...` hoạt động đúng.
- Lần đầu chạy sẽ tải weight VGG19 pretrained từ Internet.
- Đầu vào model phải là ảnh RGB đã normalize theo ImageNet.
- Đánh giá thống nhất với các model khác: dùng `diffusion-baseline/src/evaluate.py` — xem [README gốc](../../README.md).

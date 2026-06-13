# AdaIN Multi-Scale (AdaIN cải tiến)

Bản **cải tiến** của [adain_baseline](../adain_baseline/): thay vì tiêm style một lần duy nhất tại bottleneck, decoder **tiêm style tại 4 cấp độ feature** (multi-scale injection).

## 1. Động lực cải tiến

Quan sát từ baseline: ảnh giữ bố cục tốt nhưng **nhòe nét cọ tinh xảo**. Lý do nằm ở bản chất phân tầng của VGG:

- **Feature sâu** (`relu4_1`): mã hóa bố cục, phối màu *vĩ mô*.
- **Feature nông** (`relu1_1`, `relu2_1`): mã hóa nét cọ, texture, độ xoáy *vi mô*.

Baseline chỉ AdaIN tại `relu4_1` nên **bỏ sót toàn bộ phong cách vi mô**. Bản Multi-Scale khắc phục bằng cách **căn lại mean/std theo Style ở MỖI giai đoạn upsample** của decoder — tái tạo phong cách từ vi mô đến vĩ mô.

## 2. Kiến trúc

```text
Content ──► VGG19 Encoder (frozen) ──► h1, h2, h3, h4   (relu1_1 → relu4_1)
Style   ──► VGG19 Encoder (frozen) ──► (μ, σ) của 4 tầng style

Decoder:  AdaIN(h4, stats₄) → conv → AdaIN(·, stats₃) → conv×4
          → AdaIN(·, stats₂) → conv×2 → AdaIN(·, stats₁) → conv → ảnh ra
```

- Encoder VGG19 **đóng băng** — chỉ **decoder được học** (như baseline).
- Mỗi khi luồng giải mã đi qua một giai đoạn `Upsample`, dữ liệu bị "chặn lại" và được AdaIN bằng đúng thống kê (mean/std) của tầng Style **tương ứng** (từ `relu4_1` ngược về `relu1_1`).

## 3. Hai khác biệt cốt lõi so với Baseline

| | Baseline | Multi-Scale |
|---|---|---|
| Số lần tiêm AdaIN | 1 (tại `relu4_1`) | **4** (mỗi tầng decoder) |
| Vị trí Alpha blending | Feature space (bottleneck) | **Mức ảnh đầu ra** |

**Vì sao chuyển Alpha blending ra mức ảnh?** Với 4 cấp tiêm style, nội suy `alpha` *bên trong* mạng sẽ gây xung đột phân phối giữa các tầng. Do đó mạng sinh ảnh cách điệu tối đa, rồi nội suy ở **mức ảnh cuối**:

$$out = \alpha \cdot stylized + (1-\alpha) \cdot content$$

## 4. Hàm Loss (`Loss/loss.py`)

`StyleTransferLoss`: $L = L_{content} + \lambda_{style} \cdot L_{style}$ với $\lambda_{style} = 10$.

> [!IMPORTANT]
> Hàm loss **giữ nguyên dạng** với baseline (style loss = matching mean/std đa tầng). Đây là chủ ý: mọi khác biệt chất lượng đầu ra đến **thuần túy từ kiến trúc decoder** (1 cấp vs 4 cấp tiêm style), không bị nhiễu bởi yếu tố loss → đây cũng là một dạng **ablation định tính** giữa hai phiên bản.

## 5. Cấu trúc thư mục

- `DataSet/DataLoader.py`: xây dựng `Dataset` và `DataLoader`
- `Model/adain_multiscale.py`: encoder, decoder và model chính
- `Loss/loss.py`: hàm loss cho training
- `Train/trainer.py`: logic train/validate theo batch
- `Train/train.py`: pipeline huấn luyện, lưu checkpoint và ảnh mẫu
- `run_train.py`: entry script huấn luyện (CLI)
- `infer.py`: entry script inference (CLI)

## 6. Yêu cầu dữ liệu & cài đặt

Thư mục gốc chứa 2 thư mục con `content/` (ảnh thật) và `style/` (tranh Van Gogh). Chạy `bash scripts/setup_data.sh` ở repo gốc để tạo sẵn tại `data/adain/` (symlink tới dataset). Ảnh được resize + normalize theo **ImageNet** trước khi vào VGG19.

Cài dependency: `pip install -r ../../requirements.txt` (từ repo gốc).

## 7. Huấn luyện

Chạy từ thư mục `adain_multiscale`:

```bash
python run_train.py --root_dir ../../data/adain --epochs 100 --batch_size 8
```

Tham số chính: `--lr` (1e-4), `--lambda_style` (10.0), `--lambda_content` (1.0), `--val_split` (0.2), `--pair_mode` (`cycle`/`random`), `--patience`/`--min_delta` (early stopping), `--resume` (train tiếp).

Kết quả lưu trong `checkpoints/`: `history.csv`, `epoch_*.pth`, `best_model.pth`, `samples/` (ảnh content/style/result theo epoch).

## 8. Inference

```bash
python infer.py --checkpoint checkpoints/best_model.pth \
    --content_dir ../../data/archive/testA \
    --style_dir ../../data/archive/testB \
    --output_dir outputs/infer --alpha 1.0 --pair_mode cycle
```

- `--pair_mode cycle` (mặc định): mỗi content ghép 1 style → output 1-1 (dùng ngay cho `scripts/evaluate_all.sh`); `--pair_mode all` để tích chéo content×style.
- `alpha = 1.0`: style mạnh nhất; `alpha = 0.0`: giữ nguyên content; giá trị giữa: cân bằng.

## 9. Kết quả & hạn chế

So với baseline, Multi-Scale **giữ cấu trúc vật thể trọn vẹn hơn**, nét vẽ và hình khối thể hiện rõ hơn. Hạn chế còn lại: vẫn nhòe các chi tiết **quá nhỏ** (ví dụ các chấm bi trên hộp bánh donut).

## 10. Lưu ý

- Chạy lệnh từ thư mục `adain_multiscale` để các import `Train/...`, `Model/...`, `Loss/...` hoạt động đúng.
- Lần đầu chạy sẽ tải weight VGG19 pretrained từ Internet.
- Đầu vào model phải là ảnh RGB đã normalize theo ImageNet.
- Đánh giá thống nhất với các model khác: dùng `diffusion-baseline/src/evaluate.py` — xem [README gốc](../../README.md).

#!/bin/bash
# Chuẩn hóa layout dữ liệu cho cả 5 pipeline bằng symlink (idempotent).
#
# data/archive/ (gốc, phẳng):  trainA=photo, trainB=Van Gogh, testA=photo, testB=Van Gogh
#
# Tạo thêm:
#   data/cyclegan/vangogh  -> ../archive          (CycleGAN cần root/<style>/<split>)
#   data/adain/content     -> ../archive/trainA   (AdaIN cần root/{content,style})
#   data/adain/style       -> ../archive/trainB
#
# Diffusion trỏ thẳng data/archive/trainA|trainB trong config — không cần symlink.
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT_DIR/data"
ARCHIVE_DIR="$DATA_DIR/archive"

if [ ! -d "$ARCHIVE_DIR/trainA" ]; then
    echo "[Lỗi] Không tìm thấy $ARCHIVE_DIR/trainA — cần giải nén dataset Van Gogh2Photo vào data/archive/ trước."
    exit 1
fi

mkdir -p "$DATA_DIR/cyclegan" "$DATA_DIR/adain"

ln -sfn ../archive        "$DATA_DIR/cyclegan/vangogh"
ln -sfn ../archive/trainA "$DATA_DIR/adain/content"
ln -sfn ../archive/trainB "$DATA_DIR/adain/style"

echo "[OK] Symlink đã tạo:"
echo "  data/cyclegan/vangogh -> data/archive"
echo "  data/adain/content    -> data/archive/trainA"
echo "  data/adain/style      -> data/archive/trainB"
echo
echo "Số ảnh từng split:"
for d in trainA trainB testA testB; do
    printf "  %-7s %s\n" "$d:" "$(ls "$ARCHIVE_DIR/$d" | wc -l)"
done

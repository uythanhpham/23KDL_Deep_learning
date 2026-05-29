#!/bin/bash
set -e
echo "========================================="
echo "[GĐ 2] Huấn luyện Style-guided Diffusion"
echo "========================================="
python -m src.train \
    --train_config configs/train.yaml --model_config configs/model.yaml
echo "DONE: checkpoint tại checkpoints/"

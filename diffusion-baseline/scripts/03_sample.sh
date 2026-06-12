#!/bin/bash
set -e
echo "========================================="
echo "[GĐ 3] Sinh ảnh style-guided"
echo "========================================="
python -m src.sample \
    --sample_config configs/sample.yaml --model_config configs/model.yaml
echo "DONE: ảnh tại outputs/samples/"

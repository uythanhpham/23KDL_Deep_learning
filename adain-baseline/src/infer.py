# 23KDL_Deep_learning/adain-baseline/src/infer.py
"""Dùng model sinh thử 1 ảnh."""

import torch                                    # FIX: thiếu import torch
from torchvision import transforms as T         # FIX: alias "T" để tránh shadow
from PIL import Image


def stylize(model, content_path, style_path, alpha=1.0, output_path="output.jpg", size=512):
    """
    Sinh 1 ảnh style transfer và lưu ra file.

    Args:
        model       : AdaINStyleTransfer đã load weights
        content_path: đường dẫn ảnh content
        style_path  : đường dẫn ảnh style
        alpha       : mức độ style (1.0 = full style, 0.0 = giữ nguyên content)
        output_path : nơi lưu ảnh output
        size        : resize ảnh về size x size trước khi infer
    """
    # FIX: dataset dùng range [0,1] không normalize ImageNet
    # → chỉ Resize + ToTensor, KHÔNG Normalize
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),           # [0,255] PIL → [0,1] tensor
    ])

    content = transform(Image.open(content_path).convert("RGB")).unsqueeze(0)
    style   = transform(Image.open(style_path).convert("RGB")).unsqueeze(0)

    # Đưa lên cùng device với model
    device = next(model.parameters()).device
    content = content.to(device)
    style   = style.to(device)

    model.eval()
    with torch.no_grad():
        output = model(content, style, alpha=alpha)  # (1, 3, H, W), đã clamp [0,1]

    # FIX: output đã clamp [0,1] trong adain.py → chỉ cần to_pil, không cần denorm
    output_img = T.ToPILImage()(output[0].cpu())
    output_img.save(output_path)
    print(f"Saved → {output_path}")
    return output_img

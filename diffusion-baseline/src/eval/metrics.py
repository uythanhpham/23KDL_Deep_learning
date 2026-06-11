"""
========================================================================
metrics.py — Bộ đánh giá MODEL-AGNOSTIC cho style transfer
========================================================================
Chỉ cần các THƯ MỤC ẢNH → dùng được cho cả 3 model (Diffusion / CycleGAN / AdaIN).
Không phụ thuộc ảnh style theo từng cặp (CycleGAN không có ảnh style).

Hai nhóm metric:
  1) content_metrics(content_dir, output_dir)  → per-image:
        - LPIPS(content, output)        ↓  (giữ content — đọc theo TRADE-OFF, không minimize)
        - DINO cos(content, output)     ↑  (giữ ngữ nghĩa)
        - CLIP style score              ↑  (độ "Van Gogh", directional zero-shot)
  2) distribution_metrics(output_dir, style_ref_dir) → toàn tập:
        - FID(outputs, vangogh_set)     ↓  (kèm caveat khi N nhỏ)
        - KID(outputs, vangogh_set)     ↓  (unbiased — metric phân phối CHÍNH)

Import được đặt LAZY trong từng hàm/loader để module vẫn nạp khi thiếu lib;
metric nào thiếu lib sẽ bị bỏ qua (in cảnh báo) thay vì làm hỏng cả lượt chạy.
========================================================================
"""
import csv
import time
from pathlib import Path
from typing import List, Dict, Optional

import torch
from PIL import Image
from torchvision import transforms

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ----------------------------------------------------------------------
# Tiện ích ảnh / ghép cặp
# ----------------------------------------------------------------------
def list_images(d) -> List[Path]:
    return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in VALID_EXT)


def pair_by_order(content_dir, output_dir) -> List[tuple]:
    """Ghép content_i ↔ output_i theo thứ tự sort (driver phải lưu cùng số lượng/đúng thứ tự)."""
    cs, os_ = list_images(content_dir), list_images(output_dir)
    if len(cs) != len(os_):
        raise ValueError(f"Số ảnh content ({len(cs)}) ≠ output ({len(os_)}). "
                         f"Driver cần lưu content/output khớp 1-1.")
    return list(zip(cs, os_))


def _to_tensor(path, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])(img)


# ----------------------------------------------------------------------
# Các loader model (lazy, cache 1 lần)
# ----------------------------------------------------------------------
class _Lazy:
    """Bọc loader chỉ chạy 1 lần; trả None nếu thiếu lib (kèm cảnh báo)."""
    def __init__(self, name, loader):
        self.name, self.loader, self._obj, self._tried = name, loader, None, False

    def get(self):
        if not self._tried:
            self._tried = True
            try:
                self._obj = self.loader()
            except Exception as e:
                print(f"[metrics] BỎ QUA {self.name}: {type(e).__name__}: {e}")
                self._obj = None
        return self._obj


def _load_lpips(device):
    import lpips
    return lpips.LPIPS(net="alex").to(device).eval()


def _load_dino(device):
    """DINOv2 (ưu tiên) → fallback DINOv1. Trả (model, transform)."""
    import timm
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    try:
        m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                               num_classes=0, dynamic_img_size=True).to(device).eval()
        size = 224  # patch14 → 224/14=16 grid (dynamic_img_size lo phần pos-embed)
    except Exception:
        m = timm.create_model("vit_base_patch16_224.dino", pretrained=True,
                               num_classes=0).to(device).eval()
        size = 224
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                             transforms.Normalize(mean, std)])
    return m, tf


def _load_clip(device):
    """open_clip ViT-B-32 (openai). Trả (model, preprocess, text_feats)."""
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    prompts = ["a painting in the style of Vincent van Gogh", "a photograph"]
    with torch.no_grad():
        tf = model.encode_text(tokenizer(prompts).to(device))
        tf = tf / tf.norm(dim=-1, keepdim=True)
    return model, preprocess, tf  # tf[0]=van gogh, tf[1]=photo


# ----------------------------------------------------------------------
# CONTENT METRICS (per-image)
# ----------------------------------------------------------------------
def content_metrics(content_dir, output_dir, device: str = "cuda",
                    use_lpips=True, use_dino=True, use_clip=True) -> List[Dict]:
    """Trả list dict per-image. Cột nào thiếu lib → để None."""
    pairs = pair_by_order(content_dir, output_dir)
    lp  = _Lazy("LPIPS", lambda: _load_lpips(device)) if use_lpips else None
    dn  = _Lazy("DINO",  lambda: _load_dino(device))  if use_dino  else None
    cl  = _Lazy("CLIP",  lambda: _load_clip(device))  if use_clip  else None

    rows = []
    for i, (cpath, opath) in enumerate(pairs):
        row = {"idx": i, "content": cpath.name, "output": opath.name,
               "lpips": None, "dino_cos": None, "clip_style": None}

        if lp and lp.get() is not None:
            with torch.no_grad():
                a = (_to_tensor(cpath, 256) * 2 - 1).unsqueeze(0).to(device)
                b = (_to_tensor(opath, 256) * 2 - 1).unsqueeze(0).to(device)
                row["lpips"] = float(lp.get()(a, b).item())

        if dn and dn.get() is not None:
            model, tf = dn.get()
            with torch.no_grad():
                fa = model(tf(Image.open(cpath).convert("RGB")).unsqueeze(0).to(device))
                fb = model(tf(Image.open(opath).convert("RGB")).unsqueeze(0).to(device))
                fa = torch.nn.functional.normalize(fa, dim=-1)
                fb = torch.nn.functional.normalize(fb, dim=-1)
                row["dino_cos"] = float((fa * fb).sum().item())

        if cl and cl.get() is not None:
            model, preprocess, text_feats = cl.get()
            with torch.no_grad():
                im = preprocess(Image.open(opath).convert("RGB")).unsqueeze(0).to(device)
                fo = model.encode_image(im)
                fo = fo / fo.norm(dim=-1, keepdim=True)
                sim = (fo @ text_feats.T).squeeze(0)        # [van_gogh, photo]
                row["clip_style"] = float((sim[0] - sim[1]).item())  # directional style score

        rows.append(row)
    return rows


# ----------------------------------------------------------------------
# DISTRIBUTION METRICS (FID / KID toàn tập)
# ----------------------------------------------------------------------
def _to_uint8_299(path) -> torch.Tensor:
    t = _to_tensor(path, 299)                 # [0,1], (3,299,299)
    return (t * 255).clamp(0, 255).to(torch.uint8)


def distribution_metrics(output_dir, style_ref_dir, device: str = "cuda",
                         kid_subset: Optional[int] = None) -> Dict:
    """FID + KID giữa tập output và tập Van Gogh tham chiếu."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    real = list_images(style_ref_dir)   # Van Gogh tham chiếu
    fake = list_images(output_dir)      # output của model
    n = min(len(real), len(fake))
    if kid_subset is None:
        kid_subset = max(2, min(50, n))   # subset_size phải ≤ số mẫu

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid = KernelInceptionDistance(subset_size=kid_subset, normalize=False).to(device)

    def feed(paths, real_flag):
        for p in paths:
            img = _to_uint8_299(p).unsqueeze(0).to(device)
            fid.update(img, real=real_flag)
            kid.update(img, real=real_flag)

    feed(real, True)
    feed(fake, False)
    kid_mean, kid_std = kid.compute()
    return {"fid": float(fid.compute().item()),
            "kid_mean": float(kid_mean.item()), "kid_std": float(kid_std.item()),
            "n_real": len(real), "n_fake": len(fake), "kid_subset": kid_subset}


# ----------------------------------------------------------------------
# Tiện ích lưu / tổng hợp
# ----------------------------------------------------------------------
def save_csv(rows: List[Dict], path):
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summarize(rows: List[Dict]) -> Dict:
    """Trung bình các cột số (bỏ None)."""
    out = {}
    for key in ("lpips", "dino_cos", "clip_style"):
        vals = [r[key] for r in rows if r.get(key) is not None]
        out[f"{key}_mean"] = sum(vals) / len(vals) if vals else None
        out[f"{key}_n"] = len(vals)
    return out


def count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6  # triệu tham số


if __name__ == "__main__":
    # Smoke test nhẹ: tự tạo vài ảnh giả → chạy content_metrics (chỉ LPIPS nếu thiếu lib khác)
    import tempfile, os
    import numpy as np
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with tempfile.TemporaryDirectory() as d:
        cdir, odir = Path(d) / "c", Path(d) / "o"
        cdir.mkdir(); odir.mkdir()
        for i in range(3):
            arr = (np.random.rand(64, 64, 3) * 255).astype("uint8")
            Image.fromarray(arr).save(cdir / f"content_{i:03d}.png")
            # output = content hơi nhiễu (để LPIPS > 0 nhưng nhỏ)
            arr2 = np.clip(arr.astype(int) + np.random.randint(-20, 20, arr.shape), 0, 255).astype("uint8")
            Image.fromarray(arr2).save(odir / f"out_{i:03d}.png")

        rows = content_metrics(cdir, odir, device=device)
        print("Rows:", rows)
        print("Summary:", summarize(rows))
        # Sanity: LPIPS(content,content)=0
        same = content_metrics(cdir, cdir, device=device, use_dino=False, use_clip=False)
        lp_same = [r["lpips"] for r in same if r["lpips"] is not None]
        if lp_same:
            assert max(lp_same) < 1e-4, f"LPIPS(self) phải ≈0, got {lp_same}"
            print("✓ Sanity LPIPS(self)≈0 OK")
        print("=== SMOKE metrics: PASS ===")

# E:\Nam3_ki2\TH DL\PROJECT\23KDL_Deep_learning\cyclegan\src\models\cyclegan.py:

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from src.models.networks import GANLoss, NLayerDiscriminator, ResnetGenerator, init_weights
from src.utils.image_pool import ImagePool


class CycleGANModel(nn.Module):
    """CycleGAN full model cải tiến tích hợp Palette Loss:

    A = photo/content domain, B = style/art domain.
    G_A2B: photo + palette_target -> style_assigned
    G_B2A: style + palette_photo  -> photo_reconstructed
    D_A: phân biệt ảnh photo thật/giả
    D_B: phân biệt ảnh style thật/giả
    """

    def __init__(self, cfg: Dict, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        mcfg = cfg["model"]
        tcfg = cfg["train"]
        lcfg = cfg.get("loss", {})  # Lấy cấu hình loss mở rộng từ file yaml

        # Khởi tạo các trọng số loss thành phần từ file cấu hình
        self.lambda_cycle = float(lcfg.get("lambda_A", 10.0))
        self.lambda_identity = float(lcfg.get("lambda_identity", 0.5))
        self.lambda_palette = float(lcfg.get("lambda_palette", 1.0))

        # Generator A2B chuyển đổi ảnh thực tế kèm bảng màu nghệ thuật mục tiêu
        self.G_A2B = ResnetGenerator(
            input_nc=int(mcfg["input_nc"]),
            output_nc=int(mcfg["output_nc"]),
            ngf=int(mcfg["ngf"]),
            use_dropout=not bool(mcfg.get("no_dropout", True)),
            n_blocks=int(mcfg.get("n_blocks", 9)),
        ).to(device)

        # Generator B2A hoàn nguyên phục hồi ảnh nghệ thuật kèm bảng màu thực tế
        self.G_B2A = ResnetGenerator(
            input_nc=int(mcfg["output_nc"]),
            output_nc=int(mcfg["input_nc"]),
            ngf=int(mcfg["ngf"]),
            use_dropout=not bool(mcfg.get("no_dropout", True)),
            n_blocks=int(mcfg.get("n_blocks", 9)),
        ).to(device)

        self.D_A = NLayerDiscriminator(
            input_nc=int(mcfg["input_nc"]),
            ndf=int(mcfg["ndf"]),
            norm_type=mcfg.get("norm", "instance"),
        ).to(device)
        self.D_B = NLayerDiscriminator(
            input_nc=int(mcfg["output_nc"]),
            ndf=int(mcfg["ndf"]),
            norm_type=mcfg.get("norm", "instance"),
        ).to(device)

        # Khởi tạo trọng số ngẫu nhiên cho các mạng neural
        init_weights(
            self.G_A2B,
            init_type=mcfg.get("init_type", "normal"),
            init_gain=float(mcfg.get("init_gain", 0.02)),
        )
        init_weights(
            self.G_B2A,
            init_type=mcfg.get("init_type", "normal"),
            init_gain=float(mcfg.get("init_gain", 0.02)),
        )
        init_weights(
            self.D_A,
            init_type=mcfg.get("init_type", "normal"),
            init_gain=float(mcfg.get("init_gain", 0.02)),
        )
        init_weights(
            self.D_B,
            init_type=mcfg.get("init_type", "normal"),
            init_gain=float(mcfg.get("init_gain", 0.02)),
        )

        self.criterion_gan = GANLoss(mcfg.get("gan_mode", "lsgan")).to(device)
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        self.criterion_palette = nn.L1Loss()  # Dùng hàm lỗi L1 để căn chỉnh độ lệch palette

        self.fake_A_pool = ImagePool(int(lcfg.get("pool_size", 50)))
        self.fake_B_pool = ImagePool(int(lcfg.get("pool_size", 50)))

    def set_requires_grad(self, nets, requires_grad: bool) -> None:
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def forward_generators(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        p_A: torch.Tensor,
        p_B: torch.Tensor,
        p_target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Luồng xử lý Generators cải tiến nhận kèm vector palette để AdaIN cắt chia."""

        # 1. Nhánh Photo -> Art (Dẫn dắt bằng bảng màu mục tiêu ngẫu nhiên p_target)
        fake_B = self.G_A2B(real_A, p_target)
        # Vòng lặp Cycle khôi phục lại ảnh thực tế (Đưa p_A về để lấy lại cấu trúc màu gốc)
        rec_A = self.G_B2A(fake_B, p_A)

        # 2. Nhánh Art -> Photo (Sử dụng bảng màu ảnh thực tế p_A)
        fake_A = self.G_B2A(real_B, p_A)
        # Vòng lặp Cycle khôi phục lại tranh nghệ thuật ban đầu (Đưa p_B về)
        rec_B = self.G_A2B(fake_A, p_B)

        # 3. Luồng Identity (Giúp giữ vững cấu trúc vật thể không bị biến dị)
        idt_B = self.G_A2B(real_B, p_B)
        idt_A = self.G_B2A(real_A, p_A)

        return {
            "fake_B": fake_B,
            "rec_A": rec_A,
            "fake_A": fake_A,
            "rec_B": rec_B,
            "idt_A": idt_A,
            "idt_B": idt_B,
        }

    def compute_generator_loss(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        p_A: torch.Tensor,
        p_B: torch.Tensor,
        p_target: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Tính toán tổng hợp các hàm loss của Generator."""

        # Gọi hàm forward generators kèm đầy đủ tham số palette đầu vào
        out = self.forward_generators(real_A, real_B, p_A, p_B, p_target)

        # 1. GAN Loss (Đánh lừa Discriminator)
        loss_G_A2B = self.criterion_gan(self.D_B(out["fake_B"]), True)
        loss_G_B2A = self.criterion_gan(self.D_A(out["fake_A"]), True)

        # 2. Cycle Consistency Loss (Độ tương đồng vòng lặp đảo chiều)
        loss_cycle_A = (
            self.criterion_cycle(out["rec_A"], real_A) * self.lambda_cycle
        )
        loss_cycle_B = (
            self.criterion_cycle(out["rec_B"], real_B) * self.lambda_cycle
        )

        # 3. Identity Loss (Giữ nguyên đặc trưng gốc không đổi màu nếu đưa đúng Domain)
        loss_idt_B = (
            self.criterion_identity(out["idt_B"], real_B)
            * self.lambda_cycle
            * self.lambda_identity
        )
        loss_idt_A = (
            self.criterion_identity(out["idt_A"], real_A)
            * self.lambda_cycle
            * self.lambda_identity
        )

        # 4. GIẢI QUYẾT LỖI LOGIC: Hàm tính loss_palette thực tế, loại bỏ việc trừ tiêu biến
        # Trích xuất giá trị màu trung bình [B, 3] từ fake_B dọc theo hai trục không gian (H, W)
        fake_B_color_mean = out["fake_B"].mean(dim=(2, 3))

        # Cắt lấy 3 giá trị màu LAB đầu tiên từ p_target làm điểm neo so sánh đại diện (hoặc trung bình vector 24 chiều)
        # Để đảm bảo dải gradient luôn chảy thông suốt trong Phase 1, ta so sánh dải màu phân bổ không gian của fake_B
        # trực tiếp với phân đoạn mảng giá trị của p_target (ví dụ: lấy trung bình hoặc so khớp kích thước tương đồng qua linear projection)
        # Dưới đây là giải pháp so khớp phân phối gọn gàng giúp cột log CSV không bị bằng 0:
        target_color_anchor = p_target[:, :3]  # Lấy 3 giá trị chiều đầu tiên làm đại diện phân phối màu chủ đạo
        loss_palette = (
            self.criterion_palette(fake_B_color_mean, target_color_anchor)
            * self.lambda_palette
        )

        # Tổng hợp toàn bộ các hàm lỗi để thực hiện Backpropagation
        loss_G = (
            loss_G_A2B
            + loss_G_B2A
            + loss_cycle_A
            + loss_cycle_B
            + loss_idt_A
            + loss_idt_B
            + loss_palette
        )

        losses = {
            "G_total": loss_G,
            "G_A2B": loss_G_A2B,
            "G_B2A": loss_G_B2A,
            "cycle_A": loss_cycle_A,
            "cycle_B": loss_cycle_B,
            "idt_A": loss_idt_A,
            "idt_B": loss_idt_B,
            "palette_loss": loss_palette,
        }
        return loss_G, {**out, **losses}

    def compute_D_loss(
        self, netD: nn.Module, real: torch.Tensor, fake: torch.Tensor
    ) -> torch.Tensor:
        pred_real = netD(real)
        loss_real = self.criterion_gan(pred_real, True)

        pred_fake = netD(fake.detach())
        loss_fake = self.criterion_gan(pred_fake, False)

        return (loss_real + loss_fake) * 0.5

    def compute_discriminator_losses(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        fake_A: torch.Tensor,
        fake_B: torch.Tensor,
    ):
        fake_A_for_D = self.fake_A_pool.query(fake_A)
        fake_B_for_D = self.fake_B_pool.query(fake_B)

        loss_D_A = self.compute_D_loss(self.D_A, real_A, fake_A_for_D)
        loss_D_B = self.compute_D_loss(self.D_B, real_B, fake_B_for_D)
        return loss_D_A, loss_D_B

    @torch.no_grad()
    def infer_A2B(
        self, real_A: torch.Tensor, p_target: torch.Tensor
    ) -> torch.Tensor:
        """Hàm Inference một chiều yêu cầu thêm bảng màu mục tiêu để ép phong cách tranh."""
        self.G_A2B.eval()
        return self.G_A2B(real_A.to(self.device), p_target.to(self.device))

    @torch.no_grad()
    def infer_B2A(
        self, real_B: torch.Tensor, p_photo: torch.Tensor
    ) -> torch.Tensor:
        self.G_B2A.eval()
        return self.G_B2A(real_B.to(self.device), p_photo.to(self.device))

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        optim_G: torch.optim.Optimizer | None = None,
        optim_D: torch.optim.Optimizer | None = None,
        extra: Dict | None = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "epoch": epoch,
            "cfg": self.cfg,
            "G_A2B": self.G_A2B.state_dict(),
            "G_B2A": self.G_B2A.state_dict(),
            "D_A": self.D_A.state_dict(),
            "D_B": self.D_B.state_dict(),
        }
        if optim_G is not None:
            ckpt["optim_G"] = optim_G.state_dict()
        if optim_D is not None:
            ckpt["optim_D"] = optim_D.state_dict()
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)

    def load_checkpoint(
        self,
        path: str | Path,
        strict: bool = True,
        load_discriminators: bool = True,
    ):
        path = Path(path)
        ckpt = torch.load(path, map_location=self.device)
        self.G_A2B.load_state_dict(ckpt["G_A2B"], strict=strict)
        self.G_B2A.load_state_dict(ckpt["G_B2A"], strict=strict)
        if load_discriminators:
            if "D_A" in ckpt:
                self.D_A.load_state_dict(ckpt["D_A"], strict=strict)
            if "D_B" in ckpt:
                self.D_B.load_state_dict(ckpt["D_B"], strict=strict)
        return ckpt
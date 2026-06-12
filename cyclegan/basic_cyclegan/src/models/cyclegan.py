from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from src.models.networks import GANLoss, NLayerDiscriminator, ResnetGenerator, init_weights
from src.utils.image_pool import ImagePool


class CycleGANModel(nn.Module):
    """
    CycleGAN full model:
    A = photo/content, B = style/art.
    G_A2B: photo -> style
    G_B2A: style -> photo
    D_A: phân biệt ảnh photo thật/giả
    D_B: phân biệt ảnh style thật/giả
    """

    def __init__(self, cfg: Dict, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        mcfg = cfg["model"]
        tcfg = cfg["train"]

        self.lambda_cycle = float(tcfg["lambda_cycle"])
        self.lambda_identity = float(tcfg["lambda_identity"])

        self.G_A2B = ResnetGenerator(
            input_nc=int(mcfg["input_nc"]),
            output_nc=int(mcfg["output_nc"]),
            ngf=int(mcfg["ngf"]),
            norm_type=mcfg.get("norm", "instance"),
            use_dropout=not bool(mcfg.get("no_dropout", True)),
            n_blocks=int(mcfg["n_res_blocks"]),
        ).to(device)
        self.G_B2A = ResnetGenerator(
            input_nc=int(mcfg["output_nc"]),
            output_nc=int(mcfg["input_nc"]),
            ngf=int(mcfg["ngf"]),
            norm_type=mcfg.get("norm", "instance"),
            use_dropout=not bool(mcfg.get("no_dropout", True)),
            n_blocks=int(mcfg["n_res_blocks"]),
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

        init_weights(self.G_A2B)
        init_weights(self.G_B2A)
        init_weights(self.D_A)
        init_weights(self.D_B)

        self.criterion_gan = GANLoss(mcfg.get("gan_mode", "lsgan")).to(device)
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        self.fake_A_pool = ImagePool(int(tcfg.get("pool_size", 50)))
        self.fake_B_pool = ImagePool(int(tcfg.get("pool_size", 50)))

    def set_requires_grad(self, nets, requires_grad: bool) -> None:
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def forward_generators(self, real_A: torch.Tensor, real_B: torch.Tensor) -> Dict[str, torch.Tensor]:
        fake_B = self.G_A2B(real_A)
        rec_A = self.G_B2A(fake_B)

        fake_A = self.G_B2A(real_B)
        rec_B = self.G_A2B(fake_A)

        idt_B = self.G_A2B(real_B)
        idt_A = self.G_B2A(real_A)

        return {
            "fake_B": fake_B,
            "rec_A": rec_A,
            "fake_A": fake_A,
            "rec_B": rec_B,
            "idt_A": idt_A,
            "idt_B": idt_B,
        }

    def compute_generator_loss(self, real_A: torch.Tensor, real_B: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out = self.forward_generators(real_A, real_B)

        loss_G_A2B = self.criterion_gan(self.D_B(out["fake_B"]), True)
        loss_G_B2A = self.criterion_gan(self.D_A(out["fake_A"]), True)

        loss_cycle_A = self.criterion_cycle(out["rec_A"], real_A) * self.lambda_cycle
        loss_cycle_B = self.criterion_cycle(out["rec_B"], real_B) * self.lambda_cycle

        # identity loss: nếu đưa ảnh B vào G_A2B thì nên giữ B; đưa A vào G_B2A thì nên giữ A
        loss_idt_B = self.criterion_identity(out["idt_B"], real_B) * self.lambda_cycle * self.lambda_identity
        loss_idt_A = self.criterion_identity(out["idt_A"], real_A) * self.lambda_cycle * self.lambda_identity

        loss_G = loss_G_A2B + loss_G_B2A + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B

        losses = {
            "G_total": loss_G,
            "G_A2B": loss_G_A2B,
            "G_B2A": loss_G_B2A,
            "cycle_A": loss_cycle_A,
            "cycle_B": loss_cycle_B,
            "idt_A": loss_idt_A,
            "idt_B": loss_idt_B,
        }
        return loss_G, {**out, **losses}

    def compute_D_loss(self, netD: nn.Module, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        pred_real = netD(real)
        loss_real = self.criterion_gan(pred_real, True)

        pred_fake = netD(fake.detach())
        loss_fake = self.criterion_gan(pred_fake, False)

        return (loss_real + loss_fake) * 0.5

    def compute_discriminator_losses(self, real_A: torch.Tensor, real_B: torch.Tensor, fake_A: torch.Tensor, fake_B: torch.Tensor):
        fake_A_for_D = self.fake_A_pool.query(fake_A)
        fake_B_for_D = self.fake_B_pool.query(fake_B)

        loss_D_A = self.compute_D_loss(self.D_A, real_A, fake_A_for_D)
        loss_D_B = self.compute_D_loss(self.D_B, real_B, fake_B_for_D)
        return loss_D_A, loss_D_B

    @torch.no_grad()
    def infer_A2B(self, real_A: torch.Tensor) -> torch.Tensor:
        self.G_A2B.eval()
        return self.G_A2B(real_A.to(self.device))

    @torch.no_grad()
    def infer_B2A(self, real_B: torch.Tensor) -> torch.Tensor:
        self.G_B2A.eval()
        return self.G_B2A(real_B.to(self.device))

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

    def load_checkpoint(self, path: str | Path, strict: bool = True, load_discriminators: bool = True):
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

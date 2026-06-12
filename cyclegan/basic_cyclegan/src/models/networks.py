from __future__ import annotations

import functools
from typing import Callable

import torch
import torch.nn as nn


def get_norm_layer(norm_type: str = "instance") -> Callable:
    norm_type = norm_type.lower()
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    if norm_type in ("none", "identity"):
        return lambda _: nn.Identity()
    raise ValueError(f"norm_type không hỗ trợ: {norm_type}")


def init_weights(net: nn.Module, init_type: str = "normal", init_gain: float = 0.02) -> None:
    def init_func(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if hasattr(m, "weight") and ("Conv" in classname or "Linear" in classname):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            else:
                raise ValueError(f"init_type không hỗ trợ: {init_type}")
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif "BatchNorm2d" in classname:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


class ResnetBlock(nn.Module):
    def __init__(self, dim: int, norm_layer: Callable, use_dropout: bool = False):
        super().__init__()
        block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            block += [nn.Dropout(0.5)]
        block += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            norm_layer(dim),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """
    Generator CycleGAN kiểu ResNet:
    c7s1-k -> downsample x2 -> residual blocks -> upsample x2 -> c7s1-3 -> tanh.
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        norm_type: str = "instance",
        use_dropout: bool = False,
        n_blocks: int = 9,
    ):
        super().__init__()
        norm_layer = get_norm_layer(norm_type)

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        # downsample
        n_downsampling = 2
        mult = 1
        for _ in range(n_downsampling):
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]
            mult *= 2

        # residual blocks
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer, use_dropout=use_dropout)]

        # upsample
        for _ in range(n_downsampling):
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=True,
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]
            mult = int(mult / 2)

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator."""

    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        norm_type: str = "instance",
    ):
        super().__init__()
        norm_layer = get_norm_layer(norm_type)

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=True,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=True,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class GANLoss(nn.Module):
    """LSGAN hoặc vanilla BCE GAN loss."""

    def __init__(self, gan_mode: str = "lsgan"):
        super().__init__()
        self.gan_mode = gan_mode.lower()
        if self.gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif self.gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"gan_mode không hỗ trợ: {gan_mode}")

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target_value = 1.0 if target_is_real else 0.0
        return torch.full_like(prediction, fill_value=target_value, device=prediction.device)

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target)

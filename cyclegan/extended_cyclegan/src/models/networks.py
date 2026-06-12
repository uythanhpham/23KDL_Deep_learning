# CODE BỊ BÀN CỜ
# # E:\Nam3_ki2\TH DL\PROJECT\23KDL_Deep_learning\cyclegan\src\models\networks.py:
# from __future__ import annotations

# import functools
# from typing import Callable

# import torch
# import torch.nn as nn


# def get_norm_layer(norm_type: str = "instance") -> Callable:
#     norm_type = norm_type.lower()
#     if norm_type == "batch":
#         return functools.partial(
#             nn.BatchNorm2d, affine=True, track_running_stats=True
#         )
#     if norm_type == "instance":
#         return functools.partial(
#             nn.InstanceNorm2d, affine=False, track_running_stats=False
#         )
#     if norm_type in ("none", "identity"):
#         return lambda _: nn.Identity()
#     raise ValueError(f"norm_type không hỗ trợ: {norm_type}")


# def init_weights(
#     net: nn.Module, init_type: str = "normal", init_gain: float = 0.02
# ) -> None:
#     def init_func(m: nn.Module) -> None:
#         classname = m.__class__.__name__
#         if hasattr(m, "weight") and (
#             "Conv" in classname or "Linear" in classname
#         ):
#             if init_type == "normal":
#                 nn.init.normal_(m.weight.data, 0.0, init_gain)
#             elif init_type == "xavier":
#                 nn.init.xavier_normal_(m.weight.data, gain=init_gain)
#             elif init_type == "kaiming":
#                 nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
#             else:
#                 raise ValueError(f"init_type không hỗ trợ: {init_type}")
#             if getattr(m, "bias", None) is not None:
#                 nn.init.constant_(m.bias.data, 0.0)
#         elif "BatchNorm2d" in classname:
#             nn.init.normal_(m.weight.data, 1.0, init_gain)
#             nn.init.constant_(m.bias.data, 0.0)

#     net.apply(init_func)


# # =========================================================================
# # THÀNH PHẦN PALETTE-BASED STYLE TRANSFER: MAPPING NETWORK & ADAIN
# # =========================================================================


# class MappingNetwork(nn.Module):
#     """Mạng MLP biến đổi vector bảng màu 24 chiều thành một vector style gộp

#     có kích thước lớn chuẩn 10,368 chiều để cấu hình tham số cho các tầng AdaIN.
#     """

#     def __init__(self, input_dim: int = 24, style_dim: int = 10368):
#         super().__init__()
#         self.mapping = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.ReLU(True),
#             nn.Linear(256, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 1024),
#             nn.ReLU(True),
#             nn.Linear(1024, style_dim),
#         )

#     def forward(self, palette: torch.Tensor) -> torch.Tensor:
#         return self.mapping(palette)


# class AdaIN(nn.Module):
#     """Adaptive Instance Normalization.

#     Tính toán lại đặc trưng chuẩn hóa dựa trên giá trị scale (gamma) và shift
#     (beta) trích xuất trực tiếp từ chuỗi vector style.
#     """

#     def __init__(self, num_features: int):
#         super().__init__()
#         self.instance_norm = nn.InstanceNorm2d(
#             num_features, affine=False, track_running_stats=False
#         )

#     def forward(
#         self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
#     ) -> torch.Tensor:
#         normalized = self.instance_norm(x)
#         scale = scale.unsqueeze(-1).unsqueeze(-1)
#         shift = shift.unsqueeze(-1).unsqueeze(-1)
#         return normalized * scale + shift


# # =========================================================================
# # KHỐI XÂY DỰNG MẠNG GENERATOR VÀ DISCRIMINATOR
# # =========================================================================


# class ResnetBlock(nn.Module):
#     """Khối ResNet cải tiến tích hợp tầng kiểm soát màu AdaIN thay thế cho

#     InstanceNorm cố định.
#     """

#     def __init__(self, dim: int, use_dropout: bool = False):
#         super().__init__()
#         self.pad1 = nn.ReflectionPad2d(1)
#         self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True)
#         self.adain1 = AdaIN(dim)
#         self.relu = nn.ReLU(True)

#         self.use_dropout = use_dropout
#         if use_dropout:
#             self.dropout = nn.Dropout(0.5)

#         self.pad2 = nn.ReflectionPad2d(1)
#         self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True)
#         self.adain2 = AdaIN(dim)

#     def forward(
#         self,
#         x: torch.Tensor,
#         scale1: torch.Tensor,
#         shift1: torch.Tensor,
#         scale2: torch.Tensor,
#         shift2: torch.Tensor,
#     ) -> torch.Tensor:
#         residual = x

#         out = self.pad1(x)
#         out = self.conv1(out)
#         out = self.adain1(out, scale1, shift1)
#         out = self.relu(out)

#         if self.use_dropout:
#             out = self.dropout(out)

#         out = self.pad2(out)
#         out = self.conv2(out)
#         out = self.adain2(out, scale2, shift2)

#         return residual + out


# class ResnetGenerator(nn.Module):
#     """Generator CycleGAN Palette hoàn chỉnh, loại bỏ AdaIN tầng khởi tạo để bảo

#     toàn cấu trúc hình khối và cân bằng tham số cắt mảng tensor (10368).
#     """

#     def __init__(
#         self,
#         input_nc: int = 3,
#         output_nc: int = 3,
#         ngf: int = 64,
#         use_dropout: bool = False,
#         n_blocks: int = 9,
#     ):
#         super().__init__()
#         self.n_blocks = n_blocks
#         self.style_dim = 10368
#         self.mapping_net = MappingNetwork(input_dim=24, style_dim=self.style_dim)

#         # 1. Tầng khởi tạo (Mặc định không dùng AdaIN để giữ kết cấu khối thô)
#         self.start_pad = nn.ReflectionPad2d(3)
#         self.start_conv = nn.Conv2d(
#             input_nc, ngf, kernel_size=7, padding=0, bias=True
#         )
#         self.relu = nn.ReLU(True)

#         # 2. Các tầng Downsample
#         self.down_conv1 = nn.Conv2d(
#             ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=True
#         )
#         self.down_adain1 = AdaIN(ngf * 2)

#         self.down_conv2 = nn.Conv2d(
#             ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=True
#         )
#         self.down_adain2 = AdaIN(ngf * 4)

#         # 3. Khối các lớp ResNet Blocks độc lập (Tổng: 9 khối)
#         self.resnet_blocks = nn.ModuleList(
#             [ResnetBlock(ngf * 4, use_dropout=use_dropout) for _ in range(n_blocks)]
#         )

#         # 4. Các tầng Upsample
#         self.up_conv1 = nn.ConvTranspose2d(
#             ngf * 4,
#             ngf * 2,
#             kernel_size=3,
#             stride=2,
#             padding=1,
#             output_padding=1,
#             bias=True,
#         )
#         self.up_adain1 = AdaIN(ngf * 2)

#         self.up_conv2 = nn.ConvTranspose2d(
#             ngf * 2,
#             ngf,
#             kernel_size=3,
#             stride=2,
#             padding=1,
#             output_padding=1,
#             bias=True,
#         )
#         self.up_adain2 = AdaIN(ngf)

#         # 5. Khối kết xuất đầu ra ảnh kết quả
#         self.end_pad = nn.ReflectionPad2d(3)
#         self.end_conv = nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)
#         self.tanh = nn.Tanh()

#     def forward(self, x: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
#         # Chuyển đổi bảng màu sang không gian style code tổng hợp [B, 10368]
#         style_vector = self.mapping_net(palette)

#         pointer = 0

#         def get_style_params(num_features: int):
#             nonlocal pointer
#             scale = style_vector[:, pointer : pointer + num_features]
#             pointer += num_features
#             shift = style_vector[:, pointer : pointer + num_features]
#             pointer += num_features
#             return scale, shift

#         # --- LUỒNG XỬ LÝ GENERATOR MỚI: CHUẨN XÁC VỊ TRÍ CẮT ---
#         # Khởi tạo (Lớp đầu tiên bỏ AdaIN)
#         out = self.start_pad(x)
#         out = self.start_conv(out)
#         out = self.relu(out)

#         # Downsample 1 (Nhát cắt đầu tiên lấy 128x2 = 256 số từ pointer 0)
#         out = self.down_conv1(out)
#         scale, shift = get_style_params(128)
#         out = self.down_adain1(out, scale, shift)
#         out = self.relu(out)

#         # Downsample 2
#         out = self.down_conv2(out)
#         scale, shift = get_style_params(256)
#         out = self.down_adain2(out, scale, shift)
#         out = self.relu(out)

#         # Luồn qua chuỗi 9 khối ResNet kèm cặp AdaIN kép
#         for block in self.resnet_blocks:
#             s1, h1 = get_style_params(256)
#             s2, h2 = get_style_params(256)
#             out = block(out, s1, h1, s2, h2)

#         # Upsample 1
#         out = self.up_conv1(out)
#         scale, shift = get_style_params(128)
#         out = self.up_adain1(out, scale, shift)
#         out = self.relu(out)

#         # Upsample 2 (Nhát cắt cuối cùng khép lại vừa vặn tại pointer 10368)
#         out = self.up_conv2(out)
#         scale, shift = get_style_params(64)
#         out = self.up_adain2(out, scale, shift)
#         out = self.relu(out)

#         # Xuất ảnh hoàn chỉnh qua hàm Tanh
#         out = self.end_pad(out)
#         out = self.end_conv(out)
#         return self.tanh(out)


# class NLayerDiscriminator(nn.Module):
#     """PatchGAN discriminator (Giữ nguyên cấu trúc phân biệt thực/giả vùng ảnh)."""

#     def __init__(
#         self,
#         input_nc: int = 3,
#         ndf: int = 64,
#         n_layers: int = 3,
#         norm_type: str = "instance",
#     ):
#         super().__init__()
#         norm_layer = get_norm_layer(norm_type)

#         kw = 4
#         padw = 1
#         sequence = [
#             nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
#             nn.LeakyReLU(0.2, True),
#         ]

#         nf_mult = 1
#         nf_mult_prev = 1
#         for n in range(1, n_layers):
#             nf_mult_prev = nf_mult
#             max_power = min(n, 3)
#             nf_mult = min(2**max_power, 8)
#             sequence += [
#                 nn.Conv2d(
#                     ndf * nf_mult_prev,
#                     ndf * nf_mult,
#                     kernel_size=kw,
#                     stride=2,
#                     padding=padw,
#                     bias=True,
#                 ),
#                 norm_layer(ndf * nf_mult),
#                 nn.LeakyReLU(0.2, True),
#             ]

#         nf_mult_prev = nf_mult
#         max_power_layers = min(n_layers, 3)
#         nf_mult = min(2**max_power_layers, 8)
#         sequence += [
#             nn.Conv2d(
#                 ndf * nf_mult_prev,
#                 ndf * nf_mult,
#                 kernel_size=kw,
#                 stride=1,
#                 padding=padw,
#                 bias=True,
#             ),
#             norm_layer(ndf * nf_mult),
#             nn.LeakyReLU(0.2, True),
#         ]

#         sequence += [
#             nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
#         ]
#         self.model = nn.Sequential(*sequence)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.model(x)


# class GANLoss(nn.Module):
#     """LSGAN hoặc vanilla BCE GAN loss."""

#     def __init__(self, gan_mode: str = "lsgan"):
#         super().__init__()
#         self.gan_mode = gan_mode.lower()
#         if self.gan_mode == "lsgan":
#             self.loss = nn.MSELoss()
#         elif self.gan_mode == "vanilla":
#             self.loss = nn.BCEWithLogitsLoss()
#         else:
#             self.loss = nn.Identity()

#     def get_target_tensor(
#         self, prediction: torch.Tensor, target_is_real: bool
#     ) -> torch.Tensor:
#         target_value = 1.0 if target_is_real else 0.0
#         return torch.full_like(
#             prediction, fill_value=target_value, device=prediction.device
#         )

#     def forward(
#         self, prediction: torch.Tensor, target_is_real: bool
#     ) -> torch.Tensor:
#         if isinstance(self.loss, nn.Identity):
#             raise ValueError(f"gan_mode không hỗ trợ: {self.gan_mode}")
#         target = self.get_target_tensor(prediction, target_is_real)
#         return self.loss(prediction, target)







# CODE NEAREST, HẾT BÀN CỜ NHƯNG BỊ MỜ
# E:\Nam3_ki2\TH DL\PROJECT\23KDL_Deep_learning\cyclegan\src\models\networks.py:
# from __future__ import annotations

# import functools
# from typing import Callable

# import torch
# import torch.nn as nn


# def get_norm_layer(norm_type: str = "instance") -> Callable:
#     norm_type = norm_type.lower()
#     if norm_type == "batch":
#         return functools.partial(
#             nn.BatchNorm2d, affine=True, track_running_stats=True
#         )
#     if norm_type == "instance":
#         return functools.partial(
#             nn.InstanceNorm2d, affine=False, track_running_stats=False
#         )
#     if norm_type in ("none", "identity"):
#         return lambda _: nn.Identity()
#     raise ValueError(f"norm_type không hỗ trợ: {norm_type}")


# def init_weights(
#     net: nn.Module, init_type: str = "normal", init_gain: float = 0.02
# ) -> None:
#     def init_func(m: nn.Module) -> None:
#         classname = m.__class__.__name__
#         if hasattr(m, "weight") and (
#             "Conv" in classname or "Linear" in classname
#         ):
#             if init_type == "normal":
#                 nn.init.normal_(m.weight.data, 0.0, init_gain)
#             elif init_type == "xavier":
#                 nn.init.xavier_normal_(m.weight.data, gain=init_gain)
#             elif init_type == "kaiming":
#                 nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
#             else:
#                 raise ValueError(f"init_type không hỗ trợ: {init_type}")
#             if getattr(m, "bias", None) is not None:
#                 nn.init.constant_(m.bias.data, 0.0)
#         elif "BatchNorm2d" in classname:
#             nn.init.normal_(m.weight.data, 1.0, init_gain)
#             nn.init.constant_(m.bias.data, 0.0)

#     net.apply(init_func)


# class MappingNetwork(nn.Module):
#     def __init__(self, input_dim: int = 24, style_dim: int = 10368):
#         super().__init__()
#         self.mapping = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.ReLU(True),
#             nn.Linear(256, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 1024),
#             nn.ReLU(True),
#             nn.Linear(1024, style_dim),
#         )

#     def forward(self, palette: torch.Tensor) -> torch.Tensor:
#         return self.mapping(palette)


# class AdaIN(nn.Module):
#     def __init__(self, num_features: int):
#         super().__init__()
#         self.instance_norm = nn.InstanceNorm2d(
#             num_features, affine=False, track_running_stats=False
#         )

#     def forward(
#         self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
#     ) -> torch.Tensor:
#         normalized = self.instance_norm(x)
#         scale = scale.unsqueeze(-1).unsqueeze(-1)
#         shift = shift.unsqueeze(-1).unsqueeze(-1)
#         return normalized * scale + shift


# class ResnetBlock(nn.Module):
#     def __init__(self, dim: int, use_dropout: bool = False):
#         super().__init__()
#         self.pad1 = nn.ReflectionPad2d(1)
#         self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True)
#         self.adain1 = AdaIN(dim)
#         self.relu = nn.ReLU(True)

#         self.use_dropout = use_dropout
#         if use_dropout:
#             self.dropout = nn.Dropout(0.5)

#         self.pad2 = nn.ReflectionPad2d(1)
#         self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True)
#         self.adain2 = AdaIN(dim)

#     def forward(
#         self, x: torch.Tensor, s1: torch.Tensor, h1: torch.Tensor, s2: torch.Tensor, h2: torch.Tensor
#     ) -> torch.Tensor:
#         residual = x
#         out = self.pad1(x)
#         out = self.conv1(out)
#         out = self.adain1(out, s1, h1)
#         out = self.relu(out)
#         if self.use_dropout:
#             out = self.dropout(out)
#         out = self.pad2(out)
#         out = self.conv2(out)
#         out = self.adain2(out, s2, h2)
#         return residual + out


# class ResnetGenerator(nn.Module):
#     def __init__(
#         self, input_nc: int = 3, output_nc: int = 3, ngf: int = 64, use_dropout: bool = False, n_blocks: int = 9
#     ):
#         super().__init__()
#         self.n_blocks = n_blocks
#         self.style_dim = 10368
#         self.mapping_net = MappingNetwork(input_dim=24, style_dim=self.style_dim)

#         self.start_pad = nn.ReflectionPad2d(3)
#         self.start_conv = nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True)
#         self.relu = nn.ReLU(True)

#         self.down_conv1 = nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=True)
#         self.down_adain1 = AdaIN(ngf * 2)

#         self.down_conv2 = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=True)
#         self.down_adain2 = AdaIN(ngf * 4)

#         self.resnet_blocks = nn.ModuleList([ResnetBlock(ngf * 4, use_dropout=use_dropout) for _ in range(n_blocks)])

#         # Upsample 1: Upsample + Conv2d thay cho ConvTranspose2d
#         self.up_sample1 = nn.Upsample(scale_factor=2, mode='nearest')
#         self.up_conv1 = nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, stride=1, padding=1, bias=True)
#         self.up_adain1 = AdaIN(ngf * 2)

#         # Upsample 2: Upsample + Conv2d thay cho ConvTranspose2d
#         self.up_sample2 = nn.Upsample(scale_factor=2, mode='nearest')
#         self.up_conv2 = nn.Conv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1, bias=True)
#         self.up_adain2 = AdaIN(ngf)

#         self.end_pad = nn.ReflectionPad2d(3)
#         self.end_conv = nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)
#         self.tanh = nn.Tanh()

#     def forward(self, x: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
#         style_vector = self.mapping_net(palette)
#         pointer = 0

#         def get_style_params(num_features: int):
#             nonlocal pointer
#             scale = style_vector[:, pointer : pointer + num_features]
#             pointer += num_features
#             shift = style_vector[:, pointer : pointer + num_features]
#             pointer += num_features
#             return scale, shift

#         out = self.start_pad(x)
#         out = self.start_conv(out)
#         out = self.relu(out)

#         out = self.down_conv1(out)
#         scale, shift = get_style_params(128)
#         out = self.down_adain1(out, scale, shift)
#         out = self.relu(out)

#         out = self.down_conv2(out)
#         scale, shift = get_style_params(256)
#         out = self.down_adain2(out, scale, shift)
#         out = self.relu(out)

#         for block in self.resnet_blocks:
#             s1, h1 = get_style_params(256)
#             s2, h2 = get_style_params(256)
#             out = block(out, s1, h1, s2, h2)

#         # Upsample 1
#         out = self.up_sample1(out)
#         out = self.up_conv1(out)
#         scale, shift = get_style_params(128)
#         out = self.up_adain1(out, scale, shift)
#         out = self.relu(out)

#         # Upsample 2
#         out = self.up_sample2(out)
#         out = self.up_conv2(out)
#         scale, shift = get_style_params(64)
#         out = self.up_adain2(out, scale, shift)
#         out = self.relu(out)

#         out = self.end_pad(out)
#         out = self.end_conv(out)
#         return self.tanh(out)


# class NLayerDiscriminator(nn.Module):
#     def __init__(self, input_nc: int = 3, ndf: int = 64, n_layers: int = 3, norm_type: str = "instance"):
#         super().__init__()
#         norm_layer = get_norm_layer(norm_type)
#         kw, padw = 4, 1
#         sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
#         nf_mult = 1
#         for n in range(1, n_layers):
#             nf_mult_prev = nf_mult
#             nf_mult = min(2**n, 8)
#             sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=True), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
#         nf_mult_prev = nf_mult
#         nf_mult = min(2**n_layers, 8)
#         sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=True), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
#         sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
#         self.model = nn.Sequential(*sequence)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.model(x)


# class GANLoss(nn.Module):
#     def __init__(self, gan_mode: str = "lsgan"):
#         super().__init__()
#         self.gan_mode = gan_mode.lower()
#         if self.gan_mode == "lsgan":
#             self.loss = nn.MSELoss()
#         elif self.gan_mode == "vanilla":
#             self.loss = nn.BCEWithLogitsLoss()
#         else:
#             self.loss = nn.Identity()

#     def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
#         return torch.full_like(prediction, 1.0 if target_is_real else 0.0, device=prediction.device)

#     def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
#         target = self.get_target_tensor(prediction, target_is_real)
#         return self.loss(prediction, target)
    

# 
from __future__ import annotations

import functools
from typing import Callable

import torch
import torch.nn as nn


def get_norm_layer(norm_type: str = "instance") -> Callable:
    norm_type = norm_type.lower()
    if norm_type == "batch":
        return functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )
    if norm_type == "instance":
        return functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    if norm_type in ("none", "identity"):
        return lambda _: nn.Identity()
    raise ValueError(f"norm_type không hỗ trợ: {norm_type}")


def init_weights(
    net: nn.Module, init_type: str = "normal", init_gain: float = 0.02
) -> None:
    def init_func(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            "Conv" in classname or "Linear" in classname
        ):
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


class MappingNetwork(nn.Module):
    def __init__(self, input_dim: int = 24, style_dim: int = 9216):
        super().__init__()
        self.mapping = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, style_dim),
        )

    def forward(self, palette: torch.Tensor) -> torch.Tensor:
        return self.mapping(palette)


class AdaIN(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(
            num_features, affine=False, track_running_stats=False
        )

    def forward(
        self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.instance_norm(x)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return normalized * scale + shift


class ResnetBlock(nn.Module):
    def __init__(self, dim: int, use_dropout: bool = False):
        super().__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True)
        self.adain1 = AdaIN(dim)
        self.relu = nn.ReLU(True)

        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(0.5)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True)
        self.adain2 = AdaIN(dim)

    def forward(
        self, x: torch.Tensor, s1: torch.Tensor, h1: torch.Tensor, s2: torch.Tensor, h2: torch.Tensor
    ) -> torch.Tensor:
        residual = x
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.adain1(out, s1, h1)
        out = self.relu(out)
        if self.use_dropout:
            out = self.dropout(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.adain2(out, s2, h2)
        return residual + out


class ResnetGenerator(nn.Module):
    def __init__(
        self, input_nc: int = 3, output_nc: int = 3, ngf: int = 64, use_dropout: bool = False, n_blocks: int = 9
    ):
        super().__init__()
        self.n_blocks = n_blocks
        # 9 blocks * 2 AdaIN per block * 2 params (scale/shift) * 256 features = 9216
        self.style_dim = n_blocks * 2 * 2 * (ngf * 4)
        self.mapping_net = MappingNetwork(input_dim=24, style_dim=self.style_dim)

        self.start_pad = nn.ReflectionPad2d(3)
        self.start_conv = nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True)
        self.start_norm = nn.InstanceNorm2d(ngf)
        self.relu = nn.ReLU(True)

        self.down_conv1 = nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=True)
        self.down_norm1 = nn.InstanceNorm2d(ngf * 2)

        self.down_conv2 = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=True)
        self.down_norm2 = nn.InstanceNorm2d(ngf * 4)

        self.resnet_blocks = nn.ModuleList([ResnetBlock(ngf * 4, use_dropout=use_dropout) for _ in range(n_blocks)])

        self.up_sample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv1 = nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.up_norm1 = nn.InstanceNorm2d(ngf * 2)

        self.up_sample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv2 = nn.Conv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1, bias=True)
        self.up_norm2 = nn.InstanceNorm2d(ngf)

        self.end_pad = nn.ReflectionPad2d(3)
        self.end_conv = nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
        style_vector = self.mapping_net(palette)
        pointer = 0

        def get_style_params(num_features: int):
            nonlocal pointer
            scale = style_vector[:, pointer : pointer + num_features]
            pointer += num_features
            shift = style_vector[:, pointer : pointer + num_features]
            pointer += num_features
            return scale, shift

        out = self.start_pad(x)
        out = self.start_conv(out)
        out = self.start_norm(out)
        out = self.relu(out)

        out = self.down_conv1(out)
        out = self.down_norm1(out)
        out = self.relu(out)

        out = self.down_conv2(out)
        out = self.down_norm2(out)
        out = self.relu(out)

        for block in self.resnet_blocks:
            s1, h1 = get_style_params(256)
            s2, h2 = get_style_params(256)
            out = block(out, s1, h1, s2, h2)

        out = self.up_sample1(out)
        out = self.up_conv1(out)
        out = self.up_norm1(out)
        out = self.relu(out)

        out = self.up_sample2(out)
        out = self.up_conv2(out)
        out = self.up_norm2(out)
        out = self.relu(out)

        out = self.end_pad(out)
        out = self.end_conv(out)
        return self.tanh(out)


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc: int = 3, ndf: int = 64, n_layers: int = 3, norm_type: str = "instance"):
        super().__init__()
        norm_layer = get_norm_layer(norm_type)
        kw, padw = 4, 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=True), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=True), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class GANLoss(nn.Module):
    def __init__(self, gan_mode: str = "lsgan"):
        super().__init__()
        self.gan_mode = gan_mode.lower()
        if self.gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif self.gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = nn.Identity()

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        return torch.full_like(prediction, 1.0 if target_is_real else 0.0, device=prediction.device)

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target)


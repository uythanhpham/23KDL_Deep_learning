from __future__ import annotations

import random

import torch


class ImagePool:
    """
    Image buffer cho discriminator theo CycleGAN.
    Dùng fake ảnh cũ lẫn mới để giảm dao động khi train GAN.
    """

    def __init__(self, pool_size: int = 50):
        self.pool_size = int(pool_size)
        self.images: list[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.pool_size <= 0:
            return images

        return_images = []
        for image in images:
            image = image.unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(image.detach().clone())
                return_images.append(image)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.pool_size - 1)
                    old = self.images[idx].clone()
                    self.images[idx] = image.detach().clone()
                    return_images.append(old)
                else:
                    return_images.append(image)
        return torch.cat(return_images, dim=0)

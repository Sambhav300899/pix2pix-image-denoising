import torch
import torchvision
import logging
from typing import List

logger = logging.getLogger(__name__)


def gaussian_noise(img: torch.Tensor, std_div_denum: int = 1) -> torch.Tensor:
    """
    Function to add gaussian noise to an image. The noise will
    have a mean of 0 and std_dev anywhere between 0 and 1, this is
    picked randomly

    Args:
        img (py:obj:`torch.Tensor`): input image
    Retruns:
        img (py:obj:`torch.Tensor`): input image + random gaussian noise
    """
    std = torch.rand(1) / std_div_denum
    mean = 0

    logger.debug("adding noise to img with std: {std} and mean: {mean}")
    img = img + torch.randn_like(img) * std + mean
    img = img.clip(-1, 1)

    return img


class augs:
    """
    Class to add augmentations to images

    Args:
        input_size (List(int)): list specifying input image size
        gaussian_noise(bool): Flag to do random gaussian noise
        random_mirroring(bool): Flag to do random mirroring
    """

    def __init__(
        self,
        input_size: List[int] = (256, 256),
        gaussian_noise: bool = False,
        random_mirroring: bool = False,
    ) -> None:
        self.input_size = input_size
        self.gaussian_noise = gaussian_noise
        self.random_mirroring = random_mirroring

        self.resize_transform = torchvision.transforms.Resize(self.input_size)

    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Function to apply transforms

        Args:
            imgs(torch.Tensor): Batch to images to apply transform on

        Returns:
            x (torch.Tensor): x tensor
            y (torch.Tensor): y tensor
        """
        x, y = imgs
        x = self.resize_transform(x)
        y = self.resize_transform(y)

        if self.gaussian_noise:
            logger.debug("gaussian noise augmentation is being applied")
            x = gaussian_noise(x)

        if self.random_mirroring:
            logger.debug("random mirroring augmentation is being applied")
            if torch.rand(1) > 0.5:
                x = torchvision.transforms.functional.hflip(x)
                y = torchvision.transforms.functional.hflip(y)

            # if torch.rand(1) > 0.5:
            #     x = torchvision.transforms.functional.vflip(x)
            #     y = torchvision.transforms.functional.vflip(y)

        return x, y

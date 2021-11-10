import torch
from torch import nn
import logging

logger = logging.getLogger(__name__)

mean_absolute_error = nn.L1Loss()
crossentropy_loss = nn.BCELoss()


def gen_loss(
    generated_img: torch.Tensor,
    target_img: torch.Tensor,
    disc_fake_y: torch.Tensor,
    real_target: torch.Tensor,
    lambda_param: int = 100,
) -> torch.Tensor:
    """
    Calculate generator loss according to computation from original paper

    Args:
        generated_img (py:obj:`Torch.Tensor`): generated image from generator
        target_img (py:obj:`Torch.Tensor`): target actual image
        disc_fake_y (py:obj:`Torch.Tensor`): fake target for discriminator
        real_target (py:obj:`Torch.Tensor`):  real target for discriminator
        lambda_param (py:obj:`Torch.Tensor`): lambda mutliplier for MAE
    Return:
        gen_total_loss (py:obj:`Torch.Tensor`): total generator loss
    """
    gen_loss = crossentropy_loss(disc_fake_y, real_target)
    mse = mean_absolute_error(generated_img, target_img)
    gen_total_loss = gen_loss + (lambda_param * mse)

    return gen_total_loss


def disc_loss(output: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """
    Calculate discriminator loss according to computation from original paper

    Args:
        output (py:obj:`Torch.Tensor`):  output from discriminator
        label (py:obj:`Torch.Tensor`): gt for discriminator
    Return:
        (py:obj:`Torch.Tensor`): discriminator loss
    """
    return crossentropy_loss(output, label)

from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import torch


def ssim(tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
    """
    Calculate ssim between two tensors (does not work for batches)

    Args:
        tensor_1 (py:obj:`Torch.Tensor`):  tensor_1
        tensor_2 (py:obj:`Torch.Tensor`): tensor_2
    Return:
        score (float): ssim score
    """

    tensor_1 = tensor_1.permute((1, 2, 0))
    tensor_2 = tensor_2.permute((1, 2, 0))

    score = structural_similarity(tensor_1.numpy(), tensor_2.numpy(), multichannel=True, dynamic_range=1)

    return score


def psnr(tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
    """
    Calculate psnr between two tensors (does not work for batches)

    Args:
        tensor_1 (py:obj:`Torch.Tensor`):  tensor_1
        tensor_2 (py:obj:`Torch.Tensor`): tensor_2
    Return:
        score (float): psnr score
    """
    tensor_1 = tensor_1.permute((1, 2, 0))
    tensor_2 = tensor_2.permute((1, 2, 0))

    score = peak_signal_noise_ratio(tensor_1.numpy(), tensor_2.numpy(), data_range=1)

    return score


def ssim_for_batch(batch_1: torch.Tensor, batch_2: torch.Tensor) -> torch.Tensor:
    """
    Calculate ssim between two tensor batches

    Args:
        tensor_1 (py:obj:`Torch.Tensor`):  tensor_1
        tensor_2 (py:obj:`Torch.Tensor`): tensor_2
    Return:
        score (float): ssim score
    """
    ssim_list = []

    for i in range(batch_1.shape[0]):
        ssim_list.append(ssim(batch_1[i, :, :, :].squeeze(), batch_2[i, :, :, :].squeeze()))

    return sum(ssim_list) / batch_1.shape[0]


def psnr_for_batch(batch_1: torch.Tensor, batch_2: torch.Tensor) -> torch.Tensor:
    """
    Calculate psnr between two tensor batches

    Args:
        tensor_1 (py:obj:`Torch.Tensor`):  tensor_1
        tensor_2 (py:obj:`Torch.Tensor`): tensor_2
    Return:
        score (float): psnr score
    """

    psnr_list = []

    for i in range(batch_1.shape[0]):
        psnr_list.append(ssim(batch_1[i, :, :, :].squeeze(), batch_2[i, :, :, :].squeeze()))

    return sum(psnr_list) / batch_1.shape[0]

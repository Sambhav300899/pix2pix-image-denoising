import torch
import torchvision


def gaussian_noise(img):
    std = torch.rand(1)
    mean = 0

    img = img + torch.randn_like(img) * std + mean
    img = img.clip(-1, 1)

    return img


class augs:
    def __init__(
        self,
        input_size=(256, 256),
        gaussian_noise=False,
        random_mirroring=False,
    ):
        self.input_size = input_size
        self.gaussian_noise = gaussian_noise
        self.random_mirroring = random_mirroring

        self.resize_transform = torchvision.transforms.Resize(self.input_size)

    def __call__(self, imgs):
        x, y = imgs
        x = self.resize_transform(x)
        y = self.resize_transform(y)

        if self.gaussian_noise:
            x = gaussian_noise(x)

        if self.random_mirroring:
            if torch.rand(1) > 0.5:
                x = torchvision.transforms.functional.hflip(x)
                y = torchvision.transforms.functional.hflip(y)

            # if torch.rand(1) > 0.5:
            #     x = torchvision.transforms.functional.vflip(x)
            #     y = torchvision.transforms.functional.vflip(y)

        return x, y

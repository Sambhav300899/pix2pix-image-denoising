import torch
from typing import List
import logging
from pix2pix import augmentations, helper

logger = logging.getLogger(__name__)


class lfw_dataset(torch.utils.data.Dataset):
    """
    Class to load the lfw dataset for an image to image translation model. Pass
    the operations to be applied to the input image as transforms to the class.

    For model training
    target = image from lfw dataset
    input = image from lfw dataset specified transforms

    Args:
        rootdir (str): path to rootdir of dataset
        transforms (List[:py:obj:`torchvision.transforms`]): transforms to be applied to output img,
                    defaults to None

    """

    def __init__(
        self,
        img_paths: List[str],
        transforms: augmentations.augs = None,
    ) -> None:
        """

        The structure of the dataset is -

        rootdir
            '--name_of_person_1
                '--img_of_person_1_a.jpg
                '--img_of_person_1_b.jpg
                .
                .
            .
            .
            .
            '--name_of_person_n
                '--img_of_person_n_a.jpg
                .
                .
        """
        self.file_list = img_paths
        self.transforms = transforms

    def __len__(self) -> int:
        """
        get the number of samples in the dataset

        Returns:
            (int): Number of samples
        """
        return len(self.file_list)

    def __getitem__(self, idx: List[int]) -> List[torch.Tensor]:
        """
        get item at given idx from dataset

        Args:
            idx (List[int]): index of training sample
        Returns:
            ip(py:obj:`torch.Tensor`): Input image to model
            target(py:obj:`torch.Tensor`): Target image for model
        """
        if torch.is_tensor(idx):
            idx = idx.to_list()

        img_path = self.file_list[idx]
        target = torch.tensor(helper.read_img(img_path), dtype=torch.float32)
        target = (target / (255 / 2)) - 1
        target = target.permute((2, 0, 1))
        ip = target.clone()

        if self.transforms:
            ip, target = self.transforms([ip, target])

        return ip, target

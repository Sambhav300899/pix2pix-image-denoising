from pydantic import BaseSettings
from typing import List
import torch


class Dataset(BaseSettings):
    DATASET_PATH: str = "../../../generated/lfw/"
    TRAIN_VAL_TEST_SPLIT: List[float] = [0.6, 0.2, 0.2]

    if sum(TRAIN_VAL_TEST_SPLIT) != 1:
        raise ValueError("Train val test split list must sum to 1")

    IMAGE_SIZE: int = 256
    SHUFFLE_TRAIN: bool = True
    SHUFFLE_VAL: bool = False
    SHUFFLE_TEST: bool = False
    BATCH_SIZE: int = 32


dataset = Dataset()


class Augmentations(BaseSettings):
    TRAIN_AUGS: dict = dict(gaussian_noise=True, random_mirroring=True)
    VAL_AUGS: dict = dict(gaussian_noise=True)
    TEST_AUGS: dict = dict(gaussian_noise=True)


aug_config = Augmentations()


class Model(BaseSettings):
    GEN_ARGS: dict = dict(input_nc=3, output_nc=3, num_init_filters=32, use_dropout=False)

    DISC_ARGS: dict = dict(input_nc=6)


model = Model()


class Settings(BaseSettings):
    CHECKPOINT_PATH = ""
    CONTINUE_FROM_CHECKPOINT: bool = True if CHECKPOINT_PATH else False
    SAVE_DIRECTORY = "../../models"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


settings = Settings()


class Training(BaseSettings):
    GEN_LR: float = 2e-4
    DISC_LR: float = 2e-4
    EPOCHS: int = 2
    SAVE_CHECKPOINTS: bool = True


training = Training()


NOTE = ""

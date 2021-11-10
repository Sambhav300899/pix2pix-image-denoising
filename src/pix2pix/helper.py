import cv2
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)


def read_img(img_path: str) -> np.array:
    """
    Read image with opencv and convert it to RGB

    Args:
        img_path (str): path to img
    Returns:
        img (py:obj:`np.array`): loaded image
    """
    logger.debug(f"reading image from {img_path}")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def save_json(data: dict, path_to_save: str) -> None:
    """
    Save json dict to specified file

    Args:
        data (dict): dictionary with json data
        path_to_save(str): path to save json
    """
    logger.debug(f"saving {data} to {path_to_save}")

    with open(path_to_save, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

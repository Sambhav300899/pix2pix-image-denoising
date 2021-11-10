import cv2
import json


def read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def save_json(data, path_to_save):
    with open(path_to_save, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

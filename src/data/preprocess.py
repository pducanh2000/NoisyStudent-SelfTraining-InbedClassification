import random
import cv2
import numpy as np
import PIL
from PIL import ImageOps

import torch
from torchvision.transforms import ToPILImage, ToTensor, Normalize


def translate_x(image, p=0.5):
    value = image.size[0]
    if random.random() <= p:
        value = -value

    return image.transform(image.size, PIL.Image.AFFINE, (1, 0, value, 0, 1, 0))


def translate_y(image, p=0.5):
    value = image.size[1] * 0.1
    if random.random() <= p:
        value = -value

    return image.transform(image.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, value))


def rotation(image, p=0.5):
    value = 25
    if random.random() <= p:
        value = -value

    return image.rotate(value)


def cutout(img, n_holes, length):
    h = img.shape[1]
    w = img.shape[2]

    mask = np.ones((h, w), np.float32)

    for n in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.

    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    img = img * mask

    return img


def flip(image):
    return ImageOps.flip(image)


def preprocess(image, posture):
    image = cv2.equalizeHist(image)

    image = ToPILImage()(image)
    image = image.convert('RGB')
    image = image.resize((112, 224))

    image = ToTensor()(image)
    image = Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])(image)

    return image, torch.tensor(posture, dtype=torch.long)

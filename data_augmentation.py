import cv2
import random
import numpy as np
from scipy import ndimage

rand = random.Random()


def brightness(img, low, high):
    value = random.uniform(low, high)
    img = cv2.addWeighted(img, value, np.zeros_like(img), 1.5, 0)
    return img


def horizontal_flip(img, flag):
    if flag:
        return cv2.flip(img, 1)
    else:
        return img


def vertical_flip(img, flag):
    if flag:
        return cv2.flip(img, 0)
    else:
        return img


def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    img = ndimage.rotate(img, angle, mode='reflect')
    return img


def data_augmentation(img):
    img = brightness(img, 0.9, 2)
    img = horizontal_flip(img, rand.randint(0, 10) / 2 == 0)
    img = vertical_flip(img, rand.randint(0, 10) / 2 == 0)
    img = rotation(img, rand.randint(0, 30))

    return img

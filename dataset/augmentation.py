import numpy as np
import random


class RandomFlip:
    def __call__(self, imgs, seg):
        if random.random() < 1:
            imgs = np.flip(imgs, axis=2).copy()
            seg = np.flip(seg, axis=2).copy()
        return imgs, seg


class RandomRotation:
    def __call__(self, imgs, seg):
        angle = random.choice([0, 90, 180, 270])
        if angle == 0:
            return imgs, seg
        k = angle // 90
        imgs = np.rot90(imgs, k=k, axes=(1, 2)).copy()
        seg = np.rot90(seg, k=k, axes=(0, 1)).copy()
        return imgs, seg


class RandomIntensityShift:
    def __call__(self, imgs, seg):
        if random.random() < 0.5:
            shift = np.random.uniform(-0.1, 0.1)
            imgs = imgs + shift
        return imgs, seg


class RandomGaussianNoise:
    def __call__(self, imgs, seg):
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.02, imgs.shape)
            imgs = imgs + noise
        return imgs, seg


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgs, seg):
        for t in self.transforms:
            imgs, seg = t(imgs, seg)
        return imgs, seg


def get_train_transforms():
    return Compose([
        # RandomFlip(),
        RandomRotation(),
        RandomIntensityShift(),
        RandomGaussianNoise(),
    ])


def get_val_transforms():
    return Compose([])

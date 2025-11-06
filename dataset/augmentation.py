import numpy as np
import random
import scipy.ndimage as ndi


class RandomFlip:
    def __call__(self, imgs, seg):
        if random.random() < 0.5:
            imgs = np.flip(imgs, axis=2).copy()
            seg = np.flip(seg, axis=1).copy()
        return imgs, seg


class RandomRotation:
    def __call__(self, imgs, seg):
        angle = random.choice([0, 90, 180, 270])
        if angle == 0:
            return imgs, seg
        imgs = np.rot90(imgs, k=angle // 90, axes=(1, 2)).copy()
        seg = np.rot90(seg, k=angle // 90, axes=(0, 1)).copy()
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


class ElasticDeformation:
    """Simple elastic deformation using scipy"""

    def __call__(self, imgs, seg):
        if random.random() < 0.3:
            alpha = 2
            sigma = 0.2
            dx = ndi.gaussian_filter((np.random.rand(*imgs.shape[1:]) * 2 - 1), sigma) * alpha
            dy = ndi.gaussian_filter((np.random.rand(*imgs.shape[1:]) * 2 - 1), sigma) * alpha
            dz = ndi.gaussian_filter((np.random.rand(*imgs.shape[1:]) * 2 - 1), sigma) * alpha

            for c in range(imgs.shape[0]):
                imgs[c] = ndi.map_coordinates(imgs[c],
                                              np.array(np.meshgrid(
                                                  np.arange(imgs.shape[1]),
                                                  np.arange(imgs.shape[2]),
                                                  np.arange(imgs.shape[3])
                                              )) + np.stack([dx, dy, dz]), order=1)
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
        RandomFlip(),
        RandomRotation(),
        RandomIntensityShift(),
        RandomGaussianNoise(),
        ElasticDeformation()
    ])


def get_val_transforms():
    return Compose([])

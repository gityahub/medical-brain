import os
import random
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import Dataset

class BrainTumorDataset(Dataset):
    def __init__(self, data_list, modalities, transform=None, missing_prob=0.0):
        self.data_list = data_list
        self.modalities = modalities
        self.transform = transform
        self.missing_prob = missing_prob

    def __len__(self):
        return len(self.data_list)

    def _load_nii(self, path):
        img = nib.load(path).get_fdata()
        return img.astype(np.float32)

    def _normalize(self, img):
        mean, std = img.mean(), img.std()
        if std < 1e-5:
            return img
        return (img - mean) / std

    def _simulate_missing_modalities(self, images):
        num_mod = len(images)
        mask = np.ones(num_mod, dtype=np.float32)

        if random.random() < self.missing_prob:
            drop_num = random.randint(0, num_mod - 1)
            if drop_num > 0:
                drop_idx = random.sample(range(num_mod), drop_num)
                for i in drop_idx:
                    images[i] = np.zeros_like(images[i])
                    mask[i] = 0.0
        return images, mask

    def __getitem__(self, idx):
        item = self.data_list[idx]
        patient_id = item["patient_id"]
        modal_paths = item["modal_paths"]

        # ---- load modalities ----
        images = []
        for m in self.modalities:
            img_np = self._load_nii(modal_paths[m])
            img_np = self._normalize(img_np)
            images.append(img_np)

        # modal missing simulation
        images, modality_mask = self._simulate_missing_modalities(images)

        images = np.stack(images, axis=0)

        # ---- segmentation optional ----
        seg_path = item.get("seg_path", None)
        if seg_path is not None:
            seg = self._load_nii(seg_path).astype(np.int64)
        else:
            seg = None

        # ---- classification optional ----
        if "cls_label" in item:
            cls_label = item["cls_label"]
        else:
            cls_label = None

        # ---- transforms ----
        if self.transform:
            if seg is not None:
                images, seg = self.transform(images, seg)
            else:
                images, _ = self.transform(images, np.zeros_like(images[0]))

        return {
            "images": torch.tensor(images, dtype=torch.float32),
            "seg": None if seg is None else torch.tensor(seg, dtype=torch.long),
            "label": None if cls_label is None else torch.tensor(cls_label, dtype=torch.long),
            "modality_mask": torch.tensor(modality_mask, dtype=torch.float32),
            "patient_id": patient_id
        }

import os
import glob
import random
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import Dataset


class BrainTumorDataset(Dataset):
    def __init__(self, cfg, split="train", transform=None):
        self.cfg = cfg
        self.transform = transform
        self.split = split

        dataset_name = cfg.data.default_dataset.upper()
        self.root_dir = cfg.data.datasets[dataset_name].root_dir
        self.modalities = cfg.data.datasets[dataset_name].modalities
        self.missing_prob = cfg.data.datasets[dataset_name].missing_modal_prob

        if dataset_name == "ISLES":
            self.data_list = self._build_isles_list()
        elif dataset_name == "BRATS":
            self.data_list = self._build_brats_list()
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        print(f"[Dataset] Loaded {len(self.data_list)} samples from {dataset_name}/{self.split}")

    # -------------------- ISLES2022 loader --------------------
    def _build_isles_list(self):
        cases = {}
        for m in self.modalities:
            files = glob.glob(os.path.join(self.root_dir, m, self.split, "*.nii*"))
            for f in files:
                name = os.path.basename(f).replace(".nii", "").replace(".gz", "")
                pid = name.split("_")[0]
                if pid not in cases:
                    cases[pid] = {
                        "patient_id": pid,
                        "modal_paths": {}
                    }
                cases[pid]["modal_paths"][m] = f

        mask_files = glob.glob(os.path.join(self.root_dir, "mask", self.split, "*.nii*"))
        for f in mask_files:
            name = os.path.basename(f).replace(".nii", "").replace(".gz", "")
            pid = name.split("_")[0]
            if pid in cases:
                cases[pid]["seg_path"] = f

        return list(cases.values())

    # -------------------- BRATS Loader --------------------
    def _build_brats_list(self):
        cases = {}
        for m in self.modalities:
            files = glob.glob(os.path.join(self.root_dir, m, self.split, "*.nii*"))
            for f in files:
                name = os.path.basename(f).replace(".nii", "").replace(".gz", "")
                pid = name.split("_")[0]
                if pid not in cases:
                    cases[pid] = {
                        "patient_id": pid,
                        "modal_paths": {}
                    }
                cases[pid]["modal_paths"][m] = f

        seg_files = glob.glob(os.path.join(self.root_dir, "label", self.split, "*.nii*"))
        for f in seg_files:
            name = os.path.basename(f).replace(".nii", "").replace(".gz", "")
            pid = name.split("_")[0]
            if pid in cases:
                cases[pid]["seg_path"] = f

        return list(cases.values())

    # -------------------- Basic functions --------------------
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
    def _pad_or_crop_to_112(self, img):
        target = (112, 112, 112)
        out = np.zeros(target, dtype=img.dtype)

        h, w, d = img.shape
        th, tw, td = target

        # 中心起始点
        hs = max((th - h) // 2, 0)
        ws = max((tw - w) // 2, 0)
        ds = max((td - d) // 2, 0)

        # 拷贝范围
        h1 = min(h, th)
        w1 = min(w, tw)
        d1 = min(d, td)

        out[hs:hs+h1, ws:ws+w1, ds:ds+d1] = img[
            h//2 - h1//2 : h//2 + h1//2,
            w//2 - w1//2 : w//2 + w1//2,
            d//2 - d1//2 : d//2 + d1//2
        ]
        return out

    # -------------------- Main Access --------------------
    def __getitem__(self, idx):
        item = self.data_list[idx]
        patient_id = item["patient_id"]

        # 判断是否 ISLES
        is_isles = (self.cfg.data.default_dataset.upper() == "ISLES")

        images = []
        for m in self.modalities:
            path = item["modal_paths"].get(m, None)
            if path is None:
                img_np = np.zeros((112, 112, 112), dtype=np.float32) if is_isles else np.zeros((256, 256, 256), dtype=np.float32)
            else:
                img_np = self._load_nii(path)
                img_np = self._normalize(img_np)
                if is_isles:
                    img_np = self._pad_or_crop_to_112(img_np)
            images.append(img_np)

        images, modality_mask = self._simulate_missing_modalities(images)
        images = np.stack(images, axis=0)  # C,H,W,D

        # seg 同样处理
        seg = None
        if "seg_path" in item and item["seg_path"] is not None:
            seg = self._load_nii(item["seg_path"]).astype(np.int64)
            if is_isles:
                seg = self._pad_or_crop_to_112(seg)

        if self.transform:
            if seg is not None:
                images, seg = self.transform(images, seg)
            else:
                images, _ = self.transform(images, np.zeros_like(images[0]))

        return {
            "images": torch.tensor(images, dtype=torch.float32),
            "seg": None if seg is None else torch.tensor(seg, dtype=torch.long),
            "modality_mask": torch.tensor(modality_mask, dtype=torch.float32),
            "patient_id": patient_id
        }

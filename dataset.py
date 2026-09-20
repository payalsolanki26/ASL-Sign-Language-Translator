"""
PyTorch dataset + transforms  (Data Engineering -> Deep Learning hand-off)

The dataset only ever sees rows of the split manifest, so what a model trains on is
fully determined by an auditable file. Images can come from the raw folder or be
decrypted in memory from the AES-256-GCM store (no plaintext written to disk).
"""
from __future__ import annotations
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def mask_magenta(img: Image.Image) -> Image.Image:
    """Paint the magenta bounding box white. The box is drawn by the capture tool, not part of the
    sign, and its shape would give the network a shortcut cue."""
    a = np.asarray(img.convert("RGB")).copy()
    m = (a[..., 0] > 200) & (a[..., 1] < 80) & (a[..., 2] > 200)
    a[m] = 255
    return Image.fromarray(a)


def raw_loader(raw_dir: str | Path):
    return lambda rel_path: Image.open(Path(raw_dir) / rel_path).convert("RGB")


def encrypted_loader(store, pseudonymize):
    """store: EncryptedImageStore, pseudonymize: callable(rel_path)->image_id"""
    return lambda rel_path: Image.open(io.BytesIO(store.get(pseudonymize(rel_path)))).convert("RGB")


def build_transforms(image_size: int, train: bool, aug: dict):
    from torchvision import transforms as T
    tf = [T.Resize((image_size, image_size))]
    if train:
        tf += [T.RandomAffine(degrees=aug["rotation_deg"], translate=(aug["translate"],) * 2,
                              scale=tuple(aug["scale"]), fill=255),
               T.ColorJitter(aug["brightness"], aug["contrast"], aug["saturation"])]
        if aug.get("horizontal_flip"):
            tf.append(T.RandomHorizontalFlip())
    tf += [T.ToTensor(), T.Normalize(MEAN, STD)]
    if train and aug.get("random_erasing_p", 0) > 0:
        tf.append(T.RandomErasing(p=aug["random_erasing_p"], scale=(0.02, 0.10), value=1.0))
    return T.Compose(tf)


class AslDataset:
    """Map-style dataset (duck-typed so importing this module does not require torch)."""

    def __init__(self, df: pd.DataFrame, loader, transform, mask: bool = True):
        self.rel = df["rel_path"].tolist()
        self.y = df["label_id"].astype(int).tolist()
        self.loader, self.transform, self.mask = loader, transform, mask

    def __len__(self):
        return len(self.rel)

    def __getitem__(self, i):
        img = self.loader(self.rel[i])
        if self.mask:
            img = mask_magenta(img)
        return self.transform(img), self.y[i]

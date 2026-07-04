"""Dataset + augmentation for talc segmentation.

Panoramas are large but the talc zones are big, low-frequency regions, so we
train on whole images downscaled to a fixed square. With only ~40 images,
augmentation (flips, 90° rotations, small affine jitter, brightness/contrast
jitter) does the heavy lifting against overfitting.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

Image.MAX_IMAGE_PIXELS = None

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def list_pairs(pairs_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for mask_path in sorted(masks_dir.glob("*.png")):
        original = pairs_dir / "original" / f"{mask_path.stem}.JPG"
        if original.exists():
            pairs.append((original, mask_path))
    return pairs


class TalcDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]], size: int = 384, train: bool = True):
        self.pairs = pairs
        self.size = size
        self.train = train

    def __len__(self) -> int:
        return len(self.pairs)

    def _augment(self, image: np.ndarray, mask: np.ndarray):
        if random.random() < 0.5:
            image, mask = image[:, ::-1], mask[:, ::-1]
        if random.random() < 0.5:
            image, mask = image[::-1], mask[::-1]
        rot = random.choice((0, 1, 2, 3))
        if rot:
            image, mask = np.rot90(image, rot), np.rot90(mask, rot)
        image, mask = np.ascontiguousarray(image), np.ascontiguousarray(mask)
        if random.random() < 0.7:
            # brightness / contrast jitter to survive uneven illumination.
            brightness = 1.0 + random.uniform(-0.2, 0.2)
            contrast = 1.0 + random.uniform(-0.2, 0.2)
            mean = image.mean()
            image = np.clip((image - mean) * contrast + mean * brightness, 0, 255)
        return image.astype(np.float32), mask.astype(np.float32)

    def __getitem__(self, index: int):
        original_path, mask_path = self.pairs[index]
        image = Image.open(original_path).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((self.size, self.size), Image.NEAREST)
        image = np.asarray(image, dtype=np.float32)
        mask = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)

        if self.train:
            image, mask = self._augment(image, mask)

        image = (image / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
        mask = torch.from_numpy(mask[None].copy()).float()
        return image, mask

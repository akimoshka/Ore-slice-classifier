"""Turn the geologists' blue-line annotations into binary talc masks.

Every talc sample ships as a pair of pixel-aligned images:

* ``original``  — the raw micrograph;
* ``annotated`` — the same micrograph with a hand-drawn **blue contour** around
  the оталькование (talc) region.

The contour is usually *open*: it runs off one image edge and back onto another,
so ``binary_fill_holes`` alone does not close it. We therefore:

1. threshold the saturated blue pixels to recover the drawn line;
2. morphologically close small gaps in the line;
3. add a thin frame around the image so ``line ∪ frame`` becomes a set of closed
   barriers that split the picture into an "outside" region (the one hugging the
   frame the most) and one or more enclosed "talc" regions;
4. keep every enclosed region as the talc mask.

The resulting mask is paired with the **original** (line-free) image, so a model
trained on it never sees the annotation colour and cannot cheat.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def extract_blue_line(annotated: np.ndarray) -> np.ndarray:
    """Return a uint8 {0,1} mask of the hand-drawn blue annotation line."""
    rgb = annotated[:, :, :3].astype(np.int16)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    line = (b > 80) & (b - r > 35) & (b - g > 35)
    return line.astype(np.uint8)


def _enclosed_regions(line: np.ndarray, close_frac: float, min_area_frac: float):
    """Close the line by ``close_frac``, then return (mask, n_enclosed).

    ``line ∪ frame`` forms closed barriers; the component that hugs the frame the
    most is the "outside", and every other large component is enclosed talc.
    """
    height, width = line.shape
    kernel_size = max(3, int(close_frac * max(height, width))) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(line, cv2.MORPH_CLOSE, kernel)

    barrier = closed.copy()
    thickness = 3
    barrier[:thickness, :] = 1
    barrier[-thickness:, :] = 1
    barrier[:, :thickness] = 1
    barrier[:, -thickness:] = 1

    background = (barrier == 0).astype(np.uint8)
    count, labels = cv2.connectedComponents(background)

    ring = np.zeros((height, width), bool)
    r0 = 8
    ring[:r0, :] = ring[-r0:, :] = ring[:, :r0] = ring[:, -r0:] = True

    components = []
    for label in range(1, count):
        comp = labels == label
        area = int(comp.sum())
        if area < min_area_frac * height * width:
            continue
        components.append((label, area, int((comp & ring).sum())))
    if len(components) < 2:
        return np.zeros((height, width), np.uint8), 0

    outside = max(components, key=lambda item: item[2])[0]
    mask = np.zeros((height, width), np.uint8)
    for label, _area, _edge in components:
        if label != outside:
            mask[labels == label] = 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask, len(components) - 1


def talc_mask_from_annotation(
    annotated: np.ndarray,
    min_area_frac: float = 0.003,
    close_fracs: tuple[float, ...] = (0.012, 0.02, 0.03, 0.045, 0.06),
) -> np.ndarray:
    """Build a {0,1} talc mask from an annotated (blue-contour) image.

    The contour is often fragmented into several arcs. We try increasingly large
    morphological closings (``close_fracs``) and keep the first that actually
    encloses a region, so partial annotations still yield a usable mask instead
    of an empty one.
    """
    height, width = annotated.shape[:2]
    line = extract_blue_line(annotated)
    if line.sum() == 0:
        return np.zeros((height, width), np.uint8)

    for close_frac in close_fracs:
        mask, n_enclosed = _enclosed_regions(line, close_frac, min_area_frac)
        if n_enclosed >= 1:
            return mask
    return np.zeros((height, width), np.uint8)


def load_pair(original_path: Path, annotated_path: Path) -> tuple[np.ndarray, np.ndarray]:
    original = np.asarray(Image.open(original_path).convert("RGB"))
    annotated = np.asarray(Image.open(annotated_path).convert("RGB"))
    if annotated.shape[:2] != original.shape[:2]:
        annotated = np.asarray(
            Image.fromarray(annotated).resize((original.shape[1], original.shape[0]))
        )
    return original, annotated


def build_all_masks(pairs_dir: Path, out_dir: Path) -> list[dict]:
    """Generate PNG talc masks for every pair and return per-image talc shares."""
    original_dir = pairs_dir / "original"
    annotated_dir = pairs_dir / "annotated"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for annotated_path in sorted(annotated_dir.glob("*.JPG")):
        original_path = original_dir / annotated_path.name
        if not original_path.exists():
            continue
        _, annotated = load_pair(original_path, annotated_path)
        mask = talc_mask_from_annotation(annotated)
        out_path = out_dir / f"{annotated_path.stem}.png"
        Image.fromarray(mask * 255).save(out_path)
        records.append({
            "name": annotated_path.name,
            "mask": out_path.name,
            "talc_frac": float(mask.mean()),
        })
    return records


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    pairs = root / "data" / "talc_pairs"
    masks = root / "data" / "talc_masks"
    rows = build_all_masks(pairs, masks)
    shares = [r["talc_frac"] for r in rows]
    print(f"Built {len(rows)} masks -> {masks}")
    if shares:
        print(f"talc share: min={min(shares):.3f} mean={sum(shares)/len(shares):.3f} max={max(shares):.3f}")
        empty = [r["name"] for r in rows if r["talc_frac"] < 1e-4]
        if empty:
            print(f"WARNING: {len(empty)} empty masks: {empty}")

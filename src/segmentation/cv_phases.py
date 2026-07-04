"""Classical-CV segmentation of the light sulfide phases (Variant 3).

Reflected-light micrographs make this largely a brightness problem: sulfides are
the bright phase, the silicate/oxide matrix is dark. We therefore:

1. normalise illumination with CLAHE on the L* channel;
2. threshold the bright sulfides with Otsu (adapts per image);
3. clean the mask morphologically;
4. label connected grains and split them into

   * ``ordinary`` intergrowths — large, compact, weakly replaced grains, and
   * ``fine`` intergrowths — grains heavily laced with the dark non-ore phase,

   using grain size + solidity (how completely a grain fills its convex hull).
   A ragged, porous grain fills its hull poorly and reads as a fine intergrowth.

The whole thing is deterministic and explainable, which matches the geological
review workflow better than a black-box would.
"""

from __future__ import annotations

import cv2
import numpy as np

# Label ids used across the segmentation stack.
BG, ORDINARY, FINE = 0, 1, 2


def sulfide_mask(rgb: np.ndarray) -> np.ndarray:
    """Return a {0,1} mask of the bright sulfide phase."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(lightness)
    _thr, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (binary > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def classify_intergrowths(mask: np.ndarray, thickness_frac: float = 0.012) -> np.ndarray:
    """Split the sulfide mask into ORDINARY / FINE intergrowth labels.

    We use *granulometry* rather than per-grain shape stats: a morphological
    opening with a disk of radius ``R`` keeps only sulfide that is locally thick,
    i.e. compact grain cores (ORDINARY intergrowths). Dilating those cores back
    and intersecting with the mask restores the full compact grains. Whatever
    sulfide remains is thin, dark-laced filigree — the FINE intergrowths.

    ``R`` scales with image size so the split is resolution-independent.
    """
    height, width = mask.shape
    radius = max(4, int(thickness_frac * max(height, width)))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

    cores = cv2.morphologyEx(mask, cv2.MORPH_OPEN, disk)
    ordinary = (cv2.dilate(cores, disk) & mask).astype(bool)
    fine = (mask > 0) & (~ordinary)

    result = np.zeros((height, width), np.uint8)
    result[ordinary] = ORDINARY
    result[fine] = FINE
    return result


def segment_phases(rgb: np.ndarray) -> dict:
    """Full CV phase segmentation. Returns label map + area shares."""
    mask = sulfide_mask(rgb)
    labels = classify_intergrowths(mask)
    total = labels.size
    ordinary = float((labels == ORDINARY).mean())
    fine = float((labels == FINE).mean())
    sulfide = ordinary + fine
    fine_of_sulfide = fine / sulfide if sulfide > 0 else 0.0
    return {
        "labels": labels,
        "sulfide_mask": (labels > 0).astype(np.uint8),
        "sulfide_share": sulfide,
        "ordinary_share": ordinary,
        "fine_share": fine,
        "fine_of_sulfide": fine_of_sulfide,
    }

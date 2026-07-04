"""Segmentation inference: the second model the backend serves next to the
classifier.

Given a micrograph it produces a single, pixel-level **segmentation image**:

* blue   — talc zone, from the trained U-Net (``models/unet_talc.pth``);
* green  — ordinary sulfide intergrowths (classical CV);
* red    — fine sulfide intergrowths (classical CV).

It also returns pixel-area shares and the expert ore class, so the app can show
true pixel percentages instead of the tile-area proxies the classifier reports.

The classifier is untouched — this module is a separate, self-contained model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.segmentation.cv_phases import BG, FINE, ORDINARY, segment_phases
from src.segmentation.unet import ResNetUNet

Image.MAX_IMAGE_PIXELS = None

# Shared phase ids for the combined segmentation map.
TALC = 3
PHASE_COLORS = {
    ORDINARY: (34, 197, 94),    # green
    FINE: (239, 68, 68),        # red
    TALC: (59, 130, 246),       # blue
}
PHASE_TITLES = {
    ORDINARY: "Обычные срастания",
    FINE: "Тонкие срастания",
    TALC: "Тальк",
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_TITLES = {
    "ordinary": "Рядовая руда",
    "difficult": "Труднообогатимая руда",
    "talc": "Оталькованная руда",
}


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_talc_model(weights_path: Path, device: torch.device | None = None) -> ResNetUNet:
    device = device or pick_device()
    model = ResNetUNet(num_classes=1, pretrained=False)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.inference_mode()
def predict_talc(model: ResNetUNet, rgb: np.ndarray, size: int = 384, threshold: float = 0.5) -> np.ndarray:
    """Run the talc U-Net and return a {0,1} mask at the input resolution."""
    device = next(model.parameters()).device
    height, width = rgb.shape[:2]
    small = np.asarray(Image.fromarray(rgb).resize((size, size), Image.BILINEAR), dtype=np.float32)
    tensor = (small / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(tensor.transpose(2, 0, 1).copy()).float()[None].to(device)
    prob = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    prob_full = np.asarray(Image.fromarray((prob * 255).astype(np.uint8)).resize((width, height), Image.BILINEAR))
    return (prob_full > threshold * 255).astype(np.uint8)


def _downscale_for_cv(rgb: np.ndarray, max_side: int = 2000) -> tuple[np.ndarray, float]:
    height, width = rgb.shape[:2]
    scale = min(max_side / max(height, width), 1.0)
    if scale < 1.0:
        resized = np.asarray(Image.fromarray(rgb).resize((int(width * scale), int(height * scale)), Image.BILINEAR))
        return resized, scale
    return rgb, 1.0


def segment(
    rgb: np.ndarray,
    talc_model: ResNetUNet | None = None,
    talc_size: int = 384,
    talc_threshold: float = 0.5,
    ore_talc_threshold: float = 0.10,
    cv_max_side: int = 2000,
) -> dict:
    """Full segmentation. Returns the combined label map, shares and ore class."""
    work, _scale = _downscale_for_cv(rgb, cv_max_side)
    phases = segment_phases(work)
    labels = phases["labels"].copy()  # 0 bg, 1 ordinary, 2 fine

    if talc_model is not None:
        talc = predict_talc(talc_model, work, size=talc_size, threshold=talc_threshold)
        # Talc is a matrix zone; only paint it where there is no sulfide grain.
        labels[(talc > 0) & (labels == BG)] = TALC
        talc_share = float((labels == TALC).mean())
    else:
        talc_share = 0.0

    total = labels.size
    ordinary_share = float((labels == ORDINARY).mean())
    fine_share = float((labels == FINE).mean())
    sulfide_share = ordinary_share + fine_share
    fine_of_sulfide = fine_share / sulfide_share if sulfide_share > 0 else 0.0

    if talc_share > ore_talc_threshold:
        final_label = "talc"
    elif ordinary_share >= fine_share:
        final_label = "ordinary"
    else:
        final_label = "difficult"

    return {
        "labels": labels,
        "work_image": work,
        "talc_share": talc_share,
        "sulfide_share": sulfide_share,
        "ordinary_share": ordinary_share,
        "fine_share": fine_share,
        "fine_of_sulfide": fine_of_sulfide,
        "final_label": final_label,
    }


def colored_mask(labels: np.ndarray) -> np.ndarray:
    """Flat RGB image of the label map (black background)."""
    out = np.zeros((*labels.shape, 3), np.uint8)
    for label, color in PHASE_COLORS.items():
        out[labels == label] = color
    return out


def overlay(base_rgb: np.ndarray, labels: np.ndarray, alpha: float = 0.55) -> Image.Image:
    """Blend the coloured segmentation onto the (downscaled) source image."""
    out = base_rgb.astype(np.float32)
    for label, color in PHASE_COLORS.items():
        m = labels == label
        out[m] = (1 - alpha) * out[m] + alpha * np.array(color, dtype=np.float32)
    return Image.fromarray(out.astype(np.uint8))


def result_text(result: dict) -> str:
    label = CLASS_TITLES[result["final_label"]]
    return (
        f"Руда классифицирована как {label.lower()}: тальк — {result['talc_share']:.1%}, "
        f"сульфиды — {result['sulfide_share']:.1%}, из них тонких срастаний — "
        f"{result['fine_of_sulfide']:.1%} (пиксельная сегментация)."
    )


def metrics_rows(result: dict) -> list[tuple[str, str, str]]:
    """Rows for a metrics table: (metric, value, method)."""
    return [
        ("Класс руды", CLASS_TITLES[result["final_label"]], "экспертная логика"),
        ("Тальк", f'{result["talc_share"]:.1%}', "U-Net (пиксели)"),
        ("Сульфиды всего", f'{result["sulfide_share"]:.1%}', "CV (пиксели)"),
        ("Обычные срастания", f'{result["ordinary_share"]:.1%}', "CV (пиксели)"),
        ("Тонкие срастания", f'{result["fine_share"]:.1%}', "CV (пиксели)"),
        ("Доля тонких среди сульфидов", f'{result["fine_of_sulfide"]:.1%}', "CV (пиксели)"),
    ]

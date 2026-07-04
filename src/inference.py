"""Inference utilities shared by the Streamlit demo.

The trained networks are image classifiers. Panorama-level quantities below are
therefore tile-area proxies, not pixel segmentation measurements.
"""

from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Iterable

import cv2
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ore_vision_matplotlib"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import models, transforms


CLASS_NAMES = ("ordinary", "difficult", "talc")
CLASS_TITLES = {
    "ordinary": "Рядовая руда",
    "difficult": "Труднообогатимая руда",
    "talc": "Оталькованная руда",
}
CLASS_COLORS = {
    "ordinary": (34, 197, 94),
    "difficult": (239, 68, 68),
    "talc": (59, 130, 246),
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Dropout(0.25), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_model(name: str, weights_path: Path) -> nn.Module:
    if name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    elif name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASS_NAMES))
    elif name == "tinycnn":
        model = TinyCNN(len(CLASS_NAMES))
    else:
        raise ValueError(f"Неизвестная модель: {name}")
    state = torch.load(weights_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


VALID_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    result = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if result[-1] != last:
        result.append(last)
    return result


def iter_tiles(image: Image.Image, tile_size: int, overlap: int, max_tiles: int) -> Iterable[tuple]:
    width, height = image.size
    stride = max(tile_size - overlap, 1)
    xs, ys = _positions(width, tile_size, stride), _positions(height, tile_size, stride)
    coordinates = [(x, y) for y in ys for x in xs]
    if len(coordinates) > max_tiles:
        selected = np.linspace(0, len(coordinates) - 1, max_tiles, dtype=int)
        coordinates = [coordinates[index] for index in selected]
    for x, y in coordinates:
        x2, y2 = min(x + tile_size, width), min(y + tile_size, height)
        tile = image.crop((x, y, x2, y2)).convert("RGB")
        yield x, y, x2, y2, tile


def _sulfide_proxy(tile: Image.Image) -> float:
    array = np.asarray(tile.resize((256, 256)), dtype=np.uint8)
    hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    # Bright, weakly saturated pixels are a conservative proxy for light sulfides.
    return float(((gray >= 185) & (hsv[:, :, 1] <= 105)).mean())


@torch.inference_mode()
def analyze_image(
    model: nn.Module,
    image: Image.Image,
    tile_size: int = 512,
    overlap: int = 64,
    max_tiles: int = 300,
    talc_threshold: float = 0.10,
    batch_size: int = 16,
    progress: Callable[[float], None] | None = None,
) -> dict:
    image = image.convert("RGB")
    tiles = list(iter_tiles(image, tile_size, overlap, max_tiles))
    rows: list[dict] = []
    for start in range(0, len(tiles), batch_size):
        batch = tiles[start:start + batch_size]
        tensor = torch.stack([VALID_TRANSFORM(item[4]) for item in batch]).to(DEVICE)
        probabilities = torch.softmax(model(tensor), dim=1).cpu().numpy()
        for (x1, y1, x2, y2, tile), probs in zip(batch, probabilities):
            predicted = int(probs.argmax())
            rows.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "area": (x2 - x1) * (y2 - y1),
                "pred_label": CLASS_NAMES[predicted],
                "confidence": float(probs[predicted]),
                "prob_ordinary": float(probs[0]),
                "prob_difficult": float(probs[1]),
                "prob_talc": float(probs[2]),
                "sulfide_proxy": _sulfide_proxy(tile),
            })
        if progress:
            progress(min((start + len(batch)) / len(tiles), 1.0))

    frame = pd.DataFrame(rows)
    total_area = frame.area.sum()
    shares = {
        name: float(frame.loc[frame.pred_label == name, "area"].sum() / total_area)
        for name in CLASS_NAMES
    }
    non_talc = shares["ordinary"] + shares["difficult"]
    common_share = shares["ordinary"] / non_talc if non_talc else 0.0
    fine_share = shares["difficult"] / non_talc if non_talc else 0.0
    sulfide = float(np.average(frame.sulfide_proxy, weights=frame.area))
    if shares["talc"] > talc_threshold:
        final_label = "talc"
    elif shares["ordinary"] >= shares["difficult"]:
        final_label = "ordinary"
    else:
        final_label = "difficult"

    return {
        "final_label": final_label,
        "shares": shares,
        "common_share": common_share,
        "fine_share": fine_share,
        "sulfide_share": sulfide,
        "mean_confidence": float(np.average(frame.confidence, weights=frame.area)),
        "tiles": frame,
        "tile_count": len(frame),
        "sampled": len(tiles) == max_tiles,
    }


def make_overlay(image: Image.Image, tiles: pd.DataFrame, alpha: int = 92, max_side: int = 1800) -> Image.Image:
    source = image.convert("RGB")
    scale = min(max_side / max(source.size), 1.0)
    preview = source.resize((max(1, int(source.width * scale)), max(1, int(source.height * scale))))
    layer = Image.new("RGBA", preview.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    for row in tiles.itertuples():
        color = CLASS_COLORS[row.pred_label]
        box = tuple(int(value * scale) for value in (row.x1, row.y1, row.x2, row.y2))
        draw.rectangle(box, fill=(*color, alpha), outline=(*color, 190), width=2)
    return Image.alpha_composite(preview.convert("RGBA"), layer).convert("RGB")


def make_confidence_map(image: Image.Image, tiles: pd.DataFrame, max_side: int = 1800) -> Image.Image:
    source = image.convert("RGB")
    scale = min(max_side / max(source.size), 1.0)
    preview = source.resize((max(1, int(source.width * scale)), max(1, int(source.height * scale))))
    layer = Image.new("RGBA", preview.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    for row in tiles.itertuples():
        uncertainty = 1.0 - row.confidence
        color = (250, int(190 * (1 - uncertainty)), 30, int(60 + 150 * uncertainty))
        box = tuple(int(value * scale) for value in (row.x1, row.y1, row.x2, row.y2))
        draw.rectangle(box, fill=color)
    return Image.alpha_composite(preview.convert("RGBA"), layer).convert("RGB")


def metrics_frame(result: dict) -> pd.DataFrame:
    return pd.DataFrame([
        ("Класс руды", CLASS_TITLES[result["final_label"]], "экспертная логика"),
        ("Тальк", f'{result["shares"]["talc"]:.1%}', "доля тайлов"),
        ("Обычные срастания", f'{result["common_share"]:.1%}', "среди нетальковых тайлов"),
        ("Тонкие срастания", f'{result["fine_share"]:.1%}', "среди нетальковых тайлов"),
        ("Светлые сульфидные области", f'{result["sulfide_share"]:.1%}', "CV-оценка площади"),
        ("Средняя уверенность", f'{result["mean_confidence"]:.1%}', "по тайлам"),
    ], columns=["Метрика", "Значение", "Метод"])


def tiles_geojson(tiles: pd.DataFrame, image_size: tuple[int, int]) -> bytes:
    """Export tile rectangles in image pixel coordinates (no invented CRS)."""
    features = []
    for row in tiles.itertuples():
        ring = [[row.x1, row.y1], [row.x2, row.y1], [row.x2, row.y2], [row.x1, row.y2], [row.x1, row.y1]]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {
                "class": row.pred_label, "confidence": row.confidence,
                "prob_ordinary": row.prob_ordinary, "prob_difficult": row.prob_difficult,
                "prob_talc": row.prob_talc, "coordinate_space": "image_pixels",
            },
        })
    collection = {
        "type": "FeatureCollection", "name": "ore_analysis_tiles",
        "properties": {"image_width": image_size[0], "image_height": image_size[1], "crs": None},
        "features": features,
    }
    return json.dumps(collection, ensure_ascii=False, indent=2).encode("utf-8")


def make_pdf_report(filename: str, result: dict, overlay: Image.Image) -> bytes:
    buffer = BytesIO()
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="#f7f8fa")
    grid = fig.add_gridspec(3, 1, height_ratios=[0.35, 2.2, 1.15])
    title = fig.add_subplot(grid[0]); title.axis("off")
    title.text(0, .8, "ORE VISION — отчет анализа", fontsize=19, weight="bold")
    title.text(0, .25, f"Файл: {filename}", fontsize=10, color="#475569")
    picture = fig.add_subplot(grid[1]); picture.imshow(overlay); picture.axis("off")
    table_ax = fig.add_subplot(grid[2]); table_ax.axis("off")
    table = metrics_frame(result)
    table_ax.table(cellText=table.values, colLabels=table.columns, cellLoc="left", loc="upper center")
    table_ax.text(0, -0.03, "Примечание: классы и доли рассчитаны по тайлам; это не пиксельная сегментация.", fontsize=8)
    fig.tight_layout(pad=2)
    fig.savefig(buffer, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()

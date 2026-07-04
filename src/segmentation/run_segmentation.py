"""Batch segmentation CLI.

Runs the talc U-Net + CV phase segmentation over a single image or a folder and
writes, for every input:

* ``<name>_segmentation.png`` — the colour segmentation over the image;
* ``<name>_mask.png``          — the flat phase mask;

plus a single ``segmentation_metrics.csv`` with pixel shares and the ore class.
This covers the "batch processing + logging" requirement and is handy for the
demo video.

Usage:
    python -m src.segmentation.run_segmentation INPUT [--out OUT] [--weights W]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from src.segmentation import infer as seg

Image.MAX_IMAGE_PIXELS = None
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
ROOT = Path(__file__).resolve().parents[2]


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="image file or directory")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "segmentation")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "unet_talc.pth")
    parser.add_argument("--talc-threshold", type=float, default=0.10)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    model = seg.load_talc_model(args.weights) if args.weights.exists() else None
    if model is None:
        print(f"WARNING: talc weights not found at {args.weights}; running CV phases only.")

    images = collect_images(args.input)
    print(f"Segmenting {len(images)} image(s) -> {args.out}")

    rows = []
    for index, image_path in enumerate(images, start=1):
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        result = seg.segment(rgb, talc_model=model, ore_talc_threshold=args.talc_threshold)
        stem = image_path.stem
        seg.overlay(result["work_image"], result["labels"]).save(args.out / f"{stem}_segmentation.png")
        Image.fromarray(seg.colored_mask(result["labels"])).save(args.out / f"{stem}_mask.png")
        rows.append({
            "file": image_path.name,
            "ore_class": seg.CLASS_TITLES[result["final_label"]],
            "talc_pct": round(result["talc_share"] * 100, 2),
            "sulfide_pct": round(result["sulfide_share"] * 100, 2),
            "ordinary_pct": round(result["ordinary_share"] * 100, 2),
            "fine_pct": round(result["fine_share"] * 100, 2),
            "fine_of_sulfide_pct": round(result["fine_of_sulfide"] * 100, 2),
        })
        print(f"[{index}/{len(images)}] {image_path.name} -> {rows[-1]['ore_class']} "
              f"(тальк {rows[-1]['talc_pct']}%)")

    csv_path = args.out / "segmentation_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Metrics: {csv_path}")


if __name__ == "__main__":
    main()

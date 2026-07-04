"""Download the talc annotation pairs from the public Yandex.Disk dataset.

The dataset stores, for every talc (оталькованная) sample, two copies with an
identical filename:

* the original micrograph in ``Оталькованные руды/``;
* the same micrograph with a hand-drawn **blue contour** around the talc region
  in ``Оталькованные руды/Области оталькования/``.

That pair is exactly the "annotation pairs" needed to build talc masks, so this
script mirrors both folders into ``data/talc_pairs/{original,annotated}/``.

Usage:
    python -m src.segmentation.download_talc_pairs
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

PUBLIC_KEY = "https://disk.yandex.ru/d/Fo5eIM984glHaA"
API = "https://cloud-api.yandex.net/v1/disk/public/resources"
ORIGINAL_DIR = "/Фото руд по сортам. ч1/Оталькованные руды"
ANNOTATED_DIR = "/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования"

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "talc_pairs"


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def list_files(path: str) -> list[str]:
    url = f"{API}?public_key={urllib.parse.quote(PUBLIC_KEY)}&path={urllib.parse.quote(path)}&limit=500"
    data = _get_json(url)
    return [item["name"] for item in data["_embedded"]["items"] if item["type"] == "file"]


def download_file(path: str, destination: Path) -> None:
    url = f"{API}/download?public_key={urllib.parse.quote(PUBLIC_KEY)}&path={urllib.parse.quote(path)}"
    href = _get_json(url)["href"]
    with urllib.request.urlopen(href, timeout=300) as response:
        destination.write_bytes(response.read())


def main() -> None:
    original_out = OUT / "original"
    annotated_out = OUT / "annotated"
    original_out.mkdir(parents=True, exist_ok=True)
    annotated_out.mkdir(parents=True, exist_ok=True)

    annotated_names = set(list_files(ANNOTATED_DIR))
    print(f"Found {len(annotated_names)} annotated images")

    for index, name in enumerate(sorted(annotated_names), start=1):
        orig_dst = original_out / name
        anno_dst = annotated_out / name
        try:
            if not orig_dst.exists():
                download_file(f"{ORIGINAL_DIR}/{name}", orig_dst)
            if not anno_dst.exists():
                download_file(f"{ANNOTATED_DIR}/{name}", anno_dst)
            print(f"[{index}/{len(annotated_names)}] {name}")
        except Exception as exc:  # pragma: no cover - network hiccups
            print(f"[{index}/{len(annotated_names)}] FAILED {name}: {exc}")
            time.sleep(2)

    print(f"Done. Pairs in {OUT}")


if __name__ == "__main__":
    main()

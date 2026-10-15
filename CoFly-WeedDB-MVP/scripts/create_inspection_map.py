#!/usr/bin/env python
"""Create a large high-detail inspection canvas from CoFly-WeedDB frames."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "CoFly-WeedDB"
IMAGES_DIR = DATASET_ROOT / "images"
OUTPUT_DIR = DATASET_ROOT / "generated_maps"
GEO_PATTERN = re.compile(
    r"\[Lat=([+-]?\d+\.\d+),Lon=([+-]?\d+\.\d+),Alt=([+-]?\d+\.\d+)\]"
)


def load_image(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def parse_geo(name: str) -> tuple[float, float]:
    match = GEO_PATTERN.search(name)
    if not match:
        raise ValueError(f"Missing geo tags in filename: {name}")
    return float(match.group(1)), float(match.group(2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--columns", type=int, default=12)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--overlap", type=float, default=0.10)
    args = parser.parse_args()

    image_paths = sorted(IMAGES_DIR.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {IMAGES_DIR}")

    geo_items = []
    for image_path in image_paths:
        lat, lon = parse_geo(image_path.name)
        geo_items.append((image_path, lat, lon))

    geo_items.sort(key=lambda item: (-item[1], item[2], item[0].name))

    sample = load_image(geo_items[0][0])
    src_h, src_w = sample.shape[:2]
    frame_w = max(1, int(round(src_w * args.scale)))
    frame_h = max(1, int(round(src_h * args.scale)))
    step_x = max(1, int(round(frame_w * (1.0 - args.overlap))))
    step_y = max(1, int(round(frame_h * (1.0 - args.overlap))))

    rows = math.ceil(len(geo_items) / args.columns)
    canvas_w = step_x * (args.columns - 1) + frame_w
    canvas_h = step_y * (rows - 1) + frame_h
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    manifest_rows: list[dict[str, object]] = []
    for index, (image_path, lat, lon) in enumerate(geo_items):
        row = index // args.columns
        col = index % args.columns
        x = col * step_x
        y = row * step_y

        image = load_image(image_path)
        if args.scale != 1.0:
            image = cv2.resize(image, (frame_w, frame_h), interpolation=cv2.INTER_AREA)

        canvas[y:y + frame_h, x:x + frame_w] = image
        manifest_rows.append(
            {
                "index": index,
                "filename": image_path.name,
                "latitude": lat,
                "longitude": lon,
                "row": row,
                "col": col,
                "x": x,
                "y": y,
                "width": frame_w,
                "height": frame_h,
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_out = OUTPUT_DIR / "cofly_inspection_map_fullres.png"
    csv_out = OUTPUT_DIR / "cofly_inspection_map_fullres_manifest.csv"
    save_image(image_out, canvas)

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "filename",
                "latitude",
                "longitude",
                "row",
                "col",
                "x",
                "y",
                "width",
                "height",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Saved inspection map: {image_out}")
    print(f"Saved manifest: {csv_out}")
    print(f"Shape: {canvas.shape}")


if __name__ == "__main__":
    main()

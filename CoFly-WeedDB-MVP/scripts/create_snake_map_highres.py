#!/usr/bin/env python
"""Create a high-resolution snake-layout mosaic matching the first map logic."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "CoFly-WeedDB"
IMAGES_DIR = DATASET_ROOT / "images"
OUTPUT_DIR = DATASET_ROOT / "generated_maps"


def load_image(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def save_image(path: Path, image: np.ndarray, jpeg_quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = (path.suffix or ".png").lower()
    params: list[int] = []
    if ext in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    ok, encoded = cv2.imencode(ext, image, params)
    if not ok:
        raise ValueError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--columns", type=int, default=20)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--stride-ratio", type=float, default=0.75)
    parser.add_argument("--format", choices=["png", "jpg"], default="jpg")
    args = parser.parse_args()

    image_paths = sorted(IMAGES_DIR.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {IMAGES_DIR}")

    sample = load_image(image_paths[0])
    src_h, src_w = sample.shape[:2]
    frame_w = max(1, int(round(src_w * args.scale)))
    frame_h = max(1, int(round(src_h * args.scale)))
    step_x = max(1, int(round(frame_w * args.stride_ratio)))
    step_y = max(1, int(round(frame_h * args.stride_ratio)))

    rows = math.ceil(len(image_paths) / args.columns)
    canvas_w = frame_w + step_x * (args.columns - 1)
    canvas_h = frame_h + step_y * (rows - 1)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    manifest_rows: list[dict[str, object]] = []
    for index, image_path in enumerate(image_paths):
        row = index // args.columns
        col_in_row = index % args.columns
        col = col_in_row if row % 2 == 0 else args.columns - 1 - col_in_row
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
                "row": row,
                "column": col,
                "x": x,
                "y": y,
                "width": frame_w,
                "height": frame_h,
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_out = OUTPUT_DIR / f"cofly_large_test_map_highres.{args.format}"
    csv_out = OUTPUT_DIR / "cofly_large_test_map_highres_manifest.csv"
    save_image(image_out, canvas)

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "filename", "row", "column", "x", "y", "width", "height"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Saved snake mosaic: {image_out}")
    print(f"Saved manifest: {csv_out}")
    print(f"Shape: {canvas.shape}")


if __name__ == "__main__":
    main()

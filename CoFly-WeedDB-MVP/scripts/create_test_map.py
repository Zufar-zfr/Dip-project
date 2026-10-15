#!/usr/bin/env python
"""Create a large test mosaic from CoFly-WeedDB UAV frames."""

from __future__ import annotations

import math
from pathlib import Path
import csv
import re
import statistics

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


def geo_to_local_meters(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    x = (lon - lon0) * meters_per_deg_lon
    y = (lat0 - lat) * meters_per_deg_lat
    return x, y


def build_mosaic(
    scale: float = 0.75,
    assumed_stride_ratio: float = 0.25,
) -> tuple[Path, Path]:
    image_paths = sorted(IMAGES_DIR.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {IMAGES_DIR}")

    sample = load_image(image_paths[0])
    src_h, src_w = sample.shape[:2]
    frame_w = max(1, int(round(src_w * scale)))
    frame_h = max(1, int(round(src_h * scale)))

    geo_items = []
    for image_path in image_paths:
        lat, lon = parse_geo(image_path.name)
        geo_items.append((image_path, lat, lon))

    lat0 = sum(item[1] for item in geo_items) / len(geo_items)
    lon0 = sum(item[2] for item in geo_items) / len(geo_items)

    local_points = []
    for image_path, lat, lon in geo_items:
        x_m, y_m = geo_to_local_meters(lat, lon, lat0, lon0)
        local_points.append((image_path, lat, lon, x_m, y_m))

    consecutive_steps = []
    for (_, _, _, x1, y1), (_, _, _, x2, y2) in zip(local_points, local_points[1:]):
        step = math.hypot(x2 - x1, y2 - y1)
        if step > 0.01:
            consecutive_steps.append(step)
    median_step_m = statistics.median(consecutive_steps)

    stride_px = max(1, int(round(frame_w * assumed_stride_ratio)))
    pixels_per_meter = stride_px / median_step_m

    projected = []
    for image_path, lat, lon, x_m, y_m in local_points:
        x_px = x_m * pixels_per_meter
        y_px = y_m * pixels_per_meter
        projected.append((image_path, lat, lon, x_px, y_px))

    min_x = min(item[3] for item in projected)
    min_y = min(item[4] for item in projected)
    max_x = max(item[3] for item in projected)
    max_y = max(item[4] for item in projected)

    padding = 64
    canvas_width = int(math.ceil(max_x - min_x + frame_w + 2 * padding))
    canvas_height = int(math.ceil(max_y - min_y + frame_h + 2 * padding))

    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    occupancy = np.zeros((canvas_height, canvas_width, 1), dtype=np.float32)
    manifest_rows: list[dict[str, object]] = []

    for index, (image_path, lat, lon, x_px, y_px) in enumerate(projected):
        x = int(round(x_px - min_x + padding))
        y = int(round(y_px - min_y + padding))
        image = load_image(image_path)
        frame = cv2.resize(image, (frame_w, frame_h), interpolation=cv2.INTER_AREA)
        frame_f = frame.astype(np.float32)

        y2 = y + frame_h
        x2 = x + frame_w

        canvas_slice = canvas[y:y2, x:x2].astype(np.float32)
        occ_slice = occupancy[y:y2, x:x2]
        blended = np.where(occ_slice > 0, (canvas_slice + frame_f) / 2.0, frame_f)

        canvas[y:y2, x:x2] = blended.astype(np.uint8)
        occupancy[y:y2, x:x2] = 1.0

        manifest_rows.append(
            {
                "index": index,
                "filename": image_path.name,
                "latitude": lat,
                "longitude": lon,
                "x": x,
                "y": y,
                "width": frame_w,
                "height": frame_h,
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_out = OUTPUT_DIR / "cofly_large_test_map_geo_highres.png"
    csv_out = OUTPUT_DIR / "cofly_large_test_map_geo_highres_manifest.csv"

    save_image(image_out, canvas)

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "filename", "latitude", "longitude", "x", "y", "width", "height"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    return image_out, csv_out


def main() -> None:
    image_out, csv_out = build_mosaic()
    print(f"Saved mosaic: {image_out}")
    print(f"Saved manifest: {csv_out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Create a stitched panorama strip from sequential CoFly-WeedDB frames."""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "CoFly-WeedDB"
IMAGES_DIR = DATASET_ROOT / "images"
OUTPUT_DIR = DATASET_ROOT / "generated_maps"


def load_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"Failed to encode {path}")
    encoded.tofile(str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=242)
    parser.add_argument("--count", type=int, default=18)
    parser.add_argument("--scale", type=float, default=0.85)
    args = parser.parse_args()

    image_paths = sorted(IMAGES_DIR.glob("*.png"))
    selected = image_paths[args.start:args.start + args.count]
    if not selected:
        raise ValueError("No images selected")

    images = []
    manifest = []
    for image_path in selected:
        image = load_image(image_path)
        if args.scale != 1.0:
            image = cv2.resize(
                image,
                (int(image.shape[1] * args.scale), int(image.shape[0] * args.scale)),
                interpolation=cv2.INTER_AREA,
            )
        images.append(image)
        manifest.append(image_path.name)

    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    stitcher.setPanoConfidenceThresh(0.5)
    status, pano = stitcher.stitch(images)
    if status != cv2.Stitcher_OK:
        raise RuntimeError(f"Stitching failed with status {status}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_out = OUTPUT_DIR / "cofly_stitched_strip.png"
    manifest_out = OUTPUT_DIR / "cofly_stitched_strip_manifest.json"

    save_image(image_out, pano)
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved stitched strip: {image_out}")
    print(f"Saved manifest: {manifest_out}")
    print(f"Images used: {len(selected)}")
    print(f"Shape: {pano.shape}")


if __name__ == "__main__":
    main()

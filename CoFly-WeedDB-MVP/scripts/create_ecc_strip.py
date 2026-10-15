#!/usr/bin/env python
"""Create a sequential UAV mosaic using ECC alignment between adjacent frames."""

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


def build_feather_mask(height: int, width: int, edge: int = 48) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    inner_y0 = min(edge, height // 2)
    inner_x0 = min(edge, width // 2)
    inner_y1 = max(height - inner_y0, inner_y0 + 1)
    inner_x1 = max(width - inner_x0, inner_x0 + 1)
    mask[inner_y0:inner_y1, inner_x0:inner_x1] = 255
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist /= max(float(dist.max()), 1.0)
    return dist.astype(np.float32)


def to_homogeneous(matrix_2x3: np.ndarray) -> np.ndarray:
    out = np.eye(3, dtype=np.float32)
    out[:2, :] = matrix_2x3
    return out


def estimate_pair_transform(prev_img: np.ndarray, next_img: np.ndarray, scale: float) -> np.ndarray:
    prev_small = cv2.resize(prev_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    next_small = cv2.resize(next_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_small, cv2.COLOR_BGR2GRAY)

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)

    try:
        _, warp = cv2.findTransformECC(
            templateImage=prev_gray,
            inputImage=next_gray,
            warpMatrix=warp,
            motionType=cv2.MOTION_EUCLIDEAN,
            criteria=criteria,
            inputMask=None,
            gaussFiltSize=5,
        )
        warp[:, 2] /= scale
        return warp
    except cv2.error:
        orb = cv2.ORB_create(3000)
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(next_gray, None)
        if des1 is None or des2 is None:
            raise

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)[:300]
        if len(matches) < 8:
            raise

        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        affine, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
        if affine is None:
            raise
        affine = affine.astype(np.float32)
        affine[:, 2] /= scale
        return affine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=242)
    parser.add_argument("--count", type=int, default=18)
    parser.add_argument("--render-scale", type=float, default=1.0)
    parser.add_argument("--ecc-scale", type=float, default=0.5)
    args = parser.parse_args()

    image_paths = sorted(IMAGES_DIR.glob("*.png"))
    selected = image_paths[args.start:args.start + args.count]
    if not selected:
        raise ValueError("No images selected")

    images = []
    for path in selected:
        image = load_image(path)
        if args.render_scale != 1.0:
            image = cv2.resize(
                image,
                (int(image.shape[1] * args.render_scale), int(image.shape[0] * args.render_scale)),
                interpolation=cv2.INTER_AREA,
            )
        images.append(image)

    transforms = [np.eye(3, dtype=np.float32)]
    for prev_img, next_img in zip(images, images[1:]):
        warp = estimate_pair_transform(prev_img, next_img, args.ecc_scale)
        transforms.append(transforms[-1] @ to_homogeneous(warp))

    h, w = images[0].shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
    warped_corners = [cv2.perspectiveTransform(corners, mat) for mat in transforms]

    min_x = min(float(c[:, 0, 0].min()) for c in warped_corners)
    min_y = min(float(c[:, 0, 1].min()) for c in warped_corners)
    max_x = max(float(c[:, 0, 0].max()) for c in warped_corners)
    max_y = max(float(c[:, 0, 1].max()) for c in warped_corners)

    pad = 64
    tx = -min_x + pad
    ty = -min_y + pad
    out_w = int(np.ceil(max_x - min_x + 2 * pad))
    out_h = int(np.ceil(max_y - min_y + 2 * pad))

    acc = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weight = np.zeros((out_h, out_w, 1), dtype=np.float32)
    offset = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
    feather = build_feather_mask(h, w, edge=max(24, min(h, w) // 12))

    for image, mat in zip(images, transforms):
        full = offset @ mat
        warped = cv2.warpPerspective(image, full, (out_w, out_h))
        mask = cv2.warpPerspective(
            feather,
            full,
            (out_w, out_h),
        )
        mask = np.clip(mask, 0.0, 1.0)[..., None]
        acc += warped.astype(np.float32) * mask
        weight += mask

    pano = np.divide(acc, np.maximum(weight, 1.0)).astype(np.uint8)
    occupied = weight[..., 0] > 0.02
    ys, xs = np.where(occupied)
    if len(xs) and len(ys):
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        pano = pano[y0:y1, x0:x1]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_out = OUTPUT_DIR / "cofly_ecc_strip.png"
    manifest_out = OUTPUT_DIR / "cofly_ecc_strip_manifest.json"
    save_image(image_out, pano)
    manifest_out.write_text(
        json.dumps([path.name for path in selected], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved ECC strip: {image_out}")
    print(f"Saved manifest: {manifest_out}")
    print(f"Images used: {len(selected)}")
    print(f"Shape: {pano.shape}")


if __name__ == "__main__":
    main()

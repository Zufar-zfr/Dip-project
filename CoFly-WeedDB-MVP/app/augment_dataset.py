"""
Semi-synthetic augmentation for weed detection.

Builds a bank of real weed crops from the RGB masks and pastes them onto real
training images. This keeps backgrounds realistic while increasing object
count, variation in placement, and effective dataset size.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import DATASET_ROOT, PROJECT_ROOT, get_dataset_path
from utils import (
    bbox_xywh_to_xyxy,
    denormalize_bbox_yolo,
    load_image,
    load_mask,
    save_image,
    setup_logging,
)


logger = setup_logging("AugmentDataset", PROJECT_ROOT / "outputs" / "debug" / "augment.log")


@dataclass
class ObjectCrop:
    class_id: int
    image_crop: np.ndarray
    mask_crop: np.ndarray
    bbox_xyxy: Tuple[int, int, int, int]


class SemiSyntheticAugmenter:
    """Create realistic copy-paste augmentations using the original RGB masks."""

    CLASS_COLOR_MAP = {
        (128, 0, 0): 0,
        (0, 0, 128): 1,
        (128, 128, 0): 2,
    }

    def __init__(
        self,
        mode: str = "unified",
        min_area: int = 50,
        seed: int = 42,
        scale_range: Tuple[float, float] = (0.75, 1.35),
    ):
        self.mode = mode
        self.min_area = min_area
        self.rng = random.Random(seed)
        self.scale_range = scale_range
        self.images_dir = get_dataset_path("images")
        self.labels_dir = get_dataset_path("labels")
        self.object_bank: List[ObjectCrop] = []

    def build_object_bank(self, train_filenames: List[str]) -> None:
        """Extract real weed crops from the training split."""
        self.object_bank = []
        for idx, filename in enumerate(train_filenames):
            if idx % 50 == 0:
                logger.info(f"Building object bank: {idx}/{len(train_filenames)}")

            image_path = self.images_dir / filename
            mask_path = self.labels_dir / filename
            if not image_path.exists() or not mask_path.exists():
                continue

            image = load_image(str(image_path))
            mask = load_mask(str(mask_path), color=True)
            if image is None or mask is None:
                continue

            self.object_bank.extend(self._extract_objects(image, mask))

        logger.info(f"Object bank size: {len(self.object_bank)}")

    def generate_augmented_samples(
        self,
        train_filenames: List[str],
        base_labels_dir: Path,
        output_images_dir: Path,
        output_labels_dir: Path,
        copies_per_image: int = 1,
        objects_per_image: Tuple[int, int] = (1, 3),
    ) -> int:
        """Generate augmented train images and labels."""
        if not self.object_bank:
            self.build_object_bank(train_filenames)
        if not self.object_bank:
            logger.warning("Object bank is empty, skipping augmentation")
            return 0

        generated = 0
        for idx, filename in enumerate(train_filenames):
            image_path = self.images_dir / filename
            image = load_image(str(image_path))
            if image is None:
                continue

            label_path = base_labels_dir / f"{Path(filename).stem}.txt"
            existing_labels = self._load_yolo_labels(label_path, image.shape[1], image.shape[0])

            for copy_idx in range(copies_per_image):
                aug_image = image.copy()
                aug_labels = list(existing_labels)

                num_objects = self.rng.randint(objects_per_image[0], objects_per_image[1])
                pasted = self._paste_random_objects(aug_image, aug_labels, num_objects)
                if pasted == 0:
                    continue

                output_stem = f"{Path(filename).stem}_aug_{copy_idx + 1:02d}"
                save_image(aug_image, str(output_images_dir / f"{output_stem}.png"))
                self._save_yolo_labels(aug_labels, output_labels_dir / f"{output_stem}.txt")
                generated += 1

            if idx % 50 == 0:
                logger.info(f"Generated augmented samples for {idx}/{len(train_filenames)} train images")

        logger.info(f"Generated augmented train samples: {generated}")
        return generated

    def _extract_objects(self, image: np.ndarray, mask: np.ndarray) -> List[ObjectCrop]:
        objects: List[ObjectCrop] = []
        unique_colors = np.unique(mask.reshape(-1, 3), axis=0)
        for color in unique_colors:
            color_tuple = tuple(int(v) for v in color.tolist())
            if color_tuple == (0, 0, 0):
                continue
            class_id = self.CLASS_COLOR_MAP.get(color_tuple)
            if class_id is None:
                continue

            binary = np.all(mask == color, axis=2).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                pad = 4
                x0 = max(0, x - pad)
                y0 = max(0, y - pad)
                x1 = min(image.shape[1], x + w + pad)
                y1 = min(image.shape[0], y + h + pad)
                image_crop = image[y0:y1, x0:x1].copy()
                mask_crop = binary[y0:y1, x0:x1].copy()
                if mask_crop.max() == 0:
                    continue

                output_class = 0 if self.mode == "unified" else int(class_id)
                objects.append(
                    ObjectCrop(
                        class_id=output_class,
                        image_crop=image_crop,
                        mask_crop=mask_crop,
                        bbox_xyxy=(x0, y0, x1, y1),
                    )
                )
        return objects

    def _paste_random_objects(self, image: np.ndarray, labels: List[Dict], count: int) -> int:
        pasted = 0
        img_h, img_w = image.shape[:2]
        for _ in range(count):
            obj = self.rng.choice(self.object_bank)
            transformed = self._transform_object(obj)
            if transformed is None:
                continue

            obj_img, obj_mask, bbox = transformed
            h, w = obj_mask.shape[:2]
            max_x = img_w - w
            max_y = img_h - h
            if max_x <= 1 or max_y <= 1:
                continue

            placed = False
            for _attempt in range(25):
                x = self.rng.randint(0, max_x)
                y = self.rng.randint(0, max_y)
                bbox_xyxy = [x, y, x + w, y + h]
                if self._overlaps_too_much(bbox_xyxy, labels):
                    continue
                self._composite(image, obj_img, obj_mask, x, y)
                labels.append(
                    {
                        "class_id": obj.class_id,
                        "bbox_xyxy": bbox_xyxy,
                    }
                )
                pasted += 1
                placed = True
                break
            if not placed:
                continue
        return pasted

    def _transform_object(
        self,
        obj: ObjectCrop,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]]:
        scale = self.rng.uniform(*self.scale_range)
        crop = obj.image_crop
        mask = obj.mask_crop

        if self.rng.random() < 0.5:
            crop = np.ascontiguousarray(np.fliplr(crop))
            mask = np.ascontiguousarray(np.fliplr(mask))

        new_w = max(8, int(round(crop.shape[1] * scale)))
        new_h = max(8, int(round(crop.shape[0] * scale)))
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        brightness = self.rng.uniform(0.9, 1.1)
        crop = np.clip(crop.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
        if self.rng.random() < 0.3:
            crop = cv2.GaussianBlur(crop, (3, 3), 0)

        mask = (mask > 0).astype(np.uint8)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None

        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1
        crop = crop[y0:y1, x0:x1]
        mask = mask[y0:y1, x0:x1]
        return crop, mask, (0, 0, crop.shape[1], crop.shape[0])

    def _composite(self, base_image: np.ndarray, obj_image: np.ndarray, obj_mask: np.ndarray, x: int, y: int) -> None:
        h, w = obj_mask.shape
        roi = base_image[y:y + h, x:x + w].astype(np.float32)
        obj_float = obj_image.astype(np.float32)
        alpha = cv2.GaussianBlur(obj_mask.astype(np.float32), (0, 0), 1.2)
        alpha = np.clip(alpha, 0.0, 1.0)[..., None]
        blended = roi * (1.0 - alpha) + obj_float * alpha
        base_image[y:y + h, x:x + w] = blended.astype(np.uint8)

    def _overlaps_too_much(self, bbox_xyxy: List[int], labels: List[Dict], max_iou: float = 0.35) -> bool:
        x1, y1, x2, y2 = bbox_xyxy
        new_area = max(1, (x2 - x1) * (y2 - y1))
        for label in labels:
            lx1, ly1, lx2, ly2 = label["bbox_xyxy"]
            inter_x1 = max(x1, lx1)
            inter_y1 = max(y1, ly1)
            inter_x2 = min(x2, lx2)
            inter_y2 = min(y2, ly2)
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                continue
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            label_area = max(1, (lx2 - lx1) * (ly2 - ly1))
            union = new_area + label_area - inter_area
            iou = inter_area / union
            if iou > max_iou:
                return True
        return False

    def _load_yolo_labels(self, label_path: Path, img_width: int, img_height: int) -> List[Dict]:
        labels: List[Dict] = []
        if not label_path.exists():
            return labels
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            x_center, y_center, width, height = denormalize_bbox_yolo(
                x_center, y_center, width, height, img_height, img_width
            )
            x1, y1, x2, y2 = bbox_xywh_to_xyxy(x_center, y_center, width, height)
            labels.append(
                {
                    "class_id": class_id,
                    "bbox_xyxy": [
                        max(0, min(int(round(x1)), img_width - 1)),
                        max(0, min(int(round(y1)), img_height - 1)),
                        max(1, min(int(round(x2)), img_width)),
                        max(1, min(int(round(y2)), img_height)),
                    ],
                }
            )
        return labels

    def _save_yolo_labels(self, labels: List[Dict], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for label in labels:
            x1, y1, x2, y2 = label["bbox_xyxy"]
            img_w = max(x2, 1)
            img_h = max(y2, 1)
            # Actual normalization is done using the full image dimensions at write time.
            lines.append((label["class_id"], x1, y1, x2, y2))

        # Recover full image size from the target image file.
        image_path = output_path.with_suffix(".png").name
        full_image_path = output_path.parent.parent.parent / "images" / output_path.parent.name / image_path
        image = load_image(str(full_image_path))
        if image is None:
            raise RuntimeError(f"Failed to load image for writing labels: {full_image_path}")
        img_h, img_w = image.shape[:2]

        with open(output_path, "w", encoding="utf-8") as f:
            for class_id, x1, y1, x2, y2 in lines:
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

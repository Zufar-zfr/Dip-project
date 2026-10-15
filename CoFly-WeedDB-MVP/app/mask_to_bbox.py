"""
Convert mask labels to YOLO bounding box format.
Extracts connected components from masks and generates YOLO-format txt files.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
import cv2

from config import PROJECT_ROOT, DATASET_ROOT, get_dataset_path, dataset_config
from utils import (
    setup_logging, load_image, load_mask, save_image,
    normalize_bbox_yolo, bbox_xyxy_to_xywh, load_split_file
)


logger = setup_logging("MaskToBBox", PROJECT_ROOT / "outputs" / "debug" / "conversion.log")


class MaskToBBoxConverter:
    """Convert mask images to YOLO bounding box format."""
    CLASS_COLOR_MAP = {
        (128, 0, 0): 0,
        (0, 0, 128): 1,
        (128, 128, 0): 2,
    }
    
    def __init__(self, mode: str = "unified", min_area: int = 50):
        """
        Initialize converter.
        
        Args:
            mode: "unified" or "multiclass"
            min_area: minimum connected component area in pixels
        """
        self.mode = mode
        self.min_area = min_area
        self.images_dir = get_dataset_path("images")
        self.labels_dir = get_dataset_path("labels")
        self.output_dir = PROJECT_ROOT / "outputs" / "converted_labels" / mode
        self.debug_preview_dir = PROJECT_ROOT / "outputs" / "debug" / "converted_labels_preview" / mode
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug_preview_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = defaultdict(int)
    
    def convert_all(self, split_file: str = "train_split1.txt"):
        """
        Convert all masks in a split.
        
        Args:
            split_file: name of split file (e.g., "train_split1.txt")
        """
        split_path = DATASET_ROOT / split_file
        if not split_path.exists():
            logger.error(f"Split file not found: {split_path}")
            return
        
        filenames = load_split_file(str(split_path))
        logger.info(f"Converting {len(filenames)} images from {split_file}")
        
        preview_count = 0
        for idx, filename in enumerate(filenames):
            if idx % 50 == 0:
                logger.info(f"  Processed {idx}/{len(filenames)}")
            
            self.stats['total_processed'] += 1
            
            mask_path = self.labels_dir / filename
            img_path = self.images_dir / filename
            
            if not mask_path.exists():
                logger.warning(f"Mask not found: {mask_path}")
                self.stats['missing_masks'] += 1
                continue
            
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                self.stats['missing_images'] += 1
                continue
            
            # Convert this mask/image pair
            success = self.convert_single(str(img_path), str(mask_path))
            
            if success:
                self.stats['successful'] += 1
                
                # Save preview for first 20 converted images
                if preview_count < 20:
                    self._save_preview(str(img_path), str(mask_path))
                    preview_count += 1
        
        logger.info(f"Conversion complete. Stats: {dict(self.stats)}")
        self._save_conversion_stats()
    
    def convert_single(self, image_path: str, mask_path: str) -> bool:
        """
        Convert single mask to YOLO bbox format.
        
        Args:
            image_path: path to input image
            mask_path: path to input mask
            
        Returns:
            True if successful
        """
        try:
            # Load image and mask
            img = load_image(image_path)
            mask = load_mask(mask_path, color=True)
            
            if img is None or mask is None:
                return False
            
            img_height, img_width = img.shape[:2]
            
            # Find connected components
            detections = self._extract_components(mask, img_height, img_width)
            
            if not detections:
                self.stats['images_with_zero_detections'] += 1
            else:
                self.stats['total_detections'] += len(detections)
            
            # Save YOLO format labels
            filename = Path(image_path).stem
            output_txt = self.output_dir / f"{filename}.txt"
            self._save_yolo_labels(detections, output_txt, img_height, img_width)
            
            return True
        
        except Exception as e:
            logger.error(f"Error converting {image_path}: {e}")
            return False
    
    def _extract_components(self, mask: np.ndarray, img_height: int, img_width: int) -> List[dict]:
        """
        Extract connected components (objects) from mask.
        
        Args:
            mask: numpy array of shape (H, W) with class IDs
            img_height, img_width: image dimensions
            
        Returns:
            List of detection dicts with bbox_xyxy, class_id, area
        """
        detections = []
        
        if mask.ndim == 2:
            unique_classes = [int(v) for v in np.unique(mask) if int(v) != 0]
            class_masks = [
                (class_id, (mask == class_id).astype(np.uint8))
                for class_id in unique_classes
            ]
        else:
            unique_colors = np.unique(mask.reshape(-1, 3), axis=0)
            class_masks = []
            for color in unique_colors:
                color_tuple = tuple(int(v) for v in color.tolist())
                if color_tuple == (0, 0, 0):
                    continue
                if color_tuple not in self.CLASS_COLOR_MAP:
                    logger.warning(f"Unknown mask color encountered: {color_tuple}")
                    continue
                binary_mask = np.all(mask == color, axis=2).astype(np.uint8)
                class_masks.append((self.CLASS_COLOR_MAP[color_tuple], binary_mask))
        
        for class_id, binary_mask in class_masks:
            
            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by minimum area
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue
                
                # Convert to YOLO format
                x1, y1, x2, y2 = x, y, x + w, y + h
                
                # Clamp to image bounds
                x1 = max(0, min(x1, img_width - 1))
                y1 = max(0, min(y1, img_height - 1))
                x2 = max(1, min(x2, img_width))
                y2 = max(1, min(y2, img_height))
                
                # Unified mode: all classes become class 0 (weed)
                if self.mode == "unified":
                    output_class = 0
                else:
                    output_class = int(class_id)
                
                detections.append({
                    'class_id': output_class,
                    'bbox_xyxy': (x1, y1, x2, y2),
                    'area': area,
                    'original_class': int(class_id)
                })
        
        return detections
    
    def _save_yolo_labels(
        self,
        detections: List[dict],
        output_path: Path,
        img_height: int,
        img_width: int,
    ):
        """
        Save detections in YOLO format.
        
        Args:
            detections: list of detection dicts
            output_path: output txt file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for det in detections:
                x1, y1, x2, y2 = det['bbox_xyxy']

                # Convert to normalized xywh format
                x_center, y_center, width, height = bbox_xyxy_to_xywh(x1, y1, x2, y2)
                x_center, y_center, width, height = normalize_bbox_yolo(
                    x_center, y_center, width, height, img_height, img_width
                )

                # Write YOLO format: class_id x_center y_center width height
                f.write(f"{det['class_id']} {x_center:.6f} {y_center:.6f} "
                       f"{width:.6f} {height:.6f}\n")
    
    def _save_preview(self, image_path: str, mask_path: str):
        """Save debug preview with bbox overlay."""
        try:
            img = load_image(image_path)
            mask = load_mask(mask_path, color=True)
            
            if img is None or mask is None:
                return
            
            # Draw contours on image
            img_height, img_width = img.shape[:2]
            detections = self._extract_components(mask, img_height, img_width)
            
            result = img.copy()
            
            # Draw bboxes
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            for idx, det in enumerate(detections):
                x1, y1, x2, y2 = det['bbox_xyxy']
                color = colors[det['class_id'] % len(colors)]
                
                # Convert to int for drawing
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw rectangle
                cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
                
                # Draw class label
                label = f"C{det['original_class']}"
                cv2.putText(result, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add text info
            text = f"Detections: {len(detections)} | Classes: {sorted(set(d['original_class'] for d in detections))}"
            cv2.putText(result, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save
            filename = Path(image_path).stem
            output_path = self.debug_preview_dir / f"{filename}_converted.png"
            save_image(result, str(output_path))
        
        except Exception as e:
            logger.error(f"Error saving preview for {image_path}: {e}")
    
    def _save_conversion_stats(self):
        """Save conversion statistics."""
        stats_path = self.output_dir / "conversion_stats.txt"
        
        with open(stats_path, 'w') as f:
            f.write("Mask to BBox Conversion Statistics\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Mode: {self.mode}\n")
            f.write(f"Minimum area threshold: {self.min_area} pixels\n\n")
            
            f.write("Conversion Results:\n")
            for key, value in sorted(self.stats.items()):
                f.write(f"  {key}: {value}\n")
        
        logger.info(f"Saved conversion stats to {stats_path}")


def main():
    """Convert masks to bounding boxes."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert masks to YOLO bounding boxes")
    parser.add_argument("--mode", choices=["unified", "multiclass"], default="unified",
                       help="Detection mode")
    parser.add_argument("--min-area", type=int, default=50,
                       help="Minimum component area in pixels")
    parser.add_argument("--split", default="train_split1.txt",
                       help="Split file to convert")
    
    args = parser.parse_args()
    
    converter = MaskToBBoxConverter(mode=args.mode, min_area=args.min_area)
    converter.convert_all(split_file=args.split)
    
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"Labels saved to: {converter.output_dir}")
    print(f"Previews saved to: {converter.debug_preview_dir}")
    print(f"Statistics: {dict(converter.stats)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

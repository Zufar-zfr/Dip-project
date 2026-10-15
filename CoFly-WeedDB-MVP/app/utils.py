"""
Utility functions for the weed detection pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import json
import csv

import numpy as np
import cv2


def setup_logging(name: str = "WeedDetection", log_file: Optional[Path] = None):
    """Setup logging with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_image(image_path: str) -> Optional[np.ndarray]:
    """Load image from disk."""
    try:
        image_path = Path(image_path)
        raw_bytes = np.fromfile(str(image_path), dtype=np.uint8)
        img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {e}")
        return None


def load_mask(mask_path: str, color: bool = False) -> Optional[np.ndarray]:
    """Load mask image.

    Args:
        mask_path: path to the mask image
        color: when True, return an RGB mask; otherwise return grayscale
    """
    try:
        mask_path = Path(mask_path)
        raw_bytes = np.fromfile(str(mask_path), dtype=np.uint8)
        flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
        mask = cv2.imdecode(raw_bytes, flag)
        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_path}")
        if color:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        return mask
    except Exception as e:
        logging.error(f"Error loading mask {mask_path}: {e}")
        return None


def save_image(image: np.ndarray, output_path: str):
    """Save image to disk."""
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        suffix = output_path.suffix.lower() or ".png"
        ext = suffix if suffix.startswith(".") else f".{suffix}"
        success, encoded = cv2.imencode(ext, image)
        if not success:
            raise ValueError(f"Failed to encode image for {output_path}")
        encoded.tofile(str(output_path))
    except Exception as e:
        logging.error(f"Error saving image to {output_path}: {e}")


def normalize_bbox_yolo(x_center: float, y_center: float, width: float, height: float,
                        img_height: int, img_width: int) -> Tuple[float, float, float, float]:
    """
    Normalize bbox coordinates to YOLO format [0, 1].
    
    Args:
        x_center, y_center, width, height: Box coordinates in pixels
        img_height, img_width: Image dimensions
        
    Returns:
        Tuple of normalized (x_center, y_center, width, height)
    """
    return (
        x_center / img_width,
        y_center / img_height,
        width / img_width,
        height / img_height
    )


def denormalize_bbox_yolo(x_center: float, y_center: float, width: float, height: float,
                           img_height: int, img_width: int) -> Tuple[float, float, float, float]:
    """
    Denormalize YOLO bbox format back to pixel coordinates.
    
    Args:
        x_center, y_center, width, height: Normalized box coordinates
        img_height, img_width: Image dimensions
        
    Returns:
        Tuple of (x_center, y_center, width, height) in pixels
    """
    return (
        x_center * img_width,
        y_center * img_height,
        width * img_width,
        height * img_height
    )


def bbox_xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    """Convert bbox from (x1, y1, x2, y2) to (x_center, y_center, width, height)."""
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    return x_center, y_center, width, height


def bbox_xywh_to_xyxy(x_center: float, y_center: float, width: float, height: float) -> Tuple[float, float, float, float]:
    """Convert bbox from (x_center, y_center, width, height) to (x1, y1, x2, y2)."""
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    return x1, y1, x2, y2


def compute_iou(box1: Tuple[float, float, float, float],
                box2: Tuple[float, float, float, float]) -> float:
    """
    Compute Intersection over Union (IoU) for two boxes in (x1, y1, x2, y2) format.
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def nms(detections: List[dict], iou_threshold: float = 0.3) -> List[dict]:
    """
    Apply Non-Maximum Suppression to remove overlapping detections.
    
    Args:
        detections: List of detection dicts with keys: bbox_xyxy, confidence
        iou_threshold: IoU threshold for suppression
        
    Returns:
        List of detections after NMS
    """
    if not detections:
        return []
    
    # Sort by confidence descending
    sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    while sorted_dets:
        current = sorted_dets.pop(0)
        keep.append(current)
        
        remaining = []
        for det in sorted_dets:
            iou = compute_iou(current['bbox_xyxy'], det['bbox_xyxy'])
            if iou < iou_threshold:
                remaining.append(det)
        
        sorted_dets = remaining
    
    return keep


def save_detections_json(detections: List[dict], output_path: str):
    """Save detections to JSON file."""
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(detections, f, indent=2)
        logging.info(f"Saved {len(detections)} detections to {output_path}")
    except Exception as e:
        logging.error(f"Error saving JSON: {e}")


def save_detections_csv(detections: List[dict], output_path: str):
    """Save detections to CSV file."""
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not detections:
            logging.warning("No detections to save")
            return
        
        # Get all possible keys
        fieldnames = set()
        for det in detections:
            fieldnames.update(det.keys())
        fieldnames = sorted(list(fieldnames))
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detections)
        
        logging.info(f"Saved {len(detections)} detections to {output_path}")
    except Exception as e:
        logging.error(f"Error saving CSV: {e}")


def load_split_file(split_file_path: str) -> List[str]:
    """Load list of filenames from split file."""
    try:
        with open(split_file_path, 'r') as f:
            filenames = [line.strip() for line in f if line.strip()]
        return filenames
    except Exception as e:
        logging.error(f"Error loading split file {split_file_path}: {e}")
        return []


if __name__ == "__main__":
    print("Utility functions loaded successfully")

"""
Inference module for weed detection on single images and folders.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import os

import numpy as np
import cv2

from config import PROJECT_ROOT, inference_config
from utils import (
    setup_logging, load_image, save_image, denormalize_bbox_yolo,
    bbox_xywh_to_xyxy, nms, save_detections_json, save_detections_csv
)
from geo_utils import parse_coordinates_from_filename

# Try to import ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


logger = setup_logging("Inference", PROJECT_ROOT / "outputs" / "debug" / "inference.log")


class WeedDetector:
    """Run inference for weed detection."""
    
    def __init__(self, weights_path: str, conf_threshold: float = 0.25):
        """
        Initialize detector.
        
        Args:
            weights_path: path to YOLO model weights
            conf_threshold: confidence threshold for detections
        """
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics not installed. Install via: pip install ultralytics")
        
        # Handle relative paths
        weights_path_obj = Path(weights_path)
        if not weights_path_obj.is_absolute():
            weights_path_obj = PROJECT_ROOT / weights_path
        
        self.weights_path = weights_path_obj
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {self.weights_path}")
        
        logger.info(f"Loading model from {self.weights_path}")
        self.model = YOLO(str(self.weights_path))
        self.conf_threshold = conf_threshold
        
        logger.info(f"Model loaded. Confidence threshold: {conf_threshold}")

    def _is_suspicious_box(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        img_width: int,
        img_height: int,
        confidence: float,
    ) -> bool:
        """Drop low-confidence boxes that are implausibly large for this dataset."""
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        width_frac = width / max(img_width, 1)
        height_frac = height / max(img_height, 1)
        area_frac = (width * height) / max(img_width * img_height, 1)
        touches_border = x1 <= 1 or y1 <= 1 or x2 >= img_width - 1 or y2 >= img_height - 1

        if area_frac > 0.30 and confidence < 0.35:
            return True
        if width_frac > 0.40 and confidence < 0.30:
            return True
        if height_frac > 0.92 and width_frac > 0.16 and confidence < 0.40:
            return True
        if touches_border and area_frac > 0.22 and confidence < 0.28:
            return True
        return False

    def postprocess_detections(
        self,
        detections: List[Dict],
        img_width: int,
        img_height: int,
        iou_threshold: float = 0.35,
    ) -> List[Dict]:
        """Apply dataset-specific filtering and a final NMS pass."""
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox_xyxy']
            if self._is_suspicious_box(x1, y1, x2, y2, img_width, img_height, det['confidence']):
                continue
            filtered.append(det)
        return nms(filtered, iou_threshold=iou_threshold)
    
    def detect_image(self, image_path: str, save_visualization: bool = True) -> Dict:
        """
        Run detection on single image.
        
        Args:
            image_path: path to input image
            save_visualization: whether to save image with detections drawn
            
        Returns:
            Dict with detection results
        """
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return None
        
        logger.info(f"Detecting on image: {image_path.name}")
        
        # Load image
        img = load_image(str(image_path))
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        
        img_height, img_width = img.shape[:2]
        logger.info(f"Image loaded: {img_width}x{img_height}")
        
        # Run inference
        try:
            logger.info(f"Running inference with conf_threshold={self.conf_threshold}")
            results = self.model.predict(
                source=str(image_path),
                conf=self.conf_threshold,
                verbose=False
            )
            logger.info(f"Inference complete. Results: {len(results)} result(s)")
        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            return None
        
        # Parse results
        detections = []
        if results and len(results) > 0:
            result = results[0]
            
            logger.info(f"Result boxes: {result.boxes}")
            
            if result.boxes is not None and len(result.boxes) > 0:
                logger.info(f"Found {len(result.boxes)} boxes")
                for box_idx, box in enumerate(result.boxes):
                    # Get box coordinates in pixel format (x_center, y_center, width, height)
                    # YOLOv8 returns results already in pixel coordinates
                    x_center_px, y_center_px, width_px, height_px = box.xywh[0].cpu().numpy()
                    
                    # Get class and confidence
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    
                    # Convert to xyxy format (x1, y1, x2, y2)
                    x1, y1, x2, y2 = bbox_xywh_to_xyxy(
                        x_center_px, y_center_px, width_px, height_px
                    )
                    
                    # Get class name
                    class_name = self.model.names.get(class_id, f"class_{class_id}")
                    
                    # Parse geo coordinates from filename if available
                    geo_data = parse_coordinates_from_filename(image_path.name)
                    
                    detection = {
                        'image_name': image_path.name,
                        'image_width': img_width,
                        'image_height': img_height,
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox_xyxy': [float(x1), float(y1), float(x2), float(y2)],
                        'bbox_xywh': [float(x_center_px), float(y_center_px), 
                                     float(width_px), float(height_px)],
                        'center_px': [float(x_center_px), float(y_center_px)],
                        'latitude': geo_data['latitude'] if geo_data else None,
                        'longitude': geo_data['longitude'] if geo_data else None
                    }
                    detections.append(detection)
        
        detections = self.postprocess_detections(detections, img_width, img_height)
        logger.info(f"Found {len(detections)} detections")
        
        # Draw on image if requested
        visualization = None
        if save_visualization:
            visualization = self._draw_detections(img, detections)
        
        return {
            'image_path': str(image_path),
            'image_name': image_path.name,
            'detections': detections,
            'visualization': visualization
        }
    
    def detect_folder(self, folder_path: str, output_dir: Optional[str] = None) -> List[Dict]:
        """
        Run detection on all images in folder.
        
        Args:
            folder_path: path to folder with images
            output_dir: optional directory to save results
            
        Returns:
            List of detection results for each image
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            logger.error(f"Folder not found: {folder_path}")
            return []
        
        output_dir = Path(output_dir) if output_dir else inference_config.predictions_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_files = list(folder_path.glob("*.png")) + list(folder_path.glob("*.jpg"))
        logger.info(f"Found {len(image_files)} images in {folder_path}")
        
        all_detections = []
        
        for idx, img_path in enumerate(image_files):
            if idx % 10 == 0:
                logger.info(f"  Processing {idx}/{len(image_files)}")
            
            result = self.detect_image(str(img_path), save_visualization=True)
            if result:
                all_detections.extend(result['detections'])
                
                # Save visualization
                if result['visualization'] is not None:
                    vis_path = output_dir / f"vis_{img_path.stem}.png"
                    save_image(result['visualization'], str(vis_path))
        
        # Save aggregated results
        if all_detections:
            json_path = output_dir / "detections.json"
            csv_path = output_dir / "detections.csv"
            
            save_detections_json(all_detections, str(json_path))
            save_detections_csv(all_detections, str(csv_path))
        
        logger.info(f"Total detections across all images: {len(all_detections)}")
        return all_detections
    
    def _draw_detections(self, img: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on image.
        
        Args:
            img: input image
            detections: list of detection dicts
            
        Returns:
            Image with detections drawn
        """
        result = img.copy()
        
        logger.info(f"Drawing {len(detections)} detections")
        
        # Colors for different classes
        colors = {
            0: (0, 255, 0),      # Green for weed
            1: (0, 165, 255),    # Orange for weed_type_1
            2: (255, 0, 0),      # Blue for weed_type_2
        }
        
        for det_idx, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox_xyxy']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            logger.debug(f"Det {det_idx}: ({x1}, {y1}) -> ({x2}, {y2})")
            
            color = colors.get(det['class_id'], (255, 255, 0))
            
            # Draw rectangle
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det['class_name']} {det['confidence']:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Get text size for background
            (text_width, text_height) = cv2.getTextSize(label, font, 0.7, 2)[0]
            text_offset_x = 10
            text_offset_y = 30
            
            # Draw background rectangle for label
            box_coords = ((x1 + text_offset_x, y1 - text_offset_y),
                         (x1 + text_offset_x + text_width, y1 - text_offset_y + text_height))
            cv2.rectangle(result, box_coords[0], box_coords[1], color, cv2.FILLED)
            
            # Draw text
            cv2.putText(result, label, (x1 + text_offset_x, y1 - 10),
                       font, 0.7, (255, 255, 255), 2)
        
        logger.info(f"Finished drawing detections")
        return result


def main():
    """Main inference script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference for weed detection")
    parser.add_argument("--weights", default="models/unified/best.pt",
                       help="Path to model weights")
    parser.add_argument("--image", help="Path to single image")
    parser.add_argument("--folder", help="Path to folder with images")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Confidence threshold")
    parser.add_argument("--output", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.image and not args.folder:
        print("Please provide either --image or --folder")
        return
    
    # Initialize detector
    try:
        detector = WeedDetector(args.weights, conf_threshold=args.conf)
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return
    
    # Run inference
    if args.image:
        logger.info(f"Running inference on single image: {args.image}")
        result = detector.detect_image(args.image)
        
        if result:
            # Save detections
            output_dir = Path(args.output) if args.output else inference_config.predictions_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            
            json_path = output_dir / f"{Path(args.image).stem}_detections.json"
            csv_path = output_dir / f"{Path(args.image).stem}_detections.csv"
            vis_path = output_dir / f"{Path(args.image).stem}_vis.png"
            
            save_detections_json(result['detections'], str(json_path))
            save_detections_csv(result['detections'], str(csv_path))
            
            if result['visualization'] is not None:
                save_image(result['visualization'], str(vis_path))
            
            print(f"\nDetected {len(result['detections'])} weed instances")
            print(f"Results saved to: {output_dir}")
    
    elif args.folder:
        logger.info(f"Running inference on folder: {args.folder}")
        detections = detector.detect_folder(args.folder, args.output)
        
        print(f"\nDetected {len(detections)} weed instances across all images")
        if args.output:
            print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()

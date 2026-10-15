"""
Tile-based inference for processing large images.
Splits large image into overlapping tiles, runs inference, and merges results.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import math

import numpy as np
import cv2

from config import PROJECT_ROOT, tile_config
from utils import (
    setup_logging, load_image, save_image, denormalize_bbox_yolo,
    bbox_xywh_to_xyxy, nms, save_detections_json, save_detections_csv,
    compute_iou
)
from geo_utils import parse_coordinates_from_filename
from inference import WeedDetector


logger = setup_logging("TileInference", PROJECT_ROOT / "outputs" / "debug" / "tile_inference.log")


class TileBasedInference:
    """Run inference on large images using tile-based approach."""
    
    def __init__(self, detector: WeedDetector, tile_size: int = 640, 
                 overlap_ratio: float = 0.3, nms_threshold: float = 0.3):
        """
        Initialize tile-based inference.
        
        Args:
            detector: WeedDetector instance
            tile_size: size of each tile
            overlap_ratio: overlap between tiles (0.0-1.0)
            nms_threshold: NMS threshold for merging overlapping detections
        """
        self.detector = detector
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        self.nms_threshold = nms_threshold
        self.stride = int(tile_size * (1 - overlap_ratio))
        
        logger.info(f"Initialized TileBasedInference")
        logger.info(f"  Tile size: {tile_size}x{tile_size}")
        logger.info(f"  Overlap: {overlap_ratio*100:.0f}%")
        logger.info(f"  Stride: {self.stride}")
        logger.info(f"  NMS threshold: {nms_threshold}")
    
    def infer_large_image(self, image_path: str, output_dir: Optional[str] = None) -> Dict:
        """
        Run inference on large image using tiles.
        
        Args:
            image_path: path to large image
            output_dir: directory to save results
            
        Returns:
            Dict with detections and metadata
        """
        image_path = Path(image_path)
        logger.info(f"Processing large image: {image_path.name}")
        
        # Load image
        img = load_image(str(image_path))
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        
        img_height, img_width = img.shape[:2]
        logger.info(f"Image size: {img_width}x{img_height}")
        
        # Generate tiles
        tiles = self._generate_tiles(img_height, img_width)
        logger.info(f"Generated {len(tiles)} tiles")
        
        # Run inference on each tile
        all_detections = []
        for idx, tile_info in enumerate(tiles):
            if idx % 10 == 0:
                logger.info(f"  Processing tile {idx}/{len(tiles)}")
            
            tile_img = img[tile_info['y1']:tile_info['y2'], 
                          tile_info['x1']:tile_info['x2']]
            
            # Temporary save tile for inference
            temp_tile_path = PROJECT_ROOT / "outputs" / ".temp_tile.png"
            save_image(tile_img, str(temp_tile_path))
            
            # Run inference on tile
            try:
                results = self.detector.model.predict(
                    source=str(temp_tile_path),
                    conf=self.detector.conf_threshold,
                    verbose=False
                )
            except Exception as e:
                logger.warning(f"Inference on tile {idx} failed: {e}")
                continue
            finally:
                # Clean up temp file
                if temp_tile_path.exists():
                    temp_tile_path.unlink()
            
            # Convert tile detections to global image coordinates
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        # YOLO returns pixel coordinates here, not normalized values.
                        x_center_tile, y_center_tile, width_tile, height_tile = box.xywh[0].cpu().numpy()
                        
                        # Convert to global image coordinates
                        x_center_global = tile_info['x1'] + x_center_tile
                        y_center_global = tile_info['y1'] + y_center_tile
                        
                        # Get class and confidence
                        class_id = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())
                        class_name = self.detector.model.names.get(class_id, f"class_{class_id}")
                        
                        # Convert to xyxy format
                        from utils import bbox_xywh_to_xyxy
                        x1, y1, x2, y2 = bbox_xywh_to_xyxy(
                            x_center_global, y_center_global, width_tile, height_tile
                        )
                        
                        detection = {
                            'class_id': class_id,
                            'class_name': class_name,
                            'confidence': confidence,
                            'bbox_xyxy': [float(x1), float(y1), float(x2), float(y2)],
                            'bbox_xywh': [float(x_center_global), float(y_center_global),
                                         float(width_tile), float(height_tile)],
                            'center_px': [float(x_center_global), float(y_center_global)],
                            'tile_idx': idx
                        }
                        all_detections.append(detection)
        
        logger.info(f"Before NMS: {len(all_detections)} detections")
        all_detections = self.detector.postprocess_detections(all_detections, img_width, img_height)
        logger.info(f"After detector post-filter: {len(all_detections)} detections")
        
        # Merge overlapping detections using NMS
        merged_detections = self._merge_detections(all_detections)
        logger.info(f"After NMS: {len(merged_detections)} detections")
        
        # Add geo data if available
        geo_data = parse_coordinates_from_filename(image_path.name)
        for det in merged_detections:
            det['image_name'] = image_path.name
            det['image_width'] = img_width
            det['image_height'] = img_height
            if geo_data:
                det['latitude'] = geo_data['latitude']
                det['longitude'] = geo_data['longitude']
            else:
                det['latitude'] = None
                det['longitude'] = None
        
        # Draw detections on original image
        visualization = self._draw_detections(img, merged_detections)
        
        # Save results
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save visualization
            vis_path = output_dir / f"{image_path.stem}_tiled.png"
            save_image(visualization, str(vis_path))
            
            # Save detections
            json_path = output_dir / f"{image_path.stem}_detections.json"
            csv_path = output_dir / f"{image_path.stem}_detections.csv"
            
            save_detections_json(merged_detections, str(json_path))
            save_detections_csv(merged_detections, str(csv_path))
        
        return {
            'image_path': str(image_path),
            'image_name': image_path.name,
            'image_size': (img_width, img_height),
            'detections': merged_detections,
            'visualization': visualization,
            'tiles_processed': len(tiles)
        }
    
    def _generate_tiles(self, img_height: int, img_width: int) -> List[Dict]:
        """Generate tile coordinates for image."""
        tiles = []
        
        y = 0
        while y < img_height:
            y_end = min(y + self.tile_size, img_height)
            
            x = 0
            while x < img_width:
                x_end = min(x + self.tile_size, img_width)
                
                tiles.append({
                    'x1': x,
                    'y1': y,
                    'x2': x_end,
                    'y2': y_end
                })
                
                x += self.stride
                if x_end == img_width:
                    break
            
            y += self.stride
            if y_end == img_height:
                break
        
        return tiles
    
    def _merge_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Merge overlapping detections using NMS.
        
        Args:
            detections: list of detection dicts
            
        Returns:
            merged detections after NMS
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
                if iou < self.nms_threshold:
                    remaining.append(det)
            
            sorted_dets = remaining
        
        return keep
    
    def _draw_detections(self, img: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detections on image."""
        result = img.copy()
        
        colors = {
            0: (0, 255, 0),      # Green
            1: (0, 165, 255),    # Orange
            2: (255, 0, 0),      # Blue
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox_xyxy']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            color = colors.get(det['class_id'], (255, 255, 0))
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(result, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return result


def main():
    """Main tile inference script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tile-based inference on large images")
    parser.add_argument("--weights", default="models/unified/best.pt",
                       help="Path to model weights")
    parser.add_argument("--image", required=True, help="Path to large image")
    parser.add_argument("--tile-size", type=int, default=640,
                       help="Tile size")
    parser.add_argument("--overlap", type=float, default=0.3,
                       help="Tile overlap ratio")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Confidence threshold")
    parser.add_argument("--output", help="Output directory")
    
    args = parser.parse_args()
    
    # Initialize detector
    from inference import WeedDetector
    try:
        detector = WeedDetector(args.weights, conf_threshold=args.conf)
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Run tile inference
    tile_inf = TileBasedInference(
        detector,
        tile_size=args.tile_size,
        overlap_ratio=args.overlap
    )
    
    result = tile_inf.infer_large_image(args.image, output_dir=args.output)
    
    if result:
        print(f"\nProcessed large image: {result['image_name']}")
        print(f"Image size: {result['image_size']}")
        print(f"Tiles processed: {result['tiles_processed']}")
        print(f"Detections: {len(result['detections'])}")
        if args.output:
            print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()

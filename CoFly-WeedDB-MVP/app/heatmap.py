"""
Heatmap generation for weed detection results.
Supports both image-space heatmaps and geo-aggregated heatmaps.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json

import numpy as np
import cv2
from collections import defaultdict

from config import PROJECT_ROOT, heatmap_config
from utils import setup_logging, save_image, load_image
from geo_utils import parse_coordinates_from_filename, get_geo_bounds

logger = setup_logging("Heatmap", PROJECT_ROOT / "outputs" / "debug" / "heatmap.log")


class HeatmapGenerator:
    """Generate heatmaps from detection results."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize heatmap generator.
        
        Args:
            output_dir: directory to save heatmaps
        """
        self.output_dir = Path(output_dir) if output_dir else heatmap_config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized HeatmapGenerator")
        logger.info(f"  Output directory: {self.output_dir}")
    
    def generate_image_space_heatmap(self, detections: List[Dict], 
                                    image_path: Optional[str] = None,
                                    image_size: Optional[Tuple[int, int]] = None,
                                    sigma: Optional[float] = None,
                                    sensitivity: float = 1.0) -> np.ndarray:
        """
        Generate heatmap of detection density in image space.
        
        Args:
            detections: list of detection dicts with center_px
            image_path: optional path to reference image
            image_size: optional (width, height) tuple
            
        Returns:
            Heatmap numpy array
        """
        logger.info("Generating image-space heatmap...")
        
        # Get image dimensions
        if image_path:
            img = load_image(image_path)
            if img is not None:
                img_height, img_width = img.shape[:2]
            else:
                img_width, img_height = 640, 480
        elif image_size:
            img_width, img_height = image_size
        else:
            img_width, img_height = 640, 480
        
        # Build a sparse impulse map and blur it instead of looping over
        # every pixel neighborhood for every detection. This scales much
        # better for large stitched images.
        heatmap = np.zeros((img_height, img_width), dtype=np.float32)
        
        for det in detections:
            cx, cy = det['center_px']
            cx = int(round(cx))
            cy = int(round(cy))
            cx = max(0, min(cx, img_width - 1))
            cy = max(0, min(cy, img_height - 1))
            heatmap[cy, cx] += float(det.get('confidence', 1.0))
        
        sigma = max(float(sigma if sigma is not None else heatmap_config.gaussian_sigma), 1.0)
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=sigma, sigmaY=sigma)
        
        # Normalize to [0, 255]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
            gamma = max(0.1, 1.0 / max(sensitivity, 0.1))
            heatmap = np.power(heatmap, gamma)
            heatmap = (heatmap * 255).astype(np.uint8)
        else:
            heatmap = heatmap.astype(np.uint8)
        
        logger.info(f"Generated image-space heatmap: {heatmap.shape}")
        
        return heatmap
    
    def visualize_heatmap(self, heatmap: np.ndarray, image_path: Optional[str] = None,
                         image: Optional[np.ndarray] = None,
                         overlay_alpha: float = 0.4,
                         colormap_name: str = "hot") -> np.ndarray:
        """
        Visualize heatmap with optional image overlay.
        
        Args:
            heatmap: heatmap array
            image_path: optional image to overlay
            colormap_name: matplotlib colormap name
            
        Returns:
            Colored heatmap visualization
        """
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
        
        # Overlay on image if provided
        img = image
        if img is None and image_path:
            img = load_image(image_path)
        
        if img is not None:
            if heatmap_colored.shape != img.shape:
                heatmap_colored = cv2.resize(heatmap_colored, (img.shape[1], img.shape[0]))
            overlay_alpha = max(0.0, min(float(overlay_alpha), 1.0))
            base_alpha = 1.0 - overlay_alpha
            result = cv2.addWeighted(
                img,
                base_alpha,
                cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB),
                overlay_alpha,
                0,
            )
            return result
        
        return cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    def save_image_heatmap(self, detections: List[Dict], image_path: str,
                          output_name: Optional[str] = None,
                          sigma: Optional[float] = None,
                          sensitivity: float = 1.0,
                          overlay_alpha: float = 0.4):
        """
        Generate and save image-space heatmap.
        
        Args:
            detections: list of detection dicts
            image_path: path to reference image
            output_name: custom output name
        """
        # Load image
        img = load_image(image_path)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return
        
        img_height, img_width = img.shape[:2]
        
        # Generate heatmap
        heatmap = self.generate_image_space_heatmap(
            detections,
            image_size=(img_width, img_height),
            sigma=sigma,
            sensitivity=sensitivity,
        )
        
        # Visualize
        visualization = self.visualize_heatmap(
            heatmap,
            image_path=image_path,
            overlay_alpha=overlay_alpha,
        )
        
        # Save
        if output_name is None:
            output_name = f"{Path(image_path).stem}_heatmap.png"
        
        output_path = self.output_dir / output_name
        save_image(visualization, str(output_path))
        
        logger.info(f"Saved heatmap: {output_path}")
    
    def generate_geo_aggregated_heatmap(self, all_results: List[Dict]) -> Optional[Dict]:
        """
        Generate geo-aggregated heatmap from multiple images with coordinates.
        
        Args:
            all_results: list of result dicts, each with 'image_name' and 'detections'
                        where detections have latitude/longitude
            
        Returns:
            Dict with aggregated results and optional visualization
        """
        logger.info("Generating geo-aggregated heatmap...")
        
        # Aggregate by location
        geo_aggregation = defaultdict(lambda: {'count': 0, 'images': []})
        
        for result in all_results:
            image_name = result.get('image_name')
            detections = result.get('detections', [])
            
            for det in detections:
                lat = det.get('latitude')
                lon = det.get('longitude')
                
                if lat is not None and lon is not None:
                    # Use rounded coordinates as key (e.g., 2 decimal places)
                    key = (round(lat, 2), round(lon, 2))
                    geo_aggregation[key]['count'] += 1
                    if image_name not in geo_aggregation[key]['images']:
                        geo_aggregation[key]['images'].append(image_name)
        
        if not geo_aggregation:
            logger.warning("No geographic data found in detections")
            return None
        
        # Convert to list format
        aggregated_list = [
            {
                'latitude': lat,
                'longitude': lon,
                'weed_count': data['count'],
                'images': data['images'],
                'num_images': len(data['images'])
            }
            for (lat, lon), data in sorted(geo_aggregation.items(), 
                                          key=lambda x: x[1]['count'], reverse=True)
        ]
        
        logger.info(f"Aggregated results from {len(aggregated_list)} locations")
        
        # Save aggregation
        csv_path = self.output_dir / "geo_aggregation.csv"
        json_path = self.output_dir / "geo_aggregation.json"
        
        with open(csv_path, 'w') as f:
            f.write("latitude,longitude,weed_count,num_images,images\n")
            for item in aggregated_list:
                f.write(f"{item['latitude']},{item['longitude']},{item['weed_count']},"
                       f"{item['num_images']},\"{'; '.join(item['images'])}\"\n")
        
        with open(json_path, 'w') as f:
            json.dump(aggregated_list, f, indent=2)
        
        logger.info(f"Saved geo aggregation: {csv_path}, {json_path}")
        
        # Try to generate folium map if available
        try:
            import folium
            self._generate_folium_map(aggregated_list)
        except ImportError:
            logger.info("folium not available, skipping interactive map generation")
        
        return {
            'aggregated_data': aggregated_list,
            'total_locations': len(aggregated_list),
            'total_weeds': sum(item['weed_count'] for item in aggregated_list)
        }
    
    def _generate_folium_map(self, aggregated_list: List[Dict]):
        """Generate interactive folium map."""
        try:
            import folium
            
            if not aggregated_list:
                logger.warning("No data to generate map")
                return
            
            # Calculate center
            lats = [item['latitude'] for item in aggregated_list]
            lons = [item['longitude'] for item in aggregated_list]
            center_lat = np.mean(lats)
            center_lon = np.mean(lons)
            
            # Create map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=18,
                tiles='OpenStreetMap'
            )
            
            # Add markers
            for item in aggregated_list:
                color = 'red' if item['weed_count'] > 10 else (
                       'orange' if item['weed_count'] > 5 else 'green'
                )
                
                popup_text = (f"Weed count: {item['weed_count']}<br>"
                             f"Images: {item['num_images']}")
                
                folium.CircleMarker(
                    location=[item['latitude'], item['longitude']],
                    radius=min(item['weed_count'] / 2, 20),
                    popup=popup_text,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(m)
            
            # Save map
            map_path = self.output_dir / "weed_map.html"
            m.save(str(map_path))
            logger.info(f"Saved interactive map: {map_path}")
        
        except Exception as e:
            logger.warning(f"Failed to generate folium map: {e}")


def main():
    """Generate heatmaps from detection results."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate heatmaps from detections")
    parser.add_argument("--results", required=True, help="Path to results JSON file")
    parser.add_argument("--image", help="Path to reference image for image-space heatmap")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--geo", action="store_true", 
                       help="Generate geo-aggregated heatmap")
    
    args = parser.parse_args()
    
    # Load results
    with open(args.results, 'r') as f:
        detections = json.load(f)
    
    generator = HeatmapGenerator(output_dir=args.output)
    
    if args.geo:
        # Load all results (expects list of result dicts)
        results = [{'detections': detections}]
        generator.generate_geo_aggregated_heatmap(results)
    elif args.image:
        generator.save_image_heatmap(detections, args.image)
    else:
        print("Please provide either --image or --geo flag")


if __name__ == "__main__":
    main()

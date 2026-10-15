"""
Geographic utilities for parsing coordinates from filenames and GeoTIFF handling.
"""

import re
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# Regex pattern for extracting Lat/Lon/Alt from filename
# Expected format: ...[Lat=39.542124,Lon=22.644429,Alt=4.9]...
GEO_PATTERN = re.compile(
    r'\[Lat=([+-]?\d+\.\d+),Lon=([+-]?\d+\.\d+),Alt=([+-]?\d+\.\d+)\]'
)


def parse_coordinates_from_filename(filename: str) -> Optional[Dict[str, float]]:
    """
    Extract latitude, longitude, and altitude from filename.
    
    Args:
        filename: Image filename containing metadata
        
    Returns:
        Dict with keys 'latitude', 'longitude', 'altitude' or None if not found
    """
    match = GEO_PATTERN.search(filename)
    if match:
        try:
            return {
                'latitude': float(match.group(1)),
                'longitude': float(match.group(2)),
                'altitude': float(match.group(3))
            }
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse coordinates from {filename}: {e}")
            return None
    return None


def get_geo_bounds(coordinates_list: list) -> Dict[str, float]:
    """
    Calculate bounding box from list of coordinate dicts.
    
    Args:
        coordinates_list: List of dicts with 'latitude' and 'longitude' keys
        
    Returns:
        Dict with 'min_lat', 'max_lat', 'min_lon', 'max_lon'
    """
    if not coordinates_list:
        return None
    
    lats = [c['latitude'] for c in coordinates_list if c]
    lons = [c['longitude'] for c in coordinates_list if c]
    
    if not lats or not lons:
        return None
    
    return {
        'min_lat': min(lats),
        'max_lat': max(lats),
        'min_lon': min(lons),
        'max_lon': max(lons)
    }


class GeoTIFFHandler:
    """
    Placeholder for GeoTIFF support (optional, for future use).
    """
    
    def __init__(self, geotiff_path: str):
        """
        Initialize GeoTIFF handler.
        
        Args:
            geotiff_path: Path to GeoTIFF file
        """
        try:
            import rasterio
            from rasterio.crs import CRS
            self.rasterio_available = True
        except ImportError:
            logger.warning("rasterio not installed. GeoTIFF support disabled.")
            self.rasterio_available = False
        
        self.geotiff_path = Path(geotiff_path)
        
        if self.rasterio_available and self.geotiff_path.exists():
            try:
                import rasterio
                self.src = rasterio.open(self.geotiff_path)
                self.crs = self.src.crs
                self.bounds = self.src.bounds
                logger.info(f"Loaded GeoTIFF: {self.geotiff_path}")
            except Exception as e:
                logger.error(f"Failed to load GeoTIFF: {e}")
                self.src = None
        else:
            self.src = None
    
    def pixel_to_geo(self, pixel_x: int, pixel_y: int) -> Optional[Tuple[float, float]]:
        """
        Convert pixel coordinates to geographic coordinates (lat, lon).
        
        Args:
            pixel_x: Pixel x-coordinate
            pixel_y: Pixel y-coordinate
            
        Returns:
            Tuple of (latitude, longitude) or None
        """
        if not self.src or not self.rasterio_available:
            return None
        
        try:
            from rasterio.transform import Affine
            row, col = pixel_y, pixel_x
            geo_coords = self.src.transform * (col, row)
            # This returns (x, y) in the CRS, usually (lon, lat)
            return (geo_coords[1], geo_coords[0])  # Return as (lat, lon)
        except Exception as e:
            logger.error(f"Failed to convert pixel to geo: {e}")
            return None
    
    def geo_to_pixel(self, latitude: float, longitude: float) -> Optional[Tuple[int, int]]:
        """
        Convert geographic coordinates to pixel coordinates.
        
        Args:
            latitude: Latitude
            longitude: Longitude
            
        Returns:
            Tuple of (pixel_x, pixel_y) or None
        """
        if not self.src or not self.rasterio_available:
            return None
        
        try:
            from rasterio.transform import Affine
            row, col = ~self.src.transform * (longitude, latitude)
            return (int(col), int(row))
        except Exception as e:
            logger.error(f"Failed to convert geo to pixel: {e}")
            return None
    
    def close(self):
        """Close GeoTIFF file."""
        if self.src:
            self.src.close()
    
    def __del__(self):
        self.close()


if __name__ == "__main__":
    # Test coordinate parsing
    test_filename = "ID_00048_[Lat=39.542124,Lon=22.644429,Alt=4.9]_DATE_03_07_2019.png"
    coords = parse_coordinates_from_filename(test_filename)
    print("Parsed coordinates:", coords)

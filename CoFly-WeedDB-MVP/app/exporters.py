"""
Export detection results in various formats (JSON, CSV, HTML, etc.).
"""

import logging
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from config import PROJECT_ROOT
from utils import setup_logging

logger = setup_logging("Exporters", PROJECT_ROOT / "outputs" / "debug" / "exporters.log")


class DetectionExporter:
    """Export detection results in multiple formats."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize exporter.
        
        Args:
            output_dir: directory to save exports
        """
        self.output_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "outputs" / "exports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized DetectionExporter at {self.output_dir}")
    
    def export_json(self, detections: List[Dict], filename: Optional[str] = None) -> Path:
        """
        Export detections as JSON.
        
        Args:
            detections: list of detection dicts
            filename: output filename (auto-generated if not provided)
            
        Returns:
            Path to output file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detections_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        try:
            with open(output_path, 'w') as f:
                json.dump(detections, f, indent=2)
            
            logger.info(f"Exported {len(detections)} detections to {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error exporting JSON: {e}")
            raise
    
    def export_csv(self, detections: List[Dict], filename: Optional[str] = None) -> Path:
        """
        Export detections as CSV.
        
        Args:
            detections: list of detection dicts
            filename: output filename
            
        Returns:
            Path to output file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detections_{timestamp}.csv"
        
        output_path = self.output_dir / filename
        
        try:
            if not detections:
                logger.warning("No detections to export")
                return None
            
            # Get all possible keys
            fieldnames = set()
            for det in detections:
                fieldnames.update(det.keys())
            
            # Flatten nested lists/tuples for CSV
            fieldnames = sorted(list(fieldnames))
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for det in detections:
                    # Convert lists to strings
                    row = {}
                    for key, value in det.items():
                        if isinstance(value, (list, tuple)):
                            row[key] = str(value)
                        else:
                            row[key] = value
                    writer.writerow(row)
            
            logger.info(f"Exported {len(detections)} detections to {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error exporting CSV: {e}")
            raise
    
    def export_html_table(self, detections: List[Dict], 
                         filename: Optional[str] = None) -> Path:
        """
        Export detections as HTML table.
        
        Args:
            detections: list of detection dicts
            filename: output filename
            
        Returns:
            Path to output file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detections_{timestamp}.html"
        
        output_path = self.output_dir / filename
        
        try:
            html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Weed Detection Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        table { border-collapse: collapse; width: 100%; max-width: 1200px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        tr:hover { background-color: #ddd; }
        .summary { background-color: #e8f4f8; padding: 10px; margin-bottom: 20px; }
    </style>
</head>
<body>
"""
            
            # Add summary
            html += f"""
<div class="summary">
    <h2>Detection Summary</h2>
    <p>Total detections: {len(detections)}</p>
    <p>Export date/time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
</div>

<h2>Detections</h2>
<table>
    <thead>
        <tr>
"""
            
            # Get column headers
            if detections:
                fieldnames = sorted(set(k for det in detections for k in det.keys()))
                for field in fieldnames:
                    html += f"<th>{field}</th>\n"
            
            html += """
        </tr>
    </thead>
    <tbody>
"""
            
            # Add data rows
            for det in detections:
                html += "<tr>\n"
                for field in fieldnames:
                    value = det.get(field, "")
                    if isinstance(value, (list, tuple)):
                        value = ", ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in value)
                    elif isinstance(value, float):
                        value = f"{value:.4f}"
                    else:
                        value = str(value)
                    html += f"<td>{value}</td>\n"
                html += "</tr>\n"
            
            html += """
    </tbody>
</table>
</body>
</html>
"""
            
            with open(output_path, 'w') as f:
                f.write(html)
            
            logger.info(f"Exported {len(detections)} detections to {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error exporting HTML: {e}")
            raise
    
    def export_yolo_txt(self, detections: List[Dict], image_width: int, image_height: int,
                       filename: Optional[str] = None) -> Path:
        """
        Export detections in YOLO txt format (for evaluation with other YOLO tools).
        
        Args:
            detections: list of detection dicts
            image_width, image_height: image dimensions
            filename: output filename
            
        Returns:
            Path to output file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detections_{timestamp}.txt"
        
        output_path = self.output_dir / filename
        
        try:
            with open(output_path, 'w') as f:
                for det in detections:
                    x1, y1, x2, y2 = det['bbox_xyxy']
                    
                    # Normalize to YOLO format
                    x_center = ((x1 + x2) / 2) / image_width
                    y_center = ((y1 + y2) / 2) / image_height
                    width = (x2 - x1) / image_width
                    height = (y2 - y1) / image_height
                    
                    class_id = det['class_id']
                    confidence = det.get('confidence', 1.0)
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} "
                           f"{width:.6f} {height:.6f} {confidence:.6f}\n")
            
            logger.info(f"Exported {len(detections)} detections to {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error exporting YOLO txt: {e}")
            raise
    
    def export_all(self, detections: List[Dict], image_width: int = 640, 
                  image_height: int = 480, prefix: str = "") -> Dict[str, Path]:
        """
        Export detections in all formats.
        
        Args:
            detections: list of detection dicts
            image_width, image_height: image dimensions
            prefix: prefix for output filenames
            
        Returns:
            Dict mapping format names to output paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {}
        
        # JSON
        json_file = f"{prefix}detections_{timestamp}.json" if prefix else f"detections_{timestamp}.json"
        results['json'] = self.export_json(detections, json_file)
        
        # CSV
        csv_file = f"{prefix}detections_{timestamp}.csv" if prefix else f"detections_{timestamp}.csv"
        results['csv'] = self.export_csv(detections, csv_file)
        
        # HTML
        html_file = f"{prefix}detections_{timestamp}.html" if prefix else f"detections_{timestamp}.html"
        results['html'] = self.export_html_table(detections, html_file)
        
        # YOLO txt
        txt_file = f"{prefix}detections_{timestamp}.txt" if prefix else f"detections_{timestamp}.txt"
        results['txt'] = self.export_yolo_txt(detections, image_width, image_height, txt_file)
        
        logger.info(f"Exported all formats: {results}")
        return results


def main():
    """Export detection results."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Export detection results")
    parser.add_argument("--detections", required=True, help="Path to detections JSON file")
    parser.add_argument("--format", choices=["json", "csv", "html", "txt", "all"],
                       default="all", help="Export format")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=480, help="Image height")
    
    args = parser.parse_args()
    
    # Load detections
    with open(args.detections, 'r') as f:
        detections = json.load(f)
    
    exporter = DetectionExporter(output_dir=args.output)
    
    if args.format == "json":
        result = exporter.export_json(detections)
    elif args.format == "csv":
        result = exporter.export_csv(detections)
    elif args.format == "html":
        result = exporter.export_html_table(detections)
    elif args.format == "txt":
        result = exporter.export_yolo_txt(detections, args.width, args.height)
    else:  # all
        result = exporter.export_all(detections, args.width, args.height)
    
    print(f"Export complete!")
    if isinstance(result, dict):
        for fmt, path in result.items():
            print(f"  {fmt}: {path}")
    else:
        print(f"  {args.format}: {result}")


if __name__ == "__main__":
    main()

"""
Dataset inspector: analyze CoFly-WeedDB structure and generate diagnostic report.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import cv2

from config import PROJECT_ROOT, DATASET_ROOT, get_dataset_path, dataset_config
from utils import setup_logging, load_image, load_mask, save_image


logger = setup_logging("DatasetInspector", PROJECT_ROOT / "outputs" / "debug" / "inspector.log")


class DatasetInspector:
    """Inspect and analyze CoFly-WeedDB dataset."""
    
    def __init__(self):
        self.dataset_root = DATASET_ROOT
        self.images_dir = get_dataset_path("images")
        self.labels_dir = get_dataset_path("labels")
        self.debug_dir = PROJECT_ROOT / "outputs" / "debug"
        
        self.report = {
            'total_images': 0,
            'total_labels': 0,
            'matched_pairs': 0,
            'unmatched_images': [],
            'unmatched_labels': [],
            'image_sizes': defaultdict(int),
            'label_unique_values': set(),
            'label_unique_colors': set(),
            'sample_info': []
        }
    
    def inspect(self):
        """Run complete inspection."""
        logger.info("=" * 60)
        logger.info("Starting Dataset Inspection")
        logger.info("=" * 60)
        
        self._count_files()
        self._match_image_label_pairs()
        self._analyze_label_values()
        self._collect_sample_info()
        self._save_preview_samples()
        self._save_report()
        
        logger.info("=" * 60)
        logger.info("Dataset Inspection Complete")
        logger.info("=" * 60)
    
    def _count_files(self):
        """Count images and labels."""
        images = list(self.images_dir.glob("*.png"))
        labels = list(self.labels_dir.glob("*.png"))
        
        self.report['total_images'] = len(images)
        self.report['total_labels'] = len(labels)
        
        logger.info(f"Found {len(images)} images")
        logger.info(f"Found {len(labels)} labels")
    
    def _match_image_label_pairs(self):
        """Match image and label pairs."""
        images = {img.name: img for img in self.images_dir.glob("*.png")}
        labels = {lbl.name: lbl for lbl in self.labels_dir.glob("*.png")}
        
        matched = 0
        for img_name in images:
            if img_name in labels:
                matched += 1
            else:
                self.report['unmatched_images'].append(img_name)
        
        for lbl_name in labels:
            if lbl_name not in images:
                self.report['unmatched_labels'].append(lbl_name)
        
        self.report['matched_pairs'] = matched
        logger.info(f"Matched image-label pairs: {matched}")
        
        if self.report['unmatched_images']:
            logger.warning(f"Unmatched images: {len(self.report['unmatched_images'])}")
        if self.report['unmatched_labels']:
            logger.warning(f"Unmatched labels: {len(self.report['unmatched_labels'])}")
    
    def _analyze_label_values(self):
        """Analyze unique pixel values in labels."""
        logger.info("Analyzing label unique values/colors...")
        unique_values = set()
        unique_colors = set()
        
        sample_count = 0
        for label_file in self.labels_dir.glob("*.png"):
            mask_gray = load_mask(str(label_file))
            mask_rgb = load_mask(str(label_file), color=True)
            if mask_gray is not None:
                unique_vals = np.unique(mask_gray)
                unique_values.update(unique_vals.tolist())
            if mask_rgb is not None:
                colors = np.unique(mask_rgb.reshape(-1, 3), axis=0)
                unique_colors.update(tuple(int(c) for c in color) for color in colors)
                sample_count += 1
                
                if sample_count >= 50:  # Sample first 50 for speed
                    break
        
        self.report['label_unique_values'] = sorted(list(unique_values))
        self.report['label_unique_colors'] = sorted(list(unique_colors))
        logger.info(f"Unique label values found: {self.report['label_unique_values']}")
        logger.info(f"Unique label colors found: {self.report['label_unique_colors']}")
    
    def _collect_sample_info(self):
        """Collect info about random samples."""
        label_names = {lbl.name for lbl in self.labels_dir.glob("*.png")}
        images = [img_path for img_path in self.images_dir.glob("*.png") if img_path.name in label_names]
        
        sample_size = min(10, len(images))
        import random
        samples = random.sample(images, sample_size)
        
        for img_path in samples:
            img = load_image(str(img_path))
            mask = load_mask(str(self.labels_dir / img_path.name), color=True)
            
            if img is not None and mask is not None:
                flat_colors = np.unique(mask.reshape(-1, 3), axis=0)
                info = {
                    'filename': img_path.name,
                    'image_shape': img.shape,
                    'mask_shape': mask.shape,
                    'mask_unique_values': [tuple(int(c) for c in color) for color in flat_colors.tolist()],
                    'mask_value_counts': {
                        str(tuple(int(c) for c in color.tolist())): int(np.sum(np.all(mask == color, axis=2)))
                        for color in flat_colors
                    }
                }
                self.report['sample_info'].append(info)
        
        logger.info(f"Collected info for {len(self.report['sample_info'])} samples")
    
    def _save_preview_samples(self):
        """Save preview images with mask overlay."""
        logger.info("Saving preview samples...")
        preview_dir = self.debug_dir / "dataset_preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        
        label_names = {lbl.name for lbl in self.labels_dir.glob("*.png")}
        images = [img_path for img_path in self.images_dir.glob("*.png") if img_path.name in label_names]
        import random
        samples = random.sample(images, min(10, len(images)))
        
        for idx, img_path in enumerate(samples):
            img = load_image(str(img_path))
            mask = load_mask(str(self.labels_dir / img_path.name), color=True)
            
            if img is not None and mask is not None:
                # Create visualization with mask overlay
                H, W = img.shape[:2]
                
                # Colored mask overlay
                colored_mask = mask.copy()
                
                # Blend
                overlay = cv2.addWeighted(img, 0.7, colored_mask, 0.3, 0)
                
                # Add text info
                mask_colors = [tuple(int(c) for c in color) for color in np.unique(mask.reshape(-1, 3), axis=0).tolist()]
                cv2.putText(overlay, f"Mask colors: {mask_colors[:3]}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Save
                output_path = preview_dir / f"preview_{idx:02d}_{img_path.stem}.png"
                save_image(overlay, str(output_path))
        
        logger.info(f"Saved {len(samples)} preview images to {preview_dir}")
    
    def _save_report(self):
        """Save inspection report."""
        report_path = self.debug_dir / "dataset_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("CoFly-WeedDB Dataset Inspection Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Total images: {self.report['total_images']}\n")
            f.write(f"Total labels: {self.report['total_labels']}\n")
            f.write(f"Matched pairs: {self.report['matched_pairs']}\n\n")
            
            f.write(f"Unique grayscale values found: {self.report['label_unique_values']}\n")
            f.write(f"Unique RGB colors found: {self.report['label_unique_colors']}\n")
            f.write("Interpretation: masks are color-coded and should be parsed as RGB classes\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("Sample Information (first 10 samples)\n")
            f.write("=" * 60 + "\n")
            
            for idx, sample in enumerate(self.report['sample_info'], 1):
                f.write(f"\nSample {idx}: {sample['filename']}\n")
                f.write(f"  Image shape: {sample['image_shape']}\n")
                f.write(f"  Mask shape: {sample['mask_shape']}\n")
                f.write(f"  Mask classes: {sample['mask_unique_values']}\n")
                f.write(f"  Pixel counts by class:\n")
                for class_id, count in sorted(sample['mask_value_counts'].items()):
                    f.write(f"    Class {class_id}: {count} pixels\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("Unmatched Files\n")
            f.write("=" * 60 + "\n")
            
            if self.report['unmatched_images']:
                f.write(f"\nImages without labels ({len(self.report['unmatched_images'])}):\n")
                for fname in self.report['unmatched_images'][:20]:
                    f.write(f"  - {fname}\n")
                if len(self.report['unmatched_images']) > 20:
                    f.write(f"  ... and {len(self.report['unmatched_images']) - 20} more\n")
            
            if self.report['unmatched_labels']:
                f.write(f"\nLabels without images ({len(self.report['unmatched_labels'])}):\n")
                for fname in self.report['unmatched_labels'][:20]:
                    f.write(f"  - {fname}\n")
                if len(self.report['unmatched_labels']) > 20:
                    f.write(f"  ... and {len(self.report['unmatched_labels']) - 20} more\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("Recommendations for Processing\n")
            f.write("=" * 60 + "\n")
            f.write("\n1. Mode: UNIFIED WEED DETECTION (recommended for MVP)\n")
            f.write("   - All classes (0, 1, 2) will be converted to single 'weed' class\n")
            f.write("   - Provides more training examples per class\n")
            f.write("   - More stable training for baseline model\n")
            f.write("\n2. Alternative: MULTICLASS mode\n")
            f.write("   - Preserve classes 0, 1, 2 as separate weed types\n")
            f.write("   - More detailed analysis but fewer samples per class\n")
            f.write("\n3. Coordinate extraction:\n")
            f.write("   - Latitude/Longitude data available in filenames\n")
            f.write("   - Can be used for geographic heatmaps\n")
        
        logger.info(f"Saved report to {report_path}")


def main():
    """Run dataset inspection."""
    inspector = DatasetInspector()
    inspector.inspect()
    
    print("\n" + "=" * 60)
    print("Dataset inspection complete!")
    print(f"Report saved to: {inspector.debug_dir / 'dataset_report.txt'}")
    print(f"Previews saved to: {inspector.debug_dir / 'dataset_preview'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

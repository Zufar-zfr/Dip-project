#!/usr/bin/env python
"""Test detection directly."""

import logging
from pathlib import Path
from inference import WeedDetector
from utils import save_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test detection with multiple thresholds
dataset_root = Path(r"c:\Users\qwert\OneDrive\Рабочий стол\ДИПЛОМКА\CoFly-WeedDB\images")
images = list(dataset_root.glob("*.png"))[:1]

if not images:
    print("No images found!")
else:
    image_path = str(images[0])
    print(f"Testing on: {image_path.split('/')[-1]}")
    print()
    
    # Test with different confidence thresholds
    for conf_threshold in [0.01, 0.05, 0.1, 0.25]:
        detector = WeedDetector('models/unified/best.pt', conf_threshold=conf_threshold)
        result = detector.detect_image(image_path, save_visualization=True)
        
        if result:
            detections = result.get('detections', [])
            print(f"conf_threshold={conf_threshold}: {len(detections)} detections")
            if detections:
                for det in detections[:2]:
                    print(f"  - {det['class_name']}: {det['confidence']:.4f}")
        else:
            print(f"conf_threshold={conf_threshold}: ERROR")
    
    print("\n✓ Saved visualization to: outputs/test_visualization.png")


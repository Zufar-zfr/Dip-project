#!/usr/bin/env python
"""
run_train.py - Train YOLOv8 weed detection model
"""

import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from train import main

if __name__ == "__main__":
    main()

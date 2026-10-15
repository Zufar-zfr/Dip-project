#!/usr/bin/env python
"""
run_prepare.py - Prepare YOLO dataset from masks and split files
"""

import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from prepare_dataset import main

if __name__ == "__main__":
    main()

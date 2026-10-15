#!/usr/bin/env python
"""
run_infer.py - Run inference for weed detection
"""

import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from inference import main

if __name__ == "__main__":
    main()

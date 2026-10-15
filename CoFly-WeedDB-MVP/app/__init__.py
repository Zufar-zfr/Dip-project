"""
CoFly WeedDB MVP - Weed Detection System
"""

__version__ = "0.1.0"

from .config import (
    PROJECT_ROOT, DATASET_ROOT, dataset_config, training_config,
    inference_config, tile_config, heatmap_config, ui_config
)

__all__ = [
    "PROJECT_ROOT",
    "DATASET_ROOT", 
    "dataset_config",
    "training_config",
    "inference_config",
    "tile_config",
    "heatmap_config",
    "ui_config"
]

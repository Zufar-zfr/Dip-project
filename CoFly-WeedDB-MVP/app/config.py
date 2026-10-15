"""
Configuration module for CoFly WeedDB MVP project.
Centralized settings for dataset paths, training, inference, etc.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List


# Project root path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATASET_ROOT = PROJECT_ROOT.parent / "CoFly-WeedDB"  # Pre-existing dataset folder

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"

# Dataset mode
@dataclass
class DatasetConfig:
    """Configuration for dataset preparation."""
    
    # Unified mode: all weed classes (0, 1, 2) -> single class "weed"
    # Multiclass mode: preserve classes 0, 1, 2 as separate types
    mode: str = "unified"  # "unified" or "multiclass"
    
    # Split files to use
    train_split_file: str = "train_split1.txt"
    test_split_file: str = "test_split1.txt"
    
    # Minimum connected component area (pixels) to consider as valid detection
    min_area_threshold: int = 50
    
    # Classes configuration
    classes_unified: Dict[int, str] = field(default_factory=lambda: {0: "weed"})
    classes_multiclass: Dict[int, str] = field(default_factory=lambda: {
        0: "weed_type_0",
        1: "weed_type_1", 
        2: "weed_type_2"
    })
    
    def get_classes(self) -> Dict[int, str]:
        if self.mode == "unified":
            return self.classes_unified
        else:
            return self.classes_multiclass


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Model and training parameters
    model_name: str = "yolov8n"  # nano model for MVP
    imgsz: int = 640
    epochs: int = 30
    batch_size: int = 16  # Will auto-adjust if OOM
    device: str = "cpu"  # GPU device or "cpu"
    
    # Paths
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    
    # Training data paths (will be set dynamically)
    data_yaml_path: Path = None
    
    def __post_init__(self):
        if self.data_yaml_path is None:
            self.data_yaml_path = PROCESSED_DIR / "yolo_unified" / "data.yaml"


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    
    # Model weights
    weights_path: Path = field(default_factory=lambda: PROJECT_ROOT / "models" / "unified" / "best.pt")
    
    # Inference parameters
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    
    # Output paths
    predictions_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs" / "predictions")
    
    # Format output as JSON and CSV
    export_json: bool = True
    export_csv: bool = True


@dataclass
class TileInferenceConfig:
    """Configuration for tile-based inference."""
    
    # Tile size and overlap
    tile_size: int = 640
    overlap_ratio: float = 0.3  # 30% overlap
    
    # NMS/merging parameters
    nms_threshold: float = 0.3  # For merging overlapping detections from adjacent tiles
    
    # Output directory
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs" / "predictions")


@dataclass
class HeatmapConfig:
    """Configuration for heatmap generation."""
    
    # Output directory
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs" / "heatmaps")
    
    # Heatmap rendering
    gaussian_sigma: float = 30.0  # Gaussian blur sigma for density mapping
    colormap: str = "YlOrRd"  # matplotlib colormap
    
    # Geo-aggregation for multiple frames
    geospatial_enabled: bool = True
    geo_output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs" / "heatmaps" / "geo")


@dataclass
class UIConfig:
    """Configuration for Gradio UI."""
    
    host: str = "0.0.0.0"
    port: int = 7860
    share: bool = False


# Global configs
dataset_config = DatasetConfig()
training_config = TrainingConfig()
inference_config = InferenceConfig()
tile_config = TileInferenceConfig()
heatmap_config = HeatmapConfig()
ui_config = UIConfig()


def get_dataset_path(folder: str) -> Path:
    """Get absolute path to a dataset folder."""
    path = DATASET_ROOT / folder
    if not path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {path}")
    return path


def get_data_config_path(mode: str = "unified") -> Path:
    """Get path to YOLO data.yaml for given mode."""
    return PROCESSED_DIR / f"yolo_{mode}" / "data.yaml"


def ensure_output_dirs():
    """Ensure all output directories exist."""
    for config in [inference_config, heatmap_config, ui_config]:
        if hasattr(config, 'output_dir'):
            config.output_dir.mkdir(parents=True, exist_ok=True)
    
    (PROJECT_ROOT / "outputs" / "debug").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "models").mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("Project root:", PROJECT_ROOT)
    print("Dataset root:", DATASET_ROOT)
    print("Dataset config mode:", dataset_config.mode)
    print("Classes:", dataset_config.get_classes())

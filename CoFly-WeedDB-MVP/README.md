# 🌾 CoFly-WeedDB Weed Detection MVP

A complete MVP system for detecting weeds in UAV (drone) images using deep learning.

## Features

- ✅ **Automatic Dataset Analysis**: Inspects and validates the CoFly-WeedDB dataset
- ✅ **Mask to BBox Conversion**: Converts semantic segmentation masks to YOLO detection labels
- ✅ **Flexible Detection Modes**:
  - Unified: All weed types → single "weed" class (recommended for stable MVP)
  - Multiclass: Preserve individual weed types as separate classes
- ✅ **YOLOv8 Training**: Train detection models with automated dataset preparation
- ✅ **Multiple Inference Modes**:
  - Single image detection
  - Batch folder processing
  - **Tile-based inference** for large orthomosaics (automatic tiling + merging)
- ✅ **Heatmap Generation**:
  - Image-space density heatmaps
  - Geographic aggregation (using coordinates from image filenames)
  - Interactive Folium maps
- ✅ **Multi-format Export**:
  - JSON detection results
  - CSV tabular format
  - HTML visualization
  - YOLO txt format for compatibility
- ✅ **Web UI**: Gradio-based web interface for easy inference
- ✅ **Geographic Support**: Automatic extraction and visualization of coordinates from UAV image metadata

## Project Structure

```
CoFly-WeedDB-MVP/
├── app/                              # Main application modules
│   ├── __init__.py
│   ├── config.py                     # Configuration (paths, parameters, modes)
│   ├── utils.py                      # General utility functions
│   ├── geo_utils.py                  # Geographic utilities
│   ├── dataset_inspector.py          # Dataset analysis
│   ├── mask_to_bbox.py               # Mask → YOLO bbox conversion
│   ├── prepare_dataset.py            # YOLO dataset structure preparation
│   ├── train.py                      # YOLOv8 training
│   ├── inference.py                  # Single/batch inference
│   ├── tile_inference.py             # Tile-based large image processing
│   ├── heatmap.py                    # Heatmap generation
│   ├── exporters.py                  # Multi-format result export
│   └── ui.py                         # Gradio web interface
├── data/
│   ├── processed/
│   │   ├── yolo_unified/             # YOLO-format unified dataset
│   │   │   ├── images/train
│   │   │   ├── images/val
│   │   │   ├── labels/train
│   │   │   ├── labels/val
│   │   │   └── data.yaml
│   │   └── yolo_multiclass/          # YOLO-format multiclass dataset
│   └── samples/                      # Sample outputs
├── models/                           # Trained model weights
│   ├── unified/
│   │   ├── best.pt
│   │   └── last.pt
│   └── multiclass/
├── outputs/
│   ├── predictions/                  # Inference results
│   ├── heatmaps/                     # Heatmap visualizations
│   ├── exports/                      # Exported results (JSON/CSV/HTML)
│   └── debug/                        # Debug reports and previews
├── scripts/                          # Utility scripts
│   ├── run_prepare.py
│   ├── run_train.py
│   ├── run_infer.py
│   └── run_ui.py
├── main.py                           # Main entry point (CLI)
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Quick Start

### 1. Installation

```bash
# Clone or download the project
cd CoFly-WeedDB-MVP

# Install dependencies
pip install -r requirements.txt
```

**Note:** The project expects the `CoFly-WeedDB` folder to exist at the same level as this project:
```
ДИПЛОМКА/
├── CoFly-WeedDB/                     # Pre-existing dataset
└── CoFly-WeedDB-MVP/                 # This project
```

### 2. Run Complete Pipeline

```bash
# Run entire pipeline from dataset inspection to inference
python main.py all
```

This will:
1. Inspect the dataset
2. Convert masks to YOLO format
3. Prepare YOLO datasets
4. Train a YOLOv8 model
5. Run test inference

### 3. Individual Steps

```bash
# Step 1: Inspect dataset
python main.py inspect

# Step 2: Convert masks to bounding boxes
python main.py convert

# Step 3: Prepare YOLO dataset structure
python main.py prepare

# Step 4: Train model
python main.py train

# Step 5: Run inference
python main.py infer

# Step 6: Launch web UI
python main.py ui
```

## Detailed Usage

### Dataset Inspection

```bash
python main.py inspect
```

Outputs:
- `outputs/debug/dataset_report.txt` - Comprehensive dataset statistics
- `outputs/debug/dataset_preview/` - Sample visualizations with mask overlays

### Mask to BBox Conversion

```bash
# Unified mode (all weeds → single class)
cd app && python mask_to_bbox.py --mode unified --split train_split1.txt

# Multiclass mode (preserve weed types 0, 1, 2)
python mask_to_bbox.py --mode multiclass --split train_split1.txt
```

Options:
- `--mode`: "unified" (default) or "multiclass"
- `--split`: Split file name (default "train_split1.txt")
- `--min-area`: Minimum component area in pixels (default 50)

Outputs:
- `outputs/converted_labels/` - YOLO txt label files
- `outputs/debug/converted_labels_preview/` - Visual validation (first 20 images)

### Training

```bash
cd app && python train.py --mode unified --epochs 30 --batch 16
```

Options:
- `--mode`: "unified" (default) or "multiclass"
- `--epochs`: Number of training epochs (default 30)
- `--batch`: Batch size (default 16)
- `--imgsz`: Image size (default 640)
- `--device`: Device (default "0" for first GPU, or "cpu")
- `--validate`: Run validation after training

Outputs:
- `models/unified/best.pt` - Best model weights
- Training results and plots in `models/unified/train_results/`

### Inference

#### Single Image

```bash
cd app && python inference.py --image path/to/image.png
```

#### Folder of Images

```bash
cd app && python inference.py --folder path/to/images_dir
```

#### Large Image (Tile-based)

```bash
cd app && python tile_inference.py --image large_field.png --tile-size 640 --overlap 0.3
```

Options:
- `--weights`: Path to model (default "models/unified/best.pt")
- `--conf`: Confidence threshold (default 0.25)
- `--output`: Output directory for results
- `--tile-size`: Tile size for large images (default 640)
- `--overlap`: Tile overlap ratio (default 0.3)

Outputs:
- `outputs/predictions/detections.json` - JSON format
- `outputs/predictions/detections.csv` - CSV format
- `outputs/predictions/vis_*.png` - Visualization images

### Web UI

```bash
python main.py ui
```

Opens interactive Gradio interface at `http://0.0.0.0:7860`

Features:
- Single image detection
- Large image tile-based processing
- Real-time confidence threshold adjustment
- JSON/CSV results preview
- Download results

### Heatmap Generation

```bash
cd app && python heatmap.py --results outputs/predictions/detections.json --image path/to/base_image.png
```

For geographic heatmaps:
```bash
python heatmap.py --results outputs/predictions/detections.json --geo
```

Outputs:
- `outputs/heatmaps/` - Heatmap visualizations
- `outputs/heatmaps/weed_map.html` - Interactive folium map (if geocoordinates available)

## Detection Modes

### Unified Mode (Recommended for MVP)

All weed class types (0, 1, 2) are merged into a single detection class "weed".

**Advantages:**
- More training samples per class
- More stable training
- Simpler predictions
- Better for initial MVP demonstration

**Configuration:** Default mode, use `--mode unified`

### Multiclass Mode

Preserves individual weed types as separate classes:
- Class 0: weed_type_0
- Class 1: weed_type_1
- Class 2: weed_type_2

**Advantages:**
- Detailed weed analysis
- Can prioritize specific weed types

**Limitations:**
- Fewer samples per class
- May require more data/training

**Configuration:** Use `--mode multiclass`

## Data Formats

### YOLO Format Labels

```
<class_id> <x_center> <y_center> <width> <height>
```

Example:
```
0 0.5 0.5 0.3 0.4
```

### Detection JSON Output

```json
[
  {
    "image_name": "ID_00048_...png",
    "image_width": 1280,
    "image_height": 960,
    "class_id": 0,
    "class_name": "weed",
    "confidence": 0.87,
    "bbox_xyxy": [x1, y1, x2, y2],
    "bbox_xywh": [xc, yc, w, h],
    "center_px": [cx, cy],
    "latitude": 39.542124,
    "longitude": 22.644429
  }
]
```

### Detection CSV Output

```
class_id,class_name,confidence,x1,y1,x2,y2,lat,lon
0,weed,0.87,320,240,480,360,39.542124,22.644429
```

## Geographic Features

### Coordinate Extraction

The system automatically extracts Lat/Lon/Alt from UAV image filenames using the pattern:
```
[Lat=39.542124,Lon=22.644429,Alt=4.9]
```

These coordinates are:
- Automatically extracted and added to detection results
- Used for geographic aggregation in heatmaps
- Included in exported data

### Geographic Heatmaps

When processing multiple images with coordinates, the system can:
- Aggregate weed detections by location
- Generate `geo_aggregation.csv` with per-location statistics
- Create interactive Folium maps showing weed density by location

### Future GeoTIFF Support

The `geo_utils.py` module includes placeholder support for GeoTIFF files for pixel-perfect geographic mapping (currently optional for MVP).

## Output Description

### Key Output Directories

- **`outputs/debug/`** - Diagnostic reports, logs, and preview images
  - `dataset_report.txt` - Dataset summary
  - `dataset_preview/` - Masked image visualizations
  - `converted_labels_preview/` - BBox conversion validation
  - `*.log` - Detailed logs per module

- **`outputs/predictions/`** - Inference results
  - `detections.json` - Detection results in JSON
  - `detections.csv` - Detection results in CSV
  - `vis_*.png` - Visualization images

- **`outputs/heatmaps/`** - Heatmap visualizations
  - `*_heatmap.png` - Image-space heatmaps
  - `weed_map.html` - Interactive geographic map
  - `geo_aggregation.json/csv` - Geographic aggregation data

- **`models/unified/`** - Trained model weights
  - `best.pt` - Best model during training
  - `last.pt` - Last checkpoint
  - `train_results/` - Training plots and metrics

- **`data/processed/yolo_unified/`** - YOLO-format dataset
  - `images/train/`, `images/val/` - Image files
  - `labels/train/`, `labels/val/` - Label files
  - `data.yaml` - Dataset configuration

## Configuration

Edit `app/config.py` to customize:

```python
# Dataset mode
dataset_config.mode = "unified"  # or "multiclass"

# Training parameters
training_config.epochs = 30
training_config.batch_size = 16
training_config.imgsz = 640

# Inference parameters
inference_config.conf_threshold = 0.25

# Tile inference settings
tile_config.tile_size = 640
tile_config.overlap_ratio = 0.3

# Heatmap settings
heatmap_config.gaussian_sigma = 30.0
```

## Troubleshooting

### "Model not found" when running inference/UI

```bash
# Train a model first
python main.py train
```

### CUDA out of memory during training

Reduce batch size:
```bash
python train.py --batch 8  # or lower
```

Or use CPU:
```bash
python train.py --device cpu
```

### Missing dependencies

```bash
# Reinstall all dependencies
pip install -r requirements.txt --upgrade
```

### Dataset not found

Ensure the `CoFly-WeedDB` folder exists at the correct location:
```
ДИПЛОМКА/
├── CoFly-WeedDB/                     # Must exist here
└── CoFly-WeedDB-MVP/
```

### Gradio UI not accessible

Check the port:
```bash
python main.py ui --port 7860
```

Then navigate to `http://0.0.0.0:7860` (or `http://localhost:7860`)

## MVP Limitations & Future Work

### Current Limitations

- ✅ Unified mode only uses single weed class (intentional for stability)
- ✅ Tile-based inference uses simple overlap merging (not sophisticated stitching)
- ✅ GeoTIFF support is placeholder (not critical for MVP)
- ✅ No orthomosaic stitching (assumes pre-stitched inputs)

### Potential Enhancements

- [ ] Advanced NMS/merging for tile boundaries
- [ ] True GeoTIFF support with rasterio
- [ ] Automatic orthomosaic generation from image sequence
- [ ] Real-time inference on drone streams
- [ ] ROI-based inference optimization
- [ ] Ensemble methods for improved accuracy
- [ ] Quantization for edge deployment

## Citation

If you use this MVP in your research, please cite:

- **CoFly-WeedDB Dataset**: [Original paper/DOI]
- **YOLOv8**: Ultralytics (https://github.com/ultralytics/ultralytics)

## License

[Specify your license here]

## Contact & Support

For issues or questions:
- Check the `outputs/debug/` folder for detailed logs
- Review this README's Troubleshooting section
- Examine generated dataset report: `outputs/debug/dataset_report.txt`

## Timeline

**MVP Development Summary:**

1. ✅ Dataset inspection and validation
2. ✅ Mask to YOLO format conversion
3. ✅ YOLO dataset structure preparation
4. ✅ YOLOv8 model training pipeline
5. ✅ Single image and batch inference
6. ✅ Tile-based large image processing
7. ✅ Heatmap generation (image-space and geographic)
8. ✅ Multi-format result exports
9. ✅ Web UI (Gradio)
10. ✅ Documentation and CLI

---

**Status:** MVP Complete ✅

Ready for deployment and academic paper publication.

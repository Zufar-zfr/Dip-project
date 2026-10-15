# 🚀 QUICK START GUIDE

Get your weed detection system running in 5 minutes!

## Prerequisites

- Python 3.8+
- NVIDIA GPU (recommended) or CPU (slower)
- CoFly-WeedDB dataset folder at the same level as this project

## Installation

```bash
# 1. Navigate to project
cd CoFly-WeedDB-MVP

# 2. Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt
```

**⏱️ Installation time: 5-10 minutes** (depends on internet speed and GPU drivers)

## Running the Pipeline

### Option A: Complete Pipeline (Automated)

```bash
python main.py all
```

This will automatically:
1. Analyze the dataset
2. Convert masks to bbox
3. Prepare YOLO dataset
4. Train model (~20-30 min on GPU, slower on CPU)
5. Run test inference

**Total time: 30-60 minutes** depending on hardware

### Option B: Step-by-Step

```bash
# Step 1: Inspect dataset (1 min)
python main.py inspect

# Step 2: Convert masks (2 min)
python main.py convert

# Step 3: Prepare dataset (1 min)
python main.py prepare

# Step 4: Train model (20-30 min)
python main.py train

# Step 5: Test inference (1 min)
python main.py infer

# Step 6: Launch web UI (instant)
python main.py ui
```

### Option C: Skip to UI (If model already trained)

```bash
python main.py ui
```

Then open: **http://localhost:7860**

---

## 📊 Expected Results

### After `python main.py inspect`
```
outputs/debug/
├── dataset_report.txt          # Statistics about dataset
└── dataset_preview/            # Sample images with masks
```

### After `python main.py convert`
```
outputs/converted_labels/       # YOLO bbox format labels
outputs/debug/converted_labels_preview/  # Validation images
```

### After `python main.py prepare`
```
data/processed/yolo_unified/
├── images/train/               # Training images
├── images/val/                 # Validation images
├── labels/train/               # Training labels
├── labels/val/                 # Validation labels
└── data.yaml                   # Dataset config
```

### After `python main.py train`
```
models/unified/
├── best.pt                     # Best model weights (use this!)
├── last.pt                     # Last checkpoint
└── train_results/              # Training plots and metrics
```

### After `python main.py infer`
```
outputs/predictions/
├── detections.json             # Detection results
├── detections.csv              # Results in CSV
└── vis_*.png                   # Visualization images
```

---

## 🖥️ Web Interface

After running `python main.py ui`, you can:

1. **Upload an image** → See detections in real-time
2. **Adjust confidence** → Filter low-confidence detections  
3. **Process large images** → Automatic tiling for big orthomosaics
4. **Download results** → JSON, CSV, or visualizations

**URL:** http://0.0.0.0:7860 or http://localhost:7860

---

## ⚙️ Configuration

Default settings work great for MVP. To customize:

Edit `app/config.py`:

```python
# Training parameters
training_config.epochs = 30        # More epochs = longer training
training_config.batch_size = 16    # Smaller if GPU out of memory
training_config.imgsz = 640        # Image size for training

# Inference parameters
inference_config.conf_threshold = 0.25  # Lower = more detections

# Tile inference
tile_config.tile_size = 640       # Use 512 or 1024 for different image sizes
tile_config.overlap_ratio = 0.3   # More overlap = slower but better merging
```

---

## 📁 Project Structure

```
CoFly-WeedDB-MVP/              ← You are here
├── app/                       ← All code modules
├── data/processed/            ← YOLO datasets (auto-created)
├── models/                    ← Trained weights (auto-created)
├── outputs/                   ← Results (auto-created)
├── scripts/                   ← Helper scripts
├── main.py                    ← Main entry point
├── requirements.txt           ← Python dependencies
├── README.md                  ← Full documentation
└── PROJECT_SUMMARY.md         ← Detailed summary
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Module not found" | Run `pip install -r requirements.txt` |
| GPU out of memory | Use `--batch 8` or `--device cpu` |
| Dataset not found | Check CoFly-WeedDB folder at same level |
| UI not loading | Try `http://localhost:7860` instead of 0.0.0.0 |
| Training is slow | Normal! Depending on GPU, can take 20-60 min |

---

## 📞 Getting Help

1. **Check logs:** `outputs/debug/*.log`
2. **Read full docs:** See `README.md`
3. **Read summary:** See `PROJECT_SUMMARY.md`
4. **Check dataset report:** `outputs/debug/dataset_report.txt`

---

## ✅ Verify Installation

Test that everything works:

```bash
# Should print version info
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "from ultralytics import YOLO; print('YOLOv8: OK')"
```

---

## 🎯 Common Use Cases

### Use Case 1: Evaluate Dataset
```bash
python main.py inspect
# Opens outputs/debug/dataset_report.txt
```

### Use Case 2: Train Custom Model
```bash
cd app
python train.py --mode unified --epochs 50
```

### Use Case 3: Detect Weeds in Your Images
```bash
cd app
python inference.py --image your_image.png
```

### Use Case 4: Process Large Field Image
```bash
cd app
python tile_inference.py --image large_field.tif --overlap 0.3
```

### Use Case 5: Interactive Web Demo
```bash
python main.py ui
# Open http://localhost:7860 in browser
```

---

## 🎉 You're All Set!

Everything is configured and ready to go. Pick your use case above and start!

**Questions?** Check the full README.md for detailed documentation.

**Ready?** Start with: `python main.py all`

🌾 Happy weed detecting! 🚀

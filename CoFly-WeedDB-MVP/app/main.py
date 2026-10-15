#!/usr/bin/env python
"""
Main CLI entry point for CoFly WeedDB MVP.
Provides unified commands for the entire pipeline.
"""

import argparse
import logging
from pathlib import Path

# Import all modules
from config import PROJECT_ROOT
from utils import setup_logging
from dataset_inspector import DatasetInspector
from mask_to_bbox import MaskToBBoxConverter
from prepare_dataset import YOLODatasetPreparer
from train import WeedDetectionTrainer
from inference import WeedDetector
from heatmap import HeatmapGenerator
from ui import WeedDetectionUI, launch_ui

logger = setup_logging("Main", PROJECT_ROOT / "outputs" / "debug" / "main.log")


def cmd_inspect(args):
    """Inspect dataset."""
    logger.info("Starting dataset inspection...")
    inspector = DatasetInspector(mode=args.mode)
    inspector.inspect_all()
    print("✅ Dataset inspection complete. Check outputs/debug/dataset_report.txt")


def cmd_convert(args):
    """Convert masks to YOLO format."""
    logger.info(f"Converting masks (mode={args.mode})...")
    converter = MaskToBBoxConverter(mode=args.mode)
    converter.convert_all(split_file=args.split)
    print("✅ Mask conversion complete. Check outputs/converted_labels/")


def cmd_prepare(args):
    """Prepare YOLO dataset structure."""
    logger.info(f"Preparing YOLO dataset (mode={args.mode})...")
    preparer = YOLODatasetPreparer(mode=args.mode)
    preparer.prepare()
    print("✅ Dataset preparation complete. Check data/processed/yolo_{}/".format(args.mode))


def cmd_train(args):
    """Train model."""
    logger.info(f"Starting training (mode={args.mode}, epochs={args.epochs})...")
    trainer = WeedDetectionTrainer(mode=args.mode)
    trainer.epochs = args.epochs
    trainer.batch_size = args.batch
    trainer.imgsz = args.imgsz
    trainer.device = args.device
    
    best_model = trainer.train()
    if best_model:
        print(f"✅ Training complete! Model: {best_model}")
    else:
        print("❌ Training failed")


def cmd_infer(args):
    """Run inference."""
    logger.info("Running inference...")
    inference = WeedDetector(
        weights_path=args.weights,
        conf_threshold=args.conf
    )
    
    if args.image:
        inference.detect_image(args.image)
        print(f"✅ Inference complete. Output: outputs/predictions/")
    elif args.folder:
        inference.detect_folder(args.folder)
        print(f"✅ Batch inference complete. Output: outputs/predictions/")
    else:
        # Default: infer on entire dataset
        dataset_dir = PROJECT_ROOT.parent / "CoFly-WeedDB" / "images"
        inference.detect_folder(str(dataset_dir))
        print(f"✅ Inference on full dataset complete. Output: outputs/predictions/")


def cmd_heatmap(args):
    """Generate heatmaps."""
    logger.info("Generating heatmaps...")
    heatmap_gen = HeatmapGenerator()
    heatmap_gen.generate_all()
    print("✅ Heatmap generation complete. Check outputs/")


def cmd_ui(args):
    """Launch web UI."""
    logger.info("Launching Gradio UI...")
    print("🚀 Starting Gradio interface...")
    model_path = str(PROJECT_ROOT / "models" / "unified" / "best.pt")
    port = getattr(args, 'port', 7860)
    launch_ui(model_path=model_path, share=args.share, server_port=port)


def cmd_all(args):
    """Run complete pipeline."""
    logger.info("=" * 60)
    logger.info("RUNNING COMPLETE PIPELINE")
    logger.info("=" * 60)
    
    # 1. Inspect
    print("\n[1/6] Inspecting dataset...")
    try:
        cmd_inspect(args)
    except Exception as e:
        print(f"⚠️  Inspection failed: {e}")
    
    # 2. Convert
    print("\n[2/6] Converting masks...")
    try:
        cmd_convert(args)
    except Exception as e:
        print(f"⚠️  Conversion failed: {e}")
    
    # 3. Prepare
    print("\n[3/6] Preparing dataset...")
    try:
        cmd_prepare(args)
    except Exception as e:
        print(f"⚠️  Preparation failed: {e}")
    
    # 4. Train
    print("\n[4/6] Training model...")
    try:
        cmd_train(args)
    except Exception as e:
        print(f"⚠️  Training failed: {e}")
    
    # 5. Infer
    print("\n[5/6] Running inference...")
    try:
        cmd_infer(args)
    except Exception as e:
        print(f"⚠️  Inference failed: {e}")
    
    # 6. Heatmap
    print("\n[6/6] Generating heatmaps...")
    try:
        cmd_heatmap(args)
    except Exception as e:
        print(f"⚠️  Heatmap generation failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="CoFly WeedDB MVP - UAV Weed Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py inspect              # Inspect dataset
  python main.py convert --split train_split1.txt  # Convert masks
  python main.py prepare              # Prepare YOLO dataset
  python main.py train --epochs 30    # Train model
  python main.py infer                # Run inference on full dataset
  python main.py infer --image image.jpg  # Single image
  python main.py heatmap              # Generate heatmaps
  python main.py ui                   # Launch web interface
  python main.py all                  # Complete pipeline
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Inspect
    p = subparsers.add_parser("inspect", help="Inspect dataset")
    p.add_argument("--mode", default="unified", choices=["unified", "multiclass"])
    p.set_defaults(func=cmd_inspect)
    
    # Convert
    p = subparsers.add_parser("convert", help="Convert masks to YOLO format")
    p.add_argument("--mode", default="unified", choices=["unified", "multiclass"])
    p.add_argument("--split", default="train_split1.txt", help="Split file to use")
    p.set_defaults(func=cmd_convert)
    
    # Prepare
    p = subparsers.add_parser("prepare", help="Prepare YOLO dataset")
    p.add_argument("--mode", default="unified", choices=["unified", "multiclass"])
    p.set_defaults(func=cmd_prepare)
    
    # Train
    p = subparsers.add_parser("train", help="Train model")
    p.add_argument("--mode", default="unified", choices=["unified", "multiclass"])
    p.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    p.add_argument("--batch", type=int, default=16, help="Batch size")
    p.add_argument("--imgsz", type=int, default=640, help="Image size")
    p.add_argument("--device", default="cpu", help="Device (cpu or 0,1,2...)")
    p.set_defaults(func=cmd_train)
    
    # Infer
    p = subparsers.add_parser("infer", help="Run inference")
    p.add_argument("--weights", default=str(PROJECT_ROOT / "models" / "unified" / "best.pt"),
                   help="Path to model weights")
    p.add_argument("--image", help="Single image path")
    p.add_argument("--folder", help="Folder with images")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.set_defaults(func=cmd_infer)
    
    # Heatmap
    p = subparsers.add_parser("heatmap", help="Generate heatmaps")
    p.set_defaults(func=cmd_heatmap)
    
    # UI
    p = subparsers.add_parser("ui", help="Launch web interface")
    p.add_argument("--port", type=int, default=7860, help="Port to run on")
    p.add_argument("--share", action="store_true", help="Share link")
    p.set_defaults(func=cmd_ui)
    
    # All
    p = subparsers.add_parser("all", help="Run complete pipeline")
    p.add_argument("--mode", default="unified", choices=["unified", "multiclass"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_all)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        args.func(args)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()

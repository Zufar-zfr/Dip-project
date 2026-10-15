"""
Main entry point for the weed detection MVP.
Provides a simple CLI for running different pipeline stages.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add app directory to path
APP_DIR = Path(__file__).parent / "app"
sys.path.insert(0, str(APP_DIR))

from config import PROJECT_ROOT, ensure_output_dirs, get_dataset_path
from utils import setup_logging, load_split_file

# Setup logging
main_logger = setup_logging("WeedDetectionMVP", PROJECT_ROOT / "outputs" / "debug" / "main.log")


def command_inspect():
    """Run dataset inspection."""
    from dataset_inspector import DatasetInspector
    
    print("\n" + "="*60)
    print("STEP 1: Dataset Inspection")
    print("="*60)
    
    inspector = DatasetInspector()
    inspector.inspect()
    
    print("\n[OK] Dataset inspection complete!")
    print(f"Report: {PROJECT_ROOT / 'outputs' / 'debug' / 'dataset_report.txt'}")


def command_convert():
    """Convert masks to YOLO bbox format."""
    from mask_to_bbox import MaskToBBoxConverter
    
    print("\n" + "="*60)
    print("STEP 2: Convert Masks to Bounding Boxes")
    print("="*60)
    
    # Convert for both modes
    for mode in ["unified", "multiclass"]:
        print(f"\n--- Converting {mode} mode ---")
        converter = MaskToBBoxConverter(mode=mode, min_area=50)
        for split_file in ["train_split1.txt", "val_split1.txt", "test_split1.txt"]:
            print(f"  -> split: {split_file}")
            converter.convert_all(split_file=split_file)
    
    print("\n[OK] Conversion complete!")


def command_prepare():
    """Prepare YOLO dataset structure."""
    from prepare_dataset import YOLODatasetPreparer
    
    print("\n" + "="*60)
    print("STEP 3: Prepare YOLO Dataset")
    print("="*60)
    
    # Prepare both modes
    for mode in ["unified", "multiclass"]:
        print(f"\n--- Preparing {mode} mode ---")
        preparer = YOLODatasetPreparer(mode=mode)
        preparer.prepare(
            train_split="train_split1.txt",
            val_split="val_split1.txt",
            test_split="test_split1.txt",
        )
    
    print("\n[OK] Dataset preparation complete!")


def command_train():
    """Train YOLOv8 model."""
    from train import WeedDetectionTrainer
    
    print("\n" + "="*60)
    print("STEP 4: Train Model")
    print("="*60)
    
    print("\nTraining unified mode model...")
    trainer = WeedDetectionTrainer(mode="unified")
    best_model = trainer.train()
    
    if best_model:
        print(f"\n[OK] Training complete!")
        print(f"Best model: {best_model}")


def command_infer():
    """Run inference on test image."""
    from inference import WeedDetector
    import tempfile
    import shutil
    from pathlib import Path
    
    print("\n" + "="*60)
    print("STEP 5: Run Inference")
    print("="*60)
    
    model_path = PROJECT_ROOT / "models" / "unified" / "best.pt"
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("Please train a model first using: python main.py train")
        return
    
    # Get a test image
    test_split_path = PROJECT_ROOT.parent / "CoFly-WeedDB" / "test_split1.txt"
    if not test_split_path.exists():
        print(f"[ERROR] Test split not found: {test_split_path}")
        return
    
    test_files = load_split_file(str(test_split_path))
    if not test_files:
        print("[ERROR] No test files found")
        return
    
    # Use first test image
    test_image = get_dataset_path("images") / test_files[0]
    
    print(f"Using test image: {test_image.name}")
    
    # Run inference
    detector = WeedDetector(str(model_path), conf_threshold=0.25)
    result = detector.detect_image(str(test_image))
    
    if result:
        print(f"\n[OK] Inference complete!")
        print(f"Detections: {len(result['detections'])}")
        
        # Save results
        output_dir = PROJECT_ROOT / "outputs" / "predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        json_path = output_dir / "test_results.json"
        csv_path = output_dir / "test_results.csv"
        vis_path = output_dir / "test_results_visualization.png"
        
        from utils import save_detections_json, save_detections_csv, save_image
        save_detections_json(result['detections'], str(json_path))
        save_detections_csv(result['detections'], str(csv_path))
        
        if result['visualization'] is not None:
            save_image(result['visualization'], str(vis_path))
        
        print(f"Results saved to: {output_dir}")


def command_ui():
    """Launch Gradio UI."""
    from ui import launch_ui
    
    print("\n" + "="*60)
    print("Launching Web UI")
    print("="*60)
    
    model_path = PROJECT_ROOT / "models" / "unified" / "best.pt"
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("Please train a model first using: python main.py train")
        return
    
    print(f"\nUsing model: {model_path}")
    print(f"UI will be available at http://0.0.0.0:7860")
    print("Press Ctrl+C to stop.\n")
    
    launch_ui(model_path=str(model_path), share=False)


def command_all():
    """Run complete pipeline."""
    print("\n" + "="*60)
    print("Running Complete Pipeline")
    print("="*60)
    
    stages = [
        ("Inspect Dataset", command_inspect),
        ("Convert Masks", command_convert),
        ("Prepare Dataset", command_prepare),
        ("Train Model", command_train),
        ("Run Inference", command_infer),
    ]
    
    for stage_name, stage_func in stages:
        try:
            stage_func()
        except Exception as e:
            print(f"\n[ERROR] Error in {stage_name}: {e}")
            import traceback
            traceback.print_exc()
            print(f"\nSkipping remaining stages after error in {stage_name}")
            break
    
    print("\n" + "="*60)
    print("[OK] Pipeline complete!")
    print("="*60)
    print("\nNext steps:")
    print("  - Review detection results in: outputs/predictions/")
    print("  - Launch UI: python main.py ui")
    print("  - Run on large images: python main.py infer --large <path_to_image>")


def main():
    """Main CLI."""
    ensure_output_dirs()
    
    parser = argparse.ArgumentParser(
        description="CoFly-WeedDB MVP - Weed Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py all              # Run complete pipeline
  python main.py inspect          # Inspect dataset
  python main.py convert          # Convert masks to bbox
  python main.py prepare          # Prepare YOLO dataset
  python main.py train            # Train model
  python main.py infer            # Run inference
  python main.py ui               # Launch web UI
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Subcommands
    subparsers.add_parser("inspect", help="Inspect dataset structure")
    subparsers.add_parser("convert", help="Convert masks to bounding boxes")
    subparsers.add_parser("prepare", help="Prepare YOLO dataset")
    subparsers.add_parser("train", help="Train YOLOv8 model")
    subparsers.add_parser("infer", help="Run inference on test image")
    subparsers.add_parser("ui", help="Launch Gradio web UI")
    subparsers.add_parser("all", help="Run complete pipeline")
    
    args = parser.parse_args()
    
    # Map commands to functions
    command_map = {
        "inspect": command_inspect,
        "convert": command_convert,
        "prepare": command_prepare,
        "train": command_train,
        "infer": command_infer,
        "ui": command_ui,
        "all": command_all,
    }
    
    if args.command in command_map:
        try:
            command_map[args.command]()
        except Exception as e:
            print(f"\n[ERROR] Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

"""
Training module for YOLOv8 weed detection model.
"""

import logging
from pathlib import Path
from datetime import datetime

from config import PROJECT_ROOT, training_config, get_data_config_path, dataset_config
from utils import setup_logging

# Try to import ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("ultralytics not installed. Training will not be available.")


logger = setup_logging("Training", PROJECT_ROOT / "outputs" / "debug" / "training.log")


class WeedDetectionTrainer:
    """Train YOLOv8 model for weed detection."""
    
    def __init__(self, mode: str = "unified"):
        """
        Initialize trainer.
        
        Args:
            mode: "unified" or "multiclass"
        """
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics is not installed. Please install it: pip install ultralytics")
        
        self.mode = mode
        self.data_yaml = get_data_config_path(mode)
        self.model_name = training_config.model_name
        self.imgsz = training_config.imgsz
        self.epochs = training_config.epochs
        self.batch_size = training_config.batch_size
        self.device = training_config.device
        
        # Output directory for results
        self.output_dir = PROJECT_ROOT / "models" / mode
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized trainer for {mode} mode")
        logger.info(f"  Data YAML: {self.data_yaml}")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Image size: {self.imgsz}")
        logger.info(f"  Epochs: {self.epochs}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Device: {self.device}")
    
    def train(self):
        """
        Train the model.
        
        Returns:
            Path to best model weights
        """
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"Data YAML not found: {self.data_yaml}")
        
        logger.info("=" * 60)
        logger.info(f"Starting training for {self.mode} mode")
        logger.info(f"Training started at {datetime.now()}")
        logger.info("=" * 60)
        
        try:
            # Load base model
            logger.info(f"Loading base model: {self.model_name}")
            model = YOLO(f"{self.model_name}.pt")
            
            # Train
            logger.info("Starting training...")
            results = model.train(
                data=str(self.data_yaml),
                epochs=self.epochs,
                imgsz=self.imgsz,
                batch=self.batch_size,
                device=self.device,
                project=str(self.output_dir),
                name="train_results",
                patience=10,  # Early stopping patience
                save=True,
                plots=True,
                verbose=True
            )
            
            logger.info("Training complete!")
            logger.info(f"Training finished at {datetime.now()}")
            
            # Ultralytics may auto-increment the run directory name, so read it from results.
            save_dir = Path(results.save_dir)
            best_model_path = save_dir / "weights" / "best.pt"
            last_model_path = save_dir / "weights" / "last.pt"
            
            if best_model_path.exists():
                logger.info(f"Best model saved: {best_model_path}")
                
                # Copy to standard location
                import shutil
                standard_best = self.output_dir / "best.pt"
                standard_last = self.output_dir / "last.pt"
                
                shutil.copy2(best_model_path, standard_best)
                shutil.copy2(last_model_path, standard_last)
                
                logger.info(f"Copied to: {standard_best}")
                
                return standard_best
            else:
                logger.error("Best model not found!")
                return None
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def validate(self, weights_path: str):
        """
        Validate model on test set.
        
        Args:
            weights_path: path to model weights
        """
        logger.info(f"Validating model: {weights_path}")
        
        try:
            model = YOLO(weights_path)
            metrics = model.val(
                data=str(self.data_yaml),
                device=self.device,
                imgsz=self.imgsz
            )
            
            logger.info(f"Validation metrics: {metrics}")
            return metrics
        
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise


def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLOv8 weed detection model")
    parser.add_argument("--mode", choices=["unified", "multiclass"], default="unified",
                       help="Training mode")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="Image size")
    parser.add_argument("--model", default="yolov8n",
                       help="Base YOLO model name without .pt, e.g. yolov8n or yolov8s")
    parser.add_argument("--device", default="cpu",
                       help="Device (0, 1, cpu, etc.)")
    parser.add_argument("--validate", action="store_true",
                       help="Validate after training")
    
    args = parser.parse_args()
    
    # Update config  
    training_config.epochs = args.epochs
    training_config.batch_size = args.batch
    training_config.imgsz = args.imgsz
    training_config.model_name = args.model
    training_config.device = args.device
    
    # Create trainer and train
    trainer = WeedDetectionTrainer(mode=args.mode)
    best_model_path = trainer.train()
    
    if best_model_path and args.validate:
        logger.info("\nValidating trained model...")
        trainer.validate(str(best_model_path))
    
    print("\n" + "=" * 60)
    print("Training complete!")
    if best_model_path:
        print(f"Best model: {best_model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

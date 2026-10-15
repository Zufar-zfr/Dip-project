"""
Prepare YOLO dataset structure from split files and converted labels.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import yaml

from augment_dataset import SemiSyntheticAugmenter
from config import PROJECT_ROOT, DATASET_ROOT, get_dataset_path, dataset_config
from utils import setup_logging, load_split_file


logger = setup_logging("PrepareDataset", PROJECT_ROOT / "outputs" / "debug" / "prepare.log")


class YOLODatasetPreparer:
    """Prepare YOLO-compatible dataset structure."""
    
    def __init__(self, mode: str = "unified"):
        """
        Initialize preparer.
        
        Args:
            mode: "unified" or "multiclass"
        """
        self.mode = mode
        self.images_dir = get_dataset_path("images")
        self.dataset_output = PROJECT_ROOT / "data" / "processed" / f"yolo_{mode}"
        self.converted_labels_dir = PROJECT_ROOT / "outputs" / "converted_labels" / mode
        
        # YOLO structure
        self.images_train_dir = self.dataset_output / "images" / "train"
        self.images_val_dir = self.dataset_output / "images" / "val"
        self.images_test_dir = self.dataset_output / "images" / "test"
        self.labels_train_dir = self.dataset_output / "labels" / "train"
        self.labels_val_dir = self.dataset_output / "labels" / "val"
        self.labels_test_dir = self.dataset_output / "labels" / "test"
        
        # Create directories
        for d in [
            self.images_train_dir,
            self.images_val_dir,
            self.images_test_dir,
            self.labels_train_dir,
            self.labels_val_dir,
            self.labels_test_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'train_images': 0,
            'val_images': 0,
            'train_labels': 0,
            'val_labels': 0,
            'test_images': 0,
            'test_labels': 0,
            'missing_labels': 0,
            'failed_copies': 0
        }
    
    def prepare(
        self,
        train_split: str = "train_split1.txt",
        val_split: str = "val_split1.txt",
        test_split: str = "test_split1.txt",
        augment_train_copies: int = 0,
        augment_objects_per_image: Tuple[int, int] = (1, 3),
        augment_seed: int = 42,
        augment_min_area: int = 50,
    ):
        """
        Prepare YOLO dataset from split files.
        
        Args:
            train_split: name of training split file
            val_split: name of validation split file
            test_split: name of test split file
        """
        logger.info(f"Preparing YOLO dataset in {self.mode} mode")
        logger.info(f"Output directory: {self.dataset_output}")
        self._reset_output_dirs()
        
        # Load splits
        train_files = self._load_split(train_split)
        val_files = self._load_split(val_split)
        test_files = self._load_split(test_split)
        
        logger.info(f"Training files: {len(train_files)}")
        logger.info(f"Validation files: {len(val_files)}")
        logger.info(f"Test files: {len(test_files)}")
        
        # Copy training data
        logger.info("Copying training data...")
        for filename in train_files:
            self._copy_image_and_label(filename, self.images_train_dir, self.labels_train_dir)
            self.stats['train_images'] += 1
        
        # Copy validation data
        logger.info("Copying validation data...")
        for filename in val_files:
            self._copy_image_and_label(filename, self.images_val_dir, self.labels_val_dir)
            self.stats['val_images'] += 1

        # Copy test data
        logger.info("Copying test data...")
        for filename in test_files:
            self._copy_image_and_label(filename, self.images_test_dir, self.labels_test_dir)
            self.stats['test_images'] += 1

        if augment_train_copies > 0:
            logger.info(
                f"Generating semi-synthetic train augmentation: copies_per_image={augment_train_copies}, "
                f"objects_per_image={augment_objects_per_image}"
            )
            augmenter = SemiSyntheticAugmenter(
                mode=self.mode,
                min_area=augment_min_area,
                seed=augment_seed,
            )
            generated = augmenter.generate_augmented_samples(
                train_files,
                base_labels_dir=self.labels_train_dir,
                output_images_dir=self.images_train_dir,
                output_labels_dir=self.labels_train_dir,
                copies_per_image=augment_train_copies,
                objects_per_image=augment_objects_per_image,
            )
            self.stats['train_images'] += generated
            self.stats['train_labels'] += generated
            self.stats['augmented_train_images'] = generated
        
        # Generate data.yaml
        self._generate_data_yaml()
        
        logger.info(f"Dataset preparation complete. Stats: {self.stats}")

    def _reset_output_dirs(self) -> None:
        """Clear previous split contents to avoid stale files between rebuilds."""
        for directory in [
            self.images_train_dir,
            self.images_val_dir,
            self.images_test_dir,
            self.labels_train_dir,
            self.labels_val_dir,
            self.labels_test_dir,
        ]:
            for item in directory.iterdir():
                if item.is_file():
                    item.unlink()
    
    def _load_split(self, split_filename: str) -> List[str]:
        """Load filenames from split file."""
        split_path = DATASET_ROOT / split_filename
        if not split_path.exists():
            logger.error(f"Split file not found: {split_path}")
            return []
        
        filenames = load_split_file(str(split_path))
        return filenames
    
    def _copy_image_and_label(self, filename: str, img_dest_dir: Path, label_dest_dir: Path):
        """Copy image and corresponding label file."""
        try:
            # Source paths
            img_src = self.images_dir / filename
            label_src = self.converted_labels_dir / (Path(filename).stem + ".txt")
            
            # Destination paths
            img_dst = img_dest_dir / filename
            label_dst = label_dest_dir / (Path(filename).stem + ".txt")
            
            # Copy image
            if img_src.exists():
                shutil.copy2(img_src, img_dst)
            else:
                logger.warning(f"Image not found: {img_src}")
                self.stats['failed_copies'] += 1
                return
            
            # Copy label
            if label_src.exists():
                shutil.copy2(label_src, label_dst)
                split_name = label_dest_dir.name
                label_stat_key = f"{split_name}_labels"
                if label_stat_key in self.stats:
                    self.stats[label_stat_key] += 1
            else:
                logger.warning(f"Label not found: {label_src}")
                self.stats['missing_labels'] += 1
                # Still count the image copy as successful
        
        except Exception as e:
            logger.error(f"Error copying {filename}: {e}")
            self.stats['failed_copies'] += 1
    
    def _generate_data_yaml(self):
        """Generate YOLO data.yaml file."""
        # Get classes
        if self.mode == "unified":
            classes = {0: "weed"}
        else:
            classes = {
                0: "weed_type_0",
                1: "weed_type_1",
                2: "weed_type_2"
            }
        
        # Create data dict
        data_dict = {
            'path': str(self.dataset_output.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(classes),
            'names': classes
        }
        
        # Save as yaml
        yaml_path = self.dataset_output / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_dict, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved data.yaml to {yaml_path}")
        
        # Print content for verification
        logger.info("\nGenerated data.yaml content:")
        with open(yaml_path, 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                logger.info(f"  {line}")


def main():
    """Prepare YOLO datasets for both modes."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset")
    parser.add_argument("--mode", choices=["unified", "multiclass", "both"], 
                       default="unified",
                       help="Which mode(s) to prepare")
    parser.add_argument("--train-split", default="train_split1.txt",
                       help="Training split file")
    parser.add_argument("--val-split", default="val_split1.txt",
                       help="Validation split file")
    parser.add_argument("--test-split", default="test_split1.txt",
                       help="Test split file")
    parser.add_argument("--augment-train-copies", type=int, default=0,
                       help="How many semi-synthetic augmented variants to create per train image")
    parser.add_argument("--augment-min-objects", type=int, default=1,
                       help="Minimum pasted objects per augmented image")
    parser.add_argument("--augment-max-objects", type=int, default=3,
                       help="Maximum pasted objects per augmented image")
    parser.add_argument("--augment-seed", type=int, default=42,
                       help="Random seed for semi-synthetic augmentation")
    parser.add_argument("--augment-min-area", type=int, default=50,
                       help="Minimum object area used when extracting weed crops for augmentation")
    
    args = parser.parse_args()
    
    modes = ["unified"] if args.mode == "unified" else (
            ["multiclass"] if args.mode == "multiclass" else ["unified", "multiclass"]
    )
    
    for mode in modes:
        logger.info("=" * 60)
        logger.info(f"Preparing {mode} mode dataset")
        logger.info("=" * 60)
        
        preparer = YOLODatasetPreparer(mode=mode)
        preparer.prepare(
            train_split=args.train_split,
            val_split=args.val_split,
            test_split=args.test_split,
            augment_train_copies=args.augment_train_copies,
            augment_objects_per_image=(args.augment_min_objects, args.augment_max_objects),
            augment_seed=args.augment_seed,
            augment_min_area=args.augment_min_area,
        )
        
        print(f"\n{mode.upper()} mode dataset prepared!")
        print(f"  Training images: {preparer.stats['train_images']}")
        print(f"  Validation images: {preparer.stats['val_images']}")
        print(f"  Test images: {preparer.stats['test_images']}")
        print(f"  Output: {preparer.dataset_output}")
        print(f"  Missing labels: {preparer.stats['missing_labels']}")


if __name__ == "__main__":
    main()

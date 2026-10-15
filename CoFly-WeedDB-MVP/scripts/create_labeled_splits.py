#!/usr/bin/env python
"""Create deterministic train/val/test splits from all labeled images."""

from __future__ import annotations

from pathlib import Path
import random


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "CoFly-WeedDB"
IMAGES_DIR = DATASET_ROOT / "images"
LABELS_DIR = DATASET_ROOT / "labels"


def main() -> None:
    image_names = {path.name for path in IMAGES_DIR.glob("*.png")}
    labeled_names = sorted(path.name for path in LABELS_DIR.glob("*.png") if path.name in image_names)
    if not labeled_names:
        raise RuntimeError(f"No labeled images found in {LABELS_DIR}")

    rng = random.Random(42)
    rng.shuffle(labeled_names)

    total = len(labeled_names)
    train_count = int(round(total * 0.70))
    val_count = int(round(total * 0.15))
    test_count = total - train_count - val_count

    train_files = sorted(labeled_names[:train_count])
    val_files = sorted(labeled_names[train_count:train_count + val_count])
    test_files = sorted(labeled_names[train_count + val_count:])

    splits = {
        "train_split1.txt": train_files,
        "val_split1.txt": val_files,
        "test_split1.txt": test_files,
    }

    for filename, items in splits.items():
        output_path = DATASET_ROOT / filename
        output_path.write_text("\n".join(items) + "\n", encoding="utf-8")
        print(f"{filename}: {len(items)}")

    print(f"total_labeled: {total}")
    print(f"train/val/test: {train_count}/{val_count}/{test_count}")


if __name__ == "__main__":
    main()

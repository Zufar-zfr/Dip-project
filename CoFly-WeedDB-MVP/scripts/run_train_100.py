from pathlib import Path
import sys
import runpy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"

sys.path.insert(0, str(APP_DIR))
sys.argv = [
    "train.py",
    "--mode",
    "unified",
    "--epochs",
    "100",
    "--batch",
    "16",
    "--imgsz",
    "640",
    "--device",
    "cpu",
    "--validate",
]

runpy.run_path(str(APP_DIR / "train.py"), run_name="__main__")

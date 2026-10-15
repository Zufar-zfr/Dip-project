from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "CoFly-WeedDB"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "diploma_figures"


@dataclass
class ImageRecord:
    stem: str
    image_path: Path
    label_path: Path
    mask_path: Path


def ensure_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_rgb(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, img: np.ndarray) -> None:
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ext = path.suffix or ".png"
    ok, encoded = cv2.imencode(ext, bgr)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def read_yolo_boxes(label_path: Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, xc, yc, bw, bh = parts
        xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
        x1 = int((xc - bw / 2.0) * width)
        y1 = int((yc - bh / 2.0) * height)
        x2 = int((xc + bw / 2.0) * width)
        y2 = int((yc + bh / 2.0) * height)
        boxes.append((x1, y1, x2, y2))
    return boxes


def draw_boxes(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    color: tuple[int, int, int],
    title: str | None = None,
) -> np.ndarray:
    canvas = image.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
    if title:
        cv2.rectangle(canvas, (0, 0), (540, 55), (0, 0, 0), -1)
        cv2.putText(canvas, title, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def add_header(image: np.ndarray, title: str) -> np.ndarray:
    h, w = image.shape[:2]
    header = np.full((70, w, 3), 245, dtype=np.uint8)
    cv2.putText(header, title, (18, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 2, cv2.LINE_AA)
    return np.vstack([header, image])


def create_side_by_side(left: np.ndarray, right: np.ndarray, left_title: str, right_title: str) -> np.ndarray:
    target_h = min(left.shape[0], right.shape[0])

    def resize_keep_h(img: np.ndarray) -> np.ndarray:
        scale = target_h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), target_h), interpolation=cv2.INTER_AREA)

    left_r = add_header(resize_keep_h(left), left_title)
    right_r = add_header(resize_keep_h(right), right_title)
    pad = np.full((left_r.shape[0], 30, 3), 255, dtype=np.uint8)
    return np.hstack([left_r, pad, right_r])


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    result = image.copy()
    colored = np.zeros_like(image)
    nonzero = np.any(mask != 0, axis=2)
    colored[nonzero] = mask[nonzero]
    result[nonzero] = cv2.addWeighted(image[nonzero], 1.0 - alpha, colored[nonzero], alpha, 0.0)
    return result


def iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def match_counts(preds: list[tuple[int, int, int, int]], gts: list[tuple[int, int, int, int]]) -> tuple[int, int, int]:
    matched = [False] * len(gts)
    tp = 0
    fp = 0
    for pred in preds:
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gts):
            if matched[idx]:
                continue
            score = iou(pred, gt)
            if score > best_iou:
                best_iou = score
                best_idx = idx
        if best_iou >= 0.5 and best_idx >= 0:
            matched[best_idx] = True
            tp += 1
        else:
            fp += 1
    fn = len(gts) - tp
    return tp, fp, fn


def find_sample_record() -> ImageRecord:
    preview_dir = PROJECT_ROOT / "outputs" / "debug" / "dataset_preview"
    preview_path = sorted(preview_dir.glob("preview_01_*.png"))[0]
    stem = preview_path.name.replace("preview_01_", "").replace(".png", "")
    return ImageRecord(
        stem=stem,
        image_path=DATASET_ROOT / "images" / f"{stem}.png",
        label_path=DATASET_ROOT / "labels" / f"{stem}.png",
        mask_path=DATASET_ROOT / "labels" / f"{stem}.png",
    )


def make_core_figures() -> None:
    record = find_sample_record()
    raw = load_rgb(record.image_path)
    mask = load_rgb(record.mask_path)
    overlay = overlay_mask(raw, mask, alpha=0.42)

    save_rgb(OUTPUT_DIR / "figure_01_raw_image.png", raw)
    save_rgb(OUTPUT_DIR / "figure_02_mask_rgb.png", mask)
    save_rgb(OUTPUT_DIR / "figure_03_mask_overlay.png", overlay)

    converted = load_rgb(PROJECT_ROOT / "outputs" / "debug" / "converted_labels_preview" / "unified" / f"{record.stem}_converted.png")
    save_rgb(OUTPUT_DIR / "figure_04_bbox_conversion.png", converted)


def copy_existing_figure(src: Path, dst_name: str) -> None:
    img = load_rgb(src)
    save_rgb(OUTPUT_DIR / dst_name, img)


def create_threshold_plot() -> None:
    csv_path = PROJECT_ROOT / "outputs" / "metrics" / "experiment_threshold_imgsz" / "threshold_imgsz_sweep.csv"
    rows = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)

    fig, ax = plt.subplots(figsize=(10, 6))
    for imgsz in ("640", "960"):
        subset = [r for r in rows if r["imgsz"] == imgsz]
        x = [float(r["threshold"]) for r in subset]
        y = [float(r["f1"]) for r in subset]
        ax.plot(x, y, marker="o", linewidth=2.5, label=f"imgsz={imgsz}")
    ax.set_title("F1-score vs confidence threshold")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("F1-score")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure_08_threshold_sweep.png", dpi=200)
    plt.close(fig)


def create_imgsz_bar_plot() -> None:
    json_path = PROJECT_ROOT / "outputs" / "metrics" / "experiment_val_imgsz" / "summary.json"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    labels = ["Precision", "Recall", "mAP50", "mAP50-95"]
    vals_640 = [data["640"]["precision"], data["640"]["recall"], data["640"]["map50"], data["640"]["map50_95"]]
    vals_960 = [data["960"]["precision"], data["960"]["recall"], data["960"]["map50"], data["960"]["map50_95"]]

    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, vals_640, width, label="imgsz=640")
    ax.bar(x + width / 2, vals_960, width, label="imgsz=960")
    ax.set_ylim(0, 0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Comparison of inference image sizes")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure_09_imgsz_comparison.png", dpi=200)
    plt.close(fig)


def create_detection_examples() -> None:
    test_img_dir = PROJECT_ROOT / "data" / "processed" / "yolo_unified" / "images" / "test"
    test_lbl_dir = PROJECT_ROOT / "data" / "processed" / "yolo_unified" / "labels" / "test"
    model = YOLO(str(PROJECT_ROOT / "models" / "unified" / "best.pt"))

    records: list[tuple[float, float, ImageRecord, list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]] = []
    for image_path in sorted(test_img_dir.glob("*.png")):
        image = load_rgb(image_path)
        h, w = image.shape[:2]
        gt_boxes = read_yolo_boxes(test_lbl_dir / f"{image_path.stem}.txt", w, h)
        result = model.predict(source=str(image_path), imgsz=640, conf=0.20, iou=0.45, device="cpu", verbose=False, save=False)[0]
        pred_boxes: list[tuple[int, int, int, int]] = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes.xyxy.cpu().numpy():
                pred_boxes.append(tuple(int(v) for v in box[:4]))
        tp, fp, fn = match_counts(pred_boxes, gt_boxes)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        records.append((f1, fp + fn, ImageRecord(image_path.stem, image_path, test_lbl_dir / f"{image_path.stem}.txt", Path()), gt_boxes, pred_boxes))

    success = max(records, key=lambda item: (item[0], -item[1]))
    failure = max(records, key=lambda item: (item[1], -item[0]))

    for name, selected in [("figure_10_success_example.png", success), ("figure_11_failure_example.png", failure)]:
        _, _, record, gt_boxes, pred_boxes = selected
        image = load_rgb(record.image_path)
        gt_img = draw_boxes(image, gt_boxes, (40, 200, 40), "Ground truth")
        pred_img = draw_boxes(image, pred_boxes, (220, 60, 60), "Model prediction")
        pair = create_side_by_side(gt_img, pred_img, "Ground truth", "Prediction")
        save_rgb(OUTPUT_DIR / name, pair)


def write_captions() -> None:
    captions = {
        "figure_01_raw_image.png": "Рисунок 1 – Пример исходного аэрофотоснимка из набора данных CoFly-WeedDB.",
        "figure_02_mask_rgb.png": "Рисунок 2 – Пример цветной маски разметки сорных растений.",
        "figure_03_mask_overlay.png": "Рисунок 3 – Наложение маски разметки на исходное изображение.",
        "figure_04_bbox_conversion.png": "Рисунок 4 – Результат преобразования маски в bounding boxes формата YOLO.",
        "figure_05_training_curves.png": "Рисунок 5 – Кривые обучения модели night_tune1.",
        "figure_06_pr_curve.png": "Рисунок 6 – Precision-Recall кривая для лучшей модели.",
        "figure_07_confusion_matrix.png": "Рисунок 7 – Матрица ошибок лучшей модели на тестовой выборке.",
        "figure_08_threshold_sweep.png": "Рисунок 8 – Зависимость F1-меры от порога confidence при разных imgsz.",
        "figure_09_imgsz_comparison.png": "Рисунок 9 – Сравнение метрик модели при imgsz=640 и imgsz=960.",
        "figure_10_success_example.png": "Рисунок 10 – Пример корректной детекции сорных растений.",
        "figure_11_failure_example.png": "Рисунок 11 – Пример ошибки детекции: пропуски и лишние срабатывания.",
    }
    lines = [f"{name}: {caption}" for name, caption in captions.items()]
    (OUTPUT_DIR / "captions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir()
    make_core_figures()
    copy_existing_figure(PROJECT_ROOT / "runs" / "detect" / "models" / "unified" / "night_tune1" / "results.png", "figure_05_training_curves.png")
    copy_existing_figure(PROJECT_ROOT / "runs" / "detect" / "outputs" / "metrics" / "night_tune1_test" / "BoxPR_curve.png", "figure_06_pr_curve.png")
    copy_existing_figure(PROJECT_ROOT / "runs" / "detect" / "outputs" / "metrics" / "night_tune1_test" / "confusion_matrix.png", "figure_07_confusion_matrix.png")
    create_threshold_plot()
    create_imgsz_bar_plot()
    create_detection_examples()
    write_captions()
    print(f"Saved figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

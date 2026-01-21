# from ultralytics import YOLO
# from pathlib import Path

# def ensure_yaml_exists(yaml_path: str):
#     p = Path(yaml_path)
#     if not p.exists():
#         raise FileNotFoundError(f"YAML not found: {p.resolve()}")

# def train_one_run(yaml_path: str, run_name: str, epochs: int = 20):

#     model = YOLO("yolov8n.pt")

#     common = dict(
#         device="cpu",      # CUDA is False
#         imgsz=320,         # good speed on CPU
#         batch=8,           # safe for CPU/RAM
#         workers=2,         # faster loading
#         epochs=epochs,
#         exist_ok=True,     # allow reruns with same name
#         verbose=True,
#         plots=True,
#     )

#     print(f"\n===== TRAINING: {run_name} ({yaml_path}) =====")
#     results = model.train(data=yaml_path, name=run_name, **common)

#     # Validate on val split (YOLO does this during training too, but we run once for clarity)
#     print(f"\n===== VALIDATION (val): {run_name} =====")
#     model.val(data=yaml_path, split="val")

#     # Evaluate on test split (recommended for final report)
#     print(f"\n===== EVALUATION (test): {run_name} =====")
#     model.val(data=yaml_path, split="test")

#     return results


# if __name__ == "__main__":
#     # Check YAML files exist
#     ensure_yaml_exists("data.yaml")
#     ensure_yaml_exists("data_enhanced.yaml")

#     # Train original dataset
#     train_one_run("data.yaml", run_name="mouse_original_detection", epochs=20)

#     # Train enhanced dataset
#     train_one_run("data_enhanced.yaml", run_name="mouse_enhanced_detection", epochs=20)

#     print(" Done: trained + evaluated original and enhanced datasets.")
from ultralytics import YOLO
from pathlib import Path
import csv
from datetime import datetime

# ---------------- CONFIG ----------------
BASE_MODEL = "yolov8n.pt"
DEVICE = "cpu"          # you confirmed CUDA available: False
IMG_SIZE = 320
BATCH = 8
WORKERS = 2
DEFAULT_EPOCHS = 20
CONF_FOR_VAL = 0.25     # optional: keep default
IOU_FOR_VAL = 0.7       # optional: keep default

# Quick mode (set True if training too slow)
QUICK_MODE = False
# ----------------------------------------


def ensure_exists(path: str, kind: str = "path"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{kind} not found: {p.resolve()}")
    return p


def run_train_and_eval(yaml_path: str, run_name: str, epochs: int):
    # NEW model instance each run (prevents KeyError issues)
    model = YOLO(BASE_MODEL)

    # Adjust settings for quick mode
    imgsz = 256 if QUICK_MODE else IMG_SIZE
    ep = 10 if QUICK_MODE else epochs
    batch = 4 if QUICK_MODE else BATCH

    common = dict(
        device=DEVICE,
        imgsz=imgsz,
        batch=batch,
        workers=WORKERS,
        epochs=ep,
        exist_ok=True,
        verbose=True,
        plots=True,
        seed=0,
        deterministic=True,
        single_cls=True,  # single-class mouse project (safe)
    )

    print(f"\n===== TRAINING: {run_name} | data={yaml_path} =====")
    model.train(data=yaml_path, name=run_name, **common)

    print(f"\n===== VALIDATION (val): {run_name} =====")
    val_metrics = model.val(data=yaml_path, split="val", conf=CONF_FOR_VAL, iou=IOU_FOR_VAL)

    print(f"\n===== EVALUATION (test): {run_name} =====")
    test_metrics = model.val(data=yaml_path, split="test", conf=CONF_FOR_VAL, iou=IOU_FOR_VAL)

    # Save a small metrics summary for your report
    save_dir = Path(model.trainer.save_dir)  # runs/detect/<run_name>
    write_metrics_summary(save_dir, run_name, yaml_path, ep, imgsz, batch, val_metrics, test_metrics)

    return save_dir


def write_metrics_summary(save_dir: Path, run_name: str, yaml_path: str, epochs: int, imgsz: int, batch: int,
                         val_metrics, test_metrics):
    """
    Saves a CSV summary with key metrics (precision, recall, mAP50, mAP50-95).
    This is very useful for your final report comparison table.
    """
    out_csv = save_dir / "metrics_summary.csv"

    # Ultralytics exposes results in different ways depending on version.
    # We'll try robustly to read common attributes.
    def extract(metrics_obj):
        # Best-effort extraction
        P = getattr(metrics_obj.box, "mp", None) if hasattr(metrics_obj, "box") else None
        R = getattr(metrics_obj.box, "mr", None) if hasattr(metrics_obj, "box") else None
        mAP50 = getattr(metrics_obj.box, "map50", None) if hasattr(metrics_obj, "box") else None
        mAP5095 = getattr(metrics_obj.box, "map", None) if hasattr(metrics_obj, "box") else None

        # Fallbacks (some versions use different names)
        if P is None: P = getattr(metrics_obj, "mp", None)
        if R is None: R = getattr(metrics_obj, "mr", None)
        if mAP50 is None: mAP50 = getattr(metrics_obj, "map50", None)
        if mAP5095 is None: mAP5095 = getattr(metrics_obj, "map", None)

        return P, R, mAP50, mAP5095

    valP, valR, valmAP50, valmAP5095 = extract(val_metrics)
    tstP, tstR, tstmAP50, tstmAP5095 = extract(test_metrics)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": run_name,
        "data_yaml": yaml_path,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "val_precision": valP,
        "val_recall": valR,
        "val_mAP50": valmAP50,
        "val_mAP50_95": valmAP5095,
        "test_precision": tstP,
        "test_recall": tstR,
        "test_mAP50": tstmAP50,
        "test_mAP50_95": tstmAP5095,
    }

    # Write CSV
    write_header = not out_csv.exists()
    with out_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"\n✅ Saved metrics summary: {out_csv}")


if __name__ == "__main__":
    ensure_exists("data.yaml", "YAML")
    ensure_exists("data_enhanced.yaml", "YAML")

    print("\n=== START TRAINING PIPELINE ===")
    if QUICK_MODE:
        print("⚡ QUICK_MODE is ON (faster settings for CPU)")

    run_train_and_eval("data.yaml", run_name="mouse_original_detection", epochs=DEFAULT_EPOCHS)
    run_train_and_eval("data_enhanced.yaml", run_name="mouse_enhanced_detection", epochs=DEFAULT_EPOCHS)

    print("\n✅ Done: trained + evaluated original and enhanced datasets.")

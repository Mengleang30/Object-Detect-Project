from ultralytics import YOLO
from pathlib import Path

def ensure_yaml_exists(yaml_path: str):
    p = Path(yaml_path)
    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {p.resolve()}")

def train_one_run(yaml_path: str, run_name: str, epochs: int = 20):

    model = YOLO("yolov8n.pt")

    common = dict(
        device="cpu",      # CUDA is False
        imgsz=320,         # good speed on CPU
        batch=8,           # safe for CPU/RAM
        workers=2,         # faster loading
        epochs=epochs,
        exist_ok=True,     # allow reruns with same name
        verbose=True,
        plots=True,
    )

    print(f"\n===== TRAINING: {run_name} ({yaml_path}) =====")
    results = model.train(data=yaml_path, name=run_name, **common)

    # Validate on val split (YOLO does this during training too, but we run once for clarity)
    print(f"\n===== VALIDATION (val): {run_name} =====")
    model.val(data=yaml_path, split="val")

    # Evaluate on test split (recommended for final report)
    print(f"\n===== EVALUATION (test): {run_name} =====")
    model.val(data=yaml_path, split="test")

    return results


if __name__ == "__main__":
    # Check YAML files exist
    ensure_yaml_exists("data.yaml")
    ensure_yaml_exists("data_enhanced.yaml")

    # Train original dataset
    train_one_run("data.yaml", run_name="mouse_original_detection", epochs=20)

    # Train enhanced dataset
    train_one_run("data_enhanced.yaml", run_name="mouse_enhanced_detection", epochs=20)

    print(" Done: trained + evaluated original and enhanced datasets.")

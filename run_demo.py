from ultralytics import YOLO
from pathlib import Path

# ================= CONFIG =================
CONF = 0.5
PROJECT = "runs/detect"

# Mouse models
MOUSE_ORIGINAL_MODEL = "runs/detect/mouse_original_detection/weights/best.pt"
MOUSE_ENHANCED_MODEL = "runs/detect/mouse_enhanced_detection/weights/best.pt"

# Datasets
MOUSE_TEST_IMAGES = "dataset_original/images/test"
MOUSE_ENHANCED_TEST_IMAGES = "dataset_enhanced/images/test"



def check_path(path, is_model=False):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"❌ Not found: {p.resolve()}")
    if is_model and p.suffix != ".pt":
        raise ValueError(f"❌ Model file must be .pt: {p}")
    return str(p)


def run_predict(model_path, source_path, run_name):
    print(f"\n▶ Running demo: {run_name}")
    model = YOLO(model_path)
    model.predict(
        source=source_path,
        conf=CONF,
        save=True,
        project=PROJECT,
        name=run_name,
    )
    print(f"✅ Results saved to {PROJECT}/{run_name}")


def main():
    # Check paths
    mouse_original_model = check_path(MOUSE_ORIGINAL_MODEL, is_model=True)
    mouse_enhanced_model = check_path(MOUSE_ENHANCED_MODEL, is_model=True)

    mouse_test = check_path(MOUSE_TEST_IMAGES)
    mouse_enhanced_test = check_path(MOUSE_ENHANCED_TEST_IMAGES)


    # Demo 1: Mouse detection (original dataset)
    run_predict(
        mouse_original_model,
        mouse_test,
        "demo_mouse_original"
    )

    # Demo 2: Mouse detection (enhanced dataset)
    run_predict(
        mouse_enhanced_model,
        mouse_enhanced_test,
        "demo_mouse_enhanced"
    )


    print("\n🎉 DETECT COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()

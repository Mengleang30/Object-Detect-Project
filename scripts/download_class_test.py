import fiftyone as fo
import fiftyone.zoo as foz
import os

# Open Images class name (must be exact)
CLASSES = ["Cat"]

# Project paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
EXPORT_DIR = os.path.join(project_root, "dataset_cat")

# Number of samples per split (adjust if needed)
MAX_SAMPLES = {
    "train": 500,
    "validation": 100,
    "test": 100,
}

for split in ["train", "validation", "test"]:
    print(f"Downloading and exporting CAT {split} split...")

    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split=split,
        label_types=["detections"],
        classes=CLASSES,
        max_samples=MAX_SAMPLES[split],
    )

    dataset.export(
        export_dir=EXPORT_DIR,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        split=split,
        classes=CLASSES,
    )

print(f"\n✅ Cat dataset downloaded and exported to '{EXPORT_DIR}'")

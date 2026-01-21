import fiftyone as fo
import fiftyone.zoo as foz
import os

CLASSES = ["Cat"]  # correct Open Images class

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

EXPORT_DIR = os.path.join(project_root, "dataset_cat_test")

MAX_SAMPLES = 100  # enough for evaluation

print("Downloading and exporting CAT test split...")

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="test",
    label_types=["detections"],
    classes=CLASSES,
    max_samples=MAX_SAMPLES,
)

dataset.export(
    export_dir=EXPORT_DIR,
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
    split="test",
    classes=CLASSES,
)

print(f"\nCat test dataset saved to '{EXPORT_DIR}'")

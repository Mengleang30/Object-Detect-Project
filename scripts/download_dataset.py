import fiftyone as fo
import fiftyone.zoo as foz
import os

CLASSES = ["Mouse", "Cat"]
EXPORT_DIR = "dataset_original"
MAX_SAMPLES = {
    "train": 20,
    "validation": 10,
    "test": 10,
}

print("Downloading and exporting train split...")
train_dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=CLASSES,
    max_samples=MAX_SAMPLES["train"],
)
train_dataset.export(
    export_dir=EXPORT_DIR,
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
    split="train",
    classes=CLASSES,
)

print("Downloading and exporting validation split...")
val_dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    label_types=["detections"],
    classes=CLASSES,
    max_samples=MAX_SAMPLES["validation"],
)
val_dataset.export(
    export_dir=EXPORT_DIR,
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
    split="validation",
    classes=CLASSES,
)

print("Downloading and exporting test split...")
test_dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="test",
    label_types=["detections"],
    classes=CLASSES,
    max_samples=MAX_SAMPLES["test"],
)
test_dataset.export(
    export_dir=EXPORT_DIR,
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
    split="test",
    classes=CLASSES,
)

print(f"\nDataset downloaded and exported to '{EXPORT_DIR}'")

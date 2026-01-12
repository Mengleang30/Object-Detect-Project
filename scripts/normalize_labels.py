import os
import glob
from PIL import Image

def normalize_yolo_labels(image_dir, label_dir, class_name_to_id={'cat': 0, 'mouse': 1}):
    print(f"Normalizing labels in {label_dir} for images in {image_dir}")
    
    # Iterate through all image files in the directory
    for img_path in glob.glob(os.path.join(image_dir, "*.jpg")):
        base_filename = os.path.basename(img_path)
        name_without_ext = os.path.splitext(base_filename)[0]
        label_path = os.path.join(label_dir, name_without_ext + ".txt")

        if not os.path.exists(label_path):
            print(f"Warning: Label file not found for {img_path}. Skipping.")
            continue

        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Error opening image {img_path}: {e}. Skipping.")
            continue

        normalized_labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                class_name = parts[0]
                class_id = class_name_to_id.get(class_name)

                if class_id is None:
                    print(f"Warning: Unknown class name '{class_name}' in {label_path}. Skipping line.")
                    continue

                try:
                    # Assuming format: class_name x_min y_min x_max y_max
                    x_min, y_min, x_max, y_max = map(float, parts[1:])

                    # Convert to YOLO format (x_center, y_center, width, height)
                    # First, in pixel values
                    box_width_px = x_max - x_min
                    box_height_px = y_max - y_min
                    x_center_px = x_min + box_width_px / 2
                    y_center_px = y_min + box_height_px / 2

                    # Normalize
                    x_center_norm = x_center_px / img_width
                    y_center_norm = y_center_px / img_height
                    box_width_norm = box_width_px / img_width
                    box_height_norm = box_height_px / img_height

                    # Append to normalized labels
                    normalized_labels.append(
                        f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {box_width_norm:.6f} {box_height_norm:.6f}"
                    )
                except ValueError as e:
                    print(f"Error parsing coordinates in {label_path} line '{line.strip()}': {e}. Skipping line.")
                except IndexError:
                    print(f"Error: Not enough coordinates in {label_path} line '{line.strip()}'. Skipping line.")

        # Write back the normalized labels
        if normalized_labels:
            with open(label_path, 'w') as f:
                for label_line in normalized_labels:
                    f.write(label_line + '\n')
            print(f"Normalized and updated {label_path}")
        else:
            print(f"Warning: No valid labels found or processed for {label_path}. File might be empty or unchanged.")

# Main execution
if __name__ == "__main__":
    base_dataset_path = "dataset_original" 
    
    # Process training set
    train_img_path = os.path.join(base_dataset_path, "images", "train")
    train_label_path = os.path.join(base_dataset_path, "labels", "train")
    normalize_yolo_labels(train_img_path, train_label_path)

    # Process validation set
    val_img_path = os.path.join(base_dataset_path, "images", "validation")
    val_label_path = os.path.join(base_dataset_path, "labels", "validation")
    normalize_yolo_labels(val_img_path, val_label_path)

    # Process test set
    test_img_path = os.path.join(base_dataset_path, "images", "test")
    test_label_path = os.path.join(base_dataset_path, "labels", "test")
    normalize_yolo_labels(test_img_path, test_label_path)

    print("Label normalization complete.")

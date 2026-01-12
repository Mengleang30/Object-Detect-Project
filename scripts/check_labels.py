
import os

def check_labels(image_dir, label_dir, num_classes):
    image_files = {os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))}
    label_files = {os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.txt')}

    print(f"Found {len(image_files)} images in {image_dir}")
    print(f"Found {len(label_files)} labels in {label_dir}")

    # Check for images without labels
    images_without_labels = image_files - label_files
    if images_without_labels:
        print(f"\nImages without labels ({len(images_without_labels)}):")
        for image in images_without_labels:
            print(f"  - {image}")

    # Check for labels without images
    labels_without_images = label_files - image_files
    if labels_without_images:
        print(f"\nLabels without images ({len(labels_without_images)}):")
        for label in labels_without_images:
            print(f"  - {label}")

    # Check for empty or invalid label files
    invalid_labels = []
    for label_file in label_files:
        label_path = os.path.join(label_dir, f"{label_file}.txt")
        if os.path.getsize(label_path) == 0:
            invalid_labels.append(f"{label_file}.txt (empty)")
            continue

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    invalid_labels.append(f"{label_file}.txt (incorrect number of values)")
                    break
                
                class_id = int(parts[0])
                if not 0 <= class_id < num_classes:
                    invalid_labels.append(f"{label_file}.txt (invalid class id: {class_id})")
                    break

    if invalid_labels:
        print(f"\nInvalid label files ({len(invalid_labels)}):")
        for label in invalid_labels:
            print(f"  - {label}")

if __name__ == "__main__":
    # Assuming data.yaml has 'names' which gives the number of classes
    num_classes = 2  # Manually set based on the provided data.yaml
    
    print("Checking training data...")
    check_labels("dataset_original/images/train", "dataset_original/labels/train", num_classes)
    
    print("\nChecking validation data...")
    check_labels("dataset_original/images/validation", "dataset_original/labels/validation", num_classes)
    
    print("\nChecking test data...")
    check_labels("dataset_original/images/test", "dataset_original/labels/test", num_classes)

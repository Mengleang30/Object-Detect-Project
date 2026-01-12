import os
import glob

def remove_unlabeled_images(image_dir, label_dir):
    unlabeled_count = 0
    image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
    
    if not image_files:
        print(f"No images found in {image_dir}. Skipping.")
        return

    print(f"Checking for unlabeled images in {image_dir}...")
    
    for img_path in image_files:
        base_filename = os.path.basename(img_path)
        name_without_ext = os.path.splitext(base_filename)[0]
        label_path = os.path.join(label_dir, name_without_ext + ".txt")

        if not os.path.exists(label_path):
            print(f"Removing unlabeled image: {img_path}")
            os.remove(img_path)
            unlabeled_count += 1
            
    if unlabeled_count > 0:
        print(f"Removed {unlabeled_count} unlabeled images from {image_dir}.")
    else:
        print(f"No unlabeled images found in {image_dir}.")

if __name__ == "__main__":
    base_dataset_path = "dataset_original"
    
    # Process training set
    train_img_path = os.path.join(base_dataset_path, "images", "train", "mouse")
    train_label_path = os.path.join(base_dataset_path, "labels", "train", "mouse")
    remove_unlabeled_images(train_img_path, train_label_path)

    # Process validation set
    val_img_path = os.path.join(base_dataset_path, "images", "validation", "mouse")
    val_label_path = os.path.join(base_dataset_path, "labels", "validation", "mouse")
    remove_unlabeled_images(val_img_path, val_label_path)

    # Process test set
    test_img_path = os.path.join(base_dataset_path, "images", "test", "mouse")
    test_label_path = os.path.join(base_dataset_path, "labels", "test", "mouse")
    remove_unlabeled_images(test_img_path, test_label_path)

    print("Finished checking for unlabeled images.")

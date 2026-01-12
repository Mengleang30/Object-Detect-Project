import os
import glob
import random
import shutil

# Define paths
train_img_path = "dataset_original/images/train/mouse"
train_label_path = "dataset_original/labels/train/mouse"
val_img_path = "dataset_original/images/validation/mouse"
val_label_path = "dataset_original/labels/validation/mouse"

# Get all image files
img_files = glob.glob(os.path.join(train_img_path, "*.jpg"))

# Shuffle the files
random.shuffle(img_files)

# Calculate number of files for validation (20%)
val_size = int(len(img_files) * 0.2)

# Get the files for validation
val_img_files = img_files[:val_size]

# Move the files
for img_file in val_img_files:
    # Get the corresponding label file
    base_filename = os.path.basename(img_file)
    label_filename = os.path.splitext(base_filename)[0] + ".txt"
    label_file = os.path.join(train_label_path, label_filename)

    # Move image file
    shutil.move(img_file, val_img_path)

    # Move label file
    if os.path.exists(label_file):
        shutil.move(label_file, val_label_path)

print(f"Moved {len(val_img_files)} files to validation set.")

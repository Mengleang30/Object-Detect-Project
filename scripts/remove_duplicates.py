import os
import cv2
import hashlib

IMAGE_DIR = "../dataset_original/images/train"

hashes = {}
removed = 0

for img_name in os.listdir(IMAGE_DIR):
    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    img_hash = hashlib.md5(img.tobytes()).hexdigest()

    if img_hash in hashes:
        os.remove(img_path)
        removed += 1
    else:
        hashes[img_hash] = img_name

print(f"Removed {removed} duplicate images")

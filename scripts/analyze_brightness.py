import os
import glob
from PIL import Image
import numpy as np

def analyze_images(image_dirs):
    all_brightness = []
    all_contrast = []

    for image_dir in image_dirs:
        if not os.path.isdir(image_dir):
            print(f"Directory not found: {image_dir}. Skipping.")
            continue
            
        print(f"\nAnalyzing images in: {image_dir}")
        image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
        
        if not image_files:
            print("No images found to analyze.")
            continue

        for img_path in image_files:
            try:
                with Image.open(img_path).convert('L') as img: # Convert to grayscale
                    img_array = np.array(img)
                    brightness = np.mean(img_array)
                    contrast = np.std(img_array)
                    
                    all_brightness.append(brightness)
                    all_contrast.append(contrast)

                    print(f"  - {os.path.basename(img_path)}: Brightness={brightness:.2f}, Contrast={contrast:.2f}")

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    if all_brightness:
        avg_brightness = np.mean(all_brightness)
        avg_contrast = np.mean(all_contrast)
        print("\n---")
        print(f"Overall Average Brightness: {avg_brightness:.2f}")
        print(f"Overall Average Contrast: {avg_contrast:.2f}")
        print("---")
        print("Brightness is measured as the mean pixel value (0=black, 255=white).")
        print("Contrast is measured as the standard deviation of pixel values.")
    else:
        print("\nNo images were analyzed.")


if __name__ == "__main__":
    base_dataset_path = "dataset_original"
    dirs_to_analyze = [
        os.path.join(base_dataset_path, "images", "train", "mouse"),
        os.path.join(base_dataset_path, "images", "val", "mouse"),
        os.path.join(base_dataset_path, "images", "test", "mouse")
    ]
    analyze_images(dirs_to_analyze)

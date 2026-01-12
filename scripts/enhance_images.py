import os
import glob
import shutil
from PIL import Image, ImageEnhance

def enhance_and_copy_dataset(original_base, enhanced_base, contrast_factor=1.2, brightness_factor=1.2, sharpness_factor=1.5):
    """
    Enhances images from the original dataset and saves them to a new location,
    copying over the corresponding labels.
    """
    for data_split in ["train", "validation", "test"]:
        original_img_dir = os.path.join(original_base, "images", data_split)
        original_label_dir = os.path.join(original_base, "labels", data_split)
        
        enhanced_img_dir = os.path.join(enhanced_base, "images", data_split)
        enhanced_label_dir = os.path.join(enhanced_base, "labels", data_split)

        os.makedirs(enhanced_img_dir, exist_ok=True)
        os.makedirs(enhanced_label_dir, exist_ok=True)

        if not os.path.isdir(original_img_dir):
            print(f"Original directory not found: {original_img_dir}. Skipping.")
            continue
            
        print(f"\nProcessing {data_split} set...")

        # Enhance and save images
        image_files = glob.glob(os.path.join(original_img_dir, "*.jpg"))
        if not image_files:
            print(f"No images to enhance in {original_img_dir}.")
        else:
            for img_path in image_files:
                try:
                    with Image.open(img_path) as img:
                        # Enhance Contrast
                        enhancer = ImageEnhance.Contrast(img)
                        img_enhanced = enhancer.enhance(contrast_factor)
                        
                        # Enhance Brightness
                        enhancer = ImageEnhance.Brightness(img_enhanced)
                        img_enhanced = enhancer.enhance(brightness_factor)

                        # Enhance Sharpness
                        enhancer = ImageEnhance.Sharpness(img_enhanced)
                        img_enhanced = enhancer.enhance(sharpness_factor)
                        
                        # Save the enhanced image
                        new_img_path = os.path.join(enhanced_img_dir, os.path.basename(img_path))
                        img_enhanced.save(new_img_path)
                        print(f"  - Saved enhanced image to {new_img_path}")

                except Exception as e:
                    print(f"Error enhancing {img_path}: {e}")

        # Copy label files
        label_files = glob.glob(os.path.join(original_label_dir, "*.txt"))
        if not label_files:
             print(f"No labels to copy in {original_label_dir}.")
        else:
            for label_path in label_files:
                new_label_path = os.path.join(enhanced_label_dir, os.path.basename(label_path))
                shutil.copy(label_path, new_label_path)
                print(f"  - Copied label to {new_label_path}")

if __name__ == "__main__":
    original_dataset_path = "dataset_original"
    enhanced_dataset_path = "dataset_enhanced"
    
    enhance_and_copy_dataset(original_dataset_path, enhanced_dataset_path)

    print("\nFinished enhancing dataset.")

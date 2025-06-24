import os
import cv2
from config import IMG_SIZE  # (width, height)

SOURCE_DIR = './Raw'           # original dataset
OUTPUT_DIR = './images'  # save resized dataset here

def resize_and_save_all(root_dir, output_dir, target_size):
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            # Construct full path to input and corresponding output
            rel_path = os.path.relpath(subdir, root_dir)
            input_path = os.path.join(subdir, file)
            output_subdir = os.path.join(output_dir, rel_path)
            output_path = os.path.join(output_subdir, file)

            # Create output subdirectory if it doesn't exist
            os.makedirs(output_subdir, exist_ok=True)

            # Read, resize and save
            image = cv2.imread(input_path)
            if image is None:
                print(f"[WARNING] Could not read {input_path}")
                continue

            resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(output_path, resized)
            count += 1

    print(f"✅ Done. {count} images resized and saved to {output_dir}")

if __name__ == "__main__":
    resize_and_save_all(SOURCE_DIR, OUTPUT_DIR, IMG_SIZE)

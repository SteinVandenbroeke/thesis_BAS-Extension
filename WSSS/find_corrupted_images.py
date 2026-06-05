import os
from PIL import Image

# Update this to the path where your COCO images are stored
image_dir = "../../../datasets/coco_dataset/coco2017/"


def find_corrupted_images(directory):
    print("Scanning for corrupted images...")
    corrupted_files = []
    counter = 0

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(root, file)
                try:
                    with Image.open(filepath) as img:
                        img.load()  # This forces PIL to read the actual image data
                        counter+=1
                except OSError:
                    print(f"Corrupt image found: {filepath}")
                    corrupted_files.append(filepath)

    print(f"Scan complete. Found {len(corrupted_files)} corrupted images of {counter} checked images.")


if __name__ == "__main__":
    find_corrupted_images(image_dir)
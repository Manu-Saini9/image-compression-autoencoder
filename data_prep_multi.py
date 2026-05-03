# data prep for multiple images
# loads BSD500 images and makes training samples
# resizes everything to 2000x1200 

import cv2
import numpy as np
import os
import glob

# settings

# folder containing images
INPUT_FOLDER = "data/BSD500"

# how many images to use (leave None for all)
MAX_IMAGES = 100  # use first 100 images 

# augmentation - how many versions per image
AUGMENTATIONS_PER_IMAGE = 3  # flips + brightness = 3x more samples

# where to save
OUTPUT_DIR = "data/patches"


print("=" * 50)
print("DATA PREP - MULTIPLE IMAGES")
print("=" * 50)

# find images
print(f"\nSearching for images in: {INPUT_FOLDER}")
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
image_files = []

for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
    image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, '**', ext), recursive=True))

print(f"Found {len(image_files)} images")

if len(image_files) == 0:
    print("ERROR: No images found!")
    exit()

# limit to MAX_IMAGES
if MAX_IMAGES is not None and len(image_files) > MAX_IMAGES:
    image_files = image_files[:MAX_IMAGES]
    print(f"Using first {MAX_IMAGES} images")

# make output directory (clean old files first)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created folder: {OUTPUT_DIR}")
else:
    # delete old patch files
    old_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.npy')]
    if old_files:
        print(f"Deleting {len(old_files)} old patch files...")
        for f in old_files:
            os.remove(os.path.join(OUTPUT_DIR, f))
        print("Old files cleaned!")


# process each image
print(f"\nProcessing {len(image_files)} images...")
print(f"Creating {AUGMENTATIONS_PER_IMAGE} variations per image")
print(f"Total samples: {len(image_files) * AUGMENTATIONS_PER_IMAGE}")

samples = []
sample_count = 0
# use enumerate to get index for progress logging
for i, img_path in enumerate(image_files):
    # load image
    img = cv2.imread(img_path)

    if img is None:
        print(f"  Skipping {img_path} (could not load)")
        continue

    # convert BGR to RGB
    # opencv loads in BGR but we need RGB for the model
    # took me a while to figure this out lol
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize to 2000x1200 (assignment requirement)
    # couldn't find dataset with exact size so just resizing BSD500 images
    img = cv2.resize(img, (2000, 1200))

    # normalize to 0-1
    img = img.astype(np.float32) / 255.0

    # create 3 versions of each image for more training data
    # 1. original
    samples.append(img)
    sample_count += 1

    # 2. flip horizontal
    samples.append(np.fliplr(img))
    sample_count += 1

    # 3. random brightness change
    brightness = np.random.uniform(0.7, 1.3)
    bright_img = np.clip(img * brightness, 0.0, 1.0)
    samples.append(bright_img)
    sample_count += 1

    # print progress
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/{len(image_files)} images ({sample_count} samples)")

print(f"\nTotal samples created: {len(samples)}")
print(f"Each sample is shape: {samples[0].shape}")


# save all samples
print("\nSaving samples...")
for i in range(len(samples)):
    filename = os.path.join(OUTPUT_DIR, f"sample_{i:04d}.npy")
    np.save(filename, samples[i])

    if (i + 1) % 50 == 0:
        print(f"  Saved {i+1} samples...")

print(f"Saved {len(samples)} samples to {OUTPUT_DIR}/")

print("\n" + "=" * 50)
print("DONE!")
print("=" * 50)
print(f"Total samples: {len(samples)}")
print(f"From {len(image_files)} different images")
print(f"Each sample: 2000x1200x3")
print("\nPreproccessing done")

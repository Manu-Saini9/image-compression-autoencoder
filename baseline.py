# baseline compression using bicubic interpolation
# just testing how good bicubic is so we can compare with autoencoder

import cv2
import numpy as np
import os
import glob

# settings
TEST_IMAGES_DIR = "data/BSD500"
NUM_TEST_IMAGES = 10

ORIGINAL_WIDTH = 2000
ORIGINAL_HEIGHT = 1200
COMPRESSED_WIDTH = 200
COMPRESSED_HEIGHT = 120

RESULTS_DIR = "results/baseline"

# simple mse function
def calculate_mse(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    return mse

print("=" * 60)
print("BASELINE COMPRESSION - BICUBIC")
print("=" * 60)

# create results folder
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
else:
    # clean old files
    for f in os.listdir(RESULTS_DIR):
        os.remove(os.path.join(RESULTS_DIR, f))

# load test images
print(f"\nLoading test images from: {TEST_IMAGES_DIR}")
all_images = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg"))

if len(all_images) == 0:
    print("ERROR: No images found!")
    exit()

# using last 10 images as test (same ones the autoencoder will use)
test_images = all_images[-NUM_TEST_IMAGES:]
print(f"Testing on {len(test_images)} images")

# test each image with bicubic
mse_scores = []

for i, img_path in enumerate(test_images):
    print(f"\nImage {i+1}/{len(test_images)}: {os.path.basename(img_path)}")

    # load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (ORIGINAL_WIDTH, ORIGINAL_HEIGHT))
    img = img.astype(np.float32) / 255.0

    # compress with bicubic
    compressed = cv2.resize(img, (COMPRESSED_WIDTH, COMPRESSED_HEIGHT),
                           interpolation=cv2.INTER_CUBIC)

    # decompress with bicubic
    reconstructed = cv2.resize(compressed, (ORIGINAL_WIDTH, ORIGINAL_HEIGHT),
                              interpolation=cv2.INTER_CUBIC)

    # calculate MSE
    mse = calculate_mse(img, reconstructed)
    mse_scores.append(mse)

    print(f"  MSE: {mse:.6f}")

# calculate average
average_mse = np.mean(mse_scores)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Average MSE: {average_mse:.6f}")
print("=" * 60)

# save results
print(f"\nSaving to {RESULTS_DIR}/baseline_metrics.txt")

with open(os.path.join(RESULTS_DIR, "baseline_metrics.txt"), 'w') as f:
    f.write("BASELINE - BICUBIC INTERPOLATION\n")
    f.write("=" * 60 + "\n")
    f.write(f"Test images: {len(test_images)}\n")
    f.write(f"Compression: {ORIGINAL_WIDTH}x{ORIGINAL_HEIGHT} -> ")
    f.write(f"{COMPRESSED_WIDTH}x{COMPRESSED_HEIGHT}\n")
    f.write(f"\nAverage MSE: {average_mse:.6f}\n")
    f.write("=" * 60 + "\n")

print("\nDONE!")
print(f"Baseline MSE: {average_mse:.6f}")
print("Autoencoder needs to beat this score!")

# also testing specific image 97033.jpg
print("\n" + "=" * 60)
print("SPECIFIC TEST - IMAGE 97033.jpg")
print("=" * 60)

specific_image = os.path.join(TEST_IMAGES_DIR, "97033.jpg")

if os.path.exists(specific_image):
    print(f"\nTesting image: 97033.jpg")

    # load image
    img = cv2.imread(specific_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (ORIGINAL_WIDTH, ORIGINAL_HEIGHT))
    img = img.astype(np.float32) / 255.0

    # compress with bicubic
    compressed = cv2.resize(img, (COMPRESSED_WIDTH, COMPRESSED_HEIGHT),
                           interpolation=cv2.INTER_CUBIC)

    # decompress with bicubic
    reconstructed = cv2.resize(compressed, (ORIGINAL_WIDTH, ORIGINAL_HEIGHT),
                              interpolation=cv2.INTER_CUBIC)

    # calculate MSE
    mse = calculate_mse(img, reconstructed)

    print(f"MSE for 97033.jpg: {mse:.6f}")

    # save this specific result
    with open(os.path.join(RESULTS_DIR, "97033_result.txt"), 'w') as f:
        f.write("SPECIFIC IMAGE TEST - 97033.jpg\n")
        f.write("=" * 60 + "\n")
        f.write(f"Method: Bicubic Interpolation\n")
        f.write(f"Original size: {ORIGINAL_WIDTH}x{ORIGINAL_HEIGHT}\n")
        f.write(f"Compressed size: {COMPRESSED_WIDTH}x{COMPRESSED_HEIGHT}\n")
        f.write(f"\nMSE: {mse:.6f}\n")
        f.write("=" * 60 + "\n")

    print(f"Saved results to: {RESULTS_DIR}/97033_result.txt")
else:
    print(f"\nWARNING: Image 97033.jpg not found in {TEST_IMAGES_DIR}")
    print("Skipping specific test...")

print("\n" + "=" * 60)

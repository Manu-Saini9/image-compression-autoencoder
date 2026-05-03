# test the trained model on a random image
# just to see how well it works

import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob
import random

# settings
MODEL_PATH = "models/autoencoder.h5"
TEST_IMAGES_DIR = "data/BSD500"
OUTPUT_FILE = "test_result.png"

ORIGINAL_WIDTH = 2000
ORIGINAL_HEIGHT = 1200

print("=" * 60)
print("TEST AUTOENCODER ON RANDOM IMAGE")
print("=" * 60)

# load model
print(f"\nLoading model from {MODEL_PATH}...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded!")

# get the encoder part (layer 4 is the bottleneck - compressed to 200x120)
encoder = keras.Model(model.input, model.layers[4].output)

# pick a random image to test
all_images = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg"))
random_image = random.choice(all_images)
image_name = os.path.basename(random_image)

print(f"\nTesting on: {image_name}")

# load image
img = cv2.imread(random_image)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (ORIGINAL_WIDTH, ORIGINAL_HEIGHT))
original = img.copy()  # keep for display
img = img.astype(np.float32) / 255.0

# compress and reconstruct
print("Running autoencoder...")
img_batch = np.expand_dims(img, axis=0)

# get compressed representation
compressed = encoder.predict(img_batch, verbose=0)[0]
print(f"Compressed shape: {compressed.shape}")

# get reconstructed image
reconstructed = model.predict(img_batch, verbose=0)[0]

# calculate MSE
mse = np.mean((img - reconstructed) ** 2)
print(f"MSE: {mse:.6f}")

# create visualization
print("\nCreating visualization...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# original
axes[0].imshow(original)
axes[0].set_title(f"Original (2000x1200)\n{image_name}", fontsize=14, fontweight='bold')
axes[0].axis('off')

# compressed (show actual compressed representation)
# normalize compressed for display
compressed_display = (compressed - compressed.min()) / (compressed.max() - compressed.min())
axes[1].imshow(compressed_display)
axes[1].set_title(f"Compressed ({compressed.shape[1]}x{compressed.shape[0]}x{compressed.shape[2]})\nBottleneck Representation", fontsize=14, fontweight='bold')
axes[1].axis('off')

# reconstructed
reconstructed_display = (reconstructed * 255).astype(np.uint8)
axes[2].imshow(reconstructed_display)
axes[2].set_title(f"Reconstructed (2000x1200)\nMSE: {mse:.6f}", fontsize=14, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
print(f"Saved to: {OUTPUT_FILE}")

# save individual images too
print("\nSaving individual images...")

cv2.imwrite("original.png", cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
cv2.imwrite("reconstructed.png", cv2.cvtColor(reconstructed_display, cv2.COLOR_RGB2BGR))

# compressed representation as image
compressed_vis = (compressed_display * 255).astype(np.uint8)
cv2.imwrite("compressed.png", cv2.cvtColor(compressed_vis, cv2.COLOR_RGB2BGR))

print("Saved:")
print("  - original.png")
print("  - compressed.png")
print("  - reconstructed.png")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
print(f"Open {OUTPUT_FILE} to see the comparison!")

plt.show()

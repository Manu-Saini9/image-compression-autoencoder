"""
Autoencoder for image compression - BSD500
Compresses 2000x1200x3 -> 200x120x3 -> 2000x1200x3
Clean single-file implementation
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import glob

# ===== SETTINGS =====

# training data
PATCHES_DIR = "data/patches"

# test images
TEST_IMAGES_DIR = "data/BSD500"
NUM_TEST_IMAGES = 10

# training parameters
EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 0.001

# image sizes
ORIGINAL_WIDTH = 2000
ORIGINAL_HEIGHT = 1200
COMPRESSED_WIDTH = 200
COMPRESSED_HEIGHT = 120

# where to save
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "autoencoder.h5")
RESULTS_DIR = "results/autoencoder"

# ====================


print("=" * 60)
print("AUTOENCODER IMAGE COMPRESSION - BSD500")
print("=" * 60)

# create directories
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

if os.path.exists(RESULTS_DIR):
    for f in os.listdir(RESULTS_DIR):
        os.remove(os.path.join(RESULTS_DIR, f))
else:
    os.makedirs(RESULTS_DIR)


# ===== LOAD TRAINING DATA =====

print("\nLoading training data...")
patch_files = sorted([f for f in os.listdir(PATCHES_DIR) if f.endswith('.npy')])
print(f"Found {len(patch_files)} training samples")

train_data = []
for filepath in [os.path.join(PATCHES_DIR, f) for f in patch_files]:
    patch = np.load(filepath)
    train_data.append(patch)

train_data = np.array(train_data)
print(f"Training data shape: {train_data.shape}")


# ===== BUILD MODEL =====

print("\nBuilding autoencoder...")

model = keras.Sequential([
    # input
    layers.Input(shape=(ORIGINAL_HEIGHT, ORIGINAL_WIDTH, 3)),

    # encoder - compress image to 200x120
    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((5, 5)),  # 2000x1200 -> 400x240

    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),  # 400x240 -> 200x120

    layers.Conv2D(3, (3, 3), activation='relu', padding='same'),  # bottleneck 200x120x3

    # decoder - reconstruct back to 2000x1200
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),  # 200x120 -> 400x240

    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((5, 5)),  # 400x240 -> 2000x1200

    # output
    layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
])

print("Model created!")
model.summary()


# ===== COMPILE =====

print("\nCompiling model...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='mse'
)


# ===== TRAIN =====

print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)
print(f"Epochs: {EPOCHS}")
print(f"Samples: {len(train_data)}")
print("=" * 60)

history = model.fit(
    train_data,
    train_data,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=0.2,
    verbose=1
)

print("\nTraining complete!")


# ===== SAVE MODEL =====

print(f"\nSaving model to {MODEL_PATH}...")
model.save(MODEL_PATH)
print("Model saved!")


# ===== PLOT TRAINING =====

print("Creating training plot...")

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Autoencoder Training History')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "training_history.png"))
print("Saved training plot!")


# ===== TEST ON IMAGES =====

print("\n" + "=" * 60)
print("TESTING")
print("=" * 60)

# get test images (same as baseline)
all_images = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg"))
test_images = all_images[-NUM_TEST_IMAGES:]
print(f"Testing on {len(test_images)} images")

# create encoder (for compressed representation)
encoder = keras.Model(model.input, model.layers[4].output)

# test each image
test_mses = []

for i, img_path in enumerate(test_images):
    # load
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (ORIGINAL_WIDTH, ORIGINAL_HEIGHT))
    img = img.astype(np.float32) / 255.0

    # reconstruct
    img_batch = np.expand_dims(img, axis=0)
    reconstructed = model.predict(img_batch, verbose=0)[0]

    # calculate MSE
    mse = np.mean((img - reconstructed) ** 2)
    test_mses.append(mse)

    print(f"Image {i+1}: MSE = {mse:.6f}")

# average MSE
avg_mse = np.mean(test_mses)

print(f"\nAverage MSE: {avg_mse:.6f}")


# ===== SAVE RESULTS =====

print(f"\nSaving results...")

with open(os.path.join(RESULTS_DIR, "metrics.txt"), 'w') as f:
    f.write("AUTOENCODER METRICS - BSD500\n")
    f.write("=" * 60 + "\n")
    f.write(f"Training samples: {len(train_data)}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Test images: {len(test_images)}\n")
    f.write(f"\n")
    f.write(f"Final training loss: {history.history['loss'][-1]:.6f}\n")
    f.write(f"Final validation loss: {history.history['val_loss'][-1]:.6f}\n")
    f.write(f"Test MSE (avg): {avg_mse:.6f}\n")
    f.write("=" * 60 + "\n")

print("Saved metrics!")


# ===== COMPARE WITH BASELINE =====

print("\n" + "=" * 60)
print("COMPARISON WITH BASELINE")
print("=" * 60)

baseline_file = "results/baseline/baseline_metrics.txt"
if os.path.exists(baseline_file):
    with open(baseline_file, 'r') as f:
        for line in f:
            if 'Bilinear MSE' in line and 'avg' in line:
                baseline_mse = float(line.split(':')[1].strip())
                print(f"Baseline MSE:    {baseline_mse:.6f}")
                print(f"Autoencoder MSE: {avg_mse:.6f}")
                print("-" * 60)

                if avg_mse < baseline_mse:
                    improvement = ((baseline_mse - avg_mse) / baseline_mse) * 100
                    print(f"SUCCESS! {improvement:.1f}% better than baseline!")
                else:
                    worse = ((avg_mse - baseline_mse) / baseline_mse) * 100
                    print(f"Autoencoder is {worse:.1f}% worse")
                break
else:
    print("No baseline found. Run baseline.py first.")


print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
print(f"Model: {MODEL_PATH}")
print(f"Results: {RESULTS_DIR}/")

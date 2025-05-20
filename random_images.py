import os
import random
import shutil

# Paths
facade_dir = 'old_dataset/facade'
other_dir = 'old_dataset/other'
test_dir = 'old_dataset/test'

# Create the test folder if it doesn't exist
os.makedirs(test_dir, exist_ok=True)

# Get all image file paths
facade_images = [os.path.join(facade_dir, f) for f in os.listdir(facade_dir) if os.path.isfile(os.path.join(facade_dir, f))]
other_images = [os.path.join(other_dir, f) for f in os.listdir(other_dir) if os.path.isfile(os.path.join(other_dir, f))]

# Combine and shuffle
all_images = facade_images + other_images
random.shuffle(all_images)

# Select 20 random images
selected_images = all_images[:20]

# Copy to test directory
for img_path in selected_images:
    shutil.copy(img_path, test_dir)
    print(f"Copied: {img_path} -> {test_dir}")

print("✅ 20 images copied to 'old_dataset/test'")

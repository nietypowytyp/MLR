import os
import random
import shutil

def main():
    # Set paths
    train_images_dir = 'datasets/train/images'
    train_labels_dir = 'datasets/train/labels'
    valid_images_dir = 'datasets/test/images'
    valid_labels_dir = 'datasets/test/labels'

    # Create valid directories if they don't exist
    os.makedirs(valid_images_dir, exist_ok=True)
    os.makedirs(valid_labels_dir, exist_ok=True)

    # List all image files (assuming .jpg extension)
    image_files = [f for f in os.listdir(train_images_dir) if f.lower().endswith('.jpg')]

    # Calculate number of images to move (15%)
    num_to_move = max(1, int(len(image_files) * 0.15))

    # Randomly select images
    images_to_move = random.sample(image_files, num_to_move)

    for img_file in images_to_move:
        # Move image
        src_img = os.path.join(train_images_dir, img_file)
        dst_img = os.path.join(valid_images_dir, img_file)
        shutil.move(src_img, dst_img)

        # Move corresponding label
        label_file = os.path.splitext(img_file)[0] + '.txt'
        src_label = os.path.join(train_labels_dir, label_file)
        dst_label = os.path.join(valid_labels_dir, label_file)
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)

if __name__ == "__main__":
    main()
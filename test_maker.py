import os
import random
import shutil

def main():
    src_dir = r'datasets\test\images'
    dst_dir = 'test_images'
    os.makedirs(dst_dir, exist_ok=True)

    images = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    selected = random.sample(images, min(12, len(images)))

    for img in selected:
        shutil.copy(os.path.join(src_dir, img), os.path.join(dst_dir, img))

if __name__ == '__main__':
    main()
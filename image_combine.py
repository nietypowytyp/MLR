import os
from PIL import Image, ImageOps

def combine_images_from_folder(folder, margin=10, bg_color=(255, 255, 255)):
    # Get all image files in the folder (common image extensions)
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    files.sort()  # Ensure consistent order

    if len(files) != 12:
        print(f"Folder {folder} does not contain exactly 12 images. Skipping.")
        return

    # Open all images
    images = [Image.open(os.path.join(folder, f)) for f in files]

    # Optionally resize images to the same size (use the size of the first image)
    w, h = images[0].size
    images = [img.resize((w, h)) for img in images]

    # Arrange images in a 4x3 grid (4 columns, 3 rows)
    cols, rows = 4, 3

    # Calculate size of the combined image
    total_width = cols * w + (cols + 1) * margin
    total_height = rows * h + (rows + 1) * margin

    combined = Image.new('RGB', (total_width, total_height), bg_color)

    # Paste images
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = margin + col * (w + margin)
        y = margin + row * (h + margin)
        combined.paste(img, (x, y))

    # Save combined image
    out_path = os.path.join(folder, "combined.png")
    combined.save(out_path)
    print(f"Saved combined image to {out_path}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for i in range(6, 11):
        folder = os.path.join(base_dir, f"runs\detect\predict{i}")
        if os.path.isdir(folder):
            combine_images_from_folder(folder)
        else:
            print(f"Folder {folder} does not exist. Skipping.")

if __name__ == "__main__":
    main()
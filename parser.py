import os
import shutil

# Define the root folder
root_folder = 'dataset/other'

# Create target folders if they don't exist
facade_folder = os.path.join('dataset', 'facade')
other_folder = os.path.join('dataset', 'other')
os.makedirs(facade_folder, exist_ok=True)
os.makedirs(other_folder, exist_ok=True)

# Walk through all subdirectories and files
for dirpath, dirnames, filenames in os.walk(root_folder):
    for filename in filenames:
        # Skip already sorted files
        if dirpath in [facade_folder, other_folder]:
            continue

        file_path = os.path.join(dirpath, filename)

        # Determine target folder
        if 'exterior' in filename.lower():
            target_dir = facade_folder
        else:
            target_dir = other_folder

        # Avoid overwriting by making filename unique if necessary
        base_name, ext = os.path.splitext(filename)
        new_filename = filename
        counter = 1
        while os.path.exists(os.path.join(target_dir, new_filename)):
            new_filename = f"{base_name}_{counter}{ext}"
            counter += 1

        # Move file to the target directory
        shutil.move(file_path, os.path.join(target_dir, new_filename))
        print(f"Moved: {file_path} -> {os.path.join(target_dir, new_filename)}")

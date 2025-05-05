import os
import shutil
from zipfile import ZipFile

print("✅ Script started")

# Set Kaggle API credentials
kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)

# Check if kaggle.json exists in the current directory
if not os.path.exists("kaggle.json"):
    raise FileNotFoundError("❌ 'kaggle.json' not found in the current directory.")

# Move kaggle.json
shutil.move("kaggle.json", os.path.join(kaggle_dir, "kaggle.json"))
os.chmod(os.path.join(kaggle_dir, 'kaggle.json'), 0o600)

print("✅ Kaggle API credentials set up")

# Download the dataset
os.system("kaggle datasets download -d mikhailma/house-rooms-streets-image-dataset")

# Create project data directory
project_data_path = "house_rooms"
os.makedirs(project_data_path, exist_ok=True)

# Unzip the downloaded file
with ZipFile("house-rooms-streets-image-dataset.zip", 'r') as zip_ref:
    zip_ref.extractall(project_data_path)

print(f"✅ Dataset extracted to {project_data_path}")

# Remove the zip file after extraction
if os.path.exists("house-rooms-streets-image-dataset.zip"):
    os.remove("house-rooms-streets-image-dataset.zip") 
    print("✅ Zip file removed")
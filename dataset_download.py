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
os.system("kaggle datasets download -d tompaulat/modernarchitecture")

print("✅ Dataset downloaded")

# Create project data directory
project_data_path = "modern_architecture"
os.makedirs(project_data_path, exist_ok=True)

# Unzip the downloaded file
with ZipFile("modernarchitecture.zip", 'r') as zip_ref:
    zip_ref.extractall(project_data_path)

print(f"✅ Dataset extracted to {project_data_path}")

# Remove the zip file after extraction
if os.path.exists("modernarchitecture.zip"):
    os.remove("modernarchitecture.zip") 
    print("✅ Zip file removed")
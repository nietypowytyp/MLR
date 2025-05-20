# MLR

conda env create -f environment.yml
conda activate MLR_env

pip install torch==2.6.0+cu126 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu126

yolo train model=yolo11n.pt data=datasets/data.yaml epochs=10 imgsz=640 amp=False

yolo detect predict source=old_dataset/test model=runs/detect/train4/weights/best.pt
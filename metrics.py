from ultralytics import YOLO
import matplotlib.pyplot as plt

def main():    
    model_paths = [
        'runs/detect/train/weights/best.pt',
        'runs/detect/train2/weights/best.pt',
        'runs/detect/train3/weights/best.pt',
        'runs/detect/train4/weights/best.pt',
        'runs/detect/train5/weights/best.pt'
    ]


    for path in model_paths:
        model = YOLO(path)
        metrics = model.val()
        print(f"Results for model: {path}")
        print("mAP@0.5:", metrics.box.map)
        print("mAP@0.5:", metrics.box.map50)
        print("mAP@0.75:", metrics.box.map75)
        print("mAP per class:", metrics.box.maps)
        print("-" * 30)


if __name__ == "__main__":
    main()
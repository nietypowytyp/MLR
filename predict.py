from ultralytics import YOLO

def main():
    model_paths = [
        'runs/detect/train/weights/best.pt',
        'runs/detect/train2/weights/best.pt',
        'runs/detect/train3/weights/best.pt',
        'runs/detect/train4/weights/best.pt',
        'runs/detect/train5/weights/best.pt'
    ]

    source = 'test_images'

    for path in model_paths:
        model = YOLO(path)
        model.predict(source, save=True, conf=0.3)

if __name__ == "__main__":
    main()
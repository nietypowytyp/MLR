from ultralytics import YOLO

def main():
    model_11n = YOLO('runs/detect/train/weights/best.pt')
    model_11m = YOLO('runs/detect/train2/weights/best.pt')
    model_11x = YOLO('runs/detect/train3/weights/best.pt')
    model_8n = YOLO('runs/detect/train4/weights/best.pt')
    model_5n = YOLO('runs/detect/train5/weights/best.pt')

    source='datasets/test/images'

    model_11n.predict(source, save=True, conf=0.3)
    model_11m.predict(source, save=True, conf=0.3)
    model_11x.predict(source, save=True, conf=0.3)
    model_8n.predict(source, save=True, conf=0.3)
    model_5n.predict(source, save=True, conf=0.3)

if __name__ == "__main__":
    main()
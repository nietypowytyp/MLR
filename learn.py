from ultralytics import YOLO


def main():
    # Load YOLO models with different pre-trained weights
    model_11n = YOLO('yolo11n.pt')
    model_11m = YOLO('yolo11m.pt')
    model_11x = YOLO('yolo11x.pt')
    model_8n = YOLO('yolov8n.pt')
    model_5n = YOLO('yolov5n.pt')

    # Train yolo11n model
    model_11n.train(
        data='datasets/data.yaml',
        epochs=50,            # Number of training epochs
        imgsz=640,            # Input image size (pixels)
        amp=True,             # Use mixed precision for faster training
        patience=10,          # Early stopping patience (epochs)
        lr0=0.01,             # Initial learning rate
        batch=32,             # Batch size (number of images per batch)
        workers=4             # Number of data loader worker threads
    )

    # Train yolo11m model
    model_11m.train(
        data='datasets/data.yaml',
        epochs=80,
        imgsz=640,
        amp=True,
        patience=10,
        lr0=0.002,
        batch=8,
        workers=4
    )

    model_11x.train(
        data='datasets/data.yaml',
        epochs=120,
        imgsz=640,
        amp=True,
        patience=10,
        lr0=0.001,
        batch=4,
        workers=4
    )

    model_8n.train(
        data='datasets/data.yaml',
        epochs=50,
        imgsz=640,
        amp=True,
        patience=10,
        lr0=0.01,
        batch=32,
        workers=4
    )

    model_5n.train(
        data='datasets/data.yaml',
        epochs=50,
        imgsz=640,
        amp=True,
        patience=10,
        lr0=0.01,
        batch=32,
        workers=4
    )


if __name__ == "__main__":
    main()
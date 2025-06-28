from ultralytics import YOLO


def main():
    # Initialize the YOLO model with the specified weights
    model_11n = YOLO('yolo11n.pt')
    model_11m = YOLO('yolo11m.pt')
    model_11x = YOLO('yolo11x.pt')

    model_8n = YOLO('yolov8n.pt')
    model_5n = YOLO('yolov5n.pt')

    # # Train the model with the given parameters
    # model_11n.train(
    #     data='datasets/data.yaml',
    #     epochs=50,            # Increase epochs for more training
    #     imgsz=640,            # Use larger images for more detail
    #     amp=True,             # Enable mixed precision for faster training
    #     patience=10,          # Early stopping patience
    #     lr0=0.01,             # Initial learning rate
    #     batch=32,             # Increase batch size if GPU allows
    #     workers=4             # Number of data loader workers
    # )
    

    # # Train the model with the given parameters
    # model_11m.train(
    #     data='datasets/data.yaml',
    #     epochs=80,            # Increase epochs for more training
    #     imgsz=640,            # Use larger images for more detail
    #     amp=True,             # Enable mixed precision for faster training
    #     patience=10,          # Early stopping patience
    #     lr0=0.002,             # Initial learning rate
    #     batch=8,             # Increase batch size if GPU allows
    #     workers=4             # Number of data loader workers
    # )


    # # Train the model with the given parameters
    # model_11x.train(
    #     data='datasets/data.yaml',
    #     epochs=120,            # Increase epochs for more training
    #     imgsz=640,            # Use larger images for more detail
    #     amp=True,             # Enable mixed precision for faster training
    #     patience=10,          # Early stopping patience
    #     lr0=0.001,             # Initial learning rate
    #     batch=4,             # Increase batch size if GPU allows
    #     workers=4             # Number of data loader workers
    # )

    model_8n.train(
        data='datasets/data.yaml',
        epochs=50,            # Increase epochs for more training
        imgsz=640,            # Use larger images for more detail
        amp=True,             # Enable mixed precision for faster training
        patience=10,          # Early stopping patience
        lr0=0.01,             # Initial learning rate
        batch=32,             # Increase batch size if GPU allows
        workers=4             # Number of data loader workers
    )

    model_5n.train(
        data='datasets/data.yaml',
        epochs=50,            # Increase epochs for more training
        imgsz=640,            # Use larger images for more detail
        amp=True,             # Enable mixed precision for faster training
        patience=10,          # Early stopping patience
        lr0=0.01,             # Initial learning rate
        batch=32,             # Increase batch size if GPU allows
        workers=4             # Number of data loader workers
    )


if __name__ == "__main__":
    main()
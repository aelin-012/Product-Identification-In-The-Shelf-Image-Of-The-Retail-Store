from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8n.pt')  # You can change 'yolov8n.pt' to another YOLO model version (e.g., yolov8s.pt for small, yolov8m.pt for medium)

# Train the model
model.train(
    data="D:\dataset_roboflow_testing\data.yaml",  # Path to the dataset configuration YAML file
    epochs=100,                        # Number of epochs (adjust as needed)
    imgsz=640,                         # Image size for training (adjust as needed)
    batch=16,                          # Batch size (adjust as needed based on your system's capability)
    name='yolo_custom'                 # Name for the training run (optional)
)
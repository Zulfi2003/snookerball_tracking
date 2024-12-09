from ultralytics import YOLO

# Load the pre-trained YOLOv8 model (choose yolov8n.pt or yolov8s.pt)
model = YOLO('yolov8n.pt')  # or yolov8s.pt

# Start training
model.train(data='../dataset/data.yaml', epochs=50, imgsz=640, batch=16, device='cpu')
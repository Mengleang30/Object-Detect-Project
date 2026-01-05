from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Train on your mouse dataset
model.train(data="data_enhanced.yaml", epochs=50, imgsz=640, batch=16, name="mouse_and_cat_dataset")

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Train on your mouse dataset
# model.train(data="data.yaml", epochs=1, imgsz=640, batch=8, name="mouse_and_cat_detection")
model.train(data="data_enhanced.yaml", epochs=2, imgsz=640, batch=12, augment=True, name="mouse_and_cat_detection")

# model.train(
#     data="data.yaml",
#     epochs=1,
#     imgsz=320,
#     batch=32,
#     augment=False,
#     mosaic=0,
#     mixup=0,
#     hsv_h=0,
#     hsv_s=0,
#     hsv_v=0,
#     fliplr=0,
#     workers=4,
#     name="mouse_and_cat_training"
# )


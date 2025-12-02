from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')

model.train(
    data='dataset_raw/Images/Train',  # Đã có Train/Validate
    epochs=15,
    imgsz=112,
    batch=64,
    name='vietnamese_food_yolov8'
)

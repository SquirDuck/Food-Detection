from sklearn.metrics import classification_report
from PIL import Image
import numpy as np

# Load model đã train
model = YOLO('runs/classify/vietnamese_food_yolov8/weights/best.pt')

# Thư mục chứa tập validation/test (dạng ImageFolder của PyTorch: mỗi lớp là 1 folder)
val_dir = 'dataset_raw/Images/Validate'  

# Danh sách class từ tên folder
class_names = sorted(os.listdir(val_dir))  # đảm bảo đúng thứ tự như lúc train

# Mapping tên class sang chỉ số
class_to_idx = {name: idx for idx, name in enumerate(class_names)}

# Lưu nhãn thật và nhãn dự đoán
y_true = []
y_pred = []

# Duyệt qua từng class folder
for class_name in class_names:
    class_idx = class_to_idx[class_name]
    class_folder = os.path.join(val_dir, class_name)

    for image_name in os.listdir(class_folder):
        image_path = os.path.join(class_folder, image_name)

        # Dự đoán
        results = model.predict(image_path, verbose=False)[0]
        pred_idx = int(results.probs.top1)

        y_true.append(class_idx)
        y_pred.append(pred_idx)

# In classification report

print("== Classification Report ==")
print(classification_report(y_true, y_pred, target_names=class_names))

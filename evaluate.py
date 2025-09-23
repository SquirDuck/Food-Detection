# Đánh giá model
print("\n=== Evaluation Metrics ===")
print(metrics.__dict__)  # In toàn bộ thông tin metrics nếu cần debug

# Hàm tiện ích để lấy thuộc tính an toàn
def safe_get(metric_obj, attr, default='N/A'):
    return getattr(metric_obj, attr, default)

print(f"Top-1 Accuracy : {safe_get(metrics, 'top1', 0):.4f}")
print(f"Top-5 Accuracy : {safe_get(metrics, 'top5', 0):.4f}")
print(f"Precision      : {safe_get(metrics, 'precision', 0):.4f}")
print(f"Recall         : {safe_get(metrics, 'recall', 0):.4f}")
print(f"F1 Score       : {safe_get(metrics, 'f1', 0):.4f}")

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

def predict_food(img):
    # Dự đoán
    test_results = model.predict(source=img, save=False, show=False)
    result = test_results[0]

    # Lấy top5
    predictions = []
    for idx in result.probs.top5:
        name = result.names[idx]
        score = result.probs.data[idx].item()
        predictions.append(f"{name}: {score:.2%}")

    # Vẽ tên top1 lên ảnh
    top1_idx = result.probs.top1
    top1_name = result.names[top1_idx]
    top1_score = result.probs.data[top1_idx].item()

    image_pil = Image.open(img).convert("RGB")
    draw = ImageDraw.Draw(image_pil)
    text = f"{top1_name} ({top1_score:.2%})"
    draw.text((10, 10), text, fill="red")

    return image_pil, "\n".join(predictions)

iface = gr.Interface(
    fn=predict_food,
    inputs=gr.Image(type="filepath"),
    outputs=[gr.Image(type="pil"), gr.Textbox()],
    title="🍜 YOLOv8 - Nhận diện món ăn Việt Nam",
    description="Upload ảnh món ăn, YOLOv8 sẽ dự đoán tên món (Top5) và hiển thị lên ảnh."
)

iface.launch(share=True)

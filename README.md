# Food Detection

This project is about Food Image Detection using Deep Learning to recognition 30 unique foods in VietNam.

## Structure
- `notebooks/Food_detection.ipynb`: Original notebook
- `data_preprocessing.py`: Data loading and preprocessing
- `model.py`: Model definition
- `train.py`: Training script
- `evaluate.py`: Evaluation script
- `requirements.txt`: Dependencies

# Vietnamese Food Recognition

Vietnamese foods recognition with YOLOv8n-cls finetuned on Vietnamese Foods dataset (kaggle), includes Grad-CAM explainability and Gradio demo app.

This project use Deep Learning to classify and recognize food images, support related apps, webs deploy auto recommendation about diet, portion, dishes or study computer vision.

### Model & Dataset

Base model: YOLOv8n-cls (Ultralytics)

Input size: 224x224

Dataset: [30VN food](https://www.kaggle.com/datasets/quandang/vietnamese-foods)

Classes: 30 Vietnamese dishes
### Results
| Metric | Score |
|--------|--------|
| Top-1 Accuracy | 87.5% |
| Top-5 Accuracy | 96.3% |
| F1-score | 0.85 |

### Installation
```
git clone https://github.com/<yourname>/vietnamese-food-recognition.git
cd vietnamese-food-recognition
pip install -r requirements.txt
```

### Run Training

```
python model/yolo.py
```

### Evaluate

```
python model/eval.py
```

### Gradio

```
python model/grad.py
```



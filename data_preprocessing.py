!pip install ultralytics
!pip install -q kaggle
!pip install ultralytics gradio

# Upload kaggle.json
from google.colab import files
files.upload()  # Chọn tệp kaggle.json từ máy bạn

# Tạo thư mục cấu hình và di chuyển file vào đúng vị trí
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Tải bộ dữ liệu Vietnamese Foods từ Kaggle
!kaggle datasets download -d quandang/vietnamese-foods
!unzip vietnamese-foods.zip -d dataset_raw

import os
print(os.listdir('dataset_raw/Images/Validate'))
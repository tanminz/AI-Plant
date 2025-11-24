# Plant AI System 🌱

Hệ thống trí tuệ nhân tạo chuyên sâu trong nhận diện và giám sát thực vật, sử dụng ResNet50 để phát hiện bệnh lá cây.

## 🎯 Tổng quan

Plant AI System - **Plant Health Monitor** 🏥

* **Mục tiêu**: Phát hiện bệnh lá, sâu hại và nấm mốc
* **Mô hình**: ResNet50 (Transfer Learning)
* **Tính năng**:  
   * Phát hiện 39 loại bệnh trên 9+ loại cây trồng
   * Health scoring (0-100)
   * Khuyến nghị điều trị tự động
   * Giao diện web Flask

## 🚀 Cài đặt

### Yêu cầu hệ thống

* Python 3.10+
* CUDA 11.0+ (khuyến nghị)
* RAM: 8GB+ (16GB khuyến nghị)
* GPU: NVIDIA GTX 1060+ (khuyến nghị, có thể chạy CPU)

### Cài đặt dependencies

```bash
# Clone repository
git clone https://github.com/tanminz/AI-Plant.git
cd AI-Plant/plant_ai_system

# Tạo virtual environment
python -m venv .venv310
.\.venv310\Scripts\Activate.ps1  # Windows
# hoặc
source .venv310/bin/activate  # Linux/Mac

# Cài đặt dependencies
pip install -r requirements.txt
```

**Lưu ý**: Model file (`models/best_plant_model.pth`) không có trong repository. Bạn cần train lại bằng script `train_health_monitor.py` hoặc tải từ nguồn khác.

## 📁 Cấu trúc dự án

```
plant_ai_system/
├── app.py                          # Ứng dụng web Flask chính
├── train_health_monitor.py         # Script training model
├── config.json                     # Cấu hình hệ thống
├── disease_database.json            # Database thông tin bệnh
├── models/                         # Mô hình đã train
│   └── best_plant_model.pth        # ResNet50 model (cần train)
├── templates/                      # HTML templates
├── static/                         # Static files
├── data/                           # Dataset (nếu cần training)
└── utils/                          # Utilities
```

## 🎮 Sử dụng

### Chạy ứng dụng web

```bash
cd plant_ai_system
python app.py
```

Mở browser: **http://localhost:5000**

### Sử dụng trong code

```python
from app import PlantAIModel

# Khởi tạo model
plant_ai = PlantAIModel()

# Phân tích cây trồng
result = plant_ai.predict_image("path/to/plant.jpg")

# In kết quả
print(f"Predicted: {result['predicted_display_name']}")
print(f"Health Score: {result['health_analysis']['score']}/100")
```

## 🧪 Training Model

```bash
python train_health_monitor.py
```

Script sẽ train ResNet50 trên dataset và lưu model vào `models/best_plant_model.pth`

## 🔧 Cấu hình

Chỉnh sửa `config.json` để tùy chỉnh:

```json
{
  "training": {
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 100
  }
}
```

## 📊 Kết quả mẫu

```json
{
  "predicted_class": "Apple_Apple_Scab",
  "confidence": 0.852,
  "health_analysis": {
    "health_score": 45,
    "severity": "High",
    "recommendations": ["Apply fungicide", "Remove infected leaves"]
  }
}
```

## 📄 License

Distributed under the MIT License.

## 📞 Liên hệ

* **Project Link**: [https://github.com/tanminz/AI-Plant](https://github.com/tanminz/AI-Plant)


# 📁 CẤU TRÚC DỰ ÁN PLANT AI SYSTEM

## 🎯 Tổng quan

Dự án đã được làm sạch, chỉ giữ lại các file cần thiết để chạy hệ thống.

## 📂 Cấu trúc thư mục

```
plant_ai_system/
├── app.py                          # ✅ FILE CHÍNH - Ứng dụng web Flask
├── config.json                     # ✅ Cấu hình hệ thống
├── disease_database.json            # ✅ Database thông tin bệnh cây
├── train_health_monitor.py         # ✅ Script training model (tùy chọn)
│
├── models/                         # ✅ Thư mục chứa model
│   └── best_plant_model.pth        # ✅ MODEL DUY NHẤT - ResNet50 trained
│
├── templates/                      # ✅ HTML templates cho web
│   ├── base.html
│   ├── index.html
│   ├── history.html
│   └── about.html
│
├── static/                         # ✅ Static files cho web
│   ├── images/                     # Hình ảnh static
│   ├── uploads/                    # Ảnh người dùng upload
│   └── results/                    # Kết quả phân tích
│
├── data/                           # ✅ Dataset (nếu cần training lại)
│   └── health_monitoring/
│       └── mega_dataset/           # Dataset training/validation/test
│           ├── train/
│           ├── val/
│           └── test/
│
└── utils/                          # ✅ Utilities (cho training)
    ├── __init__.py
    ├── build_mega_dataset.py
    ├── data_loader.py
    ├── dataset_catalog_builder.py
    └── dataset_integration.py
```

## 🔧 Model được sử dụng

### ResNet50
- **File model**: `models/best_plant_model.pth`
- **Kiến trúc**: ResNet50 từ torchvision
- **Mục đích**: Classification bệnh lá cây (39 classes)
- **Được load trong**: `app.py` → `PlantAIModel.load_model()`

### Quy trình load model:

```python
# 1. Khi app.py khởi động:
plant_ai = PlantAIModel()

# 2. Trong __init__():
self.load_model()

# 3. Trong load_model():
checkpoint = torch.load("models/best_plant_model.pth")
self.model = models.resnet50(pretrained=False)
self.model.fc = nn.Linear(model.fc.in_features, len(classes))
self.model.load_state_dict(checkpoint['model_state_dict'])
self.model.eval()
```

## 📝 File quan trọng

### 1. `app.py` - Ứng dụng chính
- Flask web application
- Load ResNet50 model
- Xử lý upload ảnh
- Dự đoán bệnh cây
- Trả về kết quả JSON

### 2. `config.json` - Cấu hình
- Đường dẫn dataset
- Đường dẫn model
- Tham số inference
- Danh sách classes

### 3. `disease_database.json` - Database bệnh
- Thông tin chi tiết về các bệnh
- Khuyến nghị điều trị
- Mức độ nghiêm trọng

### 4. `train_health_monitor.py` - Training script
- Training ResNet50 trên dataset
- Lưu model vào `models/best_plant_model.pth`
- Chỉ cần nếu muốn retrain model

## 🚀 Chạy hệ thống

### Yêu cầu:
- Python 3.10
- PyTorch với CUDA (khuyến nghị)
- Flask
- Các dependencies trong `requirements.txt`

### Cách chạy:
```powershell
# 1. Activate virtual environment
.\.venv310\Scripts\Activate.ps1

# 2. Chạy ứng dụng
cd plant_ai_system
python app.py

# 3. Mở browser: http://localhost:5000
```

## 📊 Dataset

- **Vị trí**: `data/health_monitoring/mega_dataset/`
- **Cấu trúc**: train/val/test với 39 classes
- **Chỉ cần nếu**: Muốn training lại model

## 🗑️ Đã xóa (không cần thiết)

- ❌ Các file training khác (advanced_training.py, complete_training.py, ...)
- ❌ Các file demo/test (demo.py, test_*.py)
- ❌ Module A và Module B (không được sử dụng)
- ❌ Model checkpoints dư thừa (epoch files)
- ❌ Dataset không cần thiết (plantclef2022, plant_leaf_diseases)
- ❌ File CLI (main.py)

## ✅ Kết quả

Dự án đã được đơn giản hóa, chỉ giữ lại:
- ✅ 1 file chính: `app.py`
- ✅ 1 model: `best_plant_model.pth` (ResNet50)
- ✅ 1 script training: `train_health_monitor.py` (tùy chọn)
- ✅ Templates và static files cho web
- ✅ Dataset (nếu cần training lại)





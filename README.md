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

### Bước 1: Clone repository

```bash
git clone https://github.com/tanminz/AI-Plant.git
cd AI-Plant/plant_ai_system
```

### Bước 2: Tạo virtual environment

**Windows:**
```powershell
python -m venv .venv310
.\.venv310\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python3 -m venv .venv310
source .venv310/bin/activate
```

### Bước 3: Cài đặt dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Bước 4: Tải Model (QUAN TRỌNG)

Model file (`best_plant_model.pth`) không có trong repository do kích thước lớn (~270MB). Bạn **BẮT BUỘC** phải tải model để hệ thống hoạt động.

#### Cách 1: Tải từ Google Drive (Khuyến nghị)

1. **Tải model từ Google Drive:**
   - Link: [Google Drive - best_plant_model.pth](https://drive.google.com/drive/folders/1HkZVhltsz-9gD-stT41bHC5BaThKs4wR ) 
   - Hoặc tìm file `best_plant_model.pth` trong thư mục chia sẻ

2. **Đặt model vào đúng vị trí:**
   ```
   plant_ai_system/
   └── models/
       └── best_plant_model.pth  ← Đặt file vào đây
   ```

3. **Kiểm tra:**
   ```bash
   # Windows
   dir models\best_plant_model.pth
   
   # Linux/Mac
   ls -lh models/best_plant_model.pth
   ```
   
   File phải có kích thước khoảng **270MB**.

#### Cách 2: Train lại model (Nếu không tải được)

Nếu không thể tải model từ Drive, bạn có thể train lại:

```bash
python train_health_monitor.py
```

**Lưu ý:** Cần có dataset tại `data/health_monitoring/mega_dataset/` để train.

### Bước 5: Cấu hình (Tùy chọn)

Tạo file `.env` trong thư mục `plant_ai_system/` để sử dụng OpenAI API:

```env
OPENAI_API_KEY=your_api_key_here
```

**Lưu ý:** OpenAI API là tùy chọn, hệ thống vẫn hoạt động bình thường nếu không có.

## 📁 Cấu trúc dự án

```
plant_ai_system/
├── app.py                          # Ứng dụng web Flask chính
├── train_health_monitor.py         # Script training model
├── config.json                     # Cấu hình hệ thống
├── disease_database.json            # Database thông tin bệnh
├── models/                         # Thư mục chứa model
│   └── best_plant_model.pth        # ResNet50 model (CẦN TẢI TỪ DRIVE)
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

Sau đó mở browser và truy cập: **http://localhost:5000**

### Kiểm tra model đã load chưa

Khi chạy `app.py`, console sẽ hiển thị:
```
Model loaded: True    ← Phải là True
Classes: 39          ← Phải có 39 classes
```

Nếu hiển thị `Model loaded: False`, nghĩa là model chưa được tải đúng cách.

### Sử dụng trong code

```python
from app import PlantAIModel

# Khởi tạo model
plant_ai = PlantAIModel()

# Kiểm tra model đã load chưa
if plant_ai.model is None:
    print("Model chưa được load! Vui lòng kiểm tra file model.")
else:
    # Phân tích cây trồng
    result = plant_ai.predict_image("path/to/plant.jpg")
    print(f"Predicted: {result['predicted_display_name']}")
    print(f"Health Score: {result['health_analysis']['score']}/100")
```

## 🧪 Training Model (Tùy chọn)

Nếu muốn train lại model hoặc train với dataset mới:

```bash
python train_health_monitor.py
```

Script sẽ:
1. Load dataset từ `data/health_monitoring/mega_dataset/`
2. Train ResNet50 với các hyperparameters trong `config.json`
3. Lưu model tốt nhất vào `models/best_plant_model.pth`

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

## ⚠️ Troubleshooting

### Lỗi: Model not found

**Nguyên nhân:** File `models/best_plant_model.pth` không tồn tại hoặc không đúng vị trí.

**Giải pháp:**
1. Kiểm tra file có tồn tại: `ls models/best_plant_model.pth`
2. Tải lại model từ Google Drive: https://drive.google.com/drive/folders/1HkZVhltsz-9gD-stT41bHC5BaThKs4wR 
3. Đảm bảo file được đặt đúng: `plant_ai_system/models/best_plant_model.pth`

### Lỗi: Model is not loaded

**Nguyên nhân:** Model file bị lỗi hoặc không tương thích.

**Giải pháp:**
1. Xóa file model cũ và tải lại từ Drive
2. Hoặc train lại model bằng `train_health_monitor.py`

### Lỗi: CUDA out of memory

**Giải pháp:**
- Giảm batch_size trong `config.json`
- Hoặc chạy trên CPU (tự động fallback)

### Lỗi: Module not found

**Giải pháp:**
```bash
# Kiểm tra virtual environment đã activate
# Cài đặt lại dependencies
pip install -r requirements.txt
```

### Lỗi: Port 5000 already in use

**Giải pháp:**
- Thay đổi port trong `app.py`: `app.run(port=5001)`
- Hoặc đóng ứng dụng đang dùng port 5000

## 📄 License

Distributed under the MIT License.

## 📞 Liên hệ

* **Project Link**: [https://github.com/tanminz/AI-Plant](https://github.com/tanminz/AI-Plant)
* **Model Download**: [Google Drive Link] *(Cần cập nhật link thực tế)*

---

**Lưu ý quan trọng:** Hệ thống **KHÔNG THỂ** hoạt động nếu thiếu file model. Vui lòng tải model từ Google Drive trước khi sử dụng.

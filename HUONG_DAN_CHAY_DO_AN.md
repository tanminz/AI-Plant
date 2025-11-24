# 🌱 HƯỚNG DẪN CHẠY ĐỒ ÁN PLANT AI SYSTEM

## 📋 Mục lục
1. [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
2. [Cài đặt môi trường](#cài-đặt-môi-trường)
3. [Cấu hình dự án](#cấu-hình-dự-án)
4. [Chạy ứng dụng Web](#chạy-ứng-dụng-web)
5. [Training mô hình](#training-mô-hình)
6. [Sử dụng Command Line](#sử-dụng-command-line)
7. [Xử lý lỗi thường gặp](#xử-lý-lỗi-thường-gặp)

---

## 🖥️ Yêu cầu hệ thống

### Phần cứng tối thiểu:
- **CPU**: Intel Core i5 hoặc tương đương
- **RAM**: 8GB (khuyến nghị 16GB+)
- **GPU**: NVIDIA GTX 1060 6GB trở lên (khuyến nghị cho training)
- **Ổ cứng**: 50GB dung lượng trống

### Phần mềm:
- **Python**: 3.8 trở lên
- **CUDA**: 11.0+ (nếu có GPU)
- **cuDNN**: 8.0+ (nếu có GPU)
- **Git**: Để clone repository

---

## 🔧 Cài đặt môi trường

### ⚠️ LƯU Ý QUAN TRỌNG:
- **Python 3.10** được khuyến nghị vì CUDA hoạt động tốt nhất với Python 3.10
- Nếu bạn đã có virtual environment với Python 3.10 (ví dụ: `.venv310`), hãy sử dụng nó!

### Bước 1: Clone repository (nếu chưa có)
```bash
git clone https://github.com/tanminz/AI-Plant.git
cd AI-Plant
```

### Bước 2: Tạo virtual environment với Python 3.10

**Cách 1: Tự động (Khuyến nghị - Windows):**
```powershell
# Chạy script tự động
.\setup_venv.ps1
```

**Cách 2: Thủ công (Windows):**
```powershell
# Tạo venv với Python 3.10
python -m venv plant_ai_env
# hoặc nếu có nhiều Python version:
py -3.10 -m venv plant_ai_env

# Activate venv
.\plant_ai_env\Scripts\Activate.ps1
# Nếu gặp lỗi execution policy, chạy:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Cách 3: Sử dụng venv đã có (nếu bạn đã tạo `.venv310`):**
```powershell
# Activate venv Python 3.10 đã có
.\.venv310\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python3.10 -m venv plant_ai_env
source plant_ai_env/bin/activate
```

### Bước 3: Cài đặt dependencies

**⚠️ QUAN TRỌNG: Đảm bảo virtual environment đã được activate trước khi cài đặt!**

```bash
# Kiểm tra venv đã activate (sẽ thấy (plant_ai_env) hoặc (.venv310) ở đầu dòng)
# Nếu chưa thấy, activate lại:
# Windows:
.\plant_ai_env\Scripts\Activate.ps1
# hoặc nếu dùng .venv310:
.\.venv310\Scripts\Activate.ps1

# Nâng cấp pip
python -m pip install --upgrade pip

# Cài đặt PyTorch với CUDA (Python 3.10 + CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Cài đặt các package cơ bản
pip install -r requirements.txt

# Cài đặt Detectron2 (cho Mask R-CNN) - Chỉ cần nếu sử dụng Module B với YOLO
# Windows (CUDA 11.8):
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html

# Linux (CUDA 11.8):
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
```

### Bước 4: Kiểm tra cài đặt

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## ⚙️ Cấu hình dự án

### 1. Kiểm tra file config.json

File `plant_ai_system/config.json` đã được cấu hình sẵn. Bạn có thể chỉnh sửa nếu cần:

```json
{
  "species_model_type": "clip",
  "confidence_threshold": 0.5,
  "top_k_species": 5,
  "data_paths": {
    "plant_leaf_diseases": "data/health_monitoring/mega_dataset"
  }
}
```

### 2. Cấu hình OpenAI API (Tùy chọn)

Nếu muốn sử dụng tính năng tư vấn điều trị từ OpenAI:

1. Tạo file `.env` trong thư mục `plant_ai_system`:
```bash
OPENAI_API_KEY=your_api_key_here
```

2. Hoặc set biến môi trường:
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your_api_key_here"

# Linux/Mac
export OPENAI_API_KEY="your_api_key_here"
```

### 3. Kiểm tra dữ liệu

Đảm bảo dataset đã được đặt đúng vị trí:
- Dataset sức khỏe: `plant_ai_system/data/health_monitoring/mega_dataset/`
- Dataset nhận dạng loài: `plant_ai_system/data/plantclef2022/`

---

## 🌐 Chạy ứng dụng Web

### ⚠️ QUAN TRỌNG: Luôn activate virtual environment trước khi chạy!

### Cách 1: Chạy tự động với script (Khuyến nghị - Windows)

```powershell
# Chạy file batch (tự động activate venv và chạy app)
.\CHAY_UNG_DUNG.bat
```

### Cách 2: Chạy thủ công

```powershell
# Bước 1: Activate virtual environment
.\plant_ai_env\Scripts\Activate.ps1
# hoặc nếu dùng .venv310:
.\.venv310\Scripts\Activate.ps1

# Bước 2: Di chuyển vào thư mục plant_ai_system
cd plant_ai_system

# Bước 3: Chạy ứng dụng Flask
python app.py
```

Sau đó mở trình duyệt và truy cập: **http://localhost:5000**

### Cách 2: Chạy với Flask CLI

```bash
cd plant_ai_system
flask --app app run --host=0.0.0.0 --port=5000
```

### Tính năng Web App:
- ✅ Upload ảnh và nhận diện bệnh
- ✅ Hiển thị top 3 dự đoán
- ✅ Phân tích sức khỏe cây trồng
- ✅ Tư vấn điều trị (nếu có OpenAI API)
- ✅ Xem lịch sử phân tích
- ✅ Lọc theo loại cây

### Lưu ý:
- Model cần được train trước (file `models/best_plant_model.pth` phải tồn tại)
- Nếu chưa có model, xem phần [Training mô hình](#training-mô-hình)

---

## 🎓 Training mô hình

### 1. Training Module B - Health Monitor (Phát hiện bệnh lá)

Đây là module chính để phát hiện bệnh trên lá cây:

```bash
cd plant_ai_system
python train_health_monitor.py
```

**Cấu hình training:**
- Batch size: 32
- Learning rate: 0.001
- Epochs: 100 (có early stopping)
- Model: ResNet50

**Kết quả:**
- Model được lưu tại: `models/health_monitor/best_classification_model.pth`
- Training report: `models/health_monitor/training_report.json`

### 2. Training Module A - Species Recognition (Nhận dạng loài)

```bash
cd plant_ai_system
python train_species_recognition.py
```

**Lưu ý:** Cần có dataset PlantCLEF 2022 để training module này.

### 3. Training Complete Model (Tất cả modules)

```bash
cd plant_ai_system
python complete_training.py
```

### 4. Training với cấu hình tùy chỉnh

Bạn có thể chỉnh sửa các tham số trong file training script hoặc tạo script riêng:

```python
from train_health_monitor import HealthMonitorTrainer
import json

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

# Training config
training_config = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'early_stopping_patience': 20
}

# Initialize trainer
trainer = HealthMonitorTrainer(training_config, use_yolo=False)

# Start training
dataset_path = config['data_paths']['plant_leaf_diseases']
report = trainer.train(dataset_path)
```

---

## 💻 Sử dụng Command Line

### 1. Phân tích ảnh đơn lẻ

```bash
cd plant_ai_system
python main.py --image path/to/plant_image.jpg --output results/
```

### 2. Phân tích hàng loạt

```bash
python main.py --batch path/to/image_directory/ --output results/
```

### 3. Phân tích với dữ liệu môi trường

Tạo file `environmental_data.json`:
```json
{
  "temperature": 25.5,
  "humidity": 65.0,
  "ph": 6.8,
  "light_intensity": 1200
}
```

Chạy:
```bash
python main.py --image plant.jpg --env-data environmental_data.json
```

### 4. Sử dụng trong Python code

```python
from main import PlantAISystem

# Khởi tạo hệ thống
plant_ai = PlantAISystem()

# Phân tích cây trồng
result = plant_ai.analyze_plant("path/to/plant.jpg")

# In kết quả
print(f"Species: {result['species_recognition']['most_likely_species']}")
print(f"Health Score: {result['health_analysis']['health_score']}")
print(f"Recommendations: {result['integrated_recommendations']}")
```

---

## 🧪 Test và Demo

### 1. Test model đã train

```bash
cd plant_ai_system
python test_model.py
```

### 2. Test web app

```bash
python test_web_app.py
```

### 3. Chạy demo tự động

```bash
python auto_demo.py
```

---

## 🔍 Xử lý lỗi thường gặp

### Lỗi 1: ModuleNotFoundError: No module named 'torch'

**Nguyên nhân:** 
- Chưa activate virtual environment
- PyTorch chưa được cài đặt trong venv
- Đang dùng Python global thay vì Python trong venv

**Giải pháp:**
```powershell
# Bước 1: Activate virtual environment (QUAN TRỌNG!)
.\.venv310\Scripts\Activate.ps1
# hoặc
.\plant_ai_env\Scripts\Activate.ps1

# Bước 2: Kiểm tra Python đang dùng (phải là Python trong venv)
python --version
where python  # Windows - phải chỉ đến venv\Scripts\python.exe

# Bước 3: Cài đặt PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Bước 4: Kiểm tra lại
python -c "import torch; print('OK')"
```

**Lưu ý:** Luôn đảm bảo thấy `(.venv310)` hoặc `(plant_ai_env)` ở đầu dòng PowerShell trước khi chạy!

### Lỗi 2: CUDA out of memory

**Nguyên nhân:** GPU không đủ bộ nhớ

**Giải pháp:**
- Giảm batch size trong config
- Sử dụng CPU thay vì GPU
- Đóng các ứng dụng khác đang dùng GPU

### Lỗi 3: Model not found

**Nguyên nhân:** Chưa train model hoặc đường dẫn sai

**Giải pháp:**
```bash
# Kiểm tra model có tồn tại
ls models/best_plant_model.pth

# Nếu chưa có, chạy training
python train_health_monitor.py
```

### Lỗi 4: Dataset not found

**Nguyên nhân:** Dataset chưa được đặt đúng vị trí

**Giải pháp:**
- Kiểm tra đường dẫn trong `config.json`
- Đảm bảo cấu trúc thư mục:
  ```
  data/health_monitoring/mega_dataset/
    ├── train/
    ├── val/
    └── test/
  ```

### Lỗi 5: Port 5000 already in use

**Nguyên nhân:** Port đã được sử dụng

**Giải pháp:**
```bash
# Windows: Tìm và kill process
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:5000 | xargs kill -9

# Hoặc chạy trên port khác
python app.py  # Sửa port trong code
```

### Lỗi 6: OpenAI API không hoạt động

**Nguyên nhân:** Chưa set API key hoặc key không hợp lệ

**Giải pháp:**
- Kiểm tra file `.env` hoặc biến môi trường
- Ứng dụng vẫn chạy được nhưng không có tư vấn từ OpenAI

---

## 📊 Cấu trúc thư mục quan trọng

```
plant_ai_system/
├── app.py                          # Ứng dụng web chính
├── main.py                         # CLI application
├── config.json                     # File cấu hình
├── train_health_monitor.py         # Training script Module B
├── train_species_recognition.py    # Training script Module A
├── models/                         # Thư mục chứa models
│   ├── best_plant_model.pth       # Model chính (cần có để chạy web)
│   └── health_monitor/            # Models Module B
├── data/                          # Datasets
│   └── health_monitoring/
│       └── mega_dataset/          # Dataset training
├── static/                        # Static files cho web
│   ├── uploads/                   # Ảnh upload
│   └── results/                   # Kết quả phân tích
└── templates/                     # HTML templates
```

---

## 🚀 Quick Start (Tóm tắt nhanh)

### Để chạy Web App ngay:

**Windows (Cách nhanh nhất):**
```powershell
# Chạy file batch tự động
.\CHAY_UNG_DUNG.bat
```

**Windows (Thủ công):**
```powershell
# 1. Activate virtual environment (Python 3.10)
.\.venv310\Scripts\Activate.ps1
# hoặc
.\plant_ai_env\Scripts\Activate.ps1

# 2. Di chuyển vào thư mục
cd plant_ai_system

# 3. Chạy ứng dụng
python app.py

# 4. Mở browser: http://localhost:5000
```

**Linux/Mac:**
```bash
# 1. Activate virtual environment
source plant_ai_env/bin/activate

# 2. Di chuyển vào thư mục
cd plant_ai_system

# 3. Chạy ứng dụng
python app.py

# 4. Mở browser: http://localhost:5000
```

### Để training model:

```bash
# 1. Activate virtual environment
# 2. Di chuyển vào thư mục
cd plant_ai_system

# 3. Chạy training
python train_health_monitor.py

# 4. Đợi training hoàn thành (có thể mất vài giờ)
```

---

## 📞 Hỗ trợ

Nếu gặp vấn đề:
1. Kiểm tra lại các bước cài đặt
2. Xem phần [Xử lý lỗi thường gặp](#xử-lý-lỗi-thường-gặp)
3. Kiểm tra log trong terminal để xem lỗi chi tiết
4. Đảm bảo đã cài đặt đầy đủ dependencies

---

## 📝 Ghi chú

- **Training time:** Tùy thuộc vào GPU và kích thước dataset, có thể mất từ 2-10 giờ
- **Inference time:** ~150ms trên GPU, ~800ms trên CPU
- **Model size:** ~500MB cho classification model
- **Dataset:** Cần ít nhất 10GB dung lượng cho dataset

---

**Chúc bạn thành công với đồ án! 🌱**


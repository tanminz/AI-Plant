# 🌱 Plant AI System - Hệ Thống Nhận Diện Bệnh Cây Trồng

Hệ thống trí tuệ nhân tạo chuyên sâu trong nhận diện và giám sát sức khỏe thực vật, phát hiện bệnh lá, sâu hại và nấm mốc.

---

## 📋 Mục lục

1. [Tổng quan dự án](#tổng-quan-dự-án)
2. [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
3. [Cài đặt môi trường](#cài-đặt-môi-trường)
4. [Chạy ứng dụng Web](#chạy-ứng-dụng-web)
5. [Training mô hình](#training-mô-hình)
6. [Cấu trúc dự án](#cấu-trúc-dự-án)
7. [Xử lý lỗi thường gặp](#xử-lý-lỗi-thường-gặp)

---

## 🎯 Tổng quan dự án

Plant AI System là hệ thống web application sử dụng Deep Learning để:
- ✅ **Phát hiện bệnh lá cây trồng**: Nhận diện 39 loại bệnh và trạng thái khỏe mạnh
- ✅ **Phân tích sức khỏe cây**: Đánh giá tình trạng sức khỏe và đưa ra khuyến nghị
- ✅ **Tư vấn điều trị**: Tích hợp OpenAI API để tư vấn cách xử lý bệnh
- ✅ **Lịch sử phân tích**: Lưu trữ và xem lại các lần phân tích trước

### Công nghệ sử dụng:
- **Backend**: Flask (Python)
- **Deep Learning**: PyTorch, ResNet50
- **Frontend**: HTML, CSS, JavaScript
- **AI Model**: ResNet50 fine-tuned cho 39 classes

---

## 🖥️ Yêu cầu hệ thống

### Phần cứng tối thiểu:
- **CPU**: Intel Core i5 hoặc tương đương
- **RAM**: 8GB (khuyến nghị 16GB+)
- **GPU**: NVIDIA GTX 1060 6GB trở lên (tùy chọn, có thể chạy trên CPU)
- **Ổ cứng**: 10GB dung lượng trống (cho model và dataset)

### Phần mềm:
- **Python**: 3.8 trở lên (khuyến nghị Python 3.10)
- **CUDA**: 11.0+ (nếu có GPU, tùy chọn)
- **Hệ điều hành**: Windows 10/11, Linux, macOS

---

## 🔧 Cài đặt môi trường

### Bước 1: Kiểm tra Python

Mở PowerShell hoặc Command Prompt và kiểm tra:

```powershell
python --version
# Hoặc
py --version
```

Nếu chưa có Python, tải và cài đặt từ [python.org](https://www.python.org/downloads/)

### Bước 2: Tạo Virtual Environment

**Windows (PowerShell):**
```powershell
# Tạo virtual environment
python -m venv plant_ai_env
# hoặc nếu có nhiều Python version:
py -3.10 -m venv plant_ai_env

# Activate virtual environment
.\plant_ai_env\Scripts\Activate.ps1

# Nếu gặp lỗi execution policy, chạy:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Windows (Command Prompt):**
```cmd
python -m venv plant_ai_env
plant_ai_env\Scripts\activate.bat
```

**Linux/Mac:**
```bash
python3 -m venv plant_ai_env
source plant_ai_env/bin/activate
```

**✅ Kiểm tra đã activate thành công:**
- Bạn sẽ thấy `(plant_ai_env)` ở đầu dòng terminal
- Chạy: `where python` (Windows) hoặc `which python` (Linux/Mac) → phải chỉ đến thư mục venv

### Bước 3: Nâng cấp pip

```bash
python -m pip install --upgrade pip
```

### Bước 4: Cài đặt PyTorch

**Với GPU (CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Chỉ CPU (không có GPU):**
```bash
pip install torch torchvision torchaudio
```

**Kiểm tra cài đặt:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Bước 5: Cài đặt các dependencies

```bash
# Cài đặt từ file requirements.txt
pip install -r requirements.txt
```

**Lưu ý:** Nếu gặp lỗi với một số package, bạn có thể cài đặt từng package quan trọng:

```bash
pip install Flask>=2.3.0
pip install Pillow>=10.0.0
pip install numpy>=1.24.0
pip install python-dotenv>=1.0.0
pip install openai>=1.0.0  # Tùy chọn, chỉ cần nếu dùng OpenAI API
```

---

## 🌐 Chạy ứng dụng Web

### ⚠️ QUAN TRỌNG: Luôn activate virtual environment trước khi chạy!

### Cách 1: Chạy tự động (Khuyến nghị - Windows)

**Sử dụng file batch:**
```powershell
.\CHAY_UNG_DUNG.bat
```

File này sẽ tự động:
1. ✅ Tìm và activate virtual environment (`.venv310` hoặc `plant_ai_env`)
2. ✅ Di chuyển vào thư mục `plant_ai_system`
3. ✅ Chạy ứng dụng Flask

**Sử dụng PowerShell script:**
```powershell
.\CHAY_APP.ps1
```

### Cách 2: Chạy thủ công

**Bước 1: Activate virtual environment**
```powershell
# Windows PowerShell
.\plant_ai_env\Scripts\Activate.ps1

# Windows Command Prompt
plant_ai_env\Scripts\activate.bat

# Linux/Mac
source plant_ai_env/bin/activate
```

**Bước 2: Di chuyển vào thư mục**
```bash
cd plant_ai_system
```

**Bước 3: Chạy ứng dụng**
```bash
python app.py
```

**Bước 4: Mở trình duyệt**

Truy cập: **http://localhost:5000**

### Tính năng Web App:

- 📤 **Upload ảnh**: Upload ảnh lá cây để phân tích
- 🔍 **Nhận diện bệnh**: Hiển thị top 3 dự đoán với độ tin cậy
- 📊 **Phân tích sức khỏe**: Đánh giá tình trạng sức khỏe cây trồng
- 💡 **Tư vấn điều trị**: Khuyến nghị cách xử lý (nếu có OpenAI API)
- 📜 **Lịch sử**: Xem lại các lần phân tích trước
- 🔎 **Lọc theo loại cây**: Tìm kiếm và lọc kết quả

### Lưu ý quan trọng:

- ⚠️ **Model phải tồn tại**: File `plant_ai_system/models/best_plant_model.pth` phải có sẵn
- ⚠️ **Dataset cấu trúc**: Dataset phải được đặt đúng vị trí (xem phần [Cấu trúc dự án](#cấu-trúc-dự-án))
- ⚠️ **Port 5000**: Đảm bảo port 5000 không bị chiếm dụng

---

## 🎓 Training mô hình

### Training Health Monitor Model (Phát hiện bệnh lá)

Đây là model chính để phát hiện bệnh trên lá cây:

**Bước 1: Chuẩn bị dataset**

Dataset phải có cấu trúc:
```
plant_ai_system/data/health_monitoring/mega_dataset/
├── train/
│   ├── Apple_Apple_Scab/
│   ├── Apple_Black_Rot/
│   ├── Apple_Healthy/
│   └── ... (các class khác)
├── val/
│   └── ... (tương tự train)
└── test/
    └── ... (tương tự train)
```

**Bước 2: Chạy training**

```bash
# Activate virtual environment
.\plant_ai_env\Scripts\Activate.ps1  # Windows
# hoặc
source plant_ai_env/bin/activate  # Linux/Mac

# Di chuyển vào thư mục
cd plant_ai_system

# Chạy training
python train_health_monitor.py
```

**Cấu hình training (mặc định):**
- Model: ResNet50
- Batch size: 32
- Learning rate: 0.001
- Epochs: 100 (có early stopping)
- Optimizer: Adam

**Kết quả:**
- Model được lưu tại: `plant_ai_system/models/best_plant_model.pth`
- Training logs và metrics được hiển thị trong terminal

**Lưu ý:**
- Training có thể mất từ 2-10 giờ tùy vào GPU và kích thước dataset
- Nếu không có GPU, training sẽ chạy trên CPU (chậm hơn nhiều)
- Đảm bảo có đủ RAM (khuyến nghị 16GB+)

---

## 📁 Cấu trúc dự án

```
Plant AI/
├── plant_ai_system/              # Thư mục chính của ứng dụng
│   ├── app.py                    # Ứng dụng Flask chính ⭐
│   ├── train_health_monitor.py   # Script training model
│   ├── config.json               # File cấu hình
│   ├── disease_database.json     # Database thông tin bệnh
│   │
│   ├── models/                   # Thư mục chứa models
│   │   └── best_plant_model.pth  # Model đã train (CẦN CÓ ĐỂ CHẠY APP)
│   │
│   ├── data/                     # Datasets
│   │   └── health_monitoring/
│   │       └── mega_dataset/     # Dataset training
│   │           ├── train/        # Dữ liệu training
│   │           ├── val/          # Dữ liệu validation
│   │           └── test/         # Dữ liệu test
│   │
│   ├── static/                   # Static files cho web
│   │   ├── uploads/              # Ảnh người dùng upload
│   │   ├── results/               # Kết quả phân tích
│   │   └── images/                # Ảnh tĩnh (logo, mascot)
│   │
│   ├── templates/                 # HTML templates
│   │   ├── base.html             # Template cơ sở
│   │   ├── index.html            # Trang chủ
│   │   ├── about.html             # Trang giới thiệu
│   │   └── history.html           # Trang lịch sử
│   │
│   └── utils/                     # Utilities
│       ├── data_loader.py         # Load dữ liệu
│       └── ...
│
├── data/                          # Datasets gốc (chưa xử lý)
├── requirements.txt                # Python dependencies
├── CHAY_UNG_DUNG.bat              # Script chạy app (Windows)
├── CHAY_APP.ps1                   # Script chạy app (PowerShell)
├── setup_venv.ps1                 # Script setup môi trường
└── README.md                      # File này
```

### Các file quan trọng:

1. **`plant_ai_system/app.py`**: File chính chứa ứng dụng Flask
2. **`plant_ai_system/models/best_plant_model.pth`**: Model đã train (BẮT BUỘC phải có)
3. **`plant_ai_system/config.json`**: Cấu hình hệ thống
4. **`plant_ai_system/disease_database.json`**: Database thông tin bệnh

---

## 🔍 Xử lý lỗi thường gặp

### Lỗi 1: `ModuleNotFoundError: No module named 'torch'`

**Nguyên nhân:** 
- Chưa activate virtual environment
- PyTorch chưa được cài đặt trong venv
- Đang dùng Python global thay vì Python trong venv

**Giải pháp:**
```powershell
# Bước 1: Activate virtual environment (QUAN TRỌNG!)
.\plant_ai_env\Scripts\Activate.ps1

# Bước 2: Kiểm tra Python đang dùng
where python  # Windows - phải chỉ đến venv\Scripts\python.exe
which python  # Linux/Mac

# Bước 3: Cài đặt PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Bước 4: Kiểm tra lại
python -c "import torch; print('OK')"
```

**✅ Dấu hiệu đã activate đúng:**
- Thấy `(plant_ai_env)` ở đầu dòng terminal
- `where python` chỉ đến thư mục venv

### Lỗi 2: `FileNotFoundError: [Errno 2] No such file or directory: 'models/best_plant_model.pth'`

**Nguyên nhân:** Model chưa được train hoặc đường dẫn sai

**Giải pháp:**
```bash
# Kiểm tra model có tồn tại
ls plant_ai_system/models/best_plant_model.pth  # Linux/Mac
dir plant_ai_system\models\best_plant_model.pth  # Windows

# Nếu chưa có, chạy training
cd plant_ai_system
python train_health_monitor.py
```

### Lỗi 3: `CUDA out of memory`

**Nguyên nhân:** GPU không đủ bộ nhớ

**Giải pháp:**
- Giảm batch size trong `train_health_monitor.py`
- Sử dụng CPU thay vì GPU (sửa code để force CPU)
- Đóng các ứng dụng khác đang dùng GPU

### Lỗi 4: `Port 5000 already in use`

**Nguyên nhân:** Port 5000 đã được sử dụng

**Giải pháp:**

**Windows:**
```powershell
# Tìm process đang dùng port 5000
netstat -ano | findstr :5000

# Kill process (thay <PID> bằng số PID tìm được)
taskkill /PID <PID> /F
```

**Linux/Mac:**
```bash
# Tìm và kill process
lsof -ti:5000 | xargs kill -9
```

**Hoặc sửa port trong `app.py`:**
```python
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)  # Đổi sang port 5001
```

### Lỗi 5: `Dataset not found` hoặc `FileNotFoundError` khi training

**Nguyên nhân:** Dataset chưa được đặt đúng vị trí

**Giải pháp:**
- Kiểm tra đường dẫn trong `config.json`:
  ```json
  {
    "data_paths": {
      "plant_leaf_diseases": "data/health_monitoring/mega_dataset"
    }
  }
  ```
- Đảm bảo cấu trúc thư mục đúng:
  ```
  plant_ai_system/data/health_monitoring/mega_dataset/
    ├── train/
    ├── val/
    └── test/
  ```

### Lỗi 6: `OpenAI API không hoạt động`

**Nguyên nhân:** Chưa set API key hoặc key không hợp lệ

**Giải pháp:**
- Tạo file `.env` trong thư mục `plant_ai_system`:
  ```
  OPENAI_API_KEY=your_api_key_here
  ```
- Hoặc set biến môi trường:
  ```powershell
  # Windows PowerShell
  $env:OPENAI_API_KEY="your_api_key_here"
  
  # Linux/Mac
  export OPENAI_API_KEY="your_api_key_here"
  ```
- **Lưu ý:** Ứng dụng vẫn chạy được bình thường nếu không có OpenAI API, chỉ không có tính năng tư vấn từ AI

### Lỗi 7: `Execution Policy` trên Windows PowerShell

**Lỗi:** `cannot be loaded because running scripts is disabled on this system`

**Giải pháp:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 🚀 Quick Start (Tóm tắt nhanh)

### Để chạy Web App ngay:

**Windows (Cách nhanh nhất):**
```powershell
.\CHAY_UNG_DUNG.bat
```

**Windows (Thủ công):**
```powershell
# 1. Activate virtual environment
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
.\plant_ai_env\Scripts\Activate.ps1  # Windows
source plant_ai_env/bin/activate     # Linux/Mac

# 2. Di chuyển vào thư mục
cd plant_ai_system

# 3. Chạy training
python train_health_monitor.py

# 4. Đợi training hoàn thành (có thể mất vài giờ)
```

---

## 📊 Thông tin kỹ thuật

### Model Architecture:
- **Base Model**: ResNet50
- **Input Size**: 224x224
- **Number of Classes**: 39 (28 bệnh + 9 khỏe mạnh + 2 background)
- **Output**: Top-3 predictions với confidence scores

### Performance:
- **Inference Time**: ~150ms trên GPU, ~800ms trên CPU
- **Model Size**: ~100MB (best_plant_model.pth)
- **Accuracy**: Tùy thuộc vào dataset và training

### Supported Plant Types:
- Apple, Blueberry, Cherry, Corn
- Cashew, Cassava, Maize, Tomato
- Durian
- Background (không có lá)

---

## 📞 Hỗ trợ

Nếu gặp vấn đề:
1. ✅ Kiểm tra lại các bước cài đặt
2. ✅ Xem phần [Xử lý lỗi thường gặp](#xử-lý-lỗi-thường-gặp)
3. ✅ Kiểm tra log trong terminal để xem lỗi chi tiết
4. ✅ Đảm bảo đã cài đặt đầy đủ dependencies

---

## 📝 Ghi chú

- **Training time:** Tùy thuộc vào GPU và kích thước dataset, có thể mất từ 2-10 giờ
- **Inference time:** ~150ms trên GPU, ~800ms trên CPU
- **Model size:** ~100MB cho classification model
- **Dataset:** Cần ít nhất 5-10GB dung lượng cho dataset

---

## 📄 License

Distributed under the MIT License.

---

**Chúc bạn thành công với dự án! 🌱**

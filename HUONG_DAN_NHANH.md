# ⚡ HƯỚNG DẪN NHANH - CHẠY NGAY

## 🎯 Vấn đề bạn gặp:
```
ModuleNotFoundError: No module named 'torch'
```

## ✅ Giải pháp:

### Bước 1: Activate Virtual Environment (Python 3.10)

Bạn đã có virtual environment `.venv310` với PyTorch đã cài đặt. Chỉ cần activate nó:

**Trong PowerShell:**
```powershell
.\.venv310\Scripts\Activate.ps1
```

**Nếu gặp lỗi execution policy:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv310\Scripts\Activate.ps1
```

**Kiểm tra đã activate thành công:**
- Bạn sẽ thấy `(.venv310)` ở đầu dòng PowerShell
- Chạy: `python -c "import torch; print('OK')"` → Phải in "OK"

### Bước 2: Chạy ứng dụng

```powershell
# Di chuyển vào thư mục
cd plant_ai_system

# Chạy ứng dụng
python app.py
```

### Bước 3: Mở trình duyệt

Truy cập: **http://localhost:5000**

---

## 🚀 Cách nhanh nhất (1 lệnh):

Tôi đã tạo file `CHAY_UNG_DUNG.bat` để tự động làm tất cả:

```powershell
.\CHAY_UNG_DUNG.bat
```

File này sẽ:
1. ✅ Tự động activate virtual environment
2. ✅ Di chuyển vào thư mục đúng
3. ✅ Chạy ứng dụng

---

## ⚠️ Lưu ý quan trọng:

1. **LUÔN activate venv trước khi chạy!**
   - Phải thấy `(.venv310)` ở đầu dòng
   - Nếu không thấy → activate lại

2. **Kiểm tra Python đang dùng:**
   ```powershell
   where python
   # Phải chỉ đến: D:\Plant AI\.venv310\Scripts\python.exe
   ```

3. **Nếu vẫn lỗi:**
   - Đảm bảo đã activate venv
   - Kiểm tra PyTorch: `python -c "import torch; print(torch.__version__)"`
   - Nếu chưa có PyTorch, cài đặt:
     ```powershell
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```

---

## 📝 Tóm tắt lệnh:

```powershell
# Activate venv
.\.venv310\Scripts\Activate.ps1

# Chạy app
cd plant_ai_system
python app.py
```

**Hoặc đơn giản:**
```powershell
.\CHAY_UNG_DUNG.bat
```

---

Chúc bạn thành công! 🌱


# Script chạy ứng dụng Plant AI System
# Sử dụng: .\CHAY_APP.ps1

Write-Host "=== PLANT AI SYSTEM - CHAY UNG DUNG ===" -ForegroundColor Green
Write-Host ""

# Kiểm tra và activate virtual environment
if (Test-Path ".venv310\Scripts\Activate.ps1") {
    Write-Host "[1/3] Kich hoat virtual environment (.venv310)..." -ForegroundColor Yellow
    & .\.venv310\Scripts\Activate.ps1
} elseif (Test-Path "plant_ai_env\Scripts\Activate.ps1") {
    Write-Host "[1/3] Kich hoat virtual environment (plant_ai_env)..." -ForegroundColor Yellow
    & .\plant_ai_env\Scripts\Activate.ps1
} else {
    Write-Host "ERROR: Khong tim thay virtual environment!" -ForegroundColor Red
    Write-Host "Vui long chay: .\setup_venv.ps1" -ForegroundColor Yellow
    exit 1
}

# Kiểm tra PyTorch
Write-Host "[2/3] Kiem tra PyTorch..." -ForegroundColor Yellow
try {
    $torchCheck = python -c "import torch; print('OK')" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   PyTorch: OK" -ForegroundColor Green
    } else {
        Write-Host "   ERROR: PyTorch chua duoc cai dat!" -ForegroundColor Red
        Write-Host "   Dang cai dat PyTorch..." -ForegroundColor Yellow
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    }
} catch {
    Write-Host "   ERROR: Khong the kiem tra PyTorch" -ForegroundColor Red
}

# Di chuyển vào thư mục và chạy app
Write-Host "[3/3] Khoi dong ung dung web..." -ForegroundColor Yellow
Set-Location "plant_ai_system"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Ung dung dang chay tai:" -ForegroundColor Cyan
Write-Host "   http://localhost:5000" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Nhan Ctrl+C de dung ung dung" -ForegroundColor White
Write-Host ""

# Chạy ứng dụng
python app.py


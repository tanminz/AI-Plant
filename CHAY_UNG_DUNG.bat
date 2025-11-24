@echo off
REM Script chạy ứng dụng Plant AI System
REM Chạy file này để tự động activate venv và chạy app

echo ========================================
echo    PLANT AI SYSTEM - QUICK START
echo ========================================
echo.

REM Kiểm tra virtual environment (.venv310 hoặc plant_ai_env)
if exist ".venv310\Scripts\activate.bat" (
    set VENV_PATH=.venv310
) else if exist "plant_ai_env\Scripts\activate.bat" (
    set VENV_PATH=plant_ai_env
) else (
    echo ERROR: Virtual environment chua duoc tao!
    echo Vui long chay: .\setup_venv.ps1 truoc
    pause
    exit /b 1
)

REM Activate virtual environment
echo [1/3] Kich hoat virtual environment (%VENV_PATH%)...
call %VENV_PATH%\Scripts\activate.bat

REM Di chuyen vao thu muc
echo [2/3] Di chuyen vao thu muc plant_ai_system...
cd plant_ai_system

REM Chay ung dung
echo [3/3] Khoi dong ung dung web...
echo.
echo ========================================
echo    Ung dung dang chay tai:
echo    http://localhost:5000
echo ========================================
echo.
echo Nhan Ctrl+C de dung ung dung
echo.

python app.py

pause


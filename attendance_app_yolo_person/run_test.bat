@echo off
echo Creating virtual environment with Python 3.11...
py -3.11 -m venv venv
if errorlevel 1 (
    echo Python 3.11 not found. Please install Python 3.11 from python.org
    pause
    exit /b 1
)
call venv\Scripts\activate
echo Upgrading pip...
python -m pip install --upgrade pip
echo Installing cpu torch (will speed up ultralytics installs)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
echo Installing requirements...
pip install -r requirements.txt
echo Starting Flask server...
python app.py
pause

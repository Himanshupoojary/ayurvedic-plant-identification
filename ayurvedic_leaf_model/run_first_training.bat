@echo off
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo After placing images in data\raw\CLASS_NAME folders, run:
echo python scripts\prepare_dataset.py
echo python src\train.py --model mobilenetv2 --epochs 5
pause

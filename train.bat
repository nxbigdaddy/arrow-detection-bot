@echo off
cd /d "%~dp0"
title Train Arrow Detector
python train_yolo.py
pause

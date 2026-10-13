@echo off
cd /d "%~dp0"
title Arrow Dataset Annotation
python annotation_tool.py
if errorlevel 1 pause

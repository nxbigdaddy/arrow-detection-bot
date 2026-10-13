@echo off
cd /d "%~dp0"
title Arrow Bot GUI
python arrow_bot_gui.py
if errorlevel 1 (
  echo.
  echo The GUI exited with an error. Check the message above.
  pause
)

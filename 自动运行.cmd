@echo off
cd /d "%~dp0"
title Arrow Bot - Auto Play
echo Auto play is starting. Press Ctrl+C to stop at any time.
echo.
python arrow_bot.py --auto
echo.
echo The bot has stopped. See the message above for the reason.
pause

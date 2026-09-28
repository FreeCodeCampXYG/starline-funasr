@echo off
chcp 936 >nul
title FunASR 本地语音识别服务 - http://127.0.0.1:18466
cd /d "%~dp0"

echo.
echo ================================================
echo   FunASR 本地语音识别服务（CPU，端口 18466）
echo   接口文档: http://127.0.0.1:18466/docs
echo   模型: paraformer-zh + ct-punc 断句
echo ================================================
echo.

echo [1/2] 清理端口 18466 上的旧进程...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":18466" ^| findstr "LISTENING"') do (
    echo       结束旧进程 PID %%a
    taskkill /F /PID %%a >nul 2>&1
)

echo [2/2] 启动服务（模型加载约 30~60 秒，加载完 /health 变 ok）...
start "" http://127.0.0.1:18466/docs
.venv\Scripts\python.exe asr_server.py

echo.
echo 服务已停止。
pause

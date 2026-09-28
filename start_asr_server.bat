@echo off
chcp 936 >nul
title FunASR 本地语音识别服务 - http://127.0.0.1:18466
cd /d "%~dp0"

echo.
echo ================================================
echo   FunASR 本地语音识别服务（CPU，端口 18466）
echo   接口文档: http://127.0.0.1:18466/docs
echo   模型: paraformer-zh（iic/speech_seaco_paraformer_large）+ ct-punc 断句
echo ================================================
echo.

:: 0. 校验虚拟环境（确保用的是 .venv，而不是系统 Python）
if not exist ".venv\Scripts\python.exe" (
    echo [错误] 未检测到虚拟环境 .venv\Scripts\python.exe
    echo        请先在本目录执行：
    echo          python -m venv .venv
    echo          .venv\Scripts\pip install -e .
    pause
    exit /b 1
)
echo [环境] 使用虚拟环境 Python：
echo        %~dp0.venv\Scripts\python.exe
echo [提示] 日志中的 “download models from model hub” 只是缓存校验回显，并非重新下载。
echo        模型缓存目录固定在本项目内：%~dp0.models_cache
echo.

:: 1. 模型缓存路径固定到项目内（≈ D:\MyPro\FunASR\.models_cache）
::    默认 modelscope 会下到 C:\Users\<user>\.cache\modelscope，2GB+ 容易撑爆 C 盘，
::    故此处强制指向项目目录（与 asr_server.py 内的 os.environ.setdefault 保持一致）。
::    注意：切勿设置 MODELSCOPE_LOG_LEVEL 为字符串（如 ERROR）；
::    该变量 modelscope 内部按 int 解析（logger.py:13），非数字会导致启动崩溃。
set "MODELSCOPE_CACHE=%~dp0.models_cache"

:: 2. 清理端口 18466 上的旧进程，防止“地址已被占用”启动失败
echo [1/3] 清理端口 18466 上的旧进程...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":18466" ^| findstr "LISTENING"') do (
    echo       结束旧进程 PID %%a
    taskkill /F /PID %%a >nul 2>&1
)

:: 3. 在独立窗口启动服务（前台会阻塞，故另开窗口；窗口保留便于看实时日志/排错）
echo [2/3] 启动服务（模型加载约 30~60 秒，加载期间 /health 返回非 200）...
start "FunASR 服务窗口" cmd /k "title FunASR 服务(实时日志) & .venv\Scripts\python.exe asr_server.py"

:: 4. 轮询 /health，就绪后再打开浏览器（避免“提前开页看到连接错误”）
echo [3/3] 等待服务就绪（最多约 180 秒）...
set "READY=0"
for /L %%i in (1,1,90) do (
    .venv\Scripts\python.exe -c "import sys,urllib.request; urllib.request.urlopen('http://127.0.0.1:18466/health',timeout=2); sys.exit(0)" >nul 2>&1
    if not errorlevel 1 (
        set "READY=1"
        goto :OPEN
    )
    %SystemRoot%\System32\ping.exe -n 3 127.0.0.1 >nul 2>&1
)

:OPEN
if "%READY%"=="1" (
    echo.
    echo 服务已就绪，正在打开接口文档页...
    start "" http://127.0.0.1:18466/docs
) else (
    echo.
    echo 等待超时：服务可能启动失败，请查看“FunASR 服务窗口”中的报错。
    echo 模型缓存目录：%MODELSCOPE_CACHE%\models
)

echo.
echo 启动流程结束。服务在独立的“FunASR 服务窗口”中运行，关闭该窗口即可停止服务。
exit /b 0

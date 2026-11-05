@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo =========================================
echo   RS-Agent L2 遥感智能体 - 本地启动
echo =========================================

where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 没有找到 python，请先安装 Python 3.10 及以上版本并勾选 "Add to PATH"。
    pause
    exit /b 1
)

echo [1/2] 检查并安装依赖...
python -m pip install -q -r requirements.txt
if errorlevel 1 (
    echo [错误] 依赖安装失败，请检查网络后重试。
    pause
    exit /b 1
)

if not exist ".env" (
    echo [提示] 未找到 .env，将以纯本地模式运行：
    echo        只有关键词能匹配的问题可用，且为单轮问答。
    echo        如需大模型能力，请复制 .env.example 为 .env 并填入自己的 API Key。
    echo.
)

echo [2/2] 启动 Gradio 网页界面...
echo         浏览器打开 http://127.0.0.1:7860 ，关闭本窗口即停止服务。
python app.py
pause

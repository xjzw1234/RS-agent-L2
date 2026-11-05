#!/bin/bash
cd "$(dirname "$0")"
echo "========================================="
echo "正在启动 RS-Agent L2 (遥感智能体)"
echo "========================================="

echo "[1/2] 正在检查并安装依赖库..."
pip install -q -r requirements.txt || { echo "依赖安装失败，请检查网络后重试。"; exit 1; }

if [ ! -f ".env" ]; then
    echo "[提示] 未找到 .env，将以纯本地模式运行：只有关键词能匹配的问题可用，且为单轮问答。"
    echo "       如需大模型能力，请复制 .env.example 为 .env 并填入自己的 API Key。"
    echo
fi

echo "[2/2] 正在启动 Gradio 网页服务..."
python app.py

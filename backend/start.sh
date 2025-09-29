#!/bin/bash
# Docker容器启动脚本

echo "========================================="
echo "新闻分类系统启动中..."
echo "========================================="

# 检查模型文件是否存在
if [ -f "models/news_classifier.safetensors" ] && [ -f "models/vectorizer.pkl" ]; then
    echo "✓ 已找到预训练模型，直接启动服务..."
else
    echo "⚠ 模型文件不存在，将在启动时自动训练..."
fi

# 启动服务
echo "启动 FastAPI 服务..."
exec python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
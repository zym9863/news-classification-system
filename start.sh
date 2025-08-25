#!/bin/bash

# 新闻分类系统启动脚本
# 用于开发环境快速启动前端和后端

echo "🚀 启动新闻分类系统..."

# 检查是否安装了必要的工具
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "❌ 错误：未找到 $1 命令"
        echo "请先安装 $1"
        exit 1
    fi
}

# 检查依赖
check_command "python"
check_command "node"
check_command "pnpm"

# 启动后端
echo "📡 启动后端服务..."
cd backend
if [ ! -d "venv" ]; then
    echo "创建Python虚拟环境..."
    python -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate

# 安装依赖
echo "安装后端依赖..."
pip install -e .

# 启动后端（后台运行）
python main.py &
BACKEND_PID=$!
cd ..

# 启动前端
echo "🎨 启动前端服务..."
cd frontend

# 安装依赖
if [ ! -d "node_modules" ]; then
    echo "安装前端依赖..."
    pnpm install
fi

# 启动前端开发服务器
pnpm run dev &
FRONTEND_PID=$!
cd ..

echo "✅ 系统启动完成！"
echo "📱 前端地址: http://localhost:5173"
echo "🔗 后端地址: http://localhost:8000"
echo "📖 API文档: http://localhost:8000/docs"

# 等待用户输入来停止服务
echo ""
echo "按 Ctrl+C 或 Enter 键停止服务..."
read

# 清理进程
echo "🛑 停止服务..."
kill $BACKEND_PID 2>/dev/null
kill $FRONTEND_PID 2>/dev/null

echo "👋 服务已停止"
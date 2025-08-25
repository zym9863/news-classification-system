@echo off
REM 新闻分类系统启动脚本 - Windows版本
REM 用于开发环境快速启动前端和后端

echo 🚀 启动新闻分类系统...

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误：未找到 Python
    echo 请先安装 Python 3.12+
    pause
    exit /b 1
)

REM 检查Node.js是否安装  
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误：未找到 Node.js
    echo 请先安装 Node.js
    pause
    exit /b 1
)

REM 检查pnpm是否安装
pnpm --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误：未找到 pnpm
    echo 正在安装 pnpm...
    npm install -g pnpm
)

REM 启动后端
echo 📡 启动后端服务...
cd backend

REM 检查虚拟环境
if not exist venv (
    echo 创建Python虚拟环境...
    python -m venv venv
)

REM 激活虚拟环境并安装依赖
call venv\Scripts\activate
echo 安装后端依赖...
pip install uv
uv pip install -e .

REM 启动后端（后台运行）
start "后端服务" python main.py
cd ..

REM 启动前端
echo 🎨 启动前端服务...
cd frontend

REM 安装前端依赖
if not exist node_modules (
    echo 安装前端依赖...
    pnpm install
)

REM 启动前端开发服务器
start "前端服务" pnpm run dev
cd ..

echo ✅ 系统启动完成！
echo 📱 前端地址: http://localhost:5173
echo 🔗 后端地址: http://localhost:8000  
echo 📖 API文档: http://localhost:8000/docs
echo.
echo 按任意键退出...
pause >nul
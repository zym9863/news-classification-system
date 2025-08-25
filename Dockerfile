# 新闻分类系统 Docker 镜像
# 多阶段构建：先构建前端，然后运行后端并服务静态文件

# 阶段1: 构建前端
FROM node:18-alpine AS frontend-builder

# 设置工作目录
WORKDIR /frontend

# 安装pnpm
RUN npm install -g pnpm

# 复制前端package.json并安装依赖
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

# 复制前端源码并构建
COPY frontend/ ./
RUN pnpm run build

# 阶段2: 配置后端和最终运行环境
FROM python:3.12-slim AS backend

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装uv包管理器
RUN pip install uv

# 复制后端文件
COPY backend/ ./

# 使用uv安装Python依赖
RUN uv pip install --system -r pyproject.toml

# 从前端构建阶段复制静态文件
COPY --from=frontend-builder /frontend/dist ./static

# 创建必要的目录
RUN mkdir -p models

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# 启动命令
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
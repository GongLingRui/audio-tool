# Read-Rhyme Backend Dockerfile
# 多阶段构建，优化镜像大小

# ==================== 阶段1: 基础环境 ====================
FROM python:3.10-slim as base

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    ffmpeg \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ==================== 阶段2: 依赖安装 ====================
FROM base as dependencies

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==================== 阶段3: 最终镜像 ====================
FROM base as final

# 从依赖阶段复制已安装的包
COPY --from=dependencies /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# 复制应用代码
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini .
COPY pyproject.toml .

# 创建必要的目录
RUN mkdir -p static/uploads static/audio static/exports data

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

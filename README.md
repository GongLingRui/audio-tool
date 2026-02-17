# Read-Rhyme Backend

<div align="center">

AI-powered audiobook generation platform backend service.

![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-orange)

</div>

---

## 目录

- [项目概述](#项目概述)
- [核心功能](#核心功能)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [安装指南](#安装指南)
- [配置说明](#配置说明)
- [API文档](#api文档)
- [数据库模型](#数据库模型)
- [开发指南](#开发指南)
- [部署](#部署)

---

## 项目概述

Read-Rhyme 后端是基于 FastAPI 构建的 AI 驱动有声书生成平台服务。负责处理所有业务逻辑、数据存储、AI 模型调用和音频处理等核心功能。

### 主要特性

- 异步架构设计，高性能处理并发请求
- RESTful API 设计，完整的 Swagger 文档
- JWT 用户认证和授权
- 模块化代码组织，易于维护和扩展
- 完整的数据库迁移系统
- WebSocket 实时通信支持

---

## 核心功能

### 1. 用户认证系统
- 用户注册和登录
- JWT Token 认证
- 密码哈希存储
- 用户信息管理

### 2. 书籍管理
- 多格式电子书上传（TXT、PDF、EPUB）
- 书籍内容解析和提取
- 章节结构识别
- 书籍元数据管理
- 阅读进度跟踪

### 3. AI 脚本生成
- **大语言模型集成**：支持多种 LLM（Qwen、GLM-4、OpenAI）
- **智能分析**：
  - 角色对话识别
  - 情感标注（开心、悲伤、愤怒、平静等）
  - 场景分析
  - 旁白与对话区分
- **脚本审查**：AI 辅助脚本质量检查
- **可配置 Prompt**：自定义生成指令

### 4. 多引擎 TTS 集成
- **Edge TTS**：微软高质量语音合成
- **CosyVoice**：阿里达摩院语音合成
- **Qwen TTS**：通义千问语音服务
- **外部 TTS**：支持自定义 TTS 服务
- **语音克隆**：基于 LoRA 的个性化语音训练

### 5. 音频处理
- FFmpeg 音频处理
- 音频分段和合并
- 格式转换（MP3、WAV、M4A）
- 音频质量评估
- 并行音频生成
- 音频缓存机制

### 6. 高亮和笔记
- 文本高亮（支持多种颜色）
- 高亮内容笔记
- 章节级别组织
- 导出功能

### 7. RAG 文档问答
- 文档向量化索引
- 语义检索
- 基于文档的问答
- 引用溯源
- 网络搜索集成

### 8. 实时通信
- WebSocket 连接
- 实时进度更新
- 批量操作状态同步

---

## 技术栈

### 核心框架
- **FastAPI 0.109+**：现代化 Python Web 框架
- **Pydantic**：数据验证和设置管理
- **SQLAlchemy**：异步 ORM
- **Alembic**：数据库迁移工具

### 数据库
- **SQLite**：开发环境默认数据库
- **aiosqlite**：异步 SQLite 驱动
- 支持扩展到 PostgreSQL、MySQL

### 认证和安全
- **JWT**：JSON Web Token 认证
- **passlib**：密码哈希
- **python-multipart**：文件上传支持
- **python-jose**：JWT 处理

### AI/ML 集成
- **OpenAI SDK**：LLM API 集成
- **sentence-transformers**：文本向量化
- **FAISS**：向量检索（可选）
- **transformers**：Hugging Face 模型

### 音频处理
- **FFmpeg**：音频处理引擎
- **pydub**：音频操作库
- **librosa**：音频分析
- **edge-tts**：微软 TTS
- **httpx**：异步 HTTP 客户端

### 文档处理
- **PyPDF2**：PDF 处理
- **ebooklib**：EPUB 处理
- **beautifulsoup4**：HTML 解析

### 开发工具
- **pytest**：测试框架
- **black**：代码格式化
- **ruff**：代码检查
- **uvicorn**：ASGI 服务器

---

## 项目结构

```
backend/
├── alembic/                          # 数据库迁移
│   ├── versions/                     # 迁移脚本
│   └── env.py                        # Alembic 配置
│
├── app/
│   ├── api/                          # API 路由层
│   │   ├── __init__.py
│   │   ├── auth.py                   # 认证相关 API
│   │   │   ├── POST /register        # 用户注册
│   │   │   ├── POST /login           # 用户登录
│   │   │   └── GET /me               # 获取当前用户
│   │   │
│   │   ├── books.py                  # 书籍管理 API
│   │   │   ├── GET /                 # 获取书籍列表
│   │   │   ├── POST /upload          # 上传书籍
│   │   │   ├── GET /{id}             # 获取书籍详情
│   │   │   ├── PATCH /{id}           # 更新书籍
│   │   │   ├── DELETE /{id}          # 删除书籍
│   │   │   └── GET /{id}/content     # 获取书籍内容
│   │   │
│   │   ├── projects.py               # 项目管理 API
│   │   │   ├── GET /                 # 获取项目列表
│   │   │   ├── POST /                # 创建项目
│   │   │   ├── GET /{id}             # 获取项目详情
│   │   │   ├── PATCH /{id}           # 更新项目
│   │   │   ├── DELETE /{id}          # 删除项目
│   │   │   └── GET /{id}/progress    # 获取生成进度
│   │   │
│   │   ├── scripts.py                # 脚本生成 API
│   │   │   ├── GET /                 # 获取脚本
│   │   │   ├── POST /generate        # 生成脚本
│   │   │   ├── PATCH /               # 更新脚本
│   │   │   └── POST /review          # 审查脚本
│   │   │
│   │   ├── voices.py                 # 语音管理 API
│   │   ├── voice_styling.py          # 语音样式 API
│   │   ├── voice_advanced.py         # 高级语音 API
│   │   ├── audio.py                  # 音频处理 API
│   │   │   ├── GET /chunks           # 获取音频片段
│   │   │   ├── POST /generate-fast   # 生成所有音频
│   │   │   ├── POST /merge           # 合并音频
│   │   │   └── GET /                 # 获取最终音频
│   │   │
│   │   ├── highlights.py             # 高亮/笔记 API
│   │   ├── rag.py                    # RAG 问答 API
│   │   ├── lora_training.py          # LoRA 训练 API
│   │   ├── qwen_tts.py               # Qwen TTS API
│   │   ├── cosy_voice.py             # CosyVoice API
│   │   ├── websocket.py              # WebSocket API
│   │   └── config.py                 # 配置 API
│   │
│   ├── core/                         # 核心功能
│   │   ├── deps.py                   # 依赖注入
│   │   ├── security.py               # 安全相关（JWT、密码）
│   │   └── exceptions.py             # 异常处理
│   │
│   ├── models/                       # 数据库模型（SQLAlchemy）
│   │   ├── user.py                   # 用户模型
│   │   ├── book.py                   # 书籍模型
│   │   ├── project.py                # 项目模型
│   │   ├── script.py                 # 脚本模型
│   │   ├── chunk.py                  # 音频片段模型
│   │   ├── voice_config.py           # 语音配置模型
│   │   ├── highlight.py              # 高亮模型
│   │   ├── note.py                   # 笔记模型
│   │   └── thought.py                # 思维模型
│   │
│   ├── schemas/                      # Pydantic 数据模式
│   │   ├── user.py                   # 用户 Schema
│   │   ├── book.py                   # 书籍 Schema
│   │   ├── project.py                # 项目 Schema
│   │   ├── script.py                 # 脚本 Schema
│   │   ├── audio.py                  # 音频 Schema
│   │   ├── voice.py                  # 语音 Schema
│   │   └── highlight.py              # 高亮 Schema
│   │
│   ├── services/                     # 业务逻辑层
│   │   ├── script_generator.py       # 脚本生成服务
│   │   │   └── ScriptGenerator       # AI 脚本生成器
│   │   ├── audio_processor.py        # 音频处理服务
│   │   │   └── AudioProcessor        # 音频处理器
│   │   ├── voice_manager.py          # 语音管理服务
│   │   ├── lora_training.py          # LoRA 训练服务
│   │   ├── production_rag.py         # RAG 服务
│   │   ├── tts_engine.py             # TTS 引擎服务
│   │   ├── edge_tts_service.py       # Edge TTS 服务
│   │   ├── qwen_tts_service.py       # Qwen TTS 服务
│   │   └── voice_cloner.py           # 语音克隆服务
│   │
│   ├── utils/                        # 工具函数
│   │   ├── file_handler.py           # 文件处理
│   │   ├── text_extractor.py         # 文本提取
│   │   ├── audio_utils.py            # 音频工具
│   │   └── llm_client.py             # LLM 客户端
│   │
│   ├── config.py                     # 配置管理
│   ├── database.py                   # 数据库连接
│   └── main.py                       # 应用入口
│
├── static/                           # 静态文件
│   ├── uploads/                      # 上传的书籍
│   ├── audio/                        # 生成的音频
│   └── exports/                      # 导出文件
│
├── tests/                            # 测试文件
│   ├── test_api/                     # API 测试
│   ├── test_services/                # 服务测试
│   └── conftest.py                   # 测试配置
│
├── data/                             # 数据目录
│   └── app.db                        # SQLite 数据库
│
├── requirements.txt                  # Python 依赖
├── pyproject.toml                   # 项目配置
├── .env.example                     # 环境变量示例
└── README.md                        # 本文档
```

---

## 安装指南

### 前置要求

- Python 3.10 或更高版本
- pip (Python 包管理器)
- FFmpeg（系统依赖）

### 安装 FFmpeg

**macOS**:
```bash
brew install ffmpeg
```

**Ubuntu/Debian**:
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows**:
从 [FFmpeg 官网](https://ffmpeg.org/download.html) 下载并添加到系统 PATH

### 安装步骤

1. **克隆仓库**
```bash
git clone <repository-url>
cd backend
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **安装依赖**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **配置环境变量**
```bash
cp .env.example .env
```

编辑 `.env` 文件，配置必要的参数（详见[配置说明](#配置说明)）

5. **初始化数据库**
```bash
alembic upgrade head
```

6. **启动服务**

开发模式：
```bash
python -m app.main
```

或使用 uvicorn：
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

生产模式：
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

7. **验证安装**

访问以下地址验证服务是否正常：
- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/api/health

---

## 配置说明

### 环境变量 (.env)

```env
# ==================== 应用配置 ====================
APP_NAME=Read-Rhyme
APP_VERSION=0.1.0
DEBUG=false                          # 生产环境设为 false
SECRET_KEY=your-secret-key-here      # 应用密钥（必须修改）

# ==================== 服务器配置 ====================
HOST=0.0.0.0
PORT=8000

# ==================== 数据库配置 ====================
# SQLite（默认）
DATABASE_URL=sqlite+aiosqlite:///./data/app.db

# PostgreSQL（可选）
# DATABASE_URL=postgresql+asyncpg://user:password@localhost/dbname

# ==================== CORS 配置 ====================
CORS_ORIGINS=["http://localhost:5173","http://localhost:5174","http://localhost:8080"]

# ==================== 文件存储 ====================
UPLOAD_DIR=./static/uploads
AUDIO_DIR=./static/audio
EXPORT_DIR=./static/exports
MAX_UPLOAD_SIZE=104857600            # 100MB（字节）

# ==================== TTS 配置 ====================
# TTS 模式：edge（微软TTS）、external（外部服务）、local（本地）
TTS_MODE=edge
TTS_URL=http://localhost:7860         # 外部 TTS 服务地址
TTS_TIMEOUT=300                       # 生成超时（秒）
TTS_PARALLEL_WORKERS=2                # 并行生成数
TTS_LANGUAGE=zh-CN                    # 语言代码

# ==================== LLM 配置 ====================
# 用于脚本生成的大语言模型
# 智谱 AI（GLM-4）
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
LLM_API_KEY=your-zhipu-api-key-here   # 替换为真实 API Key
LLM_MODEL=glm-4-flash

# 或使用 OpenAI
# LLM_BASE_URL=https://api.openai.com/v1
# LLM_API_KEY=your-openai-api-key-here
# LLM_MODEL=gpt-4

# 或使用本地 Ollama
# LLM_BASE_URL=http://localhost:11434/v1
# LLM_API_KEY=local
# LLM_MODEL=qwen3-14b

# ==================== JWT 配置 ====================
JWT_SECRET_KEY=your-jwt-secret-key    # JWT 密钥（必须修改）
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=10080 # 7 天

# ==================== 可选配置 ====================
# RAG 配置
RAG_ENABLED=true
RAG_CHUNK_SIZE=500
RAG_CHUNK_OVERLAP=50

# 音频质量
AUDIO_BITRATE=128k
AUDIO_SAMPLE_RATE=24000
```

### 配置项说明

| 配置项 | 说明 | 默认值 | 必填 |
|-------|------|--------|------|
| `SECRET_KEY` | 应用密钥，用于加密 | - | 是 |
| `DATABASE_URL` | 数据库连接字符串 | SQLite | 是 |
| `TTS_MODE` | TTS 模式 | external | 是 |
| `TTS_URL` | TTS 服务地址 | http://localhost:7860 | 根据模式 |
| `LLM_BASE_URL` | LLM API 地址 | - | 是 |
| `LLM_API_KEY` | LLM API 密钥 | - | 是 |
| `LLM_MODEL` | LLM 模型名称 | - | 是 |
| `JWT_SECRET_KEY` | JWT 签名密钥 | - | 是 |

---

## API文档

### API 端点概览

#### 认证 (/api/auth)
| 方法 | 端点 | 描述 |
|-----|------|------|
| POST | `/register` | 用户注册 |
| POST | `/login` | 用户登录 |
| GET | `/me` | 获取当前用户信息 |

#### 书籍管理 (/api/books)
| 方法 | 端点 | 描述 |
|-----|------|------|
| GET | `/` | 获取书籍列表 |
| POST | `/upload` | 上传书籍文件 |
| GET | `/{id}` | 获取书籍详情 |
| PATCH | `/{id}` | 更新书籍信息 |
| DELETE | `/{id}` | 删除书籍 |
| GET | `/{id}/content` | 获取书籍内容 |
| GET | `/{id}/highlights` | 获取高亮列表 |

#### 项目管理 (/api/projects)
| 方法 | 端点 | 描述 |
|-----|------|------|
| GET | `/` | 获取项目列表 |
| POST | `/` | 创建有声书项目 |
| GET | `/{id}` | 获取项目详情 |
| PATCH | `/{id}` | 更新项目配置 |
| DELETE | `/{id}` | 删除项目 |
| GET | `/{id}/progress` | 获取生成进度 |

#### 脚本管理 (/api/projects/{id}/scripts)
| 方法 | 端点 | 描述 |
|-----|------|------|
| GET | `/` | 获取脚本内容 |
| POST | `/generate` | AI 生成脚本 |
| PATCH | `/` | 更新脚本内容 |
| POST | `/review` | AI 审查脚本 |

#### 语音管理 (/api/voices)
| 方法 | 端点 | 描述 |
|-----|------|------|
| GET | `/` | 获取可用语音列表 |
| POST | `/preview` | 语音预览 |
| GET | `/presets` | 获取情感预设 |
| POST | `/styling` | 应用语音样式 |

#### 音频处理 (/api/projects/{id}/audio)
| 方法 | 端点 | 描述 |
|-----|------|------|
| GET | `/chunks` | 获取音频片段列表 |
| POST | `/generate-fast` | 生成所有音频 |
| POST | `/merge` | 合并音频片段 |
| GET | `/` | 获取最终音频 |

#### 高亮和笔记 (/api/books/{id}/highlights)
| 方法 | 端点 | 描述 |
|-----|------|------|
| GET | `/` | 获取高亮列表 |
| POST | `/` | 创建高亮 |
| PUT | `/{highlight_id}/note` | 添加/更新笔记 |
| DELETE | `/{id}` | 删除高亮 |

#### RAG 问答 (/api/rag)
| 方法 | 端点 | 描述 |
|-----|------|------|
| POST | `/query` | 文档问答 |
| POST | `/search` | 语义搜索 |

#### WebSocket (/ws)
| 端点 | 描述 |
|------|------|
| `/ws/projects/{id}` | 项目进度实时更新 |

### 交互式文档

启动服务后访问：
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 数据库模型

### 用户表 (users)
```python
- id: UUID (主键)
- email: String (唯一)
- username: String (唯一)
- password_hash: String
- avatar_url: String (可选)
- is_active: Boolean
- created_at: DateTime
- updated_at: DateTime
```

### 书籍表 (books)
```python
- id: UUID (主键)
- user_id: UUID (外键 -> users)
- title: String
- author: String (可选)
- cover_url: String (可选)
- file_path: String
- file_type: String  # txt/pdf/epub
- total_pages: Integer (可选)
- total_chars: Integer
- progress: Float (0-1)
- created_at: DateTime
- updated_at: DateTime
```

### 项目表 (projects)
```python
- id: UUID (主键)
- book_id: UUID (外键 -> books)
- name: String
- description: String (可选)
- status: Enum  # draft/processing/completed/failed
- config: JSON  # TTS 配置
- audio_path: String (可选)
- duration: Float (可选，秒)
- created_at: DateTime
- updated_at: DateTime
```

### 脚本表 (scripts)
```python
- id: UUID (主键)
- project_id: UUID (外键 -> projects，唯一)
- content: JSONArray  # 脚本内容数组
- status: Enum  # pending/reviewed/approved
- error_message: String (可选)
- created_at: DateTime
- updated_at: DateTime
```

### 音频片段表 (chunks)
```python
- id: UUID (主键)
- project_id: UUID (外键 -> projects)
- script_id: UUID (外键 -> scripts)
- speaker: String  # 说话人
- text: String  # 文本内容
- instruct: String (可选)  # 生成指令
- emotion: String (可选)  # 情感标签
- section: String (可选)  # 章节
- status: Enum  # pending/processing/completed/failed
- audio_path: String (可选)
- duration: Float (可选，秒)
- order_index: Integer  # 排序
- created_at: DateTime
- updated_at: DateTime
```

### 高亮表 (highlights)
```python
- id: UUID (主键)
- user_id: UUID (外键 -> users)
- book_id: UUID (外键 -> books)
- chunk_id: UUID (外键 -> chunks，可选)
- text: String
- color: Enum  # yellow/green/blue/pink
- start_offset: Integer
- end_offset: Integer
- chapter: String (可选)
- created_at: DateTime
```

### 笔记表 (notes)
```python
- id: UUID (主键)
- highlight_id: UUID (外键 -> highlights，唯一)
- content: String
- created_at: DateTime
- updated_at: DateTime
```

---

## 开发指南

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/test_api/test_auth.py

# 运行测试并显示覆盖率
pytest --cov=app tests/

# 显示详细输出
pytest -v tests/
```

### 代码格式化

```bash
# 格式化代码
black app/

# 检查格式
black --check app/

# 代码检查
ruff check app/

# 自动修复
ruff check --fix app/
```

### 数据库迁移

```bash
# 创建新迁移
alembic revision --autogenerate -m "描述"

# 应用所有迁移
alembic upgrade head

# 回滚一个版本
alembic downgrade -1

# 查看迁移历史
alembic history

# 重置数据库（危险操作！）
alembic downgrade base
alembic upgrade head
```

### 添加新 API 端点

1. 在 `app/schemas/` 中创建请求/响应 Schema
2. 在 `app/models/` 中创建数据库模型（如需要）
3. 在 `app/api/` 中创建或编辑路由文件
4. 在 `app/main.py` 中注册路由

示例：
```python
# app/api/example.py
from fastapi import APIRouter, Depends
from app.schemas.example import ExampleResponse
from app.core.deps import get_current_user

router = APIRouter()

@router.get("/example", response_model=ExampleResponse)
async def get_example(current_user = Depends(get_current_user)):
    return {"message": "Hello, World!"}

# app/main.py
from app.api.example import router as example_router
app.include_router(example_router, prefix="/api/example", tags=["example"])
```

---

## 部署

### Docker 部署

```bash
# 构建镜像
docker build -t read-rhyme-backend .

# 运行容器
docker run -d \
  --name read-rhyme-backend \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/static:/app/static \
  -v $(pwd)/data:/app/data \
  read-rhyme-backend
```

### 系统服务（systemd）

创建 `/etc/systemd/system/read-rhyme-backend.service`：

```ini
[Unit]
Description=Read-Rhyme Backend
After=network.target

[Service]
Type=notify
User=www-data
WorkingDirectory=/path/to/backend
Environment="PATH=/path/to/backend/venv/bin"
ExecStart=/path/to/backend/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

启动服务：
```bash
sudo systemctl daemon-reload
sudo systemctl enable read-rhyme-backend
sudo systemctl start read-rhyme-backend
```

### Nginx 反向代理

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /ws {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## 常见问题

### Q: 数据库迁移失败？
A: 检查 `data/` 目录权限，确保应用有写入权限

### Q: TTS 生成超时？
A: 增加 `TTS_TIMEOUT` 配置值，或减少 `TTS_PARALLEL_WORKERS`

### Q: LLM API 调用失败？
A: 检查 API Key 是否有效，网络是否可达

### Q: 文件上传失败？
A: 检查 `MAX_UPLOAD_SIZE` 配置，确保文件大小在限制内

---

## 安全建议

1. **生产环境必须修改的配置**：
   - `SECRET_KEY`
   - `JWT_SECRET_KEY`
   - 使用强密码策略

2. **CORS 配置**：
   - 只允许可信域名
   - 生产环境禁用 `*` 通配符

3. **速率限制**：
   - 建议添加 API 速率限制
   - 防止滥用和 DDoS 攻击

4. **文件上传**：
   - 验证文件类型
   - 扫描恶意文件
   - 限制文件大小

---

## 许可证

MIT License

---

## 联系方式

- 作者：宫灵瑞
- 项目主页：[GitHub Repository]
- 问题反馈：[GitHub Issues]

<div align="center">

**Made with ❤️ by 宫灵瑞**

</div>

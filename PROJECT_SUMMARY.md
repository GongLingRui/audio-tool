# Read-Rhyme - AI 有声书生成项目

## 项目概述

Read-Rhyme 是一个 AI 驱动的有声书生成与阅读平台，支持用户上传文本内容，自动生成高质量的多角色有声书，并提供阅读、笔记、音频播放等功能。

## 项目结构

```
youshengshu/
├── docs/                    # 项目规划文档
│   ├── 01-project-overview.md
│   ├── 02-database-design.md
│   ├── 03-api-design.md
│   ├── 04-audio-processing.md
│   └── 05-frontend-backend-integration.md
├── backend/                 # 后端代码
│   ├── app/
│   │   ├── api/             # API 路由
│   │   ├── core/            # 核心功能
│   │   ├── models/          # 数据模型
│   │   ├── schemas/         # Pydantic 模式
│   │   ├── services/        # 业务逻辑服务
│   │   ├── utils/           # 工具函数
│   │   ├── config.py        # 配置管理
│   │   ├── database.py      # 数据库连接
│   │   └── main.py          # 应用入口
│   ├── alembic/             # 数据库迁移
│   ├── scripts/             # 初始化脚本
│   ├── static/              # 静态文件
│   ├── start.sh             # 启动脚本
│   └── requirements.txt     # Python 依赖
├── read-rhyme/             # 前端代码
│   ├── src/
│   │   ├── components/      # React 组件
│   │   ├── pages/           # 页面组件
│   │   ├── services/        # API 服务层
│   │   ├── stores/          # 状态管理
│   │   ├── types/           # TypeScript 类型
│   │   ├── hooks/           # 自定义 Hooks
│   │   ├── lib/             # 工具库
│   │   └── main.tsx         # 应用入口
│   ├── .env                 # 环境变量
│   ├── vite.config.ts       # Vite 配置
│   └── start.sh             # 启动脚本
└── alexandria-audiobook/   # 参考项目
```

## 技术栈

### 后端
- **框架**: FastAPI 0.109+
- **数据库**: SQLite + SQLAlchemy (async)
- **认证**: JWT tokens
- **音频处理**: FFmpeg, pydub, librosa
- **TTS 集成**: 外部 TTS 服务支持

### 前端
- **框架**: React 18 + TypeScript
- **构建工具**: Vite 5
- **路由**: React Router DOM 6
- **状态管理**: Zustand
- **UI 组件**: shadcn/ui + Tailwind CSS

## 快速开始

### 后端启动

```bash
cd backend

# 方式 1: 使用启动脚本（推荐）
./start.sh

# 方式 2: 手动启动
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # 首次运行时

# 初始化数据库
python scripts/init_db.py
python scripts/create_test_user.py

# 启动服务器
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

后端启动后：
- API 文档: http://localhost:8000/docs
- ReDoc 文档: http://localhost:8000/redoc

### 前端启动

```bash
cd read-rhyme

# 方式 1: 使用启动脚本（推荐）
./start.sh

# 方式 2: 手动启动
npm install
cp .env.example .env  # 首次运行时
npm run dev
```

前端启动后：http://localhost:5173

## 默认测试账号

```
邮箱: test@example.com
密码: password123
```

## API 端点

### 认证
- `POST /api/auth/register` - 用户注册
- `POST /api/auth/login` - 用户登录

### 书籍管理
- `GET /api/books` - 获取书籍列表
- `POST /api/books/upload` - 上传书籍
- `GET /api/books/{id}` - 获取书籍详情
- `GET /api/books/{id}/content` - 获取书籍内容
- `PATCH /api/books/{id}` - 更新书籍
- `DELETE /api/books/{id}` - 删除书籍

### 项目管理
- `GET /api/projects` - 获取项目列表
- `POST /api/projects` - 创建项目
- `GET /api/projects/{id}` - 获取项目详情
- `PATCH /api/projects/{id}` - 更新项目
- `DELETE /api/projects/{id}` - 删除项目

### 脚本生成
- `POST /api/projects/{id}/scripts/generate` - 生成脚本
- `GET /api/projects/{id}/scripts` - 获取脚本
- `PATCH /api/projects/{id}/scripts` - 更新脚本
- `POST /api/projects/{id}/scripts/review` - 审查脚本
- `POST /api/projects/{id}/scripts/approve` - 批准脚本
- `POST /api/projects/{id}/scripts/chunks` - 从脚本创建音频块（智能分块）

### 音频生成
- `GET /api/projects/{id}/chunks` - 获取音频块列表
- `GET /api/projects/{id}/chunks/progress` - 获取生成进度
- `POST /api/projects/{id}/chunks/generate-fast` - 批量生成音频
- `POST /api/projects/{id}/chunks/generate-batch` - 批量生成（并行/顺序）
- `PATCH /api/projects/{id}/chunks/{chunk_id}` - 更新音频块
- `POST /api/projects/{id}/audio/merge` - 合并音频
- `POST /api/projects/{id}/audio/export` - 导出音频（3种格式）
- `GET /api/projects/{id}/audio/download/{filename}` - 下载导出文件
- `GET /api/projects/{id}/audio` - 获取最终音频

### 语音配置
- `GET /api/voices` - 获取可用语音列表
- `GET /api/voices/reference` - 获取语音方向词汇库
- `POST /api/projects/{id}/voices/parse` - 解析脚本中的说话人
- `POST /api/projects/{id}/voices/config` - 设置语音配置
- `POST /api/voices/preview` - 生成语音预览
- `POST /api/voices/clone/upload` - 上传参考音频
- `POST /api/voices/design` - 设计语音

### 语音样式控制（新增）
- `GET /api/voice-styling/presets` - 获取所有情感预设
- `GET /api/voice-styling/presets/{id}` - 获取特定情感预设
- `GET /api/voice-styling/languages` - 获取支持的语言列表
- `POST /api/voice-styling/generate-styled` - 生成带情感的语音
- `POST /api/voice-styling/convert-voice` - 语音到语音转换
- `POST /api/voice-styling/batch-clone` - 批量语音克隆
- `POST /api/voice-styling/enhance-speech` - 语音增强处理

### Qwen3-TTS（新增 - Apple Silicon 优化）
- `POST /api/qwen-tts/generate` - 生成语音（MPS 加速）
- `POST /api/qwen-tts/generate-with-voice` - 语音克隆生成
- `POST /api/qwen-tts/clone-voice` - 批量语音克隆
- `GET /api/qwen-tts/voices` - 获取可用语音列表
- `GET /api/qwen-tts/languages` - 支持的语言列表
- `GET /api/qwen-tts/info` - 系统信息（MPS 状态）

### RAG 文档问答（新增 - 生产级）
- `POST /api/rag/ingest` - 摄入文档（智能分块 + 向量嵌入）
- `POST /api/rag/ingest-file` - 摄入文件（TXT/MD）
- `POST /api/rag/query` - 查询系统（语义检索 + Web 搜索 + 引用追踪）
- `DELETE /api/rag/document` - 删除文档
- `GET /api/rag/stats` - 系统统计
- `GET /api/rag/documents` - 文档列表

### 笔记管理
- `GET /api/books/{id}/highlights` - 获取高亮列表
- `POST /api/books/{id}/highlights` - 创建高亮
- `PUT /api/highlights/{id}/note` - 添加/更新笔记
- `DELETE /api/highlights/{id}` - 删除高亮
- `POST /api/books/{id}/highlights/export` - 导出笔记

### 系统配置
- `GET /api/config` - 获取系统配置
- `PATCH /api/config` - 更新系统配置

## 数据库表结构

- **users** - 用户表
- **books** - 书籍表
- **projects** - 有声书项目表
- **scripts** - 脚本表（LLM 生成）
- **voice_configs** - 语音配置表
- **chunks** - 音频块表
- **highlights** - 高亮表
- **notes** - 笔记表

## 环境配置

### 后端 (.env)
```env
APP_NAME=Read-Rhyme
DEBUG=true
SECRET_KEY=your-secret-key
DATABASE_URL=sqlite+aiosqlite:///./data/app.db
TTS_MODE=external
TTS_URL=http://localhost:7860
LLM_BASE_URL=http://localhost:11434/v1
```

### 前端 (.env)
```env
VITE_API_BASE_URL=http://localhost:8000/api
VITE_WS_BASE_URL=ws://localhost:8000/ws
```

## 功能特性

### 已实现
- ✅ 用户认证（注册/登录）
- ✅ 书籍上传（TXT/PDF/EPUB）
- ✅ 书籍管理（列表、详情、删除）
- ✅ 项目创建和管理
- ✅ 脚本生成 API（增强版，支持上下文保留、JSON修复、默认提示词）
- ✅ 音频生成 API（批量处理、智能分块）
- ✅ 语音配置 API
- ✅ 高亮和笔记 API
- ✅ 前端 API 服务层
- ✅ 状态管理集成

### 新增功能（从 Alexandria 迁移）
- ✅ **语音方向词汇系统** - 专业语音描述词汇库（VOICE_REFERENCE.md）
- ✅ **增强脚本生成** - LLM 上下文保留、批次处理、JSON清理和修复
- ✅ **默认提示词** - 专业脚本生成提示词（default_prompts.txt）
- ✅ **脚本审查** - 自动合并过分割旁白、检测问题
- ✅ **智能分块** - 自动合并相同说话人（500字符上限，保护结构文本）
- ✅ **音频时长估算** - 基于语速的预估功能
- ✅ **分块时序计算** - 自动计算每个块的开始/结束时间
- ✅ **专业音频导出** - 3种格式：
  - `combined` - 单个MP3文件（带淡入淡出、音量归一化）
  - `audacity` - Audacity项目包（.lof文件+标签）
  - `voicelines` - 单独音频文件（DAW编辑用）
- ✅ **自然停顿** - 不同说话人500ms，相同说话人250ms
- ✅ **语音设计师** - 文本描述生成语音（API支持）
- ✅ **语音克隆** - 上传参考音频克隆声音
- ✅ **语音参考API** - 获取专业词汇库

### 新增AI语音功能（2025）
- ✅ **情感控制系统** - 6种主要情感 + 4种次要参数
  - 主要情感：happiness, sadness, anger, fear, surprise, neutral
  - 次要控制：energy（能量）, tempo（语速）, pitch（音高）, volume（音量）
- ✅ **情感预设** - 10种预配置情感（neutral, happy, sad, angry, fearful, surprised, romantic, mysterious, energetic, calm）
- ✅ **语音风格控制** - 三层语音建模
  - 音色特征（timbre）：粗糙/平滑、厚重/清亮
  - 情感态度（emotion）：情绪、态度传递
  - 传递方式（delivery）：语速、节奏、重音
- ✅ **批量语音克隆** - 使用多个音频样本（5-10个）提升克隆质量
- ✅ **语音到语音转换** - 保留内容改变音色（RVC技术）
- ✅ **语音增强** - 去噪、音量归一化、压缩、LUFS目标化
- ✅ **多语言支持** - 中文、英语、日语、韩语等
- ✅ **语音方向API** - `/api/voice-styling/*` 完整API套件

### 新增 Qwen3-TTS & RAG 功能（2026）
- ✅ **MPS 加速器** - Apple Silicon GPU 优化 (M1/M2/M3/M4)
  - `app/services/mps_accelerator.py` - Metal Performance Shaders 加速
  - 自动设备选择 (MPS > CPU)
  - 动态批次大小优化
  - 内存管理和缓存清理
- ✅ **Qwen3-TTS 集成** - 生产级 TTS 服务
  - 支持 1.7B 参数模型（CustomVoice/VoiceDesign）
  - 3秒语音克隆
  - 多语言支持（中文、英文、日文等）
  - MPS 加速推理（~300ms 延迟）
  - 情感控制和语音调节
- ✅ **RAG 文档问答** - 生产级检索增强生成
  - 智能文档分块（500字符，100重叠）
  - 向量嵌入生成（Sentence-Transformers）
  - 混合检索（语义 + 关键词）
  - Web 搜索集成（DuckDuckGo）
  - 引用追踪和来源标注
  - 相关性评分和重排序
- ✅ **前端 API 服务** - `src/services/rag.ts` 和 `src/services/qwenTts.ts`

### 新增服务模块
- ✅ `app/services/chunk_service.py` - 智能分块服务
  - `group_into_chunks()` - 按说话人分组
  - `merge_consecutive_narrators()` - 合并连续旁白
  - `split_script_to_chunks()` - 完整分块流程
  - `estimate_audio_duration()` - 时长估算
  - `calculate_chunk_timing()` - 时序计算
- ✅ `app/services/mps_accelerator.py` - MPS 加速器（Apple Silicon）
  - `MPSAccelerator` - 主加速器类
  - `optimize_model_for_mps()` - 模型优化
  - `batch_process_tensors()` - 批量张量处理
  - `get_optimal_batch_size()` - 动态批次计算
- ✅ `app/services/qwen_tts_service.py` - Qwen3-TTS 服务
  - `QwenTTSService` - TTS 服务类
  - `generate_speech()` - 语音生成
  - `clone_voice()` - 语音克隆
  - `get_available_voices()` - 获取可用语音
- ✅ `app/services/production_rag.py` - RAG 系统
  - `ProductionRAG` - RAG 系统类
  - `ingest_document()` - 文档摄入
  - `retrieve()` - 语义检索
  - `web_search()` - Web 搜索
  - `query()` - 完整查询（含引用）
- ✅ `backend/default_prompts.txt` - 专业脚本生成提示词

### 待完善功能
- ⏳ TTS 引擎实际音频生成（目前为占位实现）
- ⏳ LLM 脚本生成集成
- ⏳ WebSocket 实时进度更新
- ⏳ 音频播放器与后端音频对接
- ⏳ 语音克隆功能
- ⏳ LoRA 训练功能

## 开发指南

### 添加新 API 端点

1. 在 `backend/app/schemas/` 中定义 Pydantic 模式
2. 在 `backend/app/models/` 中定义数据模型
3. 在 `backend/app/api/` 中创建路由
4. 在 `backend/app/api/__init__.py` 中注册路由
5. 在 `frontend/src/services/` 中添加对应的 API 客户端方法

### 数据库迁移

```bash
cd backend

# 创建新迁移
alembic revision --autogenerate -m "描述"

# 应用迁移
alembic upgrade head

# 回滚迁移
alembic downgrade -1
```

## 故障排除

### 后端无法启动
- 检查 Python 版本（需要 3.10+）
- 确保 virtual environment 已激活
- 检查依赖是否完整安装：`pip install -r requirements.txt`

### 前端 API 调用失败
- 确认后端已启动（http://localhost:8000）
- 检查 .env 文件中的 VITE_API_BASE_URL
- 查看浏览器控制台的网络请求错误

### 数据库错误
- 删除 `backend/data/app.db` 文件
- 重新运行 `python scripts/init_db.py`

## 许可证

MIT License

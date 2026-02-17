# Read-Rhyme 项目文档

AI驱动的有声书生成平台

## 项目概述

Read-Rhyme是一个基于AI的有声书生成平台，能够将电子书转换为高质量的多角色有声书。平台集成了先进的自然语言处理、语音合成和音频处理技术。

## 核心功能

### 1. 电子书管理
- 支持多种格式上传（EPUB, PDF, TXT, MD）
- 自动文本提取和清理
- 书籍元数据管理
- 内容预览和编辑

### 2. AI脚本生成
- 基于LLM的智能脚本生成
- 自动角色识别和对话分配
- 情感标注和场景分析
- 脚本审查和优化建议

### 3. 多角色语音合成
- 预设高质量语音库
- 自定义语音克隆
- LoRA模型训练支持
- 语音风格和情感调节

### 4. 音频处理
- 智能音频分段
- 音频质量增强
- 音频拼接和合并
- 实时进度跟踪

### 5. RAG文档问答系统
- 文档智能索引
- 语义检索
- 网络搜索集成
- 引用追踪

### 6. 高级AI功能
- **语音增强**: 降噪、音量标准化、压缩
- **语音转换**: 音调调整、速度变换、保持韵律
- **LoRA训练**: 自定义语音模型训练
- **语音设计**: 基于文本描述生成语音参数

## 技术栈

### 后端
- **框架**: FastAPI 0.109+
- **数据库**: SQLite with SQLAlchemy (async)
- **认证**: JWT tokens
- **音频处理**: FFmpeg, pydub, librosa
- **LLM集成**: OpenAI兼容API
- **TTS集成**: 外部TTS服务支持
- **向量检索**: sentence-transformers

### 前端
- **框架**: React 18 + TypeScript
- **构建工具**: Vite
- **UI组件**: shadcn/ui
- **样式**: Tailwind CSS
- **状态管理**: Zustand
- **HTTP客户端**: Axios
- **音频可视化**: wavesurfer.js

## 项目结构

```
youshengshu/
├── backend/                 # 后端API服务
│   ├── app/
│   │   ├── api/            # API路由
│   │   │   ├── auth.py     # 认证接口
│   │   │   ├── books.py    # 书籍管理
│   │   │   ├── projects.py # 项目管理
│   │   │   ├── scripts.py  # 脚本生成
│   │   │   ├── voices.py   # 语音管理
│   │   │   ├── voice_styling.py  # 语音样式
│   │   │   ├── rag.py      # RAG文档问答
│   │   │   └── lora_training.py  # LoRA训练
│   │   ├── core/           # 核心功能
│   │   ├── models/         # 数据库模型
│   │   ├── schemas/        # Pydantic schemas
│   │   ├── services/       # 业务逻辑
│   │   │   ├── audio_processor.py  # 音频处理
│   │   │   ├── lora_training.py    # LoRA训练
│   │   │   ├── script_generator.py # 脚本生成
│   │   │   └── production_rag.py   # RAG系统
│   │   └── utils/          # 工具函数
│   ├── tests/              # 测试
│   │   ├── test_api.py           # 完整API测试
│   │   └── test_api_simple.py    # 简化API测试
│   └── requirements.txt
│
└── read-rhyme/            # 前端应用
    ├── src/
    │   ├── components/     # React组件
    │   ├── pages/         # 页面组件
    │   ├── services/      # API服务
    │   ├── stores/        # 状态管理
    │   └── test/          # 测试
    │       └── services.test.ts   # 服务测试
    └── package.json
```

## 安装和配置

### 后端设置

1. 创建虚拟环境：
```bash
cd backend
python -m venv venv
source venv/bin/activate
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置环境变量：
```bash
cp .env.example .env
```

编辑 `.env` 文件：
```env
# 应用配置
SECRET_KEY=your-secret-key-here

# 数据库
DATABASE_URL=sqlite+aiosqlite:///./data/app.db

# TTS配置
TTS_MODE=external
TTS_URL=http://localhost:7860

# LLM配置
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=local
LLM_MODEL=qwen3-14b
```

4. 初始化数据库：
```bash
alembic upgrade head
```

### 前端设置

1. 安装依赖：
```bash
cd read-rhyme
npm install
```

2. 配置环境变量：
```bash
cp .env.example .env
```

编辑 `.env` 文件：
```env
VITE_API_BASE_URL=http://localhost:8000/api
```

## 运行项目

### 启动后端

```bash
cd backend
python -m app.main
```

或使用 uvicorn：
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 启动前端

```bash
cd read-rhyme
npm run dev
```

## API文档

后端运行后，访问：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 测试

### 后端测试

```bash
# 运行所有测试
cd backend
pytest tests/

# 运行简化测试（无需数据库）
pytest tests/test_api_simple.py -v -s
```

### 前端测试

```bash
cd read-rhyme
npm test
```

## 核心API端点

### 认证
- `POST /api/auth/register` - 用户注册
- `POST /api/auth/login` - 用户登录

### 书籍管理
- `GET /api/books` - 列出书籍
- `POST /api/books/upload` - 上传书籍
- `GET /api/books/{id}` - 获取书籍详情
- `GET /api/books/{id}/content` - 获取书籍内容

### 项目管理
- `GET /api/projects` - 列出项目
- `POST /api/projects` - 创建项目
- `GET /api/projects/{id}` - 获取项目详情
- `DELETE /api/projects/{id}` - 删除项目

### 脚本生成
- `POST /api/projects/{id}/scripts/generate` - 生成脚本
- `GET /api/projects/{id}/scripts` - 获取脚本
- `PATCH /api/projects/{id}/scripts` - 更新脚本

### 语音管理
- `GET /api/voices` - 列出可用语音
- `POST /api/voices/preview` - 预览语音
- `POST /api/voices/design` - 设计语音参数
- `POST /api/voices/clone/upload` - 上传克隆音频

### 语音样式
- `GET /api/voice-styling/presets` - 获取情感预设
- `POST /api/voice-styling/enhance-speech` - 增强音频
- `POST /api/voice-styling/convert-voice` - 转换语音

### RAG文档问答
- `POST /api/rag/ingest` - 索引文档
- `POST /api/rag/query` - 查询文档
- `GET /api/rag/stats` - 获取统计信息
- `DELETE /api/rag/document` - 删除文档

### LoRA训练
- `GET /api/lora/requirements` - 获取训练要求
- `GET /api/lora/config-template` - 获取配置模板
- `POST /api/lora/projects/{id}/train` - 创建训练任务
- `POST /api/lora/projects/{id}/start` - 开始训练
- `GET /api/lora/projects/{id}/progress` - 获取训练进度

## AI功能详解

### 1. 语音增强

通过 `POST /api/voice-styling/enhance-speech` 端点，可以对音频进行增强处理：

**功能：**
- 降噪处理
- 音量标准化（目标LUFS）
- 动态压缩

**参数：**
```json
{
  "audio_path": "path/to/audio.wav",
  "enhance_denoise": true,
  "enhance_volume": true,
  "add_compression": true,
  "target_lufs": -16.0
}
```

### 2. 语音转换

通过 `POST /api/voice-styling/convert-voice` 端点，可以转换语音特征：

**功能：**
- 音调调整（±12半音）
- 速度变换（0.5-2.0倍）
- 保持韵律和时间

**参数：**
```json
{
  "source_audio_path": "path/to/source.wav",
  "pitch_shift": 2.0,
  "speed_factor": 1.1,
  "preserve_timing": true
}
```

### 3. LoRA训练

通过LoRA训练API，可以训练自定义语音模型：

**训练要求：**
- 最少样本数：3个
- 推荐样本数：10个
- 每个样本时长：最少5秒，推荐15秒
- 总时长：最少30秒，推荐120秒

**训练参数：**
```json
{
  "voice_name": "custom_voice",
  "rank": 32,
  "alpha": 64,
  "num_epochs": 10,
  "learning_rate": 0.0001,
  "batch_size": 4
}
```

### 4. 语音设计

通过 `POST /api/voices/design` 端点，可以从文本描述生成语音参数：

**示例：**
```json
{
  "description": "一个温柔的中年女性声音，语速适中，带有南方口音",
  "gender": "female",
  "age_range": "middle-aged"
}
```

**返回参数：**
```json
{
  "voice_id": "designed_abc123",
  "suggested_config": {
    "emotion": "calm",
    "energy": 0.8,
    "tempo": 0.9,
    "pitch": 0.0,
    "gender": "female",
    "age_range": "middle-aged",
    "timbre": "warm"
  },
  "recommended_tts_instruction": "Female voice calm and gentle with warmth"
}
```

## 数据库模型

### 核心模型

- **User**: 用户信息
- **Book**: 书籍信息
- **Project**: 有声书项目
- **Script**: 生成的脚本
- **AudioChunk**: 音频片段
- **VoiceConfig**: 语音配置
- **Highlight**: 高亮标注
- **Thought**: 思考笔记

## 开发指南

### 添加新的API端点

1. 在 `backend/app/api/` 中创建路由文件
2. 在 `backend/app/schemas/` 中定义请求/响应模型
3. 在 `backend/app/services/` 中实现业务逻辑
4. 在 `backend/app/api/__init__.py` 中注册路由

### 添加新的前端页面

1. 在 `read-rhyme/src/pages/` 中创建页面组件
2. 在 `read-rhyme/src/services/` 中添加API服务
3. 更新路由配置

### 数据库迁移

```bash
# 创建迁移
alembic revision --autogenerate -m "description"

# 应用迁移
alembic upgrade head

# 回滚迁移
alembic downgrade -1
```

## 故障排查

### 常见问题

1. **TTS服务连接失败**
   - 检查 `TTS_URL` 配置
   - 确保TTS服务正在运行

2. **LLM API错误**
   - 检查 `LLM_BASE_URL` 和 `LLM_API_KEY`
   - 确认模型名称正确

3. **音频处理失败**
   - 确保系统已安装FFmpeg
   - 检查文件路径和权限

4. **数据库错误**
   - 运行 `alembic upgrade head` 更新数据库
   - 检查 `DATABASE_URL` 配置

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

MIT License

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送邮件到项目维护者

## 更新日志

### v1.0.0 (2025-02-16)
- 初始版本发布
- 完整的电子书到有声书转换流程
- AI语音功能集成（增强、转换、LoRA训练）
- RAG文档问答系统
- 完整的测试覆盖

### 测试结果

**后端测试** (2025-02-16):
- 13个测试全部通过
- 覆盖所有核心API端点

**前端测试** (2025-02-16):
- 22个测试全部通过
- 覆盖所有服务模块

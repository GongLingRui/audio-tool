# VoiceForge - 世界上最强的AI语音处理工具

<div align="center">

![VoiceForge Logo](https://img.shields.io/badge/VoiceForge-AI%20Voice%20Processing%20Platform-brightgreen)
![Version](https://img.shields.io/badge/version-1.0.0-orange)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![React](https://img.shields.io/badge/react-18+-cyan.svg)
![License](https://img.shields.io/badge/license-MIT-green)

**全能AI语音处理平台 - 文本转语音、音频增强、说话人分离、语音转换**

作者：宫灵瑞

[功能特性](#功能特性) • [快速开始](#快速开始) • [项目结构](#项目结构) • [技术栈](#技术栈) • [API文档](#api文档)

</div>

---

## 项目简介

VoiceForge 是一个功能完整的AI语音处理平台，提供业界领先的语音合成、音频处理和语音分析功能。平台集成了多种前沿AI语音技术，支持文本转语音、音频增强、说话人分离、语音转换等丰富功能。

### 核心亮点

- **多引擎TTS**: 支持Edge TTS、CosyVoice、Qwen TTS等多种语音合成引擎
- **音频增强**: AI驱动的降噪、音质优化、动态范围压缩
- **说话人分离**: 自动识别和分离音频中的不同说话人
- **语音转换**: 将语音转换为不同风格和音色
- **语音克隆**: 基于LoRA的个性化语音训练
- **音频分析**: 全面的音频质量分析和评估

---

## 功能特性

### 🎙️ 文本转语音 (TTS)
- **多引擎支持**
  - 微软Edge TTS - 轻量级高质量语音合成
  - 阿里CosyVoice - 最新AI语音合成模型
  - 通义千问TTS - 阿里云语音服务
- **语音定制**
  - 多种预设语音和音色
  - 音调、语速、情感调整
  - 自定义语音克隆
- **高级功能**
  - 零样本语音克隆
  - 跨语言语音合成
  - 指令式语音控制

### 🎵 音频增强
- **降噪处理**
  - 背景噪声消除
  - 静音检测和过滤
  - 频谱降噪
- **音质优化**
  - 动态范围压缩
  - 音频标准化
  - 混响消除
- **均衡器预设**
  - 低音增强
  - 高音增强
  - 人声优化
- **格式转换**
  - 支持WAV、MP3、OGG、FLAC等格式
  - 采样率转换
  - 声道转换

### 🎤 说话人分离
- **自动识别**
  - 智能检测说话人数量
  - 语音活动检测
  - 说话人时间戳提取
- **高级功能**
  - 特定说话人音频提取
  - 说话人对比验证
  - 带标签的语音转录
- **导出选项**
  - 分段导出
  - 说话人统计
  - 时间轴导出

### 🔄 语音转换
- **音色转换**
  - 深沉男声
  - 清亮女声
  - 音调调整
- **特效处理**
  - 机器人音效
  - 回声效果
  - 电话音效
- **语音分析**
  - 声纹提取
  - 相似度对比
  - 语音特征分析

### 📊 音频分析
- **质量评估**
  - 音频电平分析
  - 动态范围检测
  - 削波检测
  - 信噪比计算
- **元数据提取**
  - 时长信息
  - 采样率
  - 声道配置
  - 编码信息
- **智能建议**
  - 音质改进建议
  - 问题诊断
  - 优化方案

### 🤖 AI功能
- **语音克隆**
  - LoRA模型训练
  - 个性化声音定制
  - 声音相似度控制
- **脚本生成**
  - AI驱动的脚本分析
  - 角色识别
  - 情感标注
- **RAG问答**
  - 基于文档的智能问答
  - 语义检索
  - 引用溯源

---

## 项目结构

```
voiceforge/
├── backend/                      # FastAPI 后端服务
│   ├── app/
│   │   ├── api/                  # API 路由层
│   │   │   ├── audio_tools.py    # 音频工具API ⭐ 新增
│   │   │   ├── cosy_voice.py     # CosyVoice API
│   │   │   ├── auth.py           # 用户认证
│   │   │   ├── projects.py       # 项目管理
│   │   │   ├── audio.py          # 音频生成
│   │   │   ├── voices.py         # 语音管理
│   │   │   └── websocket.py      # WebSocket
│   │   ├── services/             # 业务逻辑层
│   │   │   ├── audio_enhancement_service.py    # 音频增强 ⭐
│   │   │   ├── speaker_diarization_service.py  # 说话人分离 ⭐
│   │   │   ├── voice_conversion_service.py     # 语音转换 ⭐
│   │   │   ├── cosy_voice.py     # CosyVoice服务
│   │   │   ├── edge_tts_service.py # Edge TTS服务
│   │   │   ├── audio_processor.py # 音频处理
│   │   │   └── lora_training.py  # LoRA训练
│   │   ├── models/               # 数据库模型
│   │   ├── schemas/              # Pydantic 数据模式
│   │   └── main.py               # 应用入口
│   ├── requirements.txt          # Python依赖
│   └── Dockerfile               # Docker配置
│
├── frontend/                     # React 前端应用
│   ├── src/
│   │   ├── pages/               # 页面组件
│   │   │   ├── AudioTools.tsx   # 音频工具页 ⭐
│   │   │   ├── VoiceStudio.tsx  # 语音工作室 ⭐
│   │   │   └── Projects.tsx     # 项目管理
│   │   ├── components/          # 通用组件
│   │   └── services/            # API服务
│   ├── package.json
│   └── Dockerfile
│
├── nginx/                       # Nginx配置
├── docker-compose.yml           # Docker Compose配置
└── README.md                    # 本文档
```

---

## 技术栈

### 后端技术栈
- **框架**: FastAPI 0.109+ (异步Python Web框架)
- **数据库**: SQLite + SQLAlchemy (异步ORM)
- **认证**: JWT Token
- **音频处理**: FFmpeg, pydub, numpy, scipy
- **TTS服务**: Edge TTS, CosyVoice, Qwen TTS
- **语音分析**: librosa, pyannote.audio (可选)

### 前端技术栈
- **框架**: React 18 + TypeScript
- **构建工具**: Vite
- **UI组件库**: shadcn/ui + Radix UI
- **样式**: Tailwind CSS
- **状态管理**: Zustand
- **音频可视化**: wavesurfer.js

---

## 快速开始

### 前置要求

- Python 3.10+
- Node.js 18+
- FFmpeg（系统依赖）
- Docker & Docker Compose（可选）

### Docker 部署（推荐）

1. **克隆项目**
```bash
git clone <repository-url>
cd voiceforge
```

2. **配置环境变量**
```bash
cp backend/.env.example backend/.env
```

编辑 `backend/.env`：
```env
# 应用配置
APP_NAME=VoiceForge
SECRET_KEY=your-secret-key-here

# TTS配置（可选）
TTS_MODE=edge
```

3. **启动服务**
```bash
docker-compose up -d
```

4. **访问应用**
- 前端: http://localhost:3000
- 后端API: http://localhost:8000
- API文档: http://localhost:8000/docs

### 本地开发

#### 后端设置

```bash
cd backend

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 配置环境
cp .env.example .env

# 初始化数据库
alembic upgrade head

# 启动服务
python -m app.main
```

#### 前端设置

```bash
cd frontend

# 安装依赖
pnpm install

# 启动开发服务器
pnpm run dev
```

---

## API文档

### 音频工具API (/api) ⭐ 新增

#### 音频增强
| 方法 | 端点 | 描述 |
|-----|------|------|
| POST | `/enhance` | 音频增强处理 |
| POST | `/analyze-quality` | 音频质量分析 |
| POST | `/convert-format` | 格式转换 |
| GET | `/capabilities` | 获取系统能力 |

#### 说话人分离
| 方法 | 端点 | 描述 |
|-----|------|------|
| POST | `/diarize` | 说话人分离 |
| POST | `/extract-speaker` | 提取特定说话人 |
| POST | `/compare-speakers` | 对比说话人 |

#### 语音转换
| 方法 | 端点 | 描述 |
|-----|------|------|
| POST | `/voice-convert` | 语音转换 |
| GET | `/voice-presets` | 获取语音预设 |
| POST | `/voice-profile` | 创建语音档案 |

#### 音频质量检查
| 方法 | 端点 | 描述 |
|-----|------|------|
| POST | `/audio-quality/check` | 检查音频质量 |
| POST | `/audio-quality/check-batch` | 批量检查 |
| GET | `/audio-quality/guidelines` | 获取录制指南 |

### 核心API端点

#### TTS服务
- `POST /api/cosy-voice/generate` - CosyVoice语音生成
- `POST /api/qwen-tts/generate` - Qwen TTS生成
- `GET /api/voices` - 获取可用语音列表

#### 项目管理
- `GET /api/projects` - 获取项目列表
- `POST /api/projects` - 创建项目
- `GET /api/projects/{id}` - 获取项目详情

详细API文档: http://localhost:8000/docs

---

## 功能演示

### 音频增强示例

```python
import requests

# 上传音频文件进行增强
with open("input.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/enhance",
        files={"file": f},
        data={
            "denoise": True,
            "normalize": True,
            "eq_preset": "vocal"
        }
    )

result = response.json()
# 返回增强后的音频路径和质量指标
```

### 说话人分离示例

```python
# 分离音频中的说话人
with open("conversation.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/diarize",
        files={"file": f},
        data={
            "min_speakers": 1,
            "max_speakers": 3
        }
    )

result = response.json()
# 返回说话人片段和时间戳
```

### 语音转换示例

```python
# 转换语音风格
with open("voice.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/voice-convert",
        files={"file": f},
        data={
            "target_voice": "robotic",
            "pitch_shift": -2
        }
    )

result = response.json()
# 返回转换后的音频
```

---

## 配置说明

### 后端配置 (.env)

```env
# 应用配置
APP_NAME=VoiceForge
APP_VERSION=1.0.0
DEBUG=false
SECRET_KEY=your-secret-key-here

# 服务器配置
HOST=0.0.0.0
PORT=8000

# 数据库配置
DATABASE_URL=sqlite+aiosqlite:///./data/app.db

# TTS配置
TTS_MODE=edge
TTS_LANGUAGE=zh-CN
TTS_PARALLEL_WORKERS=2

# JWT配置
JWT_SECRET_KEY=your-jwt-secret-key
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=10080

# LLM配置（用于脚本生成等功能）
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
LLM_API_KEY=your-api-key-here
LLM_MODEL=glm-4-flash
```

---

## 常见问题

### Q: FFmpeg未找到？
A: 确保系统已安装FFmpeg。macOS: `brew install ffmpeg`，Ubuntu: `sudo apt install ffmpeg`

### Q: 音频增强效果不理想？
A: 尝试调整不同的参数组合，如增强降噪强度或使用不同的EQ预设

### Q: 说话人分离不准确？
A: 确保音频质量良好，背景噪音较小。可以尝试调整min_speakers和max_speakers参数

### Q: 语音转换效果不明显？
A: 尝试增大pitch_shift值，或选择不同的语音预设

---

## 路线图

### v1.1 (计划中)
- [ ] DEMUCS音频源分离集成
- [ ] pyannote.audio说话人分离
- [ ] RVC语音转换
- [ ] Whisper语音转录

### v1.2 (规划中)
- [ ] 实时语音处理
- [ ] 批量音频处理
- [ ] 更多TTS引擎集成
- [ ] 语音情感识别

---

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

---

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 致谢

感谢以下开源项目和技术的支持：

- [FastAPI](https://fastapi.tiangolo.com/) - 现代化的Python Web框架
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - 阿里达摩院语音合成
- [Edge TTS](https://github.com/rany2/edge-tts) - 微软TTS接口
- [pydub](https://github.com/jiaaro/pydub) - 音频处理库
- [librosa](https://github.com/librosa/librosa) - 音频分析库

---

## 联系方式

- 作者：宫灵瑞
- 项目主页：[GitHub Repository]
- 问题反馈：[GitHub Issues]

---

<div align="center">

**Made with ❤️ by 宫灵瑞**

**世界上最强的AI语音处理工具 🚀**

[⬆ 返回顶部](#voiceforge---世界上最强的ai语音处理工具)

</div>

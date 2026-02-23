# 使用 Lovable 生成前端界面指南

## 项目概述

本文档说明如何使用 [Lovable](https://lovable.dev/) AI 工具，基于现有的 Python FastAPI 后端，生成一个完整的中文前端界面。

## 后端 API 信息

**基础 URL**: `http://localhost:8000/api`

**主要功能模块**:
- 音频质量检测
- 音频增强
- 说话人分离
- 语音转换 (RVC)
- 方言/多语言支持
- 模型量化
- ASR 语音识别
- 说话人嵌入提取

---

## Lovable 提示词 (Prompt)

### 完整提示词

复制以下提示词到 Lovable，生成完整的前端应用：

```
你是一位专业的前端开发工程师，擅长使用 React + Tailwind CSS + shadcn/ui 构建现代 Web 应用。

请为我创建一个"AI 语音处理工具平台"的中文前端界面。

## 技术栈要求
- Next.js 14 (App Router)
- TypeScript
- Tailwind CSS
- shadcn/ui 组件库
- Zustand 或 Context API 状态管理
- axios HTTP 客户端
- react-dropzone 文件上传

## 后端 API 信息
- 基础 URL: http://localhost:8000/api
- 响应格式: ApiResponse<T> = { code: number, message: string, data: T }

## 主要功能页面

### 1. 首页/工作台
- 左侧导航栏，包含所有功能模块入口
- 顶部显示用户信息、系统状态
- 主区域显示快捷操作卡片
- 最近处理记录列表

### 2. 音频质量检测 (/audio-quality)
- 音频文件上传区（支持拖拽）
- 显示检测结果：
  - 质量评分（0-100分）
  - 音频时长
  - 响度和动态范围
  - 格式信息
  - 改进建议
- 使用进度条和评分仪表盘

### 3. ASR 语音识别 (/asr)
- 上传音频文件
- 选择识别引擎：
  - Faster Whisper（推荐）
  - OpenAI Whisper
  - Groq API
- 语言选择（中文、英文、日文等）
- 显示识别结果：
  - 转录文本（可编辑）
  - 时间轴分段
  - 置信度
  - 导出为 SRT/TXT

### 4. 说话人分离 (/diarization)
- 上传音频
- 设置说话人数量范围
- 选择后端（pyannote、SpeechBrain、Basic）
- 可视化显示：
  - 波形图
  - 说话人时间轴
  - 按说话人分组显示文本

### 5. 语音转换/RVC (/rvc)
- RVC 模型管理：
  - 模型列表（卡片展示）
  - 上传新模型
  - 删除模型
- 音频转换：
  - 上传源音频
  - 选择目标模型
  - 调整参数（音调、F0方法等）
  - 试听和下载

### 6. 方言支持 (/dialect)
- 语言检测：
  - 输入文本
  - 显示检测到的方言
- 方言转换：
  - 选择目标方言（粤语、客家话、闽南语等）
  - 显示转换结果

### 7. 模型量化 (/quantization)
- 上传模型文件
- 选择量化类型：
  - INT8（推荐）
  - FP16
  - Dynamic
- 显示量化结果：
  - 原始大小 vs 量化后大小
  - 压缩比例
  - 预估加速倍数

## UI 设计要求

### 颜色主题
- 主色调: 靛蓝色 (indigo-600)
- 成功: 绿色 (green-600)
- 警告: 橙色 (orange-600)
- 错误: 红色 (red-600)
- 背景: 浅灰 (gray-50)
- 卡片: 白色带阴影

### 组件使用
- 使用 shadcn/ui 的 Card 组件作为内容容器
- 使用 Button 组件，支持主要、次要、幽灵样式
- 使用 Input、Select、Textarea 表单组件
- 使用 Progress 进度条
- 使用 Alert 警告提示
- 使用 Tabs 选项卡
- 使用 Badge 标签
- 使用 Table 表格
- 使用 Dialog/Modal 弹窗

### 布局结构
```
┌─────────────────────────────────────────┐
│  Header: Logo + 搜索 + 用户菜单          │
├──────────┬──────────────────────────────┤
│          │                               │
│  Nav     │  Main Content Area            │
│          │                               │
│  - 工作台│  - Page Title                 │
│  - 音频  │  - Action Bar                 │
│  - ASR   │  - Content                    │
│  - 转换  │                               │
│  - 模型  │                               │
│          │                               │
└──────────┴──────────────────────────────┘
```

### API 调用封装
创建 `lib/api.ts`:
```typescript
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000/api',
  timeout: 30000,
});

// 请求拦截器
api.interceptors.request.use(config => {
  const token = localStorage.getItem('token');
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// 响应拦截器
api.interceptors.response.use(
  response => response.data,
  error => {
    const message = error.response?.data?.detail || '请求失败';
    // 显示错误提示
    return Promise.reject(new Error(message));
  }
);
```

### 各页面路由结构
```
/app
  /page.tsx           # 工作台首页
  /audio-quality
    /page.tsx         # 音频质量检测
  /asr
    /page.tsx         # 语音识别
    /batch/page.tsx   # 批量识别
  /diarization
    /page.tsx         # 说话人分离
  /rvc
    /page.tsx         # RVC 管理
    /models/page.tsx  # 模型列表
  /dialect
    /page.tsx         # 方言支持
  /quantization
    /page.tsx         # 模型量化
```

### 加载和错误状态
- 所有异步操作显示 loading 骨架屏
- 网络错误显示友好的重试界面
- 使用 Toast 显示操作结果

### 移动端适配
- 使用响应式设计
- 导航栏在移动端折叠为汉堡菜单
- 表格在移动端使用卡片布局

## 数据可视化
- 使用 Recharts 图表库
- 音频波形图: 简单的 CSS 柱状图模拟
- 质量评分: 仪表盘组件
- 时间轴: 水平滚动组件

请生成完整的前端代码，包括所有页面、组件和 API 集成。代码要整洁、可维护、中文注释。
```

---

## 分步生成指南

如果一次性生成太复杂，可以分步骤让 Lovable 生成：

### 第一步：项目基础架构

```
创建一个 Next.js 14 项目，使用 TypeScript + Tailwind CSS + shadcn/ui。

要求：
1. 设置左侧导航栏布局
2. 顶部 Header 带搜索和用户菜单
3. 主内容区域使用 Card 组件
4. 添加路由结构（所有页面暂时显示"建设中"）

路由列表：
- / 工作台
- /audio-quality 音频质量
- /asr 语音识别
- /diarization 说话人分离
- /rvc 语音转换
- /dialect 方言支持
- /quantization 模型量化
```

### 第二步：音频质量检测页面

```
在 /audio-quality 路由创建音频质量检测页面。

功能：
1. 文件上传区（使用 react-dropzone）
2. 上传后显示进度条
3. 调用 POST http://localhost:8000/api/audio-quality/check
4. 显示结果：质量评分（仪表盘）、详细指标、改进建议

API 响应格式：
{
  "code": 200,
  "message": "success",
  "data": {
    "overall_score": 85,
    "duration": 120.5,
    "format": "wav",
    "sample_rate": 48000,
    "recommendations": ["建议1", "建议2"]
  }
}
```

### 第三步：ASR 语音识别页面

```
在 /asr 路由创建语音识别页面。

功能：
1. 文件上传
2. 参数选择：
   - 引擎：faster_whisper, whisper, groq
   - 语言：zh, en, ja, ko
3. 调用 POST http://localhost:8000/api/asr/transcribe
4. 显示转录文本（可复制编辑）
5. 显示时间轴分段

API 响应：
{
  "data": {
    "text": "转录的文本内容",
    "language": "zh",
    "confidence": 0.95,
    "segments": [
      {"start": 0, "end": 2.5, "text": "第一句"},
      {"start": 2.5, "end": 5.0, "text": "第二句"}
    ]
  }
}
```

### 第四步：RVC 模型管理页面

```
在 /rvc/models 创建 RVC 模型管理页面。

功能：
1. 模型列表（卡片展示）
   - 调用 GET http://localhost:8000/api/rvc/models
   - 显示模型名称、语言、状态
2. 上传新模型按钮
3. 删除模型按钮
4. 点击模型跳转到转换页面

API 响应：
{
  "data": {
    "models": [
      {
        "model_id": "model1",
        "name": "女声-温柔",
        "language": "zh-CN",
        "status": "available"
      }
    ]
  }
}
```

---

## 常用 API 端点参考

### 音频相关
| 功能 | 方法 | 端点 |
|------|------|------|
| 质量检测 | POST | /audio-quality/check |
| 音频增强 | POST | /enhancement/process |
| 格式转换 | POST | /conversion/convert |

### ASR 识别
| 功能 | 方法 | 端点 |
|------|------|------|
| 单文件识别 | POST | /asr/transcribe |
| 批量识别 | POST | /asr/batch |
| 获取支持语言 | GET | /asr/languages |
| URL识别 | POST | /asr/transcribe-url |

### RVC 转换
| 功能 | 方法 | 端点 |
|------|------|------|
| 模型列表 | GET | /rvc/models |
| 上传模型 | POST | /rvc/models/upload |
| 音频转换 | POST | /rvc/convert |
| 删除模型 | DELETE | /rvc/models/{id} |

### 说话人分离
| 功能 | 方法 | 端点 |
|------|------|------|
| 增强分离 | POST | /diarization/enhanced |
| 提取嵌入 | POST | /diarization/embeddings |
| 对比说话人 | POST | /diarization/compare-speakers |

### 方言支持
| 功能 | 方法 | 端点 |
|------|------|------|
| 语言检测 | POST | /dialect/detect-language |
| 方言转换 | POST | /dialect/convert |
| 支持列表 | GET | /dialect/supported |

### 模型量化
| 功能 | 方法 | 端点 |
|------|------|------|
| 量化模型 | POST | /quantization/quantize |
| 模型对比 | POST | /quantization/compare |
| 量化列表 | GET | /quantization/models |

---

## Lovable 使用技巧

### 1. 分阶段生成
不要一次性要求生成所有页面，先搭框架，再逐页填充

### 2. 明确技术栈
在提示词开头明确指定技术栈版本

### 3. 提供 API 示例
给 Lovable 看实际的 API 响应格式，帮助它理解数据结构

### 4. 要求代码质量
添加要求：代码整洁、TypeScript 严格模式、中文注释

### 5. 渐进式完善
生成后可以继续对话要求：
- "添加深色模式支持"
- "优化移动端显示"
- "添加更多动画效果"

---

## 生成的代码结构预览

```
frontend/
├── app/
│   ├── layout.tsx          # 根布局
│   ├── page.tsx            # 工作台首页
│   ├── audio-quality/
│   │   └── page.tsx
│   ├── asr/
│   │   ├── page.tsx
│   │   └── batch/page.tsx
│   ├── diarization/
│   │   └── page.tsx
│   ├── rvc/
│   │   ├── page.tsx
│   │   └── models/page.tsx
│   ├── dialect/
│   │   └── page.tsx
│   └── quantization/
│       └── page.tsx
├── components/
│   ├── ui/                 # shadcn/ui 组件
│   ├── AudioUploader.tsx   # 通用音频上传组件
│   ├── ProgressBar.tsx     # 进度条
│   ├── ScoreGauge.tsx      # 评分仪表盘
│   └── Timeline.tsx        # 时间轴组件
├── lib/
│   ├── api.ts              # API 客户端
│   └── utils.ts            # 工具函数
└── types/
    └── api.d.ts            # API 类型定义
```

---

## 常见问题

### Q: Lovable 生成的代码报错怎么办？
A: 把错误信息复制给 Lovable，要求修复

### Q: 如何修改已有页面？
A: 把当前代码发给 Lovable，描述需要的改动

### Q: 如何添加新功能？
A: 明确描述功能需求，提供对应的 API 端点

### Q: 如何部署？
A: Lovable 可以直接部署到 Vercel，或者导出代码自行部署

---

## 开始使用

1. 访问 [lovable.dev](https://lovable.dev/)
2. 创建新项目
3. 复制上面的"完整提示词"
4. 粘贴到 Lovable 对话框
5. 等待代码生成
6. 在 Lovable 编辑器中预览
7. 导出或部署项目

祝你生成顺利！

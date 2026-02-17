# API 接口设计文档

## 1. API 概述

### 1.1 基础信息
- **Base URL**: `http://localhost:8000/api`
- **协议**: HTTP/HTTPS
- **数据格式**: JSON
- **字符编码**: UTF-8

### 1.2 通用规范

#### 请求头
```http
Content-Type: application/json
Authorization: Bearer {access_token}
```

#### 响应格式
**成功响应**
```json
{
  "success": true,
  "data": {},
  "message": "操作成功"
}
```

**错误响应**
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "错误描述",
    "details": {}
  }
}
```

#### HTTP 状态码
| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 201 | 创建成功 |
| 400 | 请求参数错误 |
| 401 | 未授权 |
| 403 | 禁止访问 |
| 404 | 资源不存在 |
| 422 | 验证错误 |
| 500 | 服务器错误 |

### 1.3 分页规范
```json
{
  "items": [],
  "total": 100,
  "page": 1,
  "page_size": 20,
  "total_pages": 5
}
```

## 2. 认证 API

### 2.1 用户注册
```http
POST /api/auth/register
```

**请求体**
```json
{
  "email": "user@example.com",
  "username": "username",
  "password": "password123"
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "user": {
      "id": "uuid",
      "email": "user@example.com",
      "username": "username",
      "created_at": "2024-01-01T00:00:00Z"
    },
    "access_token": "jwt_token",
    "refresh_token": "refresh_token"
  }
}
```

### 2.2 用户登录
```http
POST /api/auth/login
```

**请求体**
```json
{
  "email": "user@example.com",
  "password": "password123"
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "user": {
      "id": "uuid",
      "email": "user@example.com",
      "username": "username"
    },
    "access_token": "jwt_token",
    "refresh_token": "refresh_token"
  }
}
```

### 2.3 刷新令牌
```http
POST /api/auth/refresh
```

**请求体**
```json
{
  "refresh_token": "refresh_token"
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "access_token": "new_jwt_token"
  }
}
```

## 3. 书籍管理 API

### 3.1 获取书籍列表
```http
GET /api/books?page=1&page_size=20&search=关键词&sort=created_at&order=desc
```

**查询参数**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| page | integer | 否 | 页码，默认 1 |
| page_size | integer | 否 | 每页数量，默认 20 |
| search | string | 否 | 搜索关键词 |
| sort | string | 否 | 排序字段 |
| order | string | 否 | 排序方向: asc/desc |

**响应**
```json
{
  "success": true,
  "data": {
    "items": [
      {
        "id": "uuid",
        "title": "书名",
        "author": "作者",
        "cover_url": "http://example.com/cover.jpg",
        "file_type": "txt",
        "total_pages": 100,
        "total_chars": 50000,
        "progress": 0.5,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "projects_count": 2
      }
    ],
    "total": 10,
    "page": 1,
    "page_size": 20,
    "total_pages": 1
  }
}
```

### 3.2 获取书籍详情
```http
GET /api/books/{book_id}
```

**响应**
```json
{
  "success": true,
  "data": {
    "id": "uuid",
    "user_id": "uuid",
    "title": "书名",
    "author": "作者",
    "cover_url": "http://example.com/cover.jpg",
    "file_type": "txt",
    "total_pages": 100,
    "total_chars": 50000,
    "progress": 0.5,
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z",
    "projects": [
      {
        "id": "uuid",
        "name": "项目名称",
        "status": "completed",
        "duration": 3600
      }
    ]
  }
}
```

### 3.3 上传书籍
```http
POST /api/books/upload
Content-Type: multipart/form-data
```

**表单参数**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | file | 是 | 文本文件 (txt/pdf/epub) |
| title | string | 否 | 书名（默认从文件名提取） |
| author | string | 否 | 作者 |
| cover | file | 否 | 封面图片 |

**响应**
```json
{
  "success": true,
  "data": {
    "id": "uuid",
    "title": "书名",
    "author": "作者",
    "file_type": "txt",
    "total_chars": 50000,
    "status": "processing"
  }
}
```

### 3.4 获取书籍内容
```http
GET /api/books/{book_id}/content
```

**查询参数**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| format | string | 否 | 返回格式: plain/markdown/html |
| chapter | integer | 否 | 章节索引 |

**响应**
```json
{
  "success": true,
  "data": {
    "content": "书籍内容文本...",
    "chapters": [
      {"index": 0, "title": "第一章", "offset": 0},
      {"index": 1, "title": "第二章", "offset": 1000}
    ],
    "metadata": {
      "title": "书名",
      "author": "作者",
      "total_chars": 50000
    }
  }
}
```

### 3.5 更新书籍
```http
PATCH /api/books/{book_id}
```

**请求体**
```json
{
  "title": "新书名",
  "author": "新作者",
  "cover_url": "http://example.com/new-cover.jpg",
  "progress": 0.8
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "id": "uuid",
    "title": "新书名",
    "author": "新作者",
    "updated_at": "2024-01-01T00:00:00Z"
  }
}
```

### 3.6 删除书籍
```http
DELETE /api/books/{book_id}
```

**响应**
```json
{
  "success": true,
  "data": {
    "deleted": true
  }
}
```

## 4. 项目管理 API

### 4.1 获取项目列表
```http
GET /api/projects?book_id={book_id}&status=completed
```

**查询参数**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| book_id | string | 否 | 筛选书籍 |
| status | string | 否 | 状态筛选 |
| page | integer | 否 | 页码 |
| page_size | integer | 否 | 每页数量 |

**响应**
```json
{
  "success": true,
  "data": {
    "items": [
      {
        "id": "uuid",
        "book_id": "uuid",
        "book_title": "书名",
        "name": "项目名称",
        "description": "项目描述",
        "status": "processing",
        "config": {
          "tts_mode": "external",
          "language": "zh-CN"
        },
        "duration": null,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "progress": {
          "total_chunks": 100,
          "completed_chunks": 45,
          "percentage": 45
        }
      }
    ],
    "total": 5,
    "page": 1,
    "page_size": 20
  }
}
```

### 4.2 获取项目详情
```http
GET /api/projects/{project_id}
```

**响应**
```json
{
  "success": true,
  "data": {
    "id": "uuid",
    "book_id": "uuid",
    "book_title": "书名",
    "name": "项目名称",
    "description": "项目描述",
    "status": "processing",
    "config": {
      "tts_mode": "external",
      "tts_url": "http://localhost:7860",
      "language": "zh-CN",
      "parallel_workers": 2
    },
    "audio_path": null,
    "duration": null,
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z",
    "script": {
      "id": "uuid",
      "status": "approved",
      "entries_count": 50
    },
    "voice_configs": [
      {
        "id": "uuid",
        "speaker": "NARRATOR",
        "voice_type": "custom",
        "voice_name": "Ryan",
        "language": "zh-CN"
      }
    ],
    "chunks": [
      {
        "id": "uuid",
        "speaker": "NARRATOR",
        "text": "示例文本...",
        "status": "completed",
        "audio_path": "/audio/chunks/xxx/chunk_0001.mp3",
        "duration": 5.5,
        "order_index": 0
      }
    ]
  }
}
```

### 4.3 创建项目
```http
POST /api/projects
```

**请求体**
```json
{
  "book_id": "uuid",
  "name": "项目名称",
  "description": "项目描述",
  "config": {
    "tts_mode": "external",
    "tts_url": "http://localhost:7860",
    "language": "zh-CN",
    "parallel_workers": 2
  }
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "id": "uuid",
    "name": "项目名称",
    "status": "draft",
    "created_at": "2024-01-01T00:00:00Z"
  }
}
```

### 4.4 更新项目
```http
PATCH /api/projects/{project_id}
```

**请求体**
```json
{
  "name": "新项目名称",
  "description": "新描述",
  "config": {
    "language": "en-US"
  }
}
```

### 4.5 删除项目
```http
DELETE /api/projects/{project_id}
```

## 5. 脚本管理 API

### 5.1 生成脚本
```http
POST /api/projects/{project_id}/scripts/generate
```

**请求体**
```json
{
  "system_prompt": "自定义系统提示词",
  "user_prompt": "自定义用户提示词",
  "options": {
    "max_chunk_size": 500,
    "detect_emotions": true,
    "detect_sections": true
  }
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "script_id": "uuid",
    "status": "generating",
    "message": "脚本生成中..."
  }
}
```

### 5.2 获取脚本状态
```http
GET /api/projects/{project_id}/scripts/status
```

**响应**
```json
{
  "success": true,
  "data": {
    "id": "uuid",
    "status": "completed",
    "entries_count": 50,
    "speakers": ["NARRATOR", "主角", "配角"],
    "error_message": null,
    "created_at": "2024-01-01T00:00:00Z"
  }
}
```

### 5.3 获取脚本内容
```http
GET /api/projects/{project_id}/scripts
```

**响应**
```json
{
  "success": true,
  "data": {
    "id": "uuid",
    "project_id": "uuid",
    "status": "approved",
    "content": [
      {
        "index": 0,
        "speaker": "NARRATOR",
        "text": "这是一个示例文本。",
        "instruct": "平静的叙述",
        "emotion": "neutral",
        "section": "第一章"
      }
    ],
    "created_at": "2024-01-01T00:00:00Z"
  }
}
```

### 5.4 更新脚本
```http
PATCH /api/projects/{project_id}/scripts
```

**请求体**
```json
{
  "content": [
    {
      "index": 0,
      "speaker": "NARRATOR",
      "text": "修改后的文本",
      "instruct": "平静的叙述",
      "emotion": "neutral"
    }
  ]
}
```

### 5.5 审查脚本
```http
POST /api/projects/{project_id}/scripts/review
```

**请求体**
```json
{
  "auto_fix": true,
  "check_rules": {
    "speaker_consistency": true,
    "text_continuity": true,
    "emotion_accuracy": true
  }
}
```

### 5.6 批准脚本
```http
POST /api/projects/{project_id}/scripts/approve
```

## 6. 语音配置 API

### 6.1 获取语音列表
```http
GET /api/voices
```

**响应**
```json
{
  "success": true,
  "data": {
    "custom": [
      {"id": "aiden", "name": "Aiden", "gender": "male", "language": "en-US"},
      {"id": "rachel", "name": "Rachel", "gender": "female", "language": "en-US"}
    ],
    "lora": [
      {"id": "builtin_watson", "name": "Watson", "language": "en-US"}
    ]
  }
}
```

### 6.2 获取项目语音配置
```http
GET /api/projects/{project_id}/voices
```

**响应**
```json
{
  "success": true,
  "data": {
    "voices": [
      {
        "id": "uuid",
        "speaker": "NARRATOR",
        "voice_type": "custom",
        "voice_name": "Ryan",
        "style": "calm narrator",
        "language": "zh-CN"
      }
    ]
  }
}
```

### 6.3 解析脚本中的发言人
```http
POST /api/projects/{project_id}/voices/parse
```

**响应**
```json
{
  "success": true,
  "data": {
    "speakers": ["NARRATOR", "主角", "配角"],
    "total_entries": 50
  }
}
```

### 6.4 设置语音配置
```http
POST /api/projects/{project_id}/voices/config
```

**请求体**
```json
{
  "voices": [
    {
      "speaker": "NARRATOR",
      "voice_type": "custom",
      "voice_name": "Ryan",
      "style": "calm narrator",
      "language": "zh-CN"
    },
    {
      "speaker": "主角",
      "voice_type": "clone",
      "ref_audio_path": "/uploads/voices/xxx/reference.wav",
      "style": "young male"
    }
  ]
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "updated": true,
    "count": 2
  }
}
```

### 6.5 预览语音
```http
POST /api/voices/preview
```

**请求体**
```json
{
  "text": "这是预览文本。",
  "voice_type": "custom",
  "voice_name": "Ryan",
  "instruct": "平静的语气"
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "audio_url": "/audio/previews/preview_xxx.mp3",
    "duration": 3.5
  }
}
```

### 6.6 上传参考音频（克隆语音）
```http
POST /api/voices/clone/upload
Content-Type: multipart/form-data
```

**表单参数**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| audio | file | 是 | 参考音频 (5-15秒) |
| text | string | 是 | 参考音频对应的文本 |

**响应**
```json
{
  "success": true,
  "data": {
    "audio_path": "/uploads/voices/xxx/reference.wav",
    "duration": 10.5
  }
}
```

### 6.7 语音设计（文本生成语音）
```http
POST /api/voices/design
```

**请求体**
```json
{
  "description": "一个年轻女性的声音，温柔而有活力",
  "gender": "female",
  "age_range": "20-30",
  "style": "gentle and energetic"
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "preview_url": "/audio/previews/design_xxx.mp3",
    "voice_id": "designed_voice_xxx"
  }
}
```

## 7. 音频生成 API

### 7.1 获取音频块列表
```http
GET /api/projects/{project_id}/chunks
```

**查询参数**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| speaker | string | 否 | 筛选发言人 |
| status | string | 否 | 状态筛选 |
| page | integer | 否 | 页码 |
| page_size | integer | 否 | 每页数量 |

**响应**
```json
{
  "success": true,
  "data": {
    "items": [
      {
        "id": "uuid",
        "speaker": "NARRATOR",
        "text": "示例文本...",
        "instruct": "平静的叙述",
        "section": "第一章",
        "status": "completed",
        "audio_path": "/audio/chunks/xxx/chunk_0001.mp3",
        "duration": 5.5,
        "order_index": 0
      }
    ],
    "total": 100,
    "page": 1,
    "page_size": 20
  }
}
```

### 7.2 生成单个音频块
```http
POST /api/projects/{project_id}/chunks/{chunk_id}/generate
```

**响应**
```json
{
  "success": true,
  "data": {
    "chunk_id": "uuid",
    "status": "processing",
    "message": "音频生成中..."
  }
}
```

### 7.3 批量生成音频
```http
POST /api/projects/{project_id}/chunks/generate-batch
```

**请求体**
```json
{
  "chunk_ids": ["uuid1", "uuid2", "uuid3"],
  "mode": "parallel",
  "workers": 2
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "task_id": "uuid",
    "total_chunks": 3,
    "status": "processing"
  }
}
```

### 7.4 快速批量生成
```http
POST /api/projects/{project_id}/chunks/generate-fast
```

**说明**: 使用优化模式批量生成所有待处理的音频块。

**响应**
```json
{
  "success": true,
  "data": {
    "task_id": "uuid",
    "status": "processing",
    "total_chunks": 100,
    "estimated_time": 300
  }
}
```

### 7.5 更新音频块
```http
PATCH /api/projects/{project_id}/chunks/{chunk_id}
```

**请求体**
```json
{
  "text": "修改后的文本",
  "instruct": "新的指令",
  "speaker": "新的发言人"
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "id": "uuid",
    "text": "修改后的文本",
    "updated_at": "2024-01-01T00:00:00Z"
  }
}
```

### 7.6 重新生成音频块
```http
POST /api/projects/{project_id}/chunks/{chunk_id}/regenerate
```

### 7.7 获取生成进度
```http
GET /api/projects/{project_id}/chunks/progress
```

**响应**
```json
{
  "success": true,
  "data": {
    "total": 100,
    "completed": 45,
    "processing": 2,
    "pending": 53,
    "percentage": 45,
    "estimated_time_remaining": 180
  }
}
```

### 7.8 合并音频
```http
POST /api/projects/{project_id}/audio/merge
```

**请求体**
```json
{
  "pause_between_speakers": 500,
  "pause_same_speaker": 250,
  "output_format": "mp3",
  "bitrate": "128k"
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "status": "merging",
    "message": "音频合并中..."
  }
}
```

### 7.9 获取合并后的音频
```http
GET /api/projects/{project_id}/audio
```

**响应**
```json
{
  "success": true,
  "data": {
    "audio_url": "/audio/projects/xxx/final.mp3",
    "duration": 3600,
    "file_size": 5242880,
    "format": "mp3",
    "bitrate": "128k"
  }
}
```

### 7.10 下载音频
```http
GET /api/projects/{project_id}/audio/download
```

**查询参数**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| format | string | 否 | 格式: mp3/wav/zip |
| include_metadata | boolean | 否 | 是否包含元数据 |

**响应**: 返回音频文件流

### 7.11 导出 Audacity 项目
```http
POST /api/projects/{project_id}/audio/export/audacity
```

**响应**
```json
{
  "success": true,
  "data": {
    "download_url": "/exports/xxx/audacity_project.zip",
    "files": ["project.aup", "data/"]
  }
}
```

## 8. 笔记管理 API

### 8.1 获取高亮列表
```http
GET /api/books/{book_id}/highlights
```

**查询参数**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| color | string | 否 | 颜色筛选 |
| chapter | string | 否 | 章节筛选 |
| has_note | boolean | 否 | 是否有笔记 |

**响应**
```json
{
  "success": true,
  "data": {
    "items": [
      {
        "id": "uuid",
        "text": "高亮的文本内容",
        "color": "yellow",
        "start_offset": 100,
        "end_offset": 150,
        "chapter": "第一章",
        "note": {
          "id": "uuid",
          "content": "笔记内容"
        },
        "created_at": "2024-01-01T00:00:00Z"
      }
    ],
    "total": 10
  }
}
```

### 8.2 创建高亮
```http
POST /api/books/{book_id}/highlights
```

**请求体**
```json
{
  "text": "高亮的文本",
  "color": "yellow",
  "start_offset": 100,
  "end_offset": 150,
  "chapter": "第一章",
  "chunk_id": "uuid",
  "note": "笔记内容（可选）"
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "id": "uuid",
    "text": "高亮的文本",
    "color": "yellow",
    "created_at": "2024-01-01T00:00:00Z"
  }
}
```

### 8.3 更新高亮
```http
PATCH /api/highlights/{highlight_id}
```

**请求体**
```json
{
  "color": "green"
}
```

### 8.4 删除高亮
```http
DELETE /api/highlights/{highlight_id}
```

### 8.5 添加/更新笔记
```http
PUT /api/highlights/{highlight_id}/note
```

**请求体**
```json
{
  "content": "笔记内容"
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "id": "uuid",
    "highlight_id": "uuid",
    "content": "笔记内容",
    "updated_at": "2024-01-01T00:00:00Z"
  }
}
```

### 8.6 删除笔记
```http
DELETE /api/highlights/{highlight_id}/note
```

### 8.7 导出笔记
```http
GET /api/books/{book_id}/notes/export
```

**查询参数**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| format | string | 否 | 格式: json/markdown/csv |

**响应**: 返回导出文件流

## 9. 系统配置 API

### 9.1 获取系统配置
```http
GET /api/config
```

**响应**
```json
{
  "success": true,
  "data": {
    "tts": {
      "mode": "external",
      "url": "http://localhost:7860",
      "device": "auto",
      "parallel_workers": 2,
      "language": "zh-CN"
    },
    "llm": {
      "base_url": "http://localhost:11434/v1",
      "api_key": "local",
      "model_name": "qwen3-14b"
    },
    "prompts": {
      "system_prompt": "",
      "user_prompt": ""
    }
  }
}
```

### 9.2 更新系统配置
```http
PATCH /api/config
```

**请求体**
```json
{
  "tts": {
    "mode": "external",
    "url": "http://localhost:7860",
    "language": "zh-CN"
  }
}
```

### 9.3 获取默认提示词
```http
GET /api/prompts/default
```

**响应**
```json
{
  "success": true,
  "data": {
    "script_generation": "默认的脚本生成提示词...",
    "script_review": "默认的脚本审查提示词..."
  }
}
```

### 9.4 获取系统状态
```http
GET /api/system/status
```

**响应**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "version": "1.0.0",
    "services": {
      "tts": "connected",
      "llm": "connected",
      "database": "connected"
    },
    "resources": {
      "cpu_usage": 45.2,
      "memory_usage": 62.8,
      "disk_usage": 55.3
    }
  }
}
```

## 10. WebSocket API

### 10.1 连接
```
ws://localhost:8000/ws/{project_id}
```

### 10.2 消息类型

#### 客户端 → 服务器
```json
{
  "type": "subscribe",
  "channel": "audio_progress"
}
```

#### 服务器 → 客户端
```json
{
  "type": "audio_progress",
  "data": {
    "project_id": "uuid",
    "chunk_id": "uuid",
    "status": "completed",
    "progress": {
      "completed": 45,
      "total": 100,
      "percentage": 45
    }
  }
}
```

## 11. 错误码参考

| 错误码 | 说明 |
|--------|------|
| AUTH_001 | 未授权访问 |
| AUTH_002 | 令牌过期 |
| AUTH_003 | 登录失败 |
| BOOK_001 | 书籍不存在 |
| BOOK_002 | 文件格式不支持 |
| BOOK_003 | 文件大小超限 |
| PROJECT_001 | 项目不存在 |
| PROJECT_002 | 项目状态不允许操作 |
| SCRIPT_001 | 脚本生成失败 |
| SCRIPT_002 | LLM 服务不可用 |
| AUDIO_001 | TTS 服务不可用 |
| AUDIO_002 | 音频生成失败 |
| VOICE_001 | 语音配置无效 |
| VOICE_002 | 参考音频无效 |
| VALIDATION_001 | 请求参数验证失败 |
| SERVER_001 | 服务器内部错误 |

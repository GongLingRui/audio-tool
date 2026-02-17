# 数据库设计文档

## 1. 数据库概述

### 1.1 技术选型
- **数据库**: SQLite 3
- **ORM**: SQLAlchemy 2.0 (异步支持)
- **迁移工具**: Alembic

### 1.2 数据库位置
```
backend/data/app.db
```

### 1.3 命名规范
- 表名: `snake_case`, 复数形式 (如 `users`, `books`)
- 字段名: `snake_case`
- 主键: `id` (UUID 或自增整数)
- 外键: `{resource}_id` (如 `user_id`, `book_id`)
- 时间戳: `created_at`, `updated_at`

## 2. 数据模型设计

### 2.1 ER 图

```
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│    users    │       │    books    │       │  projects   │
├─────────────┤       ├─────────────┤       ├─────────────┤
│ id          │───┐   │ id          │───┐   │ id          │
│ email       │   │   │ user_id     │◀──┘   │ book_id     │───┐
│ username    │   │   │ title       │       │ name        │   │
│ password    │   │   │ author      │       │ status      │   │
│ created_at  │   │   │ cover_url   │       │ created_at  │   │
└─────────────┘   │   │ total_pages │       │ updated_at  │   │
                  │   │ created_at  │       └─────────────┘   │
                  │   └─────────────┘                          │
                  │                                           │
                  │   ┌─────────────┐       ┌─────────────┐   │
                  │   │  scripts    │       │   chunks    │   │
                  │   ├─────────────┤       ├─────────────┤   │
                  │   │ id          │───┐   │ id          │   │
                  └──▶│ project_id  │   │   │ project_id  │◀──┘
                      │ content     │   │   │ script_id   │
                      │ status      │   │   │ speaker     │
                      │ created_at  │   │   │ text        │
                      └─────────────┘   │   │ instruct    │
                                        │   │ status      │
                                        │   │ audio_path  │
                                        │   │ duration    │
                                        │   │ order_index │
                                        │   └─────────────┘
                                        │
                ┌─────────────┐       │
                │ voice_configs│      │
                ├─────────────┤       │
                │ id          │       │
                │ project_id  │◀──────┘
                │ speaker     │
                │ voice_type  │
                │ voice_name  │
                │ style       │
                │ ref_audio   │
                │ created_at  │
                └─────────────┘

┌─────────────┐       ┌─────────────┐
│ highlights  │       │   notes     │
├─────────────┤       ├─────────────┤
│ id          │───┐   │ id          │
│ user_id     │   │   │ highlight_id│◀──┐
│ book_id     │   │   │ content     │   │
│ chunk_id    │   │   │ created_at  │   │
│ text        │   │   └─────────────┘   │
│ color       │   │                     │
│ position    │   │                     │
│ created_at  │   │                     │
└─────────────┘   │                     │
                  └─────────────────────┘
```

## 3. 表结构详细设计

### 3.1 users (用户表)

用户信息表，存储认证和用户基本信息。

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | UUID | PK, NOT NULL | 用户唯一标识 |
| email | VARCHAR(255) | UNIQUE, NOT NULL | 邮箱 |
| username | VARCHAR(50) | UNIQUE, NOT NULL | 用户名 |
| password_hash | VARCHAR(255) | NOT NULL | 密码哈希 |
| avatar_url | VARCHAR(500) | NULL | 头像 URL |
| created_at | DATETIME | NOT NULL, DEFAULT NOW | 创建时间 |
| updated_at | DATETIME | NOT NULL, DEFAULT NOW | 更新时间 |

#### 索引
- `idx_users_email`: UNIQUE (email)
- `idx_users_username`: UNIQUE (username)

#### SQLAlchemy 模型
```python
from sqlalchemy import Column, String, DateTime
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    avatar_url = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
```

### 3.2 books (书籍表)

书籍信息表，存储上传的书籍元数据。

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | UUID | PK, NOT NULL | 书籍唯一标识 |
| user_id | UUID | FK(users.id), NOT NULL | 所属用户 |
| title | VARCHAR(500) | NOT NULL | 书名 |
| author | VARCHAR(255) | NULL | 作者 |
| cover_url | VARCHAR(500) | NULL | 封面图片 URL |
| file_path | VARCHAR(500) | NOT NULL | 原始文件路径 |
| file_type | VARCHAR(20) | NOT NULL | 文件类型 (txt/pdf/epub) |
| total_pages | INTEGER | NULL | 总页数 |
| total_chars | INTEGER | NULL | 总字符数 |
| progress | FLOAT | DEFAULT 0 | 阅读进度 (0-1) |
| created_at | DATETIME | NOT NULL, DEFAULT NOW | 创建时间 |
| updated_at | DATETIME | NOT NULL, DEFAULT NOW | 更新时间 |

#### 索引
- `idx_books_user_id`: (user_id)
- `idx_books_created_at`: (created_at DESC)

#### SQLAlchemy 模型
```python
class Book(Base):
    __tablename__ = "books"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(500), nullable=False)
    author = Column(String(255), nullable=True)
    cover_url = Column(String(500), nullable=True)
    file_path = Column(String(500), nullable=False)
    file_type = Column(String(20), nullable=False)  # txt, pdf, epub
    total_pages = Column(Integer, nullable=True)
    total_chars = Column(Integer, nullable=True)
    progress = Column(Float, default=0.0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
```

### 3.3 projects (有声书项目表)

有声书生成项目，关联书籍和生成配置。

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | UUID | PK, NOT NULL | 项目唯一标识 |
| book_id | UUID | FK(books.id), NOT NULL | 关联书籍 |
| name | VARCHAR(255) | NOT NULL | 项目名称 |
| description | TEXT | NULL | 项目描述 |
| status | VARCHAR(20) | NOT NULL, DEFAULT 'draft' | 状态: draft/processing/completed/failed |
| config | JSON | NOT NULL, DEFAULT {} | 生成配置 |
| audio_path | VARCHAR(500) | NULL | 最终音频文件路径 |
| duration | FLOAT | NULL | 音频总时长(秒) |
| created_at | DATETIME | NOT NULL, DEFAULT NOW | 创建时间 |
| updated_at | DATETIME | NOT NULL, DEFAULT NOW | 更新时间 |

#### 索引
- `idx_projects_book_id`: (book_id)
- `idx_projects_status`: (status)

#### SQLAlchemy 模型
```python
class Project(Base):
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    book_id = Column(UUID(as_uuid=True), ForeignKey("books.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(20), default="draft", nullable=False, index=True)  # draft, processing, completed, failed
    config = Column(JSON, default={}, nullable=False)  # TTS配置、语音配置等
    audio_path = Column(String(500), nullable=True)
    duration = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
```

### 3.4 scripts (脚本表)

LLM 生成的标注脚本。

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | UUID | PK, NOT NULL | 脚本唯一标识 |
| project_id | UUID | FK(projects.id), NOT NULL | 关联项目 |
| content | JSON | NOT NULL | 脚本内容 (结构化数组) |
| status | VARCHAR(20) | NOT NULL, DEFAULT 'pending' | 状态: pending/reviewed/approved |
| error_message | TEXT | NULL | 错误信息 |
| created_at | DATETIME | NOT NULL, DEFAULT NOW | 创建时间 |
| updated_at | DATETIME | NOT NULL, DEFAULT NOW | 更新时间 |

#### 索引
- `idx_scripts_project_id`: UNIQUE (project_id)

#### content 字段结构
```json
[
  {
    "index": 0,
    "speaker": "NARRATOR",
    "text": "这是一个示例文本。",
    "instruct": "平静的叙述",
    "emotion": "neutral",
    "section": "第一章"
  }
]
```

#### SQLAlchemy 模型
```python
class Script(Base):
    __tablename__ = "scripts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, unique=True, index=True)
    content = Column(JSON, nullable=False)  # 结构化脚本数组
    status = Column(String(20), default="pending", nullable=False)  # pending, reviewed, approved
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
```

### 3.5 chunks (音频块表)

音频生成块，按脚本分块后的音频单元。

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | UUID | PK, NOT NULL | 块唯一标识 |
| project_id | UUID | FK(projects.id), NOT NULL | 关联项目 |
| script_id | UUID | FK(scripts.id), NOT NULL | 关联脚本 |
| speaker | VARCHAR(100) | NOT NULL | 发言人 |
| text | TEXT | NOT NULL | 文本内容 |
| instruct | VARCHAR(500) | NULL | TTS 指令 |
| emotion | VARCHAR(50) | NULL | 情绪标签 |
| section | VARCHAR(255) | NULL | 所属章节 |
| status | VARCHAR(20) | NOT NULL, DEFAULT 'pending' | 状态: pending/processing/completed/failed |
| audio_path | VARCHAR(500) | NULL | 音频文件路径 |
| duration | FLOAT | NULL | 音频时长(秒) |
| order_index | INTEGER | NOT NULL | 顺序索引 |
| created_at | DATETIME | NOT NULL, DEFAULT NOW | 创建时间 |
| updated_at | DATETIME | NOT NULL, DEFAULT NOW | 更新时间 |

#### 索引
- `idx_chunks_project_id`: (project_id)
- `idx_chunks_script_id`: (script_id)
- `idx_chunks_status`: (status)
- `idx_chunks_order`: (project_id, order_index)

#### SQLAlchemy 模型
```python
class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, index=True)
    script_id = Column(UUID(as_uuid=True), ForeignKey("scripts.id"), nullable=False, index=True)
    speaker = Column(String(100), nullable=False)
    text = Column(Text, nullable=False)
    instruct = Column(String(500), nullable=True)
    emotion = Column(String(50), nullable=True)
    section = Column(String(255), nullable=True)
    status = Column(String(20), default="pending", nullable=False, index=True)  # pending, processing, completed, failed
    audio_path = Column(String(500), nullable=True)
    duration = Column(Float, nullable=True)
    order_index = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index('idx_chunks_order', 'project_id', 'order_index'),
    )
```

### 3.6 voice_configs (语音配置表)

语音配置，每个角色的语音设置。

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | UUID | PK, NOT NULL | 配置唯一标识 |
| project_id | UUID | FK(projects.id), NOT NULL | 关联项目 |
| speaker | VARCHAR(100) | NOT NULL | 发言人名称 |
| voice_type | VARCHAR(20) | NOT NULL | 语音类型: custom/clone/lora/design |
| voice_name | VARCHAR(100) | NULL | 语音名称 |
| style | VARCHAR(255) | NULL | 风格描述 |
| ref_audio_path | VARCHAR(500) | NULL | 参考音频路径 (clone类型) |
| lora_model_path | VARCHAR(500) | NULL | LoRA 模型路径 (lora类型) |
| language | VARCHAR(20) | DEFAULT 'zh-CN' | 语言 |
| created_at | DATETIME | NOT NULL, DEFAULT NOW | 创建时间 |
| updated_at | DATETIME | NOT NULL, DEFAULT NOW | 更新时间 |

#### 索引
- `idx_voice_configs_project_id`: (project_id)
- `idx_voice_configs_project_speaker`: UNIQUE (project_id, speaker)

#### SQLAlchemy 模型
```python
class VoiceConfig(Base):
    __tablename__ = "voice_configs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, index=True)
    speaker = Column(String(100), nullable=False)
    voice_type = Column(String(20), nullable=False)  # custom, clone, lora, design
    voice_name = Column(String(100), nullable=True)
    style = Column(String(255), nullable=True)
    ref_audio_path = Column(String(500), nullable=True)
    lora_model_path = Column(String(500), nullable=True)
    language = Column(String(20), default="zh-CN", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint('project_id', 'speaker', name='uq_voice_configs_project_speaker'),
    )
```

### 3.7 highlights (高亮表)

文本高亮记录。

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | UUID | PK, NOT NULL | 高亮唯一标识 |
| user_id | UUID | FK(users.id), NOT NULL | 所属用户 |
| book_id | UUID | FK(books.id), NOT NULL | 关联书籍 |
| chunk_id | UUID | FK(chunks.id), NULL | 关联音频块 (可选) |
| text | TEXT | NOT NULL | 高亮文本内容 |
| color | VARCHAR(20) | NOT NULL | 颜色: yellow/green/blue/pink |
| start_offset | INTEGER | NOT NULL | 起始位置 |
| end_offset | INTEGER | NOT NULL | 结束位置 |
| chapter | VARCHAR(255) | NULL | 所属章节 |
| created_at | DATETIME | NOT NULL, DEFAULT NOW | 创建时间 |

#### 索引
- `idx_highlights_user_id`: (user_id)
- `idx_highlights_book_id`: (book_id)
- `idx_highlights_chunk_id`: (chunk_id)

#### SQLAlchemy 模型
```python
class Highlight(Base):
    __tablename__ = "highlights"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    book_id = Column(UUID(as_uuid=True), ForeignKey("books.id"), nullable=False, index=True)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("chunks.id"), nullable=True, index=True)
    text = Column(Text, nullable=False)
    color = Column(String(20), nullable=False)  # yellow, green, blue, pink
    start_offset = Column(Integer, nullable=False)
    end_offset = Column(Integer, nullable=False)
    chapter = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
```

### 3.8 notes (笔记表)

笔记内容，关联高亮。

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | UUID | PK, NOT NULL | 笔记唯一标识 |
| highlight_id | UUID | FK(highlights.id), NOT NULL | 关联高亮 |
| content | TEXT | NOT NULL | 笔记内容 |
| created_at | DATETIME | NOT NULL, DEFAULT NOW | 创建时间 |
| updated_at | DATETIME | NOT NULL, DEFAULT NOW | 更新时间 |

#### 索引
- `idx_notes_highlight_id`: UNIQUE (highlight_id)

#### SQLAlchemy 模型
```python
class Note(Base):
    __tablename__ = "notes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    highlight_id = Column(UUID(as_uuid=True), ForeignKey("highlights.id"), nullable=False, unique=True, index=True)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
```

## 4. 文件存储结构

```
backend/app/static/
├── uploads/              # 用户上传文件
│   ├── books/           # 书籍原始文件
│   │   ├── {user_id}/
│   │   │   └── {book_id}.txt
│   │   └── temp/        # 临时文件
│   └── voices/          # 语音参考音频
│       └── {user_id}/
│           └── {voice_id}.wav
├── audio/               # 生成的音频文件
│   ├── chunks/          # 音频块
│   │   └── {project_id}/
│   │       ├── chunk_0001.mp3
│   │       └── chunk_0002.mp3
│   ├── projects/        # 合并后的项目音频
│   │   └── {project_id}/
│   │       └── final.mp3
│   └── previews/        # 语音预览
│       └── {preview_id}.mp3
├── lora/               # LoRA 模型
│   └── {model_name}/
│       └── adapter.safetensors
└── exports/            # 导出文件
    ├── {project_id}/
    │   ├── audiobook.mp3
    │   └── audacity_project/
    │       ├── data/
    │       └── project.aup
```

## 5. 关系总结

### 5.1 一对多关系
- User → Books (一个用户可以有多个书籍)
- User → Highlights (一个用户可以有多个高亮)
- Book → Projects (一个书籍可以有多个有声书项目)
- Project → Scripts (一个项目有一个脚本)
- Project → Chunks (一个项目有多个音频块)
- Project → VoiceConfigs (一个项目有多个语音配置)
- Script → Chunks (一个脚本有多个音频块)

### 5.2 一对一关系
- Highlight → Note (一个高亮有一个笔记)

### 5.3 级联规则
- 删除 User 时级联删除其 Books、Highlights
- 删除 Book 时不级联（保留已生成的项目）
- 删除 Project 时级联删除其 Script、Chunks、VoiceConfigs
- 删除 Highlight 时级联删除其 Note
- 删除 Script 时级联删除其 Chunks

## 6. 数据库初始化脚本

```python
# backend/app/database.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.orm import declarative_base
from contextlib import asynccontextmanager
import os

DATABASE_URL = "sqlite+aiosqlite:///./data/app.db"

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False}
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()

async def init_db():
    """初始化数据库表"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db() -> AsyncSession:
    """依赖注入：获取数据库会话"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

## 7. 迁移管理 (Alembic)

```ini
# alembic.ini
[alembic]
script_location = alembic
file_template = %%(year)d%%(month).2d%%(day).2d_%%(hour).2d%%(minute).2d_%%(rev)s_%%(slug)s
sqlalchemy.url = sqlite+aiosqlite:///./data/app.db

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic
```

```python
# alembic/env.py
from asyncio import run
from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.database import Base
from app.models import *  # 导入所有模型

config = context.config
fileConfig(config.config_file_name)
target_metadata = Base.metadata

def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()

async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()

def run_migrations_online() -> None:
    run(run_async_migrations())

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

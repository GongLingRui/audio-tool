"""Application configuration."""
from pathlib import Path
from typing import List

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 使用 config 文件所在目录计算路径，避免 cwd 导致路径错误
_BASE_DIR = Path(__file__).resolve().parent.parent  # backend/app -> backend


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "Read-Rhyme"
    app_version: str = "0.1.0"
    debug: bool = False
    secret_key: str = ""
    app_mode: str = "hub"  # hub or full

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/app.db"

    # CORS - Allow all localhost origins for development
    cors_origins: List[str] = [
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:8080",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:3000",
    ]

    # File Storage - 使用绝对路径，避免不同 cwd 导致文件找不到
    upload_dir: Path = _BASE_DIR / "static" / "uploads"
    audio_dir: Path = _BASE_DIR / "static" / "audio"
    export_dir: Path = _BASE_DIR / "static" / "exports"
    max_upload_size: int = 100 * 1024 * 1024  # 100MB

    # TTS Configuration
    tts_mode: str = "external"  # local, external, edge
    tts_url: str = "http://localhost:7860"
    tts_timeout: int = 300
    tts_parallel_workers: int = 2
    tts_language: str = "zh-CN"

    # LLM Configuration
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "local"
    llm_model: str = "qwen3-14b"

    # JWT
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 1440  # 24 hours

    # Prompts
    script_generation_prompt: str = ""
    script_review_prompt: str = ""

    @model_validator(mode="after")
    def resolve_paths(self):
        """将相对路径解析为基于 backend 目录的绝对路径"""
        for name in ("upload_dir", "audio_dir", "export_dir"):
            p = getattr(self, name)
            if p and not p.is_absolute():
                setattr(self, name, (_BASE_DIR / p).resolve())
        return self

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()

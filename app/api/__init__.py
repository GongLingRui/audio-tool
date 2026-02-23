"""API routes - Refactored for general audio content creation."""
from fastapi import APIRouter

from app.api import (
    audio,
    audio_tools,
    audio_processor,
    auth,
    books,
    config,
    cosy_voice,
    dashboard,
    emotion_presets,
    lora_training,
    projects,
    rag,
    scripts,
    voices,
    voice_styling,
    voice_tools,
    qwen_tts,
    websocket,
    voice_advanced,
    rate_limit,
)

api_router = APIRouter(prefix="/api")

# Core functionality
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(books.router, prefix="/books", tags=["Books"])
api_router.include_router(projects.router, prefix="/projects", tags=["Projects"])
api_router.include_router(scripts.router, prefix="/projects", tags=["Scripts"])

# Voice features
api_router.include_router(voices.router, prefix="/voices", tags=["Voices"])
api_router.include_router(voice_styling.router, prefix="/voice-styling", tags=["Voice Styling"])
api_router.include_router(voice_advanced.router, prefix="/voice-advanced", tags=["Voice Advanced"])
api_router.include_router(lora_training.router, prefix="/lora", tags=["LoRA Training"])

# Audio processing
api_router.include_router(audio.router, prefix="/projects", tags=["Audio"])
api_router.include_router(voice_tools.router, prefix="/projects", tags=["Voice Tools"])
api_router.include_router(audio_tools.router, tags=["Audio Tools"])
api_router.include_router(audio_processor.router, prefix="/audio-processor", tags=["Audio Processor"])

# RAG
api_router.include_router(rag.router, prefix="/rag", tags=["RAG"])

# TTS engines
api_router.include_router(qwen_tts.router, prefix="/qwen-tts", tags=["Qwen3-TTS"])
api_router.include_router(cosy_voice.router, prefix="/cosy-voice", tags=["CosyVoice"])

# Real-time communication
api_router.include_router(websocket.router, tags=["WebSocket"])

# Other features
api_router.include_router(emotion_presets.router, tags=["Emotion Presets"])
api_router.include_router(config.router, prefix="/config", tags=["Config"])
api_router.include_router(rate_limit.router, prefix="/rate-limit", tags=["Rate Limit"])
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["Dashboard"])

"""API routes."""
from fastapi import APIRouter

from app.api import (
    audio,
    audio_tools,
    audio_processor,
    auth,
    books,
    config,
    cosy_voice,
    emotion_presets,
    highlights,
    lora_training,
    projects,
    scripts,
    thoughts,
    voices,
    voice_styling,
    voice_tools,
    rag,
    qwen_tts,
    websocket,
    voice_advanced,
    rate_limit,
)

api_router = APIRouter(prefix="/api")

# Include all route modules
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(books.router, prefix="/books", tags=["Books"])
api_router.include_router(projects.router, prefix="/projects", tags=["Projects"])
api_router.include_router(scripts.router, prefix="/projects", tags=["Scripts"])
api_router.include_router(voices.router, prefix="/voices", tags=["Voices"])
api_router.include_router(voice_styling.router, prefix="/voice-styling", tags=["Voice Styling"])
api_router.include_router(voice_advanced.router, prefix="/voice-advanced", tags=["Voice Advanced"])
api_router.include_router(audio.router, prefix="/projects", tags=["Audio"])
api_router.include_router(highlights.router, prefix="/highlights", tags=["Highlights"])
api_router.include_router(thoughts.router, prefix="/thoughts", tags=["Thoughts"])
api_router.include_router(config.router, prefix="/config", tags=["Config"])
api_router.include_router(rag.router, prefix="/rag", tags=["RAG - Document Q&A"])
api_router.include_router(qwen_tts.router, prefix="/qwen-tts", tags=["Qwen3-TTS"])
api_router.include_router(websocket.router, tags=["WebSocket"])
api_router.include_router(emotion_presets.router, tags=["Emotion Presets"])
api_router.include_router(voice_tools.router, prefix="/projects", tags=["Voice Tools"])
api_router.include_router(audio_tools.router, tags=["Audio Tools"])
api_router.include_router(lora_training.router, prefix="/lora", tags=["LoRA Training"])
api_router.include_router(audio_processor.router, prefix="/audio-processor", tags=["Audio Processor"])
api_router.include_router(cosy_voice.router, prefix="/cosy-voice", tags=["CosyVoice"])
api_router.include_router(rate_limit.router, prefix="/rate-limit", tags=["Rate Limit"])

"""Services module."""

# Lazy imports to avoid missing dependencies
__all__ = [
    "TTSEngine",
    "TTSEngineFactory",
    "AudioProcessor",
    "ChunkProcessor",
    "ScriptGenerator",
    "VoiceManager",
    "get_audio_cache",
    "get_tts_retry_service",
    "manager",
    "get_websocket_manager",
]


def __getattr__(name):
    """Lazy import services."""
    if name == "AudioProcessor":
        from app.services.audio_processor import AudioProcessor
        return AudioProcessor
    elif name == "ChunkProcessor":
        from app.services.chunk_processor import ChunkProcessor
        return ChunkProcessor
    elif name == "ScriptGenerator":
        from app.services.script_generator import ScriptGenerator
        return ScriptGenerator
    elif name == "TTSEngine":
        from app.services.tts_engine import TTSEngine
        return TTSEngine
    elif name == "TTSEngineFactory":
        from app.services.tts_engine import TTSEngineFactory
        return TTSEngineFactory
    elif name == "VoiceManager":
        from app.services.voice_manager import VoiceManager
        return VoiceManager
    elif name == "get_audio_cache":
        from app.services.audio_cache import get_audio_cache
        return get_audio_cache
    elif name == "get_tts_retry_service":
        from app.services.tts_retry_service import get_tts_retry_service
        return get_tts_retry_service
    elif name == "manager" or name == "get_websocket_manager":
        from app.services.websocket_manager import manager
        return manager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

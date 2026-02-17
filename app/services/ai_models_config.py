"""
AI Models Configuration for Memory-Constrained Systems
Safe model choices for M4 32GB RAM
"""

# 轻量级 TTS 模型配置
TTS_MODELS = {
    "lightweight": {
        "name": "espeak-ng",  # 本地TTS，不加载ML模型
        "memory_mb": 50,
        "description": "超轻量，无ML模型"
    },
    "small-tts": {
        "name": "speechbrain/TTS-tacotron2-ljspeech",
        "memory_mb": 300,
        "description": "小型TTS模型"
    },
    "medium-tts": {
        "name": "microsoft/speecht5_tts",
        "memory_mb": 800,
        "description": "中型TTS模型"
    },
    # 仅在有足够内存时使用
    "qwen3-tts": {
        "name": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "memory_mb": 4000,
        "description": "Qwen3-TTS (需要4GB+)"
    }
}

# 轻量级 Embedding 模型配置
EMBEDDING_MODELS = {
    "tiny": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "memory_mb": 80,
        "description": "超小模型，80MB"
    },
    "small": {
        "name": "sentence-transformers/all-MiniLM-L12-v2",
        "memory_mb": 120,
        "description": "小型模型，120MB"
    },
    "medium": {
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "memory_mb": 470,
        "description": "中型多语言模型，470MB"
    }
}

# 推荐配置 - 适合32GB M4
RECOMMENDED_CONFIG = {
    "tts_model": "small-tts",      # 使用小模型
    "embedding_model": "tiny",     # 使用最小embedding模型
    "max_concurrent_requests": 1,  # 限制并发
    "unload_idle_seconds": 300,    # 5分钟不使用自动卸载
}

def get_safe_models(available_memory_gb: float) -> dict:
    """根据可用内存返回安全的模型配置"""
    if available_memory_gb < 2:
        return {
            "tts": TTS_MODELS["lightweight"],
            "embedding": EMBEDDING_MODELS["tiny"],
            "warning": "内存严重不足，使用最轻量配置"
        }
    elif available_memory_gb < 4:
        return {
            "tts": TTS_MODELS["small-tts"],
            "embedding": EMBEDDING_MODELS["tiny"],
            "warning": "内存紧张，使用轻量配置"
        }
    elif available_memory_gb < 8:
        return {
            "tts": TTS_MODELS["medium-tts"],
            "embedding": EMBEDDING_MODELS["small"],
            "warning": "内存适中，使用标准配置"
        }
    else:
        return {
            "tts": TTS_MODELS["qwen3-tts"],
            "embedding": EMBEDDING_MODELS["medium"],
            "warning": "内存充足，可以使用完整模型"
        }

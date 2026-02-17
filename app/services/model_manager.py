"""
统一模型管理器
支持 LLM、TTS、Embedding 模型的下载、加载和测试
"""

import os
import logging
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """模型类型"""
    LLM = "llm"
    TTS = "tts"
    EMBEDDING = "embedding"


class ModelSize(Enum):
    """模型大小"""
    SMALL = "small"  # < 4B params
    MEDIUM = "medium"  # 4B - 14B params
    LARGE = "large"  # 14B - 32B params
    XLARGE = "xlarge"  # > 32B params


@dataclass
class ModelConfig:
    """模型配置"""
    model_id: str
    name: str
    model_type: ModelType
    size: ModelSize
    params: str  # e.g., "7B", "14B"
    memory_gb: int  # Estimated memory requirement
    description: str
    huggingface_id: Optional[str] = None
    ollama_name: Optional[str] = None
    requires_github: bool = False
    is_free: bool = True
    languages: List[str] = field(default_factory=list)
    features: List[str] = field(default_factory=list)


# 所有支持的模型配置
AVAILABLE_MODELS: Dict[str, ModelConfig] = {
    # ============ LLM 模型 ============
    "qwen2.5-7b-instruct": ModelConfig(
        model_id="qwen2.5-7b-instruct",
        name="Qwen2.5 7B Instruct",
        model_type=ModelType.LLM,
        size=ModelSize.MEDIUM,
        params="7B",
        memory_gb=5,
        description="Qwen2.5 7B 指令模型，适合日常任务",
        huggingface_id="Qwen/Qwen2.5-7B-Instruct",
        ollama_name="qwen2.5:7b",
        is_free=True,
        languages=["zh", "en"],
        features=["chat", "reasoning", "code"]
    ),

    "qwen2.5-14b-instruct": ModelConfig(
        model_id="qwen2.5-14b-instruct",
        name="Qwen2.5 14B Instruct",
        model_type=ModelType.LLM,
        size=ModelSize.LARGE,
        params="14B",
        memory_gb=10,
        description="Qwen2.5 14B 指令模型，更强的推理能力",
        huggingface_id="Qwen/Qwen2.5-14B-Instruct",
        ollama_name="qwen2.5:14b",
        is_free=True,
        languages=["zh", "en"],
        features=["chat", "reasoning", "code", "math"]
    ),

    "qwen2.5-32b-instruct": ModelConfig(
        model_id="qwen2.5-32b-instruct",
        name="Qwen2.5 32B Instruct",
        model_type=ModelType.LLM,
        size=ModelSize.XLARGE,
        params="32B",
        memory_gb=20,
        description="Qwen2.5 32B 指令模型，顶级性能",
        huggingface_id="Qwen/Qwen2.5-32B-Instruct",
        ollama_name="qwen2.5:32b",
        is_free=True,
        languages=["zh", "en"],
        features=["chat", "reasoning", "code", "math", "agentic"]
    ),

    "qwen2.5-72b-instruct": ModelConfig(
        model_id="qwen2.5-72b-instruct",
        name="Qwen2.5 72B Instruct",
        model_type=ModelType.LLM,
        size=ModelSize.XLARGE,
        params="72B",
        memory_gb=45,
        description="Qwen2.5 72B 指令模型，最强推理（需要量化）",
        huggingface_id="Qwen/Qwen2.5-72B-Instruct",
        ollama_name="qwen2.5:72b",
        is_free=True,
        languages=["zh", "en"],
        features=["chat", "reasoning", "code", "math", "agentic"]
    ),

    "qwen2.5-0.5b-instruct": ModelConfig(
        model_id="qwen2.5-0.5b-instruct",
        name="Qwen2.5 0.5B Instruct",
        model_type=ModelType.LLM,
        size=ModelSize.SMALL,
        params="0.5B",
        memory_gb=1,
        description="Qwen2.5 0.5B 超轻量级模型",
        huggingface_id="Qwen/Qwen2.5-0.5B-Instruct",
        ollama_name="qwen2.5:0.5b",
        is_free=True,
        languages=["zh", "en"],
        features=["chat", "basic"]
    ),

    "qwen2.5-1.5b-instruct": ModelConfig(
        model_id="qwen2.5-1.5b-instruct",
        name="Qwen2.5 1.5B Instruct",
        model_type=ModelType.LLM,
        size=ModelSize.SMALL,
        params="1.5B",
        memory_gb=2,
        description="Qwen2.5 1.5B 轻量级模型",
        huggingface_id="Qwen/Qwen2.5-1.5B-Instruct",
        ollama_name="qwen2.5:1.5b",
        is_free=True,
        languages=["zh", "en"],
        features=["chat", "basic", "reasoning"]
    ),

    "qwen2.5-3b-instruct": ModelConfig(
        model_id="qwen2.5-3b-instruct",
        name="Qwen2.5 3B Instruct",
        model_type=ModelType.LLM,
        size=ModelSize.SMALL,
        params="3B",
        memory_gb=3,
        description="Qwen2.5 3B 平衡性能和速度",
        huggingface_id="Qwen/Qwen2.5-3B-Instruct",
        ollama_name="qwen2.5:3b",
        is_free=True,
        languages=["zh", "en"],
        features=["chat", "reasoning", "code"]
    ),

    "qwen2.5-coder-7b-instruct": ModelConfig(
        model_id="qwen2.5-coder-7b-instruct",
        name="Qwen2.5 Coder 7B Instruct",
        model_type=ModelType.LLM,
        size=ModelSize.MEDIUM,
        params="7B",
        memory_gb=5,
        description="Qwen2.5 代码专用模型",
        huggingface_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        ollama_name="qwen2.5-coder:7b",
        is_free=True,
        languages=["zh", "en"],
        features=["code", "reasoning"]
    ),

    # ============ TTS 模型 ============
    "qwen3-tts-1.7b": ModelConfig(
        model_id="qwen3-tts-1.7b",
        name="Qwen3-TTS 1.7B",
        model_type=ModelType.TTS,
        size=ModelSize.MEDIUM,
        params="1.7B",
        memory_gb=4,
        description="Qwen3-Audio TTS 模型，支持语音克隆",
        huggingface_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        is_free=True,
        languages=["zh", "en"],
        features=["tts", "voice-cloning", "emotion"]
    ),

    "edge-tts": ModelConfig(
        model_id="edge-tts",
        name="Edge TTS (Free)",
        model_type=ModelType.TTS,
        size=ModelSize.SMALL,
        params="N/A",
        memory_gb=0,
        description="微软 Edge 免费在线 TTS",
        is_free=True,
        languages=["zh", "en", "ja", "ko"],
        features=["tts", "online", "multi-voice"]
    ),

    # ============ Embedding 模型 ============
    "bge-m3": ModelConfig(
        model_id="bge-m3",
        name="BGE-M3",
        model_type=ModelType.EMBEDDING,
        size=ModelSize.SMALL,
        params="N/A",
        memory_gb=2,
        description="BAAI BGE-M3 多语言embedding",
        huggingface_id="BAAI/bge-m3",
        is_free=True,
        languages=["zh", "en", "100+"],
        features=["embedding", "multilingual", "dense", "sparse", "colbert"]
    ),

    "bge-large-zh": ModelConfig(
        model_id="bge-large-zh",
        name="BGE-Large-ZH",
        model_type=ModelType.EMBEDDING,
        size=ModelSize.SMALL,
        params="N/A",
        memory_gb=1,
        description="BAAI BGE-Large 中文embedding",
        huggingface_id="BAAI/bge-large-zh-v1.5",
        is_free=True,
        languages=["zh"],
        features=["embedding", "chinese"]
    ),

    "paraphrase-multilingual": ModelConfig(
        model_id="paraphrase-multilingual",
        name="Paraphrase Multilingual MiniLM",
        model_type=ModelType.EMBEDDING,
        size=ModelSize.SMALL,
        params="N/A",
        memory_gb=0.5,
        description="多语言 paraphrase 模型（轻量）",
        huggingface_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        is_free=True,
        languages=["zh", "en", "50+"],
        features=["embedding", "multilingual", "lightweight"]
    ),

    "gte-qwen2-7b-instruct": ModelConfig(
        model_id="gte-qwen2-7b-instruct",
        name="GTE-Qwen2-7B-Instruct",
        model_type=ModelType.EMBEDDING,
        size=ModelSize.MEDIUM,
        params="7B",
        memory_gb=5,
        description="Alibaba GTE Qwen2 embedding 模型",
        huggingface_id="Alibaba-NLP/gte-Qwen2-7B-instruct",
        is_free=True,
        languages=["zh", "en"],
        features=["embedding", "high-quality"]
    ),
}


class ModelManager:
    """
    统一模型管理器
    支持模型下载、加载、切换和测试
    """

    def __init__(self, base_dir: Path = None):
        """
        Initialize model manager.

        Args:
            base_dir: Base directory for model storage
        """
        self.base_dir = base_dir or Path("./models")
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories for different model types
        self.llm_dir = self.base_dir / "llm"
        self.tts_dir = self.base_dir / "tts"
        self.embedding_dir = self.base_dir / "embedding"

        for dir_path in [self.llm_dir, self.tts_dir, self.embedding_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Track loaded models
        self.loaded_models: Dict[str, Any] = {}

        logger.info(f"ModelManager initialized with base_dir: {self.base_dir}")

    def get_available_models(
        self,
        model_type: Optional[ModelType] = None,
        max_memory_gb: Optional[int] = None,
    ) -> List[ModelConfig]:
        """
        Get available models filtered by type and memory.

        Args:
            model_type: Filter by model type
            max_memory_gb: Maximum memory in GB

        Returns:
            List of available model configs
        """
        models = list(AVAILABLE_MODELS.values())

        if model_type:
            models = [m for m in models if m.model_type == model_type]

        if max_memory_gb:
            models = [m for m in models if m.memory_gb <= max_memory_gb]

        return sorted(models, key=lambda m: m.memory_gb)

    def get_model_info(self, model_id: str) -> Optional[ModelConfig]:
        """Get model configuration by ID."""
        return AVAILABLE_MODELS.get(model_id)

    def get_recommended_models(self, memory_gb: int = 32) -> Dict[str, str]:
        """
        Get recommended models based on available memory.

        Args:
            memory_gb: Available memory in GB

        Returns:
            Dictionary with recommendations for each model type
        """
        if memory_gb >= 32:
            return {
                "llm": "qwen2.5-14b-instruct",
                "tts": "qwen3-tts-1.7b",
                "embedding": "gte-qwen2-7b-instruct",
            }
        elif memory_gb >= 16:
            return {
                "llm": "qwen2.5-7b-instruct",
                "tts": "qwen3-tts-1.7b",
                "embedding": "bge-m3",
            }
        else:
            return {
                "llm": "qwen2.5-3b-instruct",
                "tts": "edge-tts",
                "embedding": "paraphrase-multilingual",
            }

    async def check_ollama_installed(self) -> bool:
        """Check if Ollama is installed and running."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get("http://localhost:11434/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    async def get_ollama_models(self) -> List[str]:
        """Get list of models installed in Ollama."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to get Ollama models: {e}")
        return []

    async def pull_ollama_model(self, model_name: str) -> bool:
        """
        Pull a model from Ollama registry.

        Args:
            model_name: Ollama model name (e.g., "qwen2.5:7b")

        Returns:
            True if successful
        """
        try:
            import subprocess
            logger.info(f"Pulling Ollama model: {model_name}")
            process = await asyncio.create_subprocess_exec(
                "ollama", "pull", model_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                logger.info(f"Successfully pulled {model_name}")
                return True
            else:
                logger.error(f"Failed to pull {model_name}: {stderr.decode()}")
                return False
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False

    async def test_llm_model(self, model_id: str) -> Dict[str, Any]:
        """
        Test an LLM model.

        Args:
            model_id: Model identifier

        Returns:
            Test results
        """
        model_config = self.get_model_info(model_id)
        if not model_config or model_config.model_type != ModelType.LLM:
            return {"success": False, "error": "Invalid LLM model"}

        results = {
            "model_id": model_id,
            "model_name": model_config.name,
            "success": False,
            "response_time_ms": 0,
            "response": None,
            "error": None
        }

        try:
            import httpx
            import time

            test_prompt = "请用一句话介绍你自己。"

            start_time = time.time()
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": model_config.ollama_name,
                        "prompt": test_prompt,
                        "stream": False
                    }
                )
                response_time = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    data = response.json()
                    results.update({
                        "success": True,
                        "response_time_ms": round(response_time, 2),
                        "response": data.get("response", ""),
                    })
                else:
                    results["error"] = f"HTTP {response.status_code}"
        except Exception as e:
            results["error"] = str(e)

        return results

    async def test_embedding_model(self, model_id: str) -> Dict[str, Any]:
        """
        Test an embedding model.

        Args:
            model_id: Model identifier

        Returns:
            Test results
        """
        model_config = self.get_model_info(model_id)
        if not model_config or model_config.model_type != ModelType.EMBEDDING:
            return {"success": False, "error": "Invalid embedding model"}

        results = {
            "model_id": model_id,
            "model_name": model_config.name,
            "success": False,
            "embedding_dim": 0,
            "response_time_ms": 0,
            "error": None
        }

        try:
            from sentence_transformers import SentenceTransformer
            import time

            start_time = time.time()
            model = SentenceTransformer(model_config.huggingface_id)
            load_time = (time.time() - start_time) * 1000

            # Test embedding
            test_text = "这是一个测试文本。"
            start_time = time.time()
            embedding = model.encode(test_text)
            encode_time = (time.time() - start_time) * 1000

            results.update({
                "success": True,
                "embedding_dim": len(embedding),
                "load_time_ms": round(load_time, 2),
                "encode_time_ms": round(encode_time, 2),
                "total_time_ms": round(load_time + encode_time, 2),
            })

            # Clean up
            del model

        except Exception as e:
            results["error"] = str(e)

        return results

    async def test_tts_model(self, model_id: str) -> Dict[str, Any]:
        """
        Test a TTS model.

        Args:
            model_id: Model identifier

        Returns:
            Test results
        """
        model_config = self.get_model_info(model_id)
        if not model_config or model_config.model_type != ModelType.TTS:
            return {"success": False, "error": "Invalid TTS model"}

        results = {
            "model_id": model_id,
            "model_name": model_config.name,
            "success": False,
            "response_time_ms": 0,
            "audio_length_sec": 0,
            "error": None
        }

        try:
            if model_config.model_id == "edge-tts":
                import time
                import edge_tts

                start_time = time.time()
                communicate = edge_tts.Communicate(text="你好，这是一个测试。", voice="zh-CN-XiaoxiaoNeural")
                await communicate.save("/tmp/test_tts.mp3")
                duration = (time.time() - start_time) * 1000

                results.update({
                    "success": True,
                    "response_time_ms": round(duration, 2),
                })
            else:
                # Qwen3-TTS
                from app.services.qwen_tts_service import QwenTTSService
                import time

                service = QwenTTSService()
                await service.initialize()

                start_time = time.time()
                audio = await service.synthesize("你好，这是一个测试。")
                duration = (time.time() - start_time) * 1000

                results.update({
                    "success": True,
                    "response_time_ms": round(duration, 2),
                    "audio_length_sec": round(len(audio) / 24000, 2) if audio is not None else 0,
                })

        except Exception as e:
            results["error"] = str(e)

        return results

    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """
        Run comprehensive tests on all recommended models.

        Returns:
            Comprehensive test report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "platform": os.uname().machine,
                "memory_gb": 32,  # Could be detected
            },
            "ollama_installed": await self.check_ollama_installed(),
            "ollama_models": await self.get_ollama_models(),
            "tests": {}
        }

        # Test recommended models
        recommended = self.get_recommended_models(32)

        # Test LLM
        llm_id = recommended["llm"]
        logger.info(f"Testing LLM: {llm_id}")
        report["tests"]["llm"] = await self.test_llm_model(llm_id)

        # Test embedding
        emb_id = recommended["embedding"]
        logger.info(f"Testing Embedding: {emb_id}")
        report["tests"]["embedding"] = await self.test_embedding_model(emb_id)

        # Test TTS
        tts_id = recommended["tts"]
        logger.info(f"Testing TTS: {tts_id}")
        report["tests"]["tts"] = await self.test_tts_model(tts_id)

        return report

    def print_model_summary(self):
        """Print a summary of all available models."""
        print("\n" + "=" * 80)
        print("可用模型列表".center(80))
        print("=" * 80)

        for model_type in [ModelType.LLM, ModelType.TTS, ModelType.EMBEDDING]:
            models = [m for m in AVAILABLE_MODELS.values() if m.model_type == model_type]
            if not models:
                continue

            print(f"\n【{model_type.value.upper()} 模型】")
            print("-" * 80)

            for m in sorted(models, key=lambda x: x.memory_gb):
                free_tag = "✓ 免费" if m.is_free else "付费"
                feature_tag = ", ".join(m.features[:3])
                print(f"  • {m.name:40s} | {m.params:6s} | {m.memory_gb:2d}GB | {free_tag} | {feature_tag}")

        print("\n" + "=" * 80)


# Singleton instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get or create model manager singleton."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager

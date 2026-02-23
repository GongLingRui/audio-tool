"""
CosyVoice 0.5B TTS Engine Service
Supports CosyVoice2-0.5B and CosyVoice3-0.5B models
"""
import asyncio
import io
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
try:
    # NOTE: pydub depends on `audioop` (removed in Python 3.13). Import lazily and degrade gracefully.
    from pydub import AudioSegment  # type: ignore

    _PYDUB_AVAILABLE = True
    _PYDUB_IMPORT_ERROR: Exception | None = None
except Exception as e:  # pragma: no cover - environment dependent
    AudioSegment = Any  # type: ignore
    _PYDUB_AVAILABLE = False
    _PYDUB_IMPORT_ERROR = e

from app.config import settings
from app.core.exceptions import TTSError

logger = logging.getLogger(__name__)


class CosyVoiceEngine:
    """
    CosyVoice 0.5B TTS Engine

    Supports:
    - CosyVoice2-0.5B (2024)
    - CosyVoice3-0.5B-2512 (2025 latest)
    - Multi-lingual support (Chinese, English, Japanese, Cantonese, etc.)
    - Zero-shot voice cloning
    - Cross-lingual voice cloning
    - Instruction-based TTS
    - Streaming mode with ~150ms latency
    """

    # Available CosyVoice models
    MODELS = {
        "CosyVoice2-0.5B": {
            "model_id": "FunAudioLLM/CosyVoice2-0.5B",
            "huggingface": "https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B",
            "modelscope": "https://modelscope.cn/models/iic/CosyVoice2-0.5B",
            "features": ["tts", "voice_clone", "zero_shot", "instruction"],
            "languages": ["zh", "en", "ja", "yue", "ko"],
        },
        "CosyVoice3-0.5B-2512": {
            "model_id": "FunAudioLLM/CosyVoice3-0.5B-2512",
            "huggingface": "https://huggingface.co/FunAudioLLM/CosyVoice3-0.5B-2512",
            "modelscope": "https://modelscope.cn/models/iic/CosyVoice3-0.5B-2512",
            "features": ["tts", "voice_clone", "zero_shot", "instruction", "rl_optimized"],
            "languages": ["zh", "en", "ja", "yue", "ko", "de", "fr", "es"],
            "latency_ms": 150,
        },
    }

    def __init__(
        self,
        model_name: str = "CosyVoice3-0.5B-2512",
        device: str = "auto",
        load_model: bool = False,
        external_url: Optional[str] = None,
    ):
        """
        Initialize CosyVoice engine.

        Args:
            model_name: Model name (CosyVoice2-0.5B or CosyVoice3-0.5B-2512)
            device: Device to use (auto, cuda, cpu)
            load_model: Whether to load model locally (requires cosyvoice package)
            external_url: External CosyVoice server URL
        """
        self.model_name = model_name
        self.device = device
        self.external_url = external_url or getattr(settings, "COSYVOICE_URL", None)
        self.model = None
        self.model_loaded = False

        if model_name not in self.MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {list(self.MODELS.keys())}"
            )

        self.model_info = self.MODELS[model_name]

        if load_model:
            self._load_local_model()

    def _load_local_model(self):
        """Load CosyVoice model locally."""
        try:
            from cosyvoice import CosyVoice as CosyVoiceModel
            from cosyvoice.utils.file_utils import load_wav

            logger.info(f"Loading CosyVoice model: {self.model_name}")

            # Get model path from model ID
            model_path = self.model_info["model_id"]

            # Initialize model
            self.model = CosyVoiceModel(
                model_path,
                load_jit=True,
                load_onnx=False,
            )

            self.model_loaded = True
            logger.info(f"CosyVoice model loaded successfully: {self.model_name}")

        except ImportError:
            logger.warning(
                "cosyvoice package not installed. "
                "Install with: pip install cosyvoice"
            )
        except Exception as e:
            logger.error(f"Failed to load CosyVoice model: {e}")

    async def generate(
        self,
        text: str,
        speaker: str = "zh-cn-female-1",
        reference_audio: Optional[str] = None,
        instruction: Optional[str] = None,
        streaming: bool = False,
        language: str = "auto",
        speed: float = 1.0,
        temperature: float = 0.7,
    ) -> Tuple[bytes, float]:
        """
        Generate speech using CosyVoice.

        Args:
            text: Input text
            speaker: Speaker ID or name
            reference_audio: Reference audio path for voice cloning
            instruction: Style instruction (e.g., "happy", "sad", "angry")
            streaming: Enable streaming mode
            language: Language code (auto, zh, en, ja, yue, ko)
            speed: Speech speed (0.5 - 2.0)
            temperature: Sampling temperature (0.1 - 1.0)

        Returns:
            Tuple of (audio_bytes, duration_seconds)
        """
        if self.model_loaded and self.model:
            return await self._generate_local(
                text=text,
                speaker=speaker,
                reference_audio=reference_audio,
                instruction=instruction,
                streaming=streaming,
                language=language,
                speed=speed,
                temperature=temperature,
            )
        elif self.external_url:
            return await self._generate_external(
                text=text,
                speaker=speaker,
                reference_audio=reference_audio,
                instruction=instruction,
                streaming=streaming,
                language=language,
                speed=speed,
                temperature=temperature,
            )
        else:
            raise TTSError(
                "CosyVoice model not loaded and no external URL configured. "
                "Install cosyvoice package or configure COSYVOICE_URL."
            )

    async def _generate_local(
        self,
        text: str,
        speaker: str,
        reference_audio: Optional[str],
        instruction: Optional[str],
        streaming: bool,
        language: str,
        speed: float,
        temperature: float,
    ) -> Tuple[bytes, float]:
        """Generate speech locally using CosyVoice model."""
        try:
            from cosyvoice.utils.file_utils import load_wav

            # Detect language if auto
            if language == "auto":
                language = self._detect_language(text)

            # Prepare generation parameters
            kwargs = {
                "text": text,
                "speed": speed,
            }

            # Add voice cloning if reference audio provided
            if reference_audio and os.path.exists(reference_audio):
                # Load reference audio
                prompt_text = None  # Use default prompt
                prompt_speech_16k = load_wav(reference_audio)

                if instruction:
                    # Instruction-based TTS with reference
                    output = self.model.inference_sft(
                        text=text,
                        instruction=instruction,
                        prompt_text=prompt_text,
                        prompt_speech_16k=prompt_speech_16k,
                        **kwargs
                    )
                else:
                    # Zero-shot voice cloning
                    output = self.model.inference_zero_shot(
                        text=text,
                        prompt_text=prompt_text,
                        prompt_speech_16k=prompt_speech_16k,
                        **kwargs
                    )
            elif instruction:
                # Instruction-based TTS
                output = self.model.inference_sft(
                    text=text,
                    instruction=instruction,
                    **kwargs
                )
            else:
                # Standard TTS
                output = self.model.inference_speaker(
                    text=text,
                    speaker=speaker,
                    **kwargs
                )

            # Convert output to audio bytes
            audio_array = output["audio"]
            sample_rate = output.get("sample_rate", 22050)

            # Convert to WAV bytes
            audio_bytes = self._array_to_wav(audio_array, sample_rate)
            duration = len(audio_array) / sample_rate

            return audio_bytes, duration

        except Exception as e:
            logger.error(f"CosyVoice local generation failed: {e}")
            raise TTSError(f"CosyVoice generation failed: {str(e)}")

    async def _generate_external(
        self,
        text: str,
        speaker: str,
        reference_audio: Optional[str],
        instruction: Optional[str],
        streaming: bool,
        language: str,
        speed: float,
        temperature: float,
    ) -> Tuple[bytes, float]:
        """Generate speech using external CosyVoice server."""
        payload = {
            "text": text,
            "speaker": speaker,
            "model": self.model_name,
            "speed": speed,
            "temperature": temperature,
            "streaming": streaming,
        }

        if reference_audio:
            payload["reference_audio"] = reference_audio
        if instruction:
            payload["instruction"] = instruction
        if language != "auto":
            payload["language"] = language

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    f"{self.external_url}/api/generate",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                # Handle response
                if "audio" in data:
                    # Base64 or hex encoded audio
                    if data.get("encoding") == "base64":
                        import base64
                        audio_bytes = base64.b64decode(data["audio"])
                    else:
                        audio_bytes = bytes.fromhex(data["audio"])
                elif "audio_path" in data:
                    with open(data["audio_path"], "rb") as f:
                        audio_bytes = f.read()
                else:
                    raise TTSError("Invalid response format")

                duration = data.get("duration", 0)
                if duration == 0:
                    duration = self._estimate_duration(audio_bytes)

                return audio_bytes, duration

        except httpx.HTTPError as e:
            raise TTSError(f"CosyVoice server request failed: {e}")
        except Exception as e:
            raise TTSError(f"CosyVoice generation failed: {e}")

    async def clone_voice(
        self,
        text: str,
        reference_audio_path: str,
        language: str = "auto",
        instruction: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[bytes, float]:
        """
        Clone voice from reference audio.

        Args:
            text: Text to synthesize
            reference_audio_path: Path to reference audio
            language: Language code
            instruction: Optional style instruction
            speed: Speech speed

        Returns:
            Tuple of (audio_bytes, duration)
        """
        return await self.generate(
            text=text,
            speaker="clone",
            reference_audio=reference_audio_path,
            instruction=instruction,
            language=language,
            speed=speed,
        )

    async def generate_streaming(
        self,
        text: str,
        speaker: str = "zh-cn-female-1",
        reference_audio: Optional[str] = None,
        language: str = "auto",
    ):
        """
        Generate streaming audio with low latency.

        Yields audio chunks as they are generated.
        """
        # For external server, handle streaming
        if self.external_url:
            payload = {
                "text": text,
                "speaker": speaker,
                "model": self.model_name,
                "streaming": True,
            }

            if reference_audio:
                payload["reference_audio"] = reference_audio
            if language != "auto":
                payload["language"] = language

            async with httpx.AsyncClient(timeout=300) as client:
                async with client.stream(
                    "POST",
                    f"{self.external_url}/api/generate",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        yield chunk

    def get_available_speakers(self) -> List[Dict[str, Any]]:
        """Get available pre-built speakers."""
        return [
            {
                "id": "zh-cn-female-1",
                "name": "中文女声1",
                "gender": "female",
                "language": "zh",
                "description": "标准中文女声",
            },
            {
                "id": "zh-cn-male-1",
                "name": "中文男声1",
                "gender": "male",
                "language": "zh",
                "description": "标准中文男声",
            },
            {
                "id": "en-us-female-1",
                "name": "English Female 1",
                "gender": "female",
                "language": "en",
                "description": "Standard American English female voice",
            },
            {
                "id": "en-us-male-1",
                "name": "English Male 1",
                "gender": "male",
                "language": "en",
                "description": "Standard American English male voice",
            },
            {
                "id": "ja-jp-female-1",
                "name": "日本語女性1",
                "gender": "female",
                "language": "ja",
                "description": "標準日本語女性",
            },
        ]

    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get supported languages."""
        return [
            {"code": "zh", "name": "中文", "native": "普通话"},
            {"code": "en", "name": "English", "native": "English"},
            {"code": "ja", "name": "Japanese", "native": "日本語"},
            {"code": "yue", "name": "Cantonese", "native": "粤语"},
            {"code": "ko", "name": "Korean", "native": "한국어"},
            {"code": "de", "name": "German", "native": "Deutsch"},
            {"code": "fr", "name": "French", "native": "Français"},
            {"code": "es", "name": "Spanish", "native": "Español"},
        ]

    def get_style_instructions(self) -> Dict[str, List[str]]:
        """Get available style instructions."""
        return {
            "emotion": [
                "happy",
                "sad",
                "angry",
                "fearful",
                "disgusted",
                "surprised",
            ],
            "speaking_style": [
                "whispering",
                "shouting",
                "fast",
                "slow",
                "cheerful",
                "serious",
            ],
            "tone": [
                "gentle",
                "firm",
                "warm",
                "cold",
                "energetic",
                "calm",
            ],
        }

    def _detect_language(self, text: str) -> str:
        """Detect language from text."""
        # Check for Chinese characters
        if any('\u4e00' <= c <= '\u9fff' for c in text):
            # Check for Cantonese-specific characters
            cantonese_chars = set('嘅唔系咁佢哋喺咗嚟咩喎')
            if any(c in cantonese_chars for c in text):
                return "yue"
            return "zh"

        # Check for Japanese
        if any('\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' for c in text):
            return "ja"

        # Check for Korean
        if any('\uac00' <= c <= '\ud7af' for c in text):
            return "ko"

        # Default to English
        return "en"

    def _array_to_wav(self, audio_array: np.ndarray, sample_rate: int) -> bytes:
        """Convert numpy array to WAV bytes."""
        import wave

        # Ensure float32 in [-1, 1]
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        # Clip and convert to int16
        audio_array = np.clip(audio_array, -1.0, 1.0)
        audio_int16 = (audio_array * 32767).astype(np.int16)

        # Create WAV file
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())

        return buf.getvalue()

    def _estimate_duration(self, audio_bytes: bytes) -> float:
        """Estimate audio duration from bytes."""
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            return len(audio) / 1000.0
        except Exception:
            # Rough estimate: 22050Hz, 16-bit, mono
            return len(audio_bytes) / 44100.0


# Global instances
_cosy_voice_engines: Dict[str, CosyVoiceEngine] = {}


def get_cosy_voice(
    model_name: str = "CosyVoice3-0.5B-2512",
    force_new: bool = False,
) -> CosyVoiceEngine:
    """
    Get or create CosyVoice engine instance.

    Args:
        model_name: Model name to use
        force_new: Force creating new instance

    Returns:
        CosyVoiceEngine instance
    """
    if force_new or model_name not in _cosy_voice_engines:
        _cosy_voice_engines[model_name] = CosyVoiceEngine(
            model_name=model_name,
            external_url=getattr(settings, "COSYVOICE_URL", None),
        )

    return _cosy_voice_engines[model_name]


async def generate_with_cosy_voice(
    text: str,
    speaker: str = "zh-cn-female-1",
    reference_audio: Optional[str] = None,
    instruction: Optional[str] = None,
    model: str = "CosyVoice3-0.5B-2512",
    speed: float = 1.0,
) -> Tuple[bytes, float]:
    """
    Convenience function to generate speech with CosyVoice.

    Args:
        text: Input text
        speaker: Speaker ID
        reference_audio: Reference audio for cloning
        instruction: Style instruction
        model: Model to use
        speed: Speech speed

    Returns:
        Tuple of (audio_bytes, duration)
    """
    engine = get_cosy_voice(model_name=model)
    return await engine.generate(
        text=text,
        speaker=speaker,
        reference_audio=reference_audio,
        instruction=instruction,
        speed=speed,
    )

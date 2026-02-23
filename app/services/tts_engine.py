"""TTS Engine service for audio generation."""
import io
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

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


class TTSMode(str, Enum):
    """TTS mode options."""
    LOCAL = "local"
    EXTERNAL = "external"
    EDGE = "edge"


class VoiceType(str, Enum):
    """Voice type options."""
    CUSTOM = "custom"
    CLONE = "clone"
    LORA = "lora"
    DESIGN = "design"


class TTSEngine(ABC):
    """Abstract TTS engine."""

    @abstractmethod
    async def generate(
        self,
        text: str,
        speaker: str,
        instruct: str | None = None,
        voice_config: dict | None = None,
    ) -> tuple[bytes, float]:
        """
        Generate audio from text.

        Args:
            text: Text to convert
            speaker: Speaker name
            instruct: TTS instruction
            voice_config: Voice configuration

        Returns:
            Tuple of (audio_data, duration_seconds)
        """
        pass

    @abstractmethod
    async def get_voices(self) -> list[dict]:
        """Get available voices."""
        pass


class ExternalTTSEngine(TTSEngine):
    """External TTS server engine."""

    def __init__(self, config: dict | None = None):
        self.base_url = config.get("tts_url", settings.tts_url) if config else settings.tts_url
        self.timeout = config.get("timeout", settings.tts_timeout) if config else settings.tts_timeout

    async def generate(
        self,
        text: str,
        speaker: str,
        instruct: str | None = None,
        voice_config: dict | None = None,
    ) -> tuple[bytes, float]:
        """Generate audio using external TTS service."""
        voice_type = voice_config.get("voice_type", "custom") if voice_config else "custom"

        payload = {
            "text": text,
            "speaker": speaker,
            "voice_type": voice_type,
        }

        if instruct:
            payload["instruction"] = instruct

        if voice_config:
            if voice_type == "clone":
                payload["ref_audio_path"] = voice_config.get("ref_audio_path")
            elif voice_type == "lora":
                payload["lora_model_path"] = voice_config.get("lora_model_path")
            elif voice_type == "design":
                payload["description"] = voice_config.get("description")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                response.raise_for_status()

                data = response.json()

                # Handle different response formats
                if "audio" in data:
                    audio_hex = data["audio"]
                    audio_bytes = bytes.fromhex(audio_hex)
                elif "audio_path" in data:
                    # Read from file
                    with open(data["audio_path"], "rb") as f:
                        audio_bytes = f.read()
                else:
                    raise TTSError("Invalid TTS response format")

                duration = data.get("duration", 0)

                # If duration not provided, calculate it
                if duration == 0 and audio_bytes:
                    duration = self._estimate_duration(audio_bytes)

                return audio_bytes, duration

        except httpx.HTTPError as e:
            raise TTSError(f"TTS request failed: {str(e)}")
        except Exception as e:
            raise TTSError(f"TTS generation failed: {str(e)}")

    def _estimate_duration(self, audio_bytes: bytes) -> float:
        """Estimate audio duration from bytes."""
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            return len(audio) / 1000.0
        except Exception:
            # Rough estimate: assume 16kHz, 16-bit, mono
            return len(audio_bytes) / 32000.0

    async def get_voices(self) -> list[dict]:
        """Get available voices from external service."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(f"{self.base_url}/api/voices")
                response.raise_for_status()
                return response.json().get("voices", [])
        except Exception:
            # Return default voices
            return [
                {"id": "ryan", "name": "Ryan", "gender": "male", "language": "en-US"},
                {"id": "rachel", "name": "Rachel", "gender": "female", "language": "en-US"},
            ]

    async def generate_batch(
        self,
        items: list[dict],
        workers: int = 2,
    ) -> list[tuple[bytes, float]]:
        """Generate multiple audio files."""
        # For now, sequential generation
        results = []
        for item in items:
            result = await self.generate(
                text=item["text"],
                speaker=item.get("speaker", "NARRATOR"),
                instruct=item.get("instruct"),
                voice_config=item.get("voice_config"),
            )
            results.append(result)
        return results


class LocalTTSEngine(TTSEngine):
    """Local TTS engine using Edge TTS service."""

    def __init__(self, config: dict | None = None):
        self.device = config.get("device", "auto") if config else "auto"
        self.model = None
        # Import Edge TTS engine
        from app.services.edge_tts_engine import create_edge_tts_engine
        self.edge_tts = create_edge_tts_engine(config)

    async def generate(
        self,
        text: str,
        speaker: str,
        instruct: str | None = None,
        voice_config: dict | None = None,
    ) -> tuple[bytes, float]:
        """Generate audio using Edge TTS service."""
        # Default voices for different locales
        default_voices = {
            "zh-CN": "zh-CN-XiaoxiaoNeural",
            "en-US": "en-US-JennyNeural",
        }

        # 选择语音：
        # - 如果传入的是真正的 Edge 声音 ID（包含 locale，例如 "zh-CN-XiaoxiaoNeural"），直接使用；
        # - 如果传入的是逻辑标签（如 "NARRATOR"、"CHARACTER" 等全大写且不含 '-'），视为“未指定”，按文本内容自动选择；
        # - 如果为空，也按文本内容自动选择。
        voice = (speaker or "").strip()

        def _is_edge_voice_id(name: str) -> bool:
            # 简单判断：包含语言前缀和连字符，例如 "zh-CN-xxx" 或 "en-US-xxx"
            return "-" in name and any(name.startswith(prefix) for prefix in ("zh-CN-", "en-US-"))

        use_auto_detect = False
        if not voice:
            use_auto_detect = True
        else:
            # 像 "NARRATOR" 这类全大写、无连字符的标签，强制走自动检测
            if voice.isupper() and "-" not in voice:
                use_auto_detect = True

        if use_auto_detect or not _is_edge_voice_id(voice):
            # Detect language from text（正确判断中文字符）
            has_chinese = any("\u4e00" <= c <= "\u9fff" for c in text)
            voice = default_voices["zh-CN"] if has_chinese else default_voices["en-US"]

        try:
            return await self.edge_tts.generate(
                text=text,
                speaker=voice,
                rate="+0%",
                pitch="+0Hz",
                volume="+0%",
            )
        except Exception as e:
            # Fallback to simple audio if Edge TTS fails
            print(f"Edge TTS failed, using fallback: {e}")
            return self._generate_fallback(text)

    def _generate_fallback(self, text: str) -> tuple[bytes, float]:
        """Generate fallback audio using basic TTS."""
        try:
            # Try using pyttsx3 if available
            import pyttsx3
            engine = pyttsx3.init()

            # Save to bytes
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_path = temp_file.name

            engine.save_to_file(temp_path)
            engine.runAndWait()

            with open(temp_path, "rb") as f:
                audio_data = f.read()

            import os
            os.unlink(temp_path)

            duration = len(text) * 0.15  # Rough estimate
            return audio_data, max(0.5, duration)

        except ImportError:
            # Final fallback - generate WAV with tone sequence representing speech
            sample_rate = 22050
            duration = max(1.0, len(text) * 0.08)
            samples = int(sample_rate * duration)

            # Generate a more interesting sound pattern (simulated speech)
            audio = np.zeros(samples, dtype=np.float32)

            # Create tone patterns based on text
            for i, char in enumerate(text[:100]):  # Limit to first 100 chars
                char_freq = 200 + (ord(char) % 200)  # Frequency based on character
                start_sample = int(i * samples / len(text))
                end_sample = min(start_sample + int(samples / len(text)), samples)

                if start_sample < samples:
                    t = np.linspace(start_sample / sample_rate, end_sample / sample_rate,
                                    end_sample - start_sample)
                    # Create a brief tone with decay
                    tone = np.sin(2 * np.pi * char_freq * t) * np.exp(-5 * (t - t[0]))
                    audio[start_sample:end_sample] += tone * 0.3

            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.8

            return self._to_wav(audio, sample_rate), duration

    def _to_wav(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Convert audio array to WAV bytes."""
        import wave

        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            audio_int16 = (audio * 32767).astype(np.int16)
            wf.writeframes(audio_int16.tobytes())
        return buf.getvalue()

    async def get_voices(self) -> list[dict]:
        """Get available local voices."""
        return [
            {"id": "aiden", "name": "Aiden", "gender": "male", "language": "en-US"},
            {"id": "rachel", "name": "Rachel", "gender": "female", "language": "en-US"},
        ]


class TTSEngineFactory:
    """Factory for creating TTS engines."""

    @staticmethod
    def create(mode: TTSMode | str, config: dict | None = None) -> TTSEngine:
        """Create TTS engine based on mode."""
        if isinstance(mode, str):
            mode = TTSMode(mode.lower())

        if mode == TTSMode.EXTERNAL:
            return ExternalTTSEngine(config)
        elif mode == TTSMode.LOCAL:
            return LocalTTSEngine(config)
        else:
            raise ValueError(f"Unsupported TTS mode: {mode}")

"""
ASR (Automatic Speech Recognition) Service
Advanced speech-to-text with multiple backend support
"""
import asyncio
import logging
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import importlib.util

logger = logging.getLogger(__name__)


class ASRBackend(Enum):
    """Available ASR backends."""
    WHISPER = "whisper"  # OpenAI Whisper (original)
    FASTER_WHISPER = "faster_whisper"  # Faster Whisper implementation
    GROQ = "groq"  # Groq Whisper API
    AZURE = "azure"  # Azure Speech Service
    GOOGLE = "google"  # Google Cloud Speech-to-Text
    BASIC = "basic"  # Fallback basic implementation


@dataclass
class ASRResult:
    """Result of speech recognition."""
    text: str  # Transcribed text
    language: str  # Detected language code
    confidence: float  # Overall confidence (0-1)
    duration: float  # Audio duration in seconds
    processing_time: float  # Time taken to process
    backend: ASRBackend  # Backend used
    segments: List[Dict[str, Any]]  # Individual segments with timestamps
    word_segments: Optional[List[Dict[str, Any]]] = None  # Word-level timestamps

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "language": self.language,
            "confidence": self.confidence,
            "duration": self.duration,
            "processing_time": self.processing_time,
            "backend": self.backend.value,
            "segments": self.segments,
            "word_segments": self.word_segments,
        }


class ASRService:
    """
    Advanced Speech Recognition Service with multiple backend support.

    Features:
    - OpenAI Whisper (original/transformers)
    - Faster Whisper (CTranslate2 optimized)
    - Groq Whisper API (ultra-fast)
    - Azure Speech Service
    - Google Cloud Speech-to-Text
    - Multi-language support (90+ languages)
    - Word-level timestamps
    - Speaker diarization integration
    - Punctuation restoration
    - Custom vocabulary support
    """

    def __init__(
        self,
        preferred_backend: ASRBackend = ASRBackend.FASTER_WHISPER,
        model_size: str = "base",  # tiny, base, small, medium, large-v3
        compute_type: str = "default",  # int8, float16, default
        device: str = "auto",  # auto, cpu, cuda
        api_key: Optional[str] = None,  # For cloud APIs
        language: Optional[str] = None,  # Default language (auto-detect if None)
    ):
        """
        Initialize ASR service.

        Args:
            preferred_backend: Preferred ASR backend
            model_size: Whisper model size (tiny/base/small/medium/large-v3)
            compute_type: Compute type for optimization
            device: Device to use for inference
            api_key: API key for cloud services
            language: Default language (e.g., "zh", "en", "ja")
        """
        self.preferred_backend = preferred_backend
        self.model_size = model_size
        self.compute_type = compute_type
        self.device = device
        self.api_key = api_key
        self.default_language = language

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            f"Initialized ASR service with backend: {preferred_backend.value}, "
            f"model: {model_size}"
        )

        # Lazy loaded models
        self._whisper_model = None
        self._faster_whisper_model = None

    def _auto_device_for_whisper(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _auto_device_for_faster_whisper(self) -> str:
        # ctranslate2 does not support MPS, so prefer cpu on Apple Silicon.
        if self.device != "auto":
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _get_audio_duration(self, audio_path: str) -> float:
        """Best-effort duration without relying on torchaudio backends."""
        try:
            from app.utils.audio_decode import ffmpeg_available, probe_audio

            if ffmpeg_available():
                info = probe_audio(str(audio_path))
                if info.duration:
                    return float(info.duration)
        except Exception:
            pass

        try:
            from app.utils.wav_audio import read_wav_info

            info = read_wav_info(str(audio_path))
            if info.duration:
                return float(info.duration)
        except Exception:
            pass

        return 0.0

    async def _get_whisper_model(self):
        """Lazy load OpenAI Whisper model."""
        if self._whisper_model is None:
            try:
                import whisper

                self._whisper_model = whisper.load_model(
                    self.model_size,
                    device=self._auto_device_for_whisper(),
                )
                self.logger.info(f"Whisper model loaded: {self.model_size}")

            except ImportError:
                self.logger.warning("Whisper not installed")
                self._whisper_model = False
            except Exception as e:
                self.logger.error(f"Error loading Whisper: {e}")
                self._whisper_model = False

        return self._whisper_model if self._whisper_model is not False else None

    async def _get_faster_whisper_model(self):
        """Lazy load Faster Whisper model."""
        if self._faster_whisper_model is None:
            try:
                from faster_whisper import WhisperModel

                self._faster_whisper_model = WhisperModel(
                    self.model_size,
                    device=self._auto_device_for_faster_whisper(),
                    compute_type=self.compute_type,
                )
                self.logger.info(f"Faster Whisper model loaded: {self.model_size}")

            except ImportError:
                self.logger.warning("Faster Whisper not installed")
                self._faster_whisper_model = False
            except Exception as e:
                self.logger.error(f"Error loading Faster Whisper: {e}")
                self._faster_whisper_model = False

        return self._faster_whisper_model if self._faster_whisper_model is not False else None

    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",  # transcribe or translate
        timestamps: str = "segment",  # segment, word, none
        vad_filter: bool = True,  # Voice Activity Detection
        backend: Optional[ASRBackend] = None,
        **kwargs,
    ) -> ASRResult:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            language: Language code (auto-detect if None)
            task: Task type (transcribe or translate to English)
            timestamps: Timestamp granularity (segment, word, none)
            vad_filter: Apply VAD filter to remove silence
            backend: Force specific backend
            **kwargs: Additional backend-specific parameters

        Returns:
            ASRResult with transcription and metadata
        """
        start_time = time.time()
        backend = backend or self.preferred_backend
        language = language or self.default_language

        try:
            if backend == ASRBackend.WHISPER:
                result = await self._transcribe_whisper(
                    audio_path, language, task, timestamps, vad_filter, **kwargs
                )
            elif backend == ASRBackend.FASTER_WHISPER:
                result = await self._transcribe_faster_whisper(
                    audio_path, language, task, timestamps, vad_filter, **kwargs
                )
            elif backend == ASRBackend.GROQ:
                result = await self._transcribe_groq(
                    audio_path, language, **kwargs
                )
            elif backend == ASRBackend.AZURE:
                result = await self._transcribe_azure(
                    audio_path, language, **kwargs
                )
            elif backend == ASRBackend.GOOGLE:
                result = await self._transcribe_google(
                    audio_path, language, **kwargs
                )
            else:
                result = await self._transcribe_basic(
                    audio_path, language
                )

            processing_time = time.time() - start_time
            result.processing_time = processing_time

            return result

        except Exception as e:
            self.logger.error(f"ASR error with {backend.value}: {e}")
            # Fallback to basic implementation
            self.logger.warning("Falling back to basic ASR")
            return await self._transcribe_basic(audio_path, language)

    async def _transcribe_whisper(
        self,
        audio_path: str,
        language: Optional[str],
        task: str,
        timestamps: str,
        vad_filter: bool,
        **kwargs,
    ) -> ASRResult:
        """Transcribe using OpenAI Whisper."""
        model = await self._get_whisper_model()

        if model is None:
            raise RuntimeError("Whisper not available")

        # Transcribe
        options = {
            "task": task,
            "language": language,
            "word_timestamps": (timestamps == "word"),
            **kwargs,
        }

        result = model.transcribe(str(audio_path), **options)

        duration = self._get_audio_duration(str(audio_path))

        # Process segments
        segments = []
        word_segments = []

        for segment in result.get("segments", []):
            segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip(),
                "confidence": segment.get("avg_logprob", 0.0),
            })

            # Word-level timestamps if available
            if timestamps == "word" and "words" in segment:
                for word in segment["words"]:
                    word_segments.append({
                        "word": word["word"],
                        "start": word["start"],
                        "end": word["end"],
                        "confidence": word.get("probability", 0.0),
                    })

        return ASRResult(
            text=result["text"].strip(),
            language=result.get("language", "unknown"),
            confidence=result.get("avg_logprob", 0.0),
            duration=duration,
            processing_time=0.0,
            backend=ASRBackend.WHISPER,
            segments=segments,
            word_segments=word_segments if timestamps == "word" else None,
        )

    async def _transcribe_faster_whisper(
        self,
        audio_path: str,
        language: Optional[str],
        task: str,
        timestamps: str,
        vad_filter: bool,
        **kwargs,
    ) -> ASRResult:
        """Transcribe using Faster Whisper (optimized)."""
        model = await self._get_faster_whisper_model()

        if model is None:
            raise RuntimeError("Faster Whisper not available")

        # Transcribe
        segments_info, info = model.transcribe(
            str(audio_path),
            language=language,
            task=task,
            word_timestamps=(timestamps == "word"),
            vad_filter=vad_filter,
            **kwargs,
        )

        # Collect results
        segments = []
        word_segments = []
        full_text = []

        for segment in segments_info:
            text = segment.text.strip()
            full_text.append(text)

            segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": text,
                "confidence": segment.avg_logprob,
            })

            # Word-level timestamps
            if timestamps == "word" and hasattr(segment, 'words'):
                for word in segment.words:
                    word_segments.append({
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "confidence": word.probability,
                    })

        detected_language = info.language if hasattr(info, 'language') else language or "unknown"
        duration = info.duration if hasattr(info, 'duration') and info.duration else self._get_audio_duration(str(audio_path))

        return ASRResult(
            text=" ".join(full_text),
            language=detected_language,
            confidence=0.9,  # Faster Whisper doesn't provide overall confidence
            duration=duration,
            processing_time=0.0,
            backend=ASRBackend.FASTER_WHISPER,
            segments=segments,
            word_segments=word_segments if timestamps == "word" else None,
        )

    async def _transcribe_groq(
        self,
        audio_path: str,
        language: Optional[str],
        **kwargs,
    ) -> ASRResult:
        """Transcribe using Groq Whisper API (ultra-fast)."""
        if not self.api_key:
            raise ValueError("API key required for Groq")

        try:
            from groq import Groq

            client = Groq(api_key=self.api_key)

            # Read audio file
            with open(audio_path, "rb") as f:
                audio_data = f.read()

            # Transcribe
            transcription = client.audio.transcriptions.create(
                file=(Path(audio_path).name, audio_data),
                model="whisper-large-v3",
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["word"],
                **kwargs,
            )

            # Process segments
            segments = []
            word_segments = []

            if hasattr(transcription, 'segments'):
                for seg in transcription.segments:
                    segments.append({
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text,
                        "confidence": getattr(seg, 'confidence', 0.0),
                    })

            if hasattr(transcription, 'words'):
                for word in transcription.words:
                    word_segments.append({
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "confidence": getattr(word, 'confidence', 0.0),
                    })

            return ASRResult(
                text=transcription.text,
                language=transcription.language if hasattr(transcription, 'language') else language or "unknown",
                confidence=0.95,  # Groq Whisper is generally accurate
                duration=getattr(transcription, 'duration', 0.0),
                processing_time=0.0,
                backend=ASRBackend.GROQ,
                segments=segments,
                word_segments=word_segments,
            )

        except ImportError:
            raise RuntimeError("Groq SDK not installed: pip install groq")

    async def _transcribe_azure(
        self,
        audio_path: str,
        language: Optional[str],
        **kwargs,
    ) -> ASRResult:
        """Transcribe using Azure Speech Service."""
        if not self.api_key:
            raise ValueError("API key required for Azure Speech")

        try:
            import azure.cognitiveservices.speech as speechsdk

            # Create speech config
            speech_config = speechsdk.SpeechConfig(
                subscription=self.api_key,
                region=kwargs.get("region", "eastus")
            )

            # Configure language
            if language:
                speech_config.speech_recognition_language = language

            # Audio config
            audio_config = speechsdk.audio.AudioConfig(filename=audio_path)

            # Create recognizer
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config
            )

            # Recognize
            result = recognizer.recognize_once()

            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return ASRResult(
                    text=result.text,
                    language=language or "unknown",
                    confidence=0.9,
                    duration=0.0,
                    processing_time=0.0,
                    backend=ASRBackend.AZURE,
                    segments=[{
                        "start": 0,
                        "end": 0,
                        "text": result.text,
                        "confidence": 0.9,
                    }],
                )
            else:
                raise RuntimeError(f"Azure recognition failed: {result.reason}")

        except ImportError:
            raise RuntimeError("Azure Speech SDK not installed: pip install azure-cognitiveservices-speech")

    async def _transcribe_google(
        self,
        audio_path: str,
        language: Optional[str],
        **kwargs,
    ) -> ASRResult:
        """Transcribe using Google Cloud Speech-to-Text."""
        if not self.api_key:
            raise ValueError("API credentials required for Google Speech")

        try:
            from google.cloud import speech

            client = speech.SpeechClient.from_service_account_file(self.api_key)

            # Read audio
            with open(audio_path, "rb") as f:
                audio_data = f.read()

            # Configure
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
                sample_rate_hertz=16000,
                language_code=language or "zh-CN",
                enable_word_time_offsets=True,
                enable_automatic_punctuation=True,
            )

            audio = speech.RecognitionAudio(content=audio_data)

            # Recognize
            response = client.recognize(config=config, audio=audio)

            # Process results
            segments = []
            word_segments = []
            full_text = []

            for result in response.results:
                alternative = result.alternatives[0]
                full_text.append(alternative.transcript)

                segments.append({
                    "start": 0,
                    "end": 0,
                    "text": alternative.transcript,
                    "confidence": alternative.confidence,
                })

                # Word-level timestamps
                for word in alternative.words:
                    word_segments.append({
                        "word": word.word,
                        "start": word.start_time.total_seconds(),
                        "end": word.end_time.total_seconds(),
                    })

            return ASRResult(
                text=" ".join(full_text),
                language=language or "unknown",
                confidence=0.9,
                duration=0.0,
                processing_time=0.0,
                backend=ASRBackend.GOOGLE,
                segments=segments,
                word_segments=word_segments,
            )

        except ImportError:
            raise RuntimeError("Google Cloud Speech not installed: pip install google-cloud-speech")

    async def _transcribe_basic(
        self,
        audio_path: str,
        language: Optional[str],
    ) -> ASRResult:
        """Basic ASR fallback using simple energy-based VAD.

        This fallback intentionally avoids optional deps (pydub/audioop/ffmpeg).
        If system ffmpeg is available, it supports common formats via streaming decode.
        """
        from pathlib import Path

        import numpy as np

        path = Path(audio_path)

        duration = 0.0
        sample_rate = 16000

        # Choose a sample iterator:
        # - if ffmpeg available: decode any format
        # - else: WAV-only via stdlib
        if path.suffix.lower() == ".wav":
            from app.utils.wav_audio import read_wav_info, iter_wav_mono_samples

            info = read_wav_info(str(path))
            duration = info.duration
            sample_rate = info.sample_rate or 16000

            def sample_iter():
                for block in iter_wav_mono_samples(str(path), block_frames=4096):
                    yield block

        else:
            try:
                from app.utils.audio_decode import ffmpeg_available, iter_audio_mono_float32, probe_audio
            except Exception:
                ffmpeg_available = lambda: False  # type: ignore

            if not ffmpeg_available():
                return ASRResult(
                    text="[基础ASR模式] 当前环境未安装可解码 MP3/FLAC/M4A 的依赖（ffmpeg）。请先安装 ffmpeg 或将音频转换为 WAV。",
                    language=language or "unknown",
                    confidence=0.0,
                    duration=0.0,
                    processing_time=0.0,
                    backend=ASRBackend.BASIC,
                    segments=[],
                )

            try:
                info = probe_audio(str(path))
                duration = info.duration
            except Exception:
                duration = 0.0

            def sample_iter():
                for block in iter_audio_mono_float32(str(path), sample_rate=sample_rate, chunk_samples=8192):
                    yield block

        # Simple VAD: compute RMS over ~20ms windows and threshold.
        window_ms = 20
        hop_ms = 10
        window = max(1, int(sample_rate * (window_ms / 1000.0)))
        hop = max(1, int(sample_rate * (hop_ms / 1000.0)))

        # Collect a downsampled energy contour.
        energies: list[float] = []
        samples_total = 0
        buffer = np.zeros((0,), dtype=np.float32)
        for block in sample_iter():
            if block.size == 0:
                continue
            buffer = np.concatenate([buffer, block])
            while buffer.size >= window:
                frame = buffer[:window]
                buffer = buffer[hop:]
                rms = float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0
                energies.append(rms)
            samples_total += int(block.size)

        if not energies:
            return ASRResult(
                text="[基础ASR模式 - 未检测到语音活动]",
                language=language or "unknown",
                confidence=0.0,
                duration=duration,
                processing_time=0.0,
                backend=ASRBackend.BASIC,
                segments=[],
            )

        # Threshold: median * 2 (with floor), tuned for speech.
        median = float(np.median(np.array(energies, dtype=np.float32)))
        threshold = max(0.01, median * 2.0)

        segments: list[dict[str, Any]] = []
        in_speech = False
        start_t = 0.0

        def t_at(i: int) -> float:
            return i * (hop_ms / 1000.0)

        min_speech_sec = 0.3
        for i, e in enumerate(energies):
            if e >= threshold and not in_speech:
                in_speech = True
                start_t = t_at(i)
            elif e < threshold and in_speech:
                in_speech = False
                end_t = t_at(i)
                if end_t - start_t >= min_speech_sec:
                    segments.append(
                        {
                            "start": round(start_t, 3),
                            "end": round(end_t, 3),
                            "text": "[语音片段]",
                            "confidence": 0.3,
                        }
                    )

        if in_speech:
            end_t = t_at(len(energies))
            if end_t - start_t >= min_speech_sec:
                segments.append(
                    {
                        "start": round(start_t, 3),
                        "end": round(end_t, 3),
                        "text": "[语音片段]",
                        "confidence": 0.3,
                    }
                )

        text = "[基础ASR模式 - 仅检测到语音活动，无法转录文本]"
        if not segments:
            text = "[基础ASR模式 - 未检测到语音活动]"

        return ASRResult(
            text=text,
            language=language or "unknown",
            confidence=0.3 if segments else 0.0,
            duration=duration,
            processing_time=0.0,
            backend=ASRBackend.BASIC,
            segments=segments,
        )

    async def transcribe_batch(
        self,
        audio_paths: List[str],
        **kwargs,
    ) -> List[ASRResult]:
        """
        Transcribe multiple audio files in batch.

        Args:
            audio_paths: List of audio file paths
            **kwargs: Additional parameters for transcribe()

        Returns:
            List of ASRResult
        """
        tasks = [
            self.transcribe(path, **kwargs)
            for path in audio_paths
        ]
        return await asyncio.gather(*tasks)

    async def get_supported_languages(self) -> Dict[str, str]:
        """
        Get list of supported languages.

        Returns:
            Dictionary mapping language codes to names
        """
        return {
            "zh": "Chinese (Mandarin)",
            "zh-CN": "Chinese (Simplified)",
            "zh-TW": "Chinese (Traditional)",
            "yue": "Chinese (Cantonese)",
            "en": "English",
            "ja": "Japanese",
            "ko": "Korean",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ar": "Arabic",
            "hi": "Hindi",
            "th": "Thai",
            "vi": "Vietnamese",
            "ms": "Malay",
            "id": "Indonesian",
            "tr": "Turkish",
            "pl": "Polish",
            "nl": "Dutch",
            "sv": "Swedish",
            "no": "Norwegian",
            "da": "Danish",
            "fi": "Finnish",
        }

    def get_supported_backends(self) -> List[Dict[str, Any]]:
        """Get list of supported backends with availability info."""
        has_faster = importlib.util.find_spec("faster_whisper") is not None
        has_whisper = importlib.util.find_spec("whisper") is not None
        backends = [
            {
                "name": "faster_whisper",
                "display_name": "Faster Whisper",
                "description": "优化版Whisper，速度快5-8倍",
                "requires": "pip install faster-whisper",
                "available": bool(has_faster),
                "recommended": True,
            },
            {
                "name": "whisper",
                "display_name": "Whisper (OpenAI)",
                "description": "OpenAI原始Whisper模型",
                "requires": "pip install openai-whisper",
                "available": bool(has_whisper),
                "recommended": True,
            },
            {
                "name": "groq",
                "display_name": "Groq Whisper API",
                "description": "云端API，超快速推理",
                "requires": "pip install groq + API key",
                "available": bool(self.api_key),
                "recommended": True,
            },
            {
                "name": "azure",
                "display_name": "Azure Speech Service",
                "description": "微软Azure语音服务",
                "requires": "pip install azure-cognitiveservices-speech + API key",
                "available": False,
                "recommended": False,
            },
            {
                "name": "google",
                "display_name": "Google Cloud Speech-to-Text",
                "description": "谷歌云语音转文字",
                "requires": "pip install google-cloud-speech + credentials",
                "available": False,
                "recommended": False,
            },
            {
                "name": "basic",
                "display_name": "基础模式",
                "description": "仅语音活动检测，不转录文本",
                "requires": "无",
                "available": True,
                "recommended": False,
            },
        ]

        return backends

    async def get_installation_instructions(self, backend: str) -> str:
        """Get installation instructions for a backend."""
        instructions = {
            "faster_whisper": """
# 安装 Faster Whisper（推荐）
pip install faster-whisper

# 如果需要 GPU 加速
pip install faster-whisper[cuda]

# 模型会自动下载到 ~/.cache/huggingface/
""",
            "whisper": """
# 安装 OpenAI Whisper
pip install openai-whisper

# 安装依赖
pip install torch torchvision torchaudio

# 模型会自动下载
""",
            "groq": """
# 安装 Groq SDK
pip install groq

# 获取 API Key: https://console.groq.com/
# 设置环境变量
export GROQ_API_KEY="your_api_key_here"
""",
            "azure": """
# 安装 Azure Speech SDK
pip install azure-cognitiveservices-speech

# 需要配置 Azure Speech Service 资源
# 获取 API Key 和 Region
""",
            "google": """
# 安装 Google Cloud Speech
pip install google-cloud-speech

# 配置认证
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
""",
            "basic": """
# 基础模式无需安装
# 仅提供语音活动检测功能
""",
        }

        return instructions.get(backend, "Unknown backend")


# Global instance
_asr_service: Optional[ASRService] = None


def get_asr_service(
    backend: ASRBackend = ASRBackend.FASTER_WHISPER,
    model_size: str = "base",
    api_key: Optional[str] = None,
) -> ASRService:
    """Get global ASR service instance."""
    global _asr_service
    if _asr_service is None:
        _asr_service = ASRService(
            preferred_backend=backend,
            model_size=model_size,
            api_key=api_key,
        )
    return _asr_service

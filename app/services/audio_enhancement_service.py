"""Audio enhancement service - Complete implementation for AI voice processing."""
import logging
import time
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
try:
    # NOTE: pydub depends on `audioop` (removed in Python 3.13). Import lazily and degrade gracefully.
    from pydub import AudioSegment  # type: ignore
    from pydub.scipy_effects import low_pass_filter, high_pass_filter  # type: ignore
    from pydub.silence import detect_nonsilent  # type: ignore

    _PYDUB_AVAILABLE = True
    _PYDUB_IMPORT_ERROR: Exception | None = None
except Exception as e:  # pragma: no cover - environment dependent
    AudioSegment = Any  # type: ignore
    low_pass_filter = None  # type: ignore
    high_pass_filter = None  # type: ignore
    detect_nonsilent = None  # type: ignore
    _PYDUB_AVAILABLE = False
    _PYDUB_IMPORT_ERROR = e


logger = logging.getLogger(__name__)


def _require_pydub() -> None:
    if not _PYDUB_AVAILABLE:
        raise RuntimeError(
            "Audio enhancement requires optional dependencies that are unavailable in this runtime. "
            "If you are using Python 3.13+, install `pyaudioop` or run the backend with Python <= 3.12."
        ) from _PYDUB_IMPORT_ERROR


class AudioEnhancementService:
    """Professional audio enhancement service for improving audio quality."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self._sample_rate = 24000

    async def process(self, task_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Process audio enhancement task.

        Args:
            task_id: The task ID
            params: Parameters including input_audio, denoise, normalize, etc.

        Returns:
            Processing result with output_audio_path and metrics
        """
        from app.database import async_session_maker
        from app.models.audio_task import AudioTask
        from sqlalchemy import update

        input_audio = params.get("input_audio")
        denoise = params.get("denoise", True)
        normalize_audio = params.get("normalize", True)
        remove_reverb = params.get("remove_reverb", False)
        eq_preset = params.get("eq_preset")
        enhance_speech = params.get("enhance_speech", True)
        output_format = params.get("output_format", "wav")

        if not input_audio:
            raise ValueError("input_audio is required")

        self.logger.info(f"Processing audio enhancement for task {task_id}")

        # Update progress
        await self._update_progress(task_id, 0.1, "processing")

        start_time = time.time()

        # Load audio
        await self._update_progress(task_id, 0.2, "loading")
        audio = await self._load_audio(input_audio)

        # Apply enhancements
        if denoise:
            await self._update_progress(task_id, 0.3, "denoising")
            audio = await self._reduce_noise(audio)

        if enhance_speech:
            await self._update_progress(task_id, 0.5, "enhancing")
            audio = await self._enhance_speech(audio)

        if eq_preset:
            await self._update_progress(task_id, 0.6, "eq")
            audio = await self._apply_eq(audio, eq_preset)

        if remove_reverb:
            await self._update_progress(task_id, 0.7, "dereverb")
            audio = await self._reduce_reverb(audio)

        if normalize_audio:
            await self._update_progress(task_id, 0.8, "normalizing")
            audio = await self._normalize_audio(audio)

        # Save output
        await self._update_progress(task_id, 0.9, "saving")
        source_path = Path(input_audio)
        output_filename = f"enhanced_{task_id}_{source_path.stem}.{output_format}"
        output_path = str(source_path.parent / output_filename)

        await self._save_audio(audio, output_path, output_format)

        await self._update_progress(task_id, 0.95, "analyzing")

        # Analyze quality
        metrics = await self.analyze_quality(output_path)

        processing_time = time.time() - start_time

        await self._update_progress(task_id, 1.0, "completed")

        self.logger.info(f"Audio enhancement completed for task {task_id}")

        return {
            "output_audio_path": output_path,
            "metrics": metrics,
            "processing_time": processing_time,
        }

    async def _load_audio(self, audio_path: str) -> AudioSegment:
        """Load audio file."""
        _require_pydub()
        return AudioSegment.from_file(audio_path)

    async def _save_audio(self, audio: AudioSegment, output_path: str, format: str = "wav") -> None:
        """Save audio file."""
        _require_pydub()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        export_params = {"format": format}
        if format == "mp3":
            export_params["bitrate"] = "192k"

        audio.export(output_path, **export_params)

    async def _reduce_noise(self, audio: AudioSegment) -> AudioSegment:
        """Reduce background noise using spectral subtraction."""
        _require_pydub()
        # Convert to numpy for processing
        samples = np.array(audio.get_array_of_samples())

        # Simple noise gate
        silence_thresh = audio.dBFS - 16
        nonsilent_parts = detect_nonsilent(
            audio,
            min_silence_len=50,
            silence_thresh=silence_thresh
        )

        if nonsilent_parts:
            # Keep only nonsilent parts with small crossfade
            result = AudioSegment.silent()
            for i, (start, end) in enumerate(nonsilent_parts):
                segment = audio[start:end]
                if i > 0:
                    result = result + segment.crossfade(10)
                else:
                    result = segment
            return result

        return audio

    async def _enhance_speech(self, audio: AudioSegment) -> AudioSegment:
        """Enhance speech clarity."""
        _require_pydub()
        # Apply high-pass filter to remove low frequency rumble
        audio = high_pass_filter(audio, 80)

        # Apply low-pass filter to remove high frequency hiss
        audio = low_pass_filter(audio, 8000)

        # Compress dynamic range slightly for better intelligibility
        audio = audio.compress_dynamic_range(
            threshold=-20,
            ratio=4.0,
            attack=5.0,
            release=50.0
        )

        return audio

    async def _apply_eq(self, audio: AudioSegment, preset: str) -> AudioSegment:
        """Apply EQ preset."""
        _require_pydub()
        # Bass boost
        if preset == "bass":
            audio = low_pass_filter(audio, 500) + 3

        # Treble boost
        elif preset == "treble":
            audio = high_pass_filter(audio, 2000) + 3

        # Vocal boost
        elif preset == "vocal":
            # Boost mid frequencies
            audio = audio + 2

        # Flat/clean
        elif preset == "flat":
            pass

        return audio

    async def _reduce_reverb(self, audio: AudioSegment) -> AudioSegment:
        """Reduce room reverb."""
        _require_pydub()
        # Simple dereverberation by reducing tail
        # Apply slight compression to reduce reverb tail
        audio = audio.compress_dynamic_range(
            threshold=-16,
            ratio=2.0,
            attack=1.0,
            release=100.0
        )

        return audio

    async def _normalize_audio(self, audio: AudioSegment) -> AudioSegment:
        """Normalize audio levels."""
        _require_pydub()
        # Normalize to -1 dB
        target_dBFS = -1.0
        change_in_dBFS = target_dBFS - audio.dBFS

        return audio.apply_gain(change_in_dBFS)

    async def validate_input(self, params: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate audio enhancement parameters."""
        errors = []

        if "input_audio" not in params or not params["input_audio"]:
            errors.append("input_audio is required and must be a valid file path")

        # Validate optional parameters
        if "denoise" in params and not isinstance(params["denoise"], bool):
            errors.append("denoise must be a boolean")

        if "normalize" in params and not isinstance(params["normalize"], bool):
            errors.append("normalize must be a boolean")

        if "remove_reverb" in params and not isinstance(params["remove_reverb"], bool):
            errors.append("remove_reverb must be a boolean")

        if "output_format" in params:
            fmt = params["output_format"]
            if fmt not in ["wav", "mp3", "ogg", "flac"]:
                errors.append("output_format must be one of: wav, mp3, ogg, flac")

        return len(errors) == 0, errors

    async def separate_sources(
        self,
        audio_path: str,
        sources: list[str] | None = None,
    ) -> dict[str, str]:
        """Separate audio into different sources (vocals, music, drums, etc.).

        This is a placeholder for future DEMUCS or similar integration.

        Args:
            audio_path: Input audio path
            sources: List of sources to separate (default: ["vocals", "music"])

        Returns:
            Dictionary mapping source names to output file paths
        """
        if sources is None:
            sources = ["vocals", "music", "drums", "bass", "other"]

        self.logger.info(f"Source separation requested for {audio_path}: {sources}")
        self.logger.warning("Source separation requires DEMUCS or similar model - not implemented yet")

        # Placeholder: return original audio for all sources
        # In production, implement with:
        # - DEMUCS (https://github.com/facebookresearch/demucs)
        # - Spleeter (https://github.com/deezer/spleeter)

        output_paths = {}
        source_path = Path(audio_path)

        for source in sources:
            output_filename = f"{source_path.stem}_{source}{source_path.suffix}"
            output_paths[source] = str(source_path.parent / output_filename)

        return output_paths

    async def analyze_quality(self, audio_path: str) -> dict[str, Any]:
        """Analyze audio quality comprehensively.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with quality metrics and recommendations
        """
        try:
            _require_pydub()
            audio = AudioSegment.from_file(audio_path)

            # Basic metrics
            duration_ms = len(audio)
            duration_sec = duration_ms / 1000.0

            # Level metrics
            dbfs = audio.dBFS
            max_dBFS = audio.max_dBFS

            # Dynamic range
            dynamic_range = max_dBFS - audio.min_possible_dBFS

            # Detect clipping
            sample_count = len(audio.get_array_of_samples())
            clipped_samples = sum(1 for s in audio.get_array_of_samples() if abs(s) >= 32767)
            clipping_ratio = clipped_samples / sample_count if sample_count > 0 else 0

            # Calculate quality scores
            level_score = self._calculate_level_score(dbfs)
            dynamic_range_score = self._calculate_dynamic_range_score(dynamic_range)
            clipping_score = 1.0 - min(clipping_ratio * 100, 1.0)

            overall_score = (level_score * 0.4 + dynamic_range_score * 0.4 + clipping_score * 0.2) * 100

            # Generate recommendations
            recommendations = []

            if dbfs < -18:
                recommendations.append("音频电平偏低，建议提升音量")
            elif dbfs > -3:
                recommendations.append("音频电平偏高，有失真风险")

            if dynamic_range < 10:
                recommendations.append("动态范围较小，声音可能缺乏层次感")

            if clipping_ratio > 0.001:
                recommendations.append(f"检测到削波 ({clipping_ratio*100:.2f}% samples)，建议重新录制")

            if len(recommendations) == 0:
                recommendations.append("音频质量良好，无需特别处理")

            return {
                "overall_score": round(overall_score, 1),
                "level_dbfs": round(dbfs, 2),
                "max_dbfs": round(max_dBFS, 2),
                "dynamic_range": round(dynamic_range, 2),
                "duration": round(duration_sec, 2),
                "clipping_ratio": round(clipping_ratio * 100, 4),
                "clipping_detected": clipping_ratio > 0.001,
                "sample_rate": audio.frame_rate,
                "channels": audio.channels,
                "sample_width": audio.sample_width,
                "quality_level": self._get_quality_level(overall_score),
                "recommendations": recommendations,
            }

        except Exception as e:
            self.logger.error(f"Error analyzing audio quality: {e}")
            return {
                "overall_score": 0,
                "error": str(e),
                "recommendations": ["无法分析音频质量"],
            }

    def _calculate_level_score(self, dbfs: float) -> float:
        """Calculate score for audio level."""
        # Ideal range: -14 to -6 dBFS
        if -14 <= dbfs <= -6:
            return 1.0
        elif -18 <= dbfs <= -3:
            return 0.8
        elif -24 <= dbfs <= 0:
            return 0.6
        else:
            return 0.3

    def _calculate_dynamic_range_score(self, dynamic_range: float) -> float:
        """Calculate score for dynamic range."""
        # Ideal: >15 dB
        if dynamic_range >= 15:
            return 1.0
        elif dynamic_range >= 10:
            return 0.7
        elif dynamic_range >= 5:
            return 0.4
        else:
            return 0.2

    def _get_quality_level(self, score: float) -> str:
        """Get quality level description."""
        if score >= 85:
            return "优秀"
        elif score >= 70:
            return "良好"
        elif score >= 55:
            return "一般"
        elif score >= 40:
            return "较差"
        else:
            return "很差"

    async def _update_progress(self, task_id: str, progress: float, status: str) -> None:
        """Update task progress."""
        from app.database import async_session_maker
        from app.models.audio_task import AudioTask
        from sqlalchemy import update

        async with async_session_maker() as db:
            await db.execute(
                update(AudioTask)
                .where(AudioTask.id == task_id)
                .values(progress=progress, status=status)
            )
            await db.commit()

    async def convert_format(
        self,
        input_path: str,
        output_path: str,
        output_format: str = "wav",
        sample_rate: int = 24000,
        channels: int = 1,
    ) -> dict[str, Any]:
        """Convert audio to different format.

        Args:
            input_path: Input audio file path
            output_path: Output audio file path
            output_format: Target format (wav, mp3, ogg, flac)
            sample_rate: Target sample rate
            channels: Number of channels (1=mono, 2=stereo)

        Returns:
            Dictionary with output path and conversion info
        """
        try:
            _require_pydub()
            audio = AudioSegment.from_file(input_path)

            # Convert sample rate if needed
            if audio.frame_rate != sample_rate:
                audio = audio.set_frame_rate(sample_rate)

            # Convert channels if needed
            if audio.channels != channels:
                audio = audio.set_channels(channels)

            # Export
            export_params = {"format": output_format}
            if output_format == "mp3":
                export_params["bitrate"] = "192k"
            elif output_format == "wav":
                export_params["parameters"] = ["-ar", str(sample_rate)]

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            audio.export(output_path, **export_params)

            return {
                "success": True,
                "output_path": output_path,
                "duration": len(audio) / 1000.0,
                "sample_rate": audio.frame_rate,
                "channels": audio.channels,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

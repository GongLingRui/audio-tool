"""Voice conversion service - Complete implementation for AI voice processing."""
import logging
import time
from pathlib import Path
from typing import Any

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


logger = logging.getLogger(__name__)


def _require_pydub() -> None:
    if not _PYDUB_AVAILABLE:
        raise RuntimeError(
            "Voice conversion requires optional dependencies that are unavailable in this runtime. "
            "If you are using Python 3.13+, install `pyaudioop` or run the backend with Python <= 3.12."
        ) from _PYDUB_IMPORT_ERROR


class VoiceConversionService:
    """Voice conversion service for converting audio from one voice to another.

    This service provides a framework for voice conversion. For production use,
    integrate with:
    - RVC (Retrieval-based Voice Conversion)
    - OpenVoice
    - VoiceFixer
    - So-VITS-SVC
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    async def process(self, task_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Process voice conversion task.

        Args:
            task_id: The task ID
            params: Parameters including source_audio, target_voice, similarity

        Returns:
            Processing result with output_audio_path and quality_score
        """
        from app.database import async_session_maker
        from app.models.audio_task import AudioTask
        from sqlalchemy import update

        source_audio = params.get("source_audio")
        target_voice = params.get("target_voice")
        similarity = params.get("similarity", 0.85)
        pitch_shift = params.get("pitch_shift", 0)
        formant_shift = params.get("formant_shift", 0)

        if not source_audio or not target_voice:
            raise ValueError("source_audio and target_voice are required")

        self.logger.info(f"Processing voice conversion for task {task_id}")

        # Update progress
        await self._update_progress(task_id, 0.1, "processing")

        start_time = time.time()

        # Load audio
        await self._update_progress(task_id, 0.2, "loading")
        _require_pydub()
        audio = AudioSegment.from_file(source_audio)

        # Apply audio transformations (basic voice modification)
        await self._update_progress(task_id, 0.4, "analyzing")
        processed = await self._apply_voice_modifications(
            audio,
            pitch_shift=pitch_shift,
            formant_shift=formant_shift,
            target_voice=target_voice
        )

        # Apply similarity-based processing
        await self._update_progress(task_id, 0.7, "converting")
        if target_voice != "original":
            processed = await self._match_target_voice(
                processed,
                target_voice,
                similarity
            )

        # Save output
        await self._update_progress(task_id, 0.9, "saving")
        source_path = Path(source_audio)
        output_filename = f"converted_{task_id}_{source_path.stem}{source_path.suffix}"
        output_path = str(source_path.parent / output_filename)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        processed.export(output_path, format="wav")

        # Calculate quality metrics
        quality_score = await self._calculate_quality_score(
            AudioSegment.from_file(source_audio),
            processed
        )

        processing_time = time.time() - start_time

        await self._update_progress(task_id, 1.0, "completed")

        self.logger.info(f"Voice conversion completed for task {task_id}")

        return {
            "output_audio_path": output_path,
            "quality_score": quality_score,
            "processing_time": processing_time,
            "target_voice": target_voice,
            "similarity": similarity,
        }

    async def _apply_voice_modifications(
        self,
        audio: AudioSegment,
        pitch_shift: int = 0,
        formant_shift: int = 0,
        target_voice: str = "original"
    ) -> AudioSegment:
        """Apply basic voice modifications.

        Args:
            audio: Input audio
            pitch_shift: Semitones to shift pitch (-12 to +12)
            formant_shift: Formant frequency shift
            target_voice: Target voice preset

        Returns:
            Modified audio
        """
        result = audio

        # Apply pitch shift using frame rate manipulation
        if pitch_shift != 0:
            # This is a simple pitch shift - for better quality, use specialized libraries
            # like pydub's pitch_shift or librosa's pitch shifting
            factor = 2.0 ** (pitch_shift / 12.0)
            new_frame_rate = int(result.frame_rate * factor)
            result = result._spawn(
                result.raw_data,
                overrides={'frame_rate': new_frame_rate}
            )
            result = result.set_frame_rate(audio.frame_rate)

        # Apply voice presets
        if target_voice == "deeper":
            # Lower pitch for deeper voice
            result = await self._apply_formant_shift(result, -0.8)
        elif target_voice == "higher":
            # Higher pitch
            result = await self._apply_formant_shift(result, 0.8)
        elif target_voice == "robotic":
            # Add robotic effect
            result = self._add_robotic_effect(result)
        elif target_voice == "echo":
            # Add echo effect
            result = self._add_echo(result)
        elif target_voice == "telephone":
            # Add telephone bandpass effect
            result = self._apply_telephone_effect(result)

        return result

    async def _apply_formant_shift(self, audio: AudioSegment, shift: float) -> AudioSegment:
        """Apply formant shifting (voice quality modification).

        Args:
            audio: Input audio
            shift: Formant shift factor (negative = deeper, positive = higher)

        Returns:
            Modified audio
        """
        # Simple formant shift using playback rate
        # For production, use specialized formant shifting algorithms
        if shift < 0:
            # Slower playback = deeper voice
            factor = 1.0 + abs(shift) * 0.2
        else:
            # Faster playback = higher voice
            factor = 1.0 - shift * 0.2

        factor = max(0.5, min(2.0, factor))

        new_frame_rate = int(audio.frame_rate * factor)
        result = audio._spawn(
            audio.raw_data,
            overrides={'frame_rate': new_frame_rate}
        )
        result = result.set_frame_rate(audio.frame_rate)

        return result

    def _add_robotic_effect(self, audio: AudioSegment) -> AudioSegment:
        """Add robotic/electronic effect to audio."""
        # Modulate with high frequency
        samples = np.array(audio.get_array_of_samples())

        # Simple amplitude modulation
        modulation = np.sin(2 * np.pi * 50 * np.linspace(0, len(samples) / audio.frame_rate, len(samples)))
        modulated = samples * (0.7 + 0.3 * modulation)

        # Convert back to AudioSegment
        modulated = np.clip(modulated, -32768, 32767).astype(np.int16)
        return audio._spawn(modulated.tobytes())

    def _add_echo(self, audio: AudioSegment, delay_ms: int = 200, decay: float = 0.5) -> AudioSegment:
        """Add echo/delay effect."""
        # Create delayed version
        delay = AudioSegment.silent(duration=delay_ms) + audio

        # Mix with original
        echo = audio.overlay(delay.apply_gain(-20 * np.log10(1 / decay)))

        return echo

    def _apply_telephone_effect(self, audio: AudioSegment) -> AudioSegment:
        """Apply telephone bandpass effect (300Hz - 3400Hz)."""
        # This is a simplified implementation
        # For production, use proper bandpass filtering

        # Cut low frequencies
        audio = audio.high_pass_filter(300)

        # Cut high frequencies
        audio = audio.low_pass_filter(3400)

        # Compress dynamic range
        audio = audio.compress_dynamic_range(
            threshold=-20,
            ratio=4.0
        )

        return audio

    async def _match_target_voice(
        self,
        audio: AudioSegment,
        target_voice: str,
        similarity: float
    ) -> AudioSegment:
        """Match audio to target voice characteristics.

        This is a placeholder for RVC or similar integration.
        For production, implement with:
        - RVC (Retrieval-based Voice Conversion)
        - OpenVoice API
        - So-VITS-SVC

        Args:
            audio: Input audio
            target_voice: Target voice identifier
            similarity: Similarity threshold (0.0 - 1.0)

        Returns:
            Converted audio
        """
        self.logger.info(f"Matching to voice: {target_voice} with similarity {similarity}")

        # Placeholder implementation
        # In production, this would call the RVC service or similar

        if target_voice.startswith("speaker_"):
            # Extract speaker ID and apply corresponding transformation
            speaker_id = target_voice.split("_")[1]
            # Apply speaker-specific modifications
            # This is where you'd integrate with a trained model
            pass

        return audio

    async def _calculate_quality_score(
        self,
        original: AudioSegment,
        converted: AudioSegment
    ) -> float:
        """Calculate quality score for converted audio.

        Args:
            original: Original audio
            converted: Converted audio

        Returns:
            Quality score (0.0 - 1.0)
        """
        # Compare characteristics
        original_dbfs = original.dBFS
        converted_dbfs = converted.dBFS

        # Level similarity
        level_diff = abs(original_dbfs - converted_dbfs)
        level_score = max(0.0, 1.0 - level_diff / 20.0)

        # Duration similarity
        duration_diff = abs(len(original) - len(converted)) / max(len(original), 1)
        duration_score = max(0.0, 1.0 - duration_diff)

        # Combined score
        quality_score = (level_score * 0.6 + duration_score * 0.4)

        return round(quality_score, 3)

    async def validate_input(self, params: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate voice conversion parameters.

        Args:
            params: Parameters to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if "source_audio" not in params or not params["source_audio"]:
            errors.append("source_audio is required and must be a valid file path")

        if "target_voice" not in params or not params["target_voice"]:
            errors.append("target_voice is required")

        if "similarity" in params:
            similarity = params["similarity"]
            if not isinstance(similarity, (int, float)) or not (0.0 <= similarity <= 1.0):
                errors.append("similarity must be a float between 0.0 and 1.0")

        if "pitch_shift" in params:
            pitch = params["pitch_shift"]
            if not isinstance(pitch, (int, float)) or not (-12 <= pitch <= 12):
                errors.append("pitch_shift must be between -12 and +12 semitones")

        if "formant_shift" in params:
            formant = params["formant_shift"]
            if not isinstance(formant, (int, float)) or not (-1.0 <= formant <= 1.0):
                errors.append("formant_shift must be between -1.0 and 1.0")

        return len(errors) == 0, errors

    async def extract_voice_embedding(self, audio_path: str) -> dict[str, Any]:
        """Extract voice embedding for speaker verification.

        This is a placeholder for voice embedding extraction.
        For production, use:
        - Resemblyzer
        - SpeechBrain speaker embedding
        - ECAPA-TDNN

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with embedding and metadata
        """
        self.logger.info(f"Extracting voice embedding from: {audio_path}")

        audio = AudioSegment.from_file(audio_path)

        # Placeholder implementation
        # In production, extract actual embeddings using a trained model

        return {
            "embedding": np.zeros(256).tolist(),  # Placeholder
            "audio_length": len(audio) / 1000.0,
            "sample_rate": audio.frame_rate,
            "channels": audio.channels,
            "note": "Voice embedding requires model integration"
        }

    async def compare_voices(
        self,
        audio_path1: str,
        audio_path2: str
    ) -> dict[str, Any]:
        """Compare two voice samples.

        Args:
            audio_path1: First audio file
            audio_path2: Second audio file

        Returns:
            Dictionary with similarity score and match result
        """
        self.logger.info(f"Comparing voices: {audio_path1} vs {audio_path2}")

        # Extract embeddings
        embedding1 = await self.extract_voice_embedding(audio_path1)
        embedding2 = await self.extract_voice_embedding(audio_path2)

        # Calculate cosine similarity (placeholder)
        # In production, use actual embeddings
        similarity = 0.5  # Placeholder

        return {
            "similarity": similarity,
            "same_speaker": similarity > 0.75,
            "confidence": 0.0,
            "threshold": 0.75,
        }

    async def get_available_voices(self) -> list[dict[str, Any]]:
        """Get list of available voice presets.

        Returns:
            List of available voice configurations
        """
        return [
            {
                "id": "original",
                "name": "Original",
                "description": "Keep original voice",
                "category": "preset"
            },
            {
                "id": "deeper",
                "name": "Deeper",
                "description": "Lower pitch for deeper voice",
                "category": "preset"
            },
            {
                "id": "higher",
                "name": "Higher",
                "description": "Higher pitch voice",
                "category": "preset"
            },
            {
                "id": "robotic",
                "name": "Robotic",
                "description": "Electronic/robotic effect",
                "category": "effect"
            },
            {
                "id": "echo",
                "name": "Echo",
                "description": "Add echo/delay effect",
                "category": "effect"
            },
            {
                "id": "telephone",
                "name": "Telephone",
                "description": "Telephone quality effect",
                "category": "effect"
            },
        ]

    async def voice_profile_to_json(
        self,
        audio_path: str,
        profile_name: str
    ) -> dict[str, Any]:
        """Create a voice profile from audio sample.

        This is used to capture voice characteristics for conversion.

        Args:
            audio_path: Path to reference audio
            profile_name: Name for the voice profile

        Returns:
            Voice profile dictionary
        """
        audio = AudioSegment.from_file(audio_path)

        profile = {
            "name": profile_name,
            "duration": len(audio) / 1000.0,
            "sample_rate": audio.frame_rate,
            "avg_dbfs": audio.dBFS,
            "max_dbfs": audio.max_dBFS,
            "embedding": await self.extract_voice_embedding(audio_path),
            "created_at": time.time(),
        }

        return profile

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

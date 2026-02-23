"""
Audio Mixer - Background music and sound effects mixing
Professional audio mixing with ducking, crossfade, and volume control
"""
import io
import logging
import tempfile
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
try:
    # NOTE: pydub depends on `audioop` (removed in Python 3.13). Import lazily and degrade gracefully.
    from pydub import AudioSegment  # type: ignore
    from pydub.silence import detect_nonsilent  # type: ignore
    from pydub.utils import ratio_to_db  # type: ignore

    _PYDUB_AVAILABLE = True
    _PYDUB_IMPORT_ERROR: Exception | None = None
except Exception as e:  # pragma: no cover - environment dependent
    AudioSegment = Any  # type: ignore
    detect_nonsilent = None  # type: ignore
    ratio_to_db = None  # type: ignore
    _PYDUB_AVAILABLE = False
    _PYDUB_IMPORT_ERROR = e

logger = logging.getLogger(__name__)


from dataclasses import dataclass


def _require_pydub() -> None:
    if not _PYDUB_AVAILABLE:
        raise RuntimeError(
            "Audio mixing requires optional dependencies that are unavailable in this runtime. "
            "If you are using Python 3.13+, install `pyaudioop` or run the backend with Python <= 3.12."
        ) from _PYDUB_IMPORT_ERROR


@dataclass
class SoundEffect:
    """Sound effect definition."""
    file_path: str
    time: float  # Time in seconds to insert effect
    volume: float = 0.5  # Volume multiplier (0.0 - 1.0)
    duration: Optional[float] = None  # Override effect duration
    fade_in: float = 0.0  # Fade in duration in seconds
    fade_out: float = 0.0  # Fade out duration in seconds


@dataclass
class MusicTrack:
    """Background music track definition."""
    file_path: str
    volume: float = 0.2  # Base volume (0.0 - 1.0)
    loop: bool = True  # Loop to fill duration
    fade_in: float = 0.0  # Fade in duration in seconds
    fade_out: float = 0.0  # Fade out duration in seconds
    start_time: float = 0.0  # Start position in music file


class AudioMixer:
    """
    Professional audio mixer for TTS output.

    Features:
    - Background music mixing with looping
    - Sound effects at specific timestamps
    - Automatic ducking (lower music during speech)
    - Crossfading between segments
    - Volume normalization and balancing
    - Multiple audio format support
    """

    def __init__(self, sample_rate: int = 24000, channels: int = 1):
        """
        Initialize audio mixer.

        Args:
            sample_rate: Output sample rate
            channels: Number of channels (1=mono, 2=stereo)
        """
        self.sample_rate = sample_rate
        self.channels = channels

    async def mix_audio(
        self,
        speech_audio: bytes,
        background_music: Optional[bytes] = None,
        background_music_path: Optional[str] = None,
        music_volume: float = 0.2,
        sound_effects: Optional[List[SoundEffect]] = None,
        ducking: bool = True,
        ducking_amount: float = 6.0,  # dB to reduce during speech
        crossfade: float = 0.0,
        output_format: str = "mp3",
        normalize: bool = True,
    ) -> bytes:
        """
        Mix speech audio with background music and sound effects.

        Args:
            speech_audio: Main speech audio bytes
            background_music: Background music audio bytes (optional)
            background_music_path: Path to background music file (optional)
            music_volume: Music volume multiplier (0.0 - 1.0)
            sound_effects: List of sound effects to add
            ducking: Apply ducking (reduce music during speech)
            ducking_amount: dB to reduce music during speech activity
            crossfade: Crossfade duration between music segments (seconds)
            output_format: Output audio format (mp3, wav, ogg, flac)
            normalize: Normalize final output

        Returns:
            Mixed audio bytes
        """
        # Load speech
        speech = AudioSegment.from_file(io.BytesIO(speech_audio))
        speech_duration = len(speech) / 1000.0  # seconds

        # Convert to target format
        speech = self._convert_format(speech)

        result = speech

        # Mix background music
        if background_music or background_music_path:
            if background_music_path:
                music = AudioSegment.from_file(background_music_path)
            else:
                music = AudioSegment.from_file(io.BytesIO(background_music))

            music = self._convert_format(music)

            # Mix with music
            result = await self._mix_with_music(
                speech,
                music,
                music_volume,
                ducking,
                ducking_amount,
                crossfade,
            )

        # Add sound effects
        if sound_effects:
            result = await self._add_sound_effects(result, sound_effects)

        # Normalize if requested
        if normalize:
            result = self._normalize_audio(result)

        # Export
        output = io.BytesIO()
        export_params = {"format": output_format}
        if output_format == "mp3":
            export_params["bitrate"] = "192k"
        elif output_format == "wav":
            export_params["parameters"] = ["-ar", str(self.sample_rate)]

        result.export(output, **export_params)
        return output.read()

    async def _mix_with_music(
        self,
        speech: AudioSegment,
        music: AudioSegment,
        music_volume: float,
        ducking: bool,
        ducking_amount: float,
        crossfade: float,
    ) -> AudioSegment:
        """Mix speech with background music."""
        speech_duration = len(speech)
        music_duration = len(music)

        # Loop music to match speech duration
        if music_duration < speech_duration:
            loops_needed = int(np.ceil(speech_duration / music_duration))
            music_parts = []
            for i in range(loops_needed):
                music_parts.append(music)

            # Crossfade between loops if specified
            if crossfade > 0:
                crossfade_ms = int(crossfade * 1000)
                looped_music = music_parts[0]
                for part in music_parts[1:]:
                    looped_music = looped_music.append(part, crossfade=crossfade_ms)
            else:
                looped_music = sum(music_parts)
        else:
            # Truncate music to speech duration
            looped_music = music[:speech_duration]

        # Ensure music matches speech duration
        if len(looped_music) > speech_duration:
            looped_music = looped_music[:speech_duration]
        elif len(looped_music) < speech_duration:
            # Pad with silence
            silence_duration = speech_duration - len(looped_music)
            looped_music = looped_music + AudioSegment.silent(duration=silence_duration)

        # Apply music volume
        if music_volume != 1.0:
            volume_change = ratio_to_db(music_volume)
            looped_music = looped_music.apply_gain(volume_change)

        # Apply ducking if requested
        if ducking:
            looped_music = self._apply_ducking(looped_music, speech, ducking_amount)

        # Mix
        result = speech.overlay(looped_music)

        return result

    def _apply_ducking(
        self,
        music: AudioSegment,
        speech: AudioSegment,
        ducking_amount: float,
    ) -> AudioSegment:
        """
        Apply ducking to music based on speech activity.

        Reduces music volume during speech segments.
        """
        # Detect speech activity (non-silent parts)
        speech_active = detect_nonsilent(
            speech,
            min_silence_len=100,  # 100ms
            silence_thresh=speech.dbFS - 16,
        )

        if not speech_active:
            # No speech detected, return music as-is
            return music

        # Create ducked music
        ducked_music = music

        # Sort speech segments by start time
        speech_active.sort()

        for start_ms, end_ms in speech_active:
            # Ensure segment is within music bounds
            if start_ms >= len(music):
                continue

            end_ms = min(end_ms, len(music))

            # Apply ducking to this segment
            segment = music[start_ms:end_ms]

            # Get current segment volume
            current_dBFS = segment.dBFS

            # Reduce volume by ducking_amount
            ducked_segment = segment.apply_gain(-ducking_amount)

            # Replace in music
            ducked_music = ducked_music[:start_ms] + ducked_segment + ducked_music[end_ms:]

        return ducked_music

    async def _add_sound_effects(
        self,
        audio: AudioSegment,
        sound_effects: List[SoundEffect],
    ) -> AudioSegment:
        """Add sound effects at specified times."""
        for effect in sound_effects:
            try:
                # Load effect audio
                effect_audio = AudioSegment.from_file(effect.file_path)
                effect_audio = self._convert_format(effect_audio)

                # Apply volume
                if effect.volume != 1.0:
                    volume_change = ratio_to_db(effect.volume)
                    effect_audio = effect_audio.apply_gain(volume_change)

                # Apply fade in/out
                if effect.fade_in > 0:
                    fade_in_ms = int(effect.fade_in * 1000)
                    effect_audio = effect_audio.fade_in(fade_in_ms)

                if effect.fade_out > 0:
                    fade_out_ms = int(effect.fade_out * 1000)
                    effect_duration = len(effect_audio)
                    effect_audio = effect_audio.fade_out(min(fade_out_ms, effect_duration))

                # Override duration if specified
                if effect.duration:
                    target_ms = int(effect.duration * 1000)
                    if len(effect_audio) > target_ms:
                        effect_audio = effect_audio[:target_ms]
                    elif len(effect_audio) < target_ms:
                        # Extend with silence or loop
                        silence_needed = target_ms - len(effect_audio)
                        effect_audio = effect_audio + AudioSegment.silent(duration=silence_needed)

                # Insert at specified time
                effect_time_ms = int(effect.time * 1000)

                # Ensure audio is long enough
                if len(audio) < effect_time_ms + len(effect_audio):
                    # Extend audio with silence
                    silence_needed = effect_time_ms + len(effect_audio) - len(audio)
                    audio = audio + AudioSegment.silent(duration=silence_needed)

                # Overlay effect
                audio = audio.overlay(effect_audio, position=effect_time_ms)

            except Exception as e:
                logger.error(f"Error adding sound effect {effect.file_path}: {e}")

        return audio

    def _convert_format(self, audio: AudioSegment) -> AudioSegment:
        """Convert audio to target format."""
        # Convert sample rate if needed
        if audio.frame_rate != self.sample_rate:
            audio = audio.set_frame_rate(self.sample_rate)

        # Convert channels if needed
        if audio.channels != self.channels:
            audio = audio.set_channels(self.channels)

        return audio

    def _normalize_audio(self, audio: AudioSegment, target_dBFS: float = -1.0) -> AudioSegment:
        """Normalize audio to target level."""
        current_dBFS = audio.dBFS
        change_in_dBFS = target_dBFS - current_dBFS
        return audio.apply_gain(change_in_dBFS)

    async def create_audio_crossfade(
        self,
        audio1: bytes,
        audio2: bytes,
        crossfade_duration: float = 1.0,
    ) -> bytes:
        """
        Create crossfade between two audio segments.

        Args:
            audio1: First audio segment
            audio2: Second audio segment
            crossfade_duration: Crossfade duration in seconds

        Returns:
            Crossfaded audio bytes
        """
        seg1 = AudioSegment.from_file(io.BytesIO(audio1))
        seg2 = AudioSegment.from_file(io.BytesIO(audio2))

        # Convert to target format
        seg1 = self._convert_format(seg1)
        seg2 = self._convert_format(seg2)

        # Apply crossfade
        crossfade_ms = int(crossfade_duration * 1000)

        # Limit crossfade to shorter segment
        max_crossfade = min(len(seg1), len(seg2)) // 2
        crossfade_ms = min(crossfade_ms, max_crossfade)

        result = seg1.append(seg2, crossfade=crossfade_ms)

        # Export
        output = io.BytesIO()
        result.export(output, format="mp3", bitrate="192k")
        return output.read()

    async def splice_audio(
        self,
        segments: List[bytes],
        crossfade: float = 0.5,
    ) -> bytes:
        """
        Splice multiple audio segments together with crossfade.

        Args:
            segments: List of audio segments to splice
            crossfade: Crossfade duration between segments (seconds)

        Returns:
            Spliced audio bytes
        """
        if not segments:
            return b""

        if len(segments) == 1:
            return segments[0]

        # Load and convert first segment
        result = AudioSegment.from_file(io.BytesIO(segments[0]))
        result = self._convert_format(result)

        # Append remaining segments
        for segment_bytes in segments[1:]:
            segment = AudioSegment.from_file(io.BytesIO(segment_bytes))
            segment = self._convert_format(segment)

            # Append with crossfade
            crossfade_ms = int(crossfade * 1000)
            max_crossfade = min(len(result), len(segment)) // 2
            crossfade_ms = min(crossfade_ms, max_crossfade)

            result = result.append(segment, crossfade=crossfade_ms)

        # Export
        output = io.BytesIO()
        result.export(output, format="mp3", bitrate="192k")
        return output.read()

    async def extract_segment(
        self,
        audio: bytes,
        start_time: float,
        end_time: float,
        fade_in: float = 0.0,
        fade_out: float = 0.0,
    ) -> bytes:
        """
        Extract a segment from audio.

        Args:
            audio: Input audio bytes
            start_time: Start time in seconds
            end_time: End time in seconds
            fade_in: Fade in duration in seconds
            fade_out: Fade out duration in seconds

        Returns:
            Extracted segment bytes
        """
        audio_seg = AudioSegment.from_file(io.BytesIO(audio))
        audio_seg = self._convert_format(audio_seg)

        # Extract segment
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)

        segment = audio_seg[start_ms:end_ms]

        # Apply fades
        if fade_in > 0:
            fade_in_ms = int(fade_in * 1000)
            segment = segment.fade_in(fade_in_ms)

        if fade_out > 0:
            fade_out_ms = int(fade_out * 1000)
            segment = segment.fade_out(fade_out_ms)

        # Export
        output = io.BytesIO()
        segment.export(output, format="mp3", bitrate="192k")
        return output.read()

    async def calculate_volume_curve(
        self,
        audio: bytes,
        curve_type: str = "fade_in",
        duration: float = 1.0,
    ) -> bytes:
        """
        Apply volume curve to audio.

        Args:
            audio: Input audio bytes
            curve_type: Type of curve (fade_in, fade_out, fade_in_out)
            duration: Duration of curve in seconds

        Returns:
            Processed audio bytes
        """
        audio_seg = AudioSegment.from_file(io.BytesIO(audio))
        audio_seg = self._convert_format(audio_seg)

        duration_ms = int(duration * 1000)

        if curve_type == "fade_in":
            result = audio_seg.fade_in(duration_ms)
        elif curve_type == "fade_out":
            result = audio_seg.fade_out(duration_ms)
        elif curve_type == "fade_in_out":
            fade_duration = min(duration_ms, len(audio_seg) // 2)
            result = audio_seg.fade_in(fade_duration).fade_out(fade_duration)
        else:
            result = audio_seg

        # Export
        output = io.BytesIO()
        result.export(output, format="mp3", bitrate="192k")
        return output.read()


class SoundEffectLibrary:
    """Library of predefined sound effects."""

    EFFECTS = {
        # UI Sounds
        "click": {"category": "ui", "duration": 0.1, "volume": 0.3},
        "notification": {"category": "ui", "duration": 0.5, "volume": 0.4},
        "success": {"category": "ui", "duration": 0.3, "volume": 0.5},
        "error": {"category": "ui", "duration": 0.4, "volume": 0.5},

        # Ambient Sounds
        "rain": {"category": "ambient", "volume": 0.3, "loop": True},
        "wind": {"category": "ambient", "volume": 0.2, "loop": True},
        "forest": {"category": "ambient", "volume": 0.3, "loop": True},
        "ocean": {"category": "ambient", "volume": 0.4, "loop": True},
        "city": {"category": "ambient", "volume": 0.5, "loop": True},

        # Action Sounds
        "footsteps": {"category": "action", "volume": 0.3},
        "door_open": {"category": "action", "duration": 0.5, "volume": 0.6},
        "door_close": {"category": "action", "duration": 0.4, "volume": 0.5},
        "page_turn": {"category": "action", "duration": 0.3, "volume": 0.4},

        # Emotional Atmosphere
        "tension": {"category": "atmosphere", "volume": 0.2, "loop": True},
        "mystery": {"category": "atmosphere", "volume": 0.2, "loop": True},
        "romantic": {"category": "atmosphere", "volume": 0.25, "loop": True},
        "action": {"category": "atmosphere", "volume": 0.4, "loop": True},
    }

    @classmethod
    def get_effect(cls, name: str) -> Optional[Dict[str, Any]]:
        """Get effect definition by name."""
        return cls.EFFECTS.get(name)

    @classmethod
    def list_effects(cls, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available effects, optionally filtered by category."""
        if category:
            return [
                {"name": name, **info}
                for name, info in cls.EFFECTS.items()
                if info.get("category") == category
            ]
        return [{"name": name, **info} for name, info in cls.EFFECTS.items()]


class MusicLibrary:
    """Library of predefined background music."""

    MUSIC = {
        # Atmospheric
        "calm_piano": {
            "category": "atmospheric",
            "mood": "calm",
            "volume": 0.15,
            "description": "Calm piano background",
        },
        "ambient_pad": {
            "category": "atmospheric",
            "mood": "neutral",
            "volume": 0.2,
            "description": "Ambient pad drone",
        },
        "nature_sounds": {
            "category": "atmospheric",
            "mood": "calm",
            "volume": 0.25,
            "description": "Nature ambient sounds",
        },

        # Story-specific
        "suspense": {
            "category": "story",
            "mood": "tense",
            "volume": 0.2,
            "description": "Suspenseful underscore",
        },
        "dramatic": {
            "category": "story",
            "mood": "dramatic",
            "volume": 0.25,
            "description": "Dramatic orchestral",
        },
        "mystery": {
            "category": "story",
            "mood": "mysterious",
            "volume": 0.18,
            "description": "Mysterious atmosphere",
        },
        "romantic": {
            "category": "story",
            "mood": "romantic",
            "volume": 0.2,
            "description": "Romantic strings",
        },
        "action": {
            "category": "story",
            "mood": "energetic",
            "volume": 0.3,
            "description": "Action percussion",
        },
        "melancholy": {
            "category": "story",
            "mood": "sad",
            "volume": 0.15,
            "description": "Melancholic cello",
        },
    }

    @classmethod
    def get_music(cls, name: str) -> Optional[Dict[str, Any]]:
        """Get music definition by name."""
        return cls.MUSIC.get(name)

    @classmethod
    def list_music(cls, category: Optional[str] = None, mood: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available music, optionally filtered."""
        results = []
        for name, info in cls.MUSIC.items():
            if category and info.get("category") != category:
                continue
            if mood and info.get("mood") != mood:
                continue
            results.append({"name": name, **info})
        return results


# Global instance
_mixer: Optional[AudioMixer] = None


def get_audio_mixer(sample_rate: int = 24000, channels: int = 1) -> AudioMixer:
    """Get global audio mixer instance."""
    global _mixer
    if _mixer is None:
        _mixer = AudioMixer(sample_rate=sample_rate, channels=channels)
    return _mixer

"""Audio utility functions."""
import os
from pathlib import Path


def get_audio_duration(file_path: str) -> float:
    """Get audio file duration in seconds."""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000.0
    except Exception:
        return 0.0


def validate_audio_file(file_path: str) -> dict:
    """Validate audio file and return metadata."""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(file_path)

        return {
            "valid": True,
            "duration": len(audio) / 1000.0,
            "channels": audio.channels,
            "sample_rate": audio.frame_rate,
            "sample_width": audio.sample_width,
            "format": Path(file_path).suffix.lower().lstrip("."),
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
        }


def get_audio_file_info(file_path: str) -> dict | None:
    """Get detailed audio file information."""
    if not os.path.exists(file_path):
        return None

    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(file_path)

        return {
            "file_path": file_path,
            "file_size": os.path.getsize(file_path),
            "duration": len(audio) / 1000.0,
            "channels": audio.channels,
            "sample_rate": audio.frame_rate,
            "sample_width": audio.sample_width,
            "frame_count": audio.frame_count(),
            "frame_width": audio.frame_width,
        }
    except Exception:
        return None

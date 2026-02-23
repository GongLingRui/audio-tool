"""Audio decoding helpers.

Goal: make common non-WAV formats (mp3/flac/m4a/ogg/...) usable without relying
on pydub/audioop, by decoding via the system `ffmpeg`/`ffprobe` binaries when
available.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@dataclass(frozen=True)
class DecodedAudioInfo:
    duration: float
    channels: int
    sample_rate: int


def probe_audio(path: str) -> DecodedAudioInfo:
    """Probe audio via ffprobe. Raises if ffprobe missing or probe fails."""
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg/ffprobe not available")

    p = Path(path)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=channels,sample_rate,duration",
        "-of",
        "json",
        str(p),
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    data = json.loads(out)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError("No audio stream found")

    stream = streams[0]
    channels = int(stream.get("channels") or 0)
    sample_rate = int(stream.get("sample_rate") or 0)
    duration = float(stream.get("duration") or 0.0)
    return DecodedAudioInfo(duration=duration, channels=channels, sample_rate=sample_rate)


def iter_audio_mono_float32(
    path: str,
    *,
    sample_rate: int = 16000,
    chunk_samples: int = 4096,
) -> Iterator[np.ndarray]:
    """Decode audio to mono float32 [-1, 1] via ffmpeg and stream as numpy arrays.

    Raises if ffmpeg is missing or decoding fails.
    """
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg/ffprobe not available")

    p = Path(path)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(p),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]

    # Use a subprocess and stream from stdout to avoid holding the entire decoded
    # audio in memory.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    assert proc.stderr is not None

    bytes_per_sample = 4  # float32
    read_size = chunk_samples * bytes_per_sample

    try:
        while True:
            buf = proc.stdout.read(read_size)
            if not buf:
                break
            # Ensure aligned to float32 samples.
            if len(buf) % bytes_per_sample != 0:
                buf = buf[: len(buf) - (len(buf) % bytes_per_sample)]
            if not buf:
                break
            arr = np.frombuffer(buf, dtype=np.float32)
            yield np.clip(arr, -1.0, 1.0).astype(np.float32)

        rc = proc.wait()
        if rc != 0:
            err = proc.stderr.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"ffmpeg decode failed (rc={rc}): {err.strip()}")
    finally:
        try:
            proc.kill()
        except Exception:
            pass

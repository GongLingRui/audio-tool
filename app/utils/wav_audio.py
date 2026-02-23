"""WAV utilities.

This module provides lightweight WAV parsing + basic loudness metrics without
external dependencies (ffmpeg/pydub/audioop). It is used as a fallback on
Python 3.13+ or environments where those optional deps are unavailable.
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class WavInfo:
    duration: float
    channels: int
    sample_rate: int
    sample_width: int
    frame_count: int


def read_wav_info(path: str) -> WavInfo:
    p = Path(path)
    with wave.open(str(p), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        frame_count = wf.getnframes()
        duration = frame_count / float(sample_rate) if sample_rate else 0.0
    return WavInfo(
        duration=duration,
        channels=channels,
        sample_rate=sample_rate,
        sample_width=sample_width,
        frame_count=frame_count,
    )


def _bytes_to_float_mono(
    raw: bytes,
    channels: int,
    sample_width: int,
) -> np.ndarray:
    """Convert interleaved PCM bytes to mono float32 [-1, 1]."""
    if not raw:
        return np.zeros((0,), dtype=np.float32)

    if sample_width == 1:
        # 8-bit PCM is unsigned.
        data_u8 = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data_u8 - 128.0) / 128.0
    elif sample_width == 2:
        data_i16 = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        data = data_i16 / 32768.0
    elif sample_width == 4:
        data_i32 = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
        data = data_i32 / 2147483648.0
    elif sample_width == 3:
        # 24-bit little endian signed
        a = np.frombuffer(raw, dtype=np.uint8)
        a = a.reshape(-1, 3)
        signed = (a[:, 0].astype(np.int32) | (a[:, 1].astype(np.int32) << 8) | (a[:, 2].astype(np.int32) << 16))
        signed = (signed ^ 0x800000) - 0x800000
        data = signed.astype(np.float32) / 8388608.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    return np.clip(data, -1.0, 1.0).astype(np.float32)


def iter_wav_mono_samples(
    path: str,
    *,
    block_frames: int = 4096,
) -> Iterator[np.ndarray]:
    """Yield mono float32 blocks from a WAV file."""
    p = Path(path)
    with wave.open(str(p), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        while True:
            raw = wf.readframes(block_frames)
            if not raw:
                break
            yield _bytes_to_float_mono(raw, channels=channels, sample_width=sample_width)


def dbfs_from_amp(amp: float) -> float:
    if amp <= 0:
        return -120.0
    return 20.0 * math.log10(min(1.0, max(amp, 1e-12)))


def analyze_wav_loudness(path: str) -> dict[str, float]:
    """Compute basic loudness metrics for a WAV file."""
    peak = 0.0
    sumsq = 0.0
    count = 0

    for block in iter_wav_mono_samples(path):
        if block.size == 0:
            continue
        abs_block = np.abs(block)
        peak = max(peak, float(abs_block.max(initial=0.0)))
        sumsq += float(np.square(block).sum())
        count += int(block.size)

    rms = math.sqrt(sumsq / count) if count else 0.0
    peak_db = dbfs_from_amp(peak)
    rms_db = dbfs_from_amp(rms)
    dynamic_range_db = max(0.0, peak_db - rms_db)

    return {
        "peak_db": peak_db,
        "rms_db": rms_db,
        "dynamic_range_db": dynamic_range_db,
    }


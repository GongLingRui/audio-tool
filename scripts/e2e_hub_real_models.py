#!/usr/bin/env python
"""End-to-end smoke test for the voice-studio-hub feature set.

Runs the same endpoints the frontend uses, with real audio files generated on
macOS via `say` and converted via `ffmpeg`.

Usage:
  python backend/scripts/e2e_hub_real_models.py

Optional env vars:
  ASR_BACKEND=faster_whisper|whisper|basic
  DIAR_BACKEND=pyannote|speechbrain|basic
  HF_TOKEN=...                    (for pyannote, if installed)
  RVC_MODEL_PTH=/abs/path/model.pth
  RVC_INDEX_FILE=/abs/path/model.index
  RVC_MODEL_ID=my_model
  RVC_MODEL_NAME="My Model"
  RVC_PROJECT_PATH=/abs/path/to/rvc/repo  (enables real RVC inference if supported)
  RUN_QUANT=1                      (runs quantization smoke if torch installed)
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path
import sys

from httpx import ASGITransport, AsyncClient


def require_cmd(name: str) -> None:
    from shutil import which

    if which(name) is None:
        raise SystemExit(f"Missing required command: {name}")


async def main() -> None:
    require_cmd("ffmpeg")
    require_cmd("ffprobe")
    require_cmd("say")

    # Ensure backend root is importable (app/ package lives under backend/app).
    backend_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(backend_root))

    # Import app late so env vars can be set before Settings instantiation if needed.
    from app.main import app

    tmpdir = Path(tempfile.gettempdir()) / "voiceforge_e2e"
    tmpdir.mkdir(parents=True, exist_ok=True)

    # Generate speech audio using macOS TTS (real audio, no network).
    aiff = tmpdir / "speech.aiff"
    aiff2 = tmpdir / "speech2.aiff"
    wav = tmpdir / "speech.wav"
    mp3 = tmpdir / "speech.mp3"
    m4a = tmpdir / "speech.m4a"
    flac = tmpdir / "speech.flac"
    wav2 = tmpdir / "speech2.wav"
    mp3_2 = tmpdir / "speech2.mp3"
    zh_aiff = tmpdir / "zh.aiff"
    zh_wav = tmpdir / "zh.wav"
    zh_mp3 = tmpdir / "zh.mp3"

    text = "Hello world. This is a test."
    subprocess.check_call(["say", "-v", "Samantha", "-o", str(aiff), text])
    subprocess.check_call(["say", "-v", "Alex", "-o", str(aiff2), text])
    subprocess.check_call(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(aiff), str(wav)])
    subprocess.check_call(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav), str(mp3)])
    subprocess.check_call(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav), str(m4a)])
    subprocess.check_call(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav), str(flac)])
    subprocess.check_call(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(aiff2), str(wav2)])
    subprocess.check_call(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav2), str(mp3_2)])

    # Chinese sample (used for ModelScope/FunASR diarization which relies on timestamp-capable models)
    subprocess.check_call(["say", "-v", "Ting-Ting", "-o", str(zh_aiff), "你好，这是一个测试。我们在做说话人分离。"])
    subprocess.check_call(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(zh_aiff), str(zh_wav)])
    subprocess.check_call(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(zh_wav), str(zh_mp3)])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Dashboard
        r = await client.get("/api/dashboard/tasks?limit=3")
        r.raise_for_status()

        # Backend capability probe
        r = await client.get("/api/diarization/backends")
        r.raise_for_status()

        # Dialect
        r = await client.post("/api/dialect/detect-language", data={"text": "这里什么都没有。Hello."})
        r.raise_for_status()
        r = await client.post("/api/dialect/convert", data={"text": "这里有什么？", "target_dialect": "zh-HK"})
        r.raise_for_status()
        r = await client.get("/api/dialect/supported")
        r.raise_for_status()

        # Audio quality + ASR + diarization on real non-WAV formats.
        audio_cases = [
            ("mp3", mp3, "audio/mpeg"),
            ("m4a", m4a, "audio/mp4"),
            ("flac", flac, "audio/flac"),
        ]

        asr_backend = os.getenv("ASR_BACKEND", "faster_whisper")
        diar_backend = os.getenv("DIAR_BACKEND", "speechbrain")

        for ext, path, content_type in audio_cases:
            # Audio quality
            with path.open("rb") as f:
                r = await client.post(
                    "/api/audio-quality/check",
                    files={"file": (f"speech.{ext}", f, content_type)},
                    data={"detailed": "true"},
                )
            r.raise_for_status()

            # ASR (print once)
            with path.open("rb") as f:
                r = await client.post(
                    "/api/asr/transcribe",
                    files={"audio_file": (f"speech.{ext}", f, content_type)},
                    data={
                        "backend": asr_backend,
                        "language": "en",
                        "timestamps": "segment",
                        "vad_filter": "true",
                    },
                )
            r.raise_for_status()
            if ext == "mp3":
                asr = r.json()["data"]
                print("ASR backend:", asr.get("backend"), "text:", (asr.get("text") or "")[:80])

            # Diarization
            with path.open("rb") as f:
                r = await client.post(
                    "/api/diarization/enhanced",
                    files={"audio_file": (f"speech.{ext}", f, content_type)},
                    data={"backend": diar_backend, "min_speakers": "1", "max_speakers": "2"},
                )
            r.raise_for_status()

        # Optional: pyannote diarization if HF token is provided in env
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            with mp3.open("rb") as f:
                r = await client.post(
                    "/api/diarization/enhanced",
                    files={"audio_file": ("speech.mp3", f, "audio/mpeg")},
                    data={"backend": "pyannote", "min_speakers": "1", "max_speakers": "2", "huggingface_token": hf_token},
                )
            r.raise_for_status()

        # Optional: ModelScope/FunASR diarization (large models download on first use)
        if os.getenv("RUN_MODELSCOPE_DIAR") == "1":
            with zh_mp3.open("rb") as f:
                r = await client.post(
                    "/api/diarization/enhanced",
                    files={"audio_file": ("zh.mp3", f, "audio/mpeg")},
                    data={"backend": "modelscope", "min_speakers": "1", "max_speakers": "2"},
                )
            r.raise_for_status()

        # ASR Whisper backend (real model) on mp3 with word timestamps
        with mp3.open("rb") as f:
            r = await client.post(
                "/api/asr/transcribe",
                files={"audio_file": ("speech.mp3", f, "audio/mpeg")},
                data={
                    "backend": "whisper",
                    "language": "en",
                    "timestamps": "word",
                    "vad_filter": "true",
                },
            )
        r.raise_for_status()

        # Diarization embeddings (uses best available backend)
        with mp3.open("rb") as f:
            r = await client.post(
                "/api/diarization/embeddings",
                files={"audio_file": ("speech.mp3", f, "audio/mpeg")},
                data={"min_speakers": "1", "max_speakers": "2"},
            )
        r.raise_for_status()

        # Compare speakers between two different TTS voices
        with mp3.open("rb") as f1, mp3_2.open("rb") as f2:
            r = await client.post(
                "/api/diarization/compare-speakers",
                files={
                    "audio1": ("speech1.mp3", f1, "audio/mpeg"),
                    "audio2": ("speech2.mp3", f2, "audio/mpeg"),
                },
            )
        r.raise_for_status()

        # RVC: optional real-model test (requires real model file)
        r = await client.get("/api/rvc/models")
        r.raise_for_status()
        if os.getenv("RVC_MODEL_PTH"):
            model_pth = Path(os.environ["RVC_MODEL_PTH"])
            model_id = os.getenv("RVC_MODEL_ID", model_pth.stem)
            model_name = os.getenv("RVC_MODEL_NAME", model_id)
            index_path = os.getenv("RVC_INDEX_FILE")

            files = {"model_file": (model_pth.name, model_pth.read_bytes(), "application/octet-stream")}
            if index_path:
                ip = Path(index_path)
                files["index_file"] = (ip.name, ip.read_bytes(), "application/octet-stream")

            r = await client.post(
                "/api/rvc/models/upload",
                files=files,
                data={
                    "model_id": model_id,
                    "name": model_name,
                    "description": "e2e-upload",
                    "language": "en-US",
                    "gender": "female",
                },
            )
            r.raise_for_status()

            # Convert (will be real only if backend is configured with an RVC project path)
            with mp3.open("rb") as f:
                r = await client.post(
                    "/api/rvc/convert",
                    files={"audio_file": ("speech.mp3", f, "audio/mpeg")},
                    data={"model_id": model_id, "pitch_shift": "0", "f0_method": "rmvpe"},
                )
            r.raise_for_status()
            assert r.headers.get("content-type", "").startswith("audio/")

        # Quantization smoke (real torch model object)
        try:
            import torch
            import torch.nn as nn
        except Exception as e:
            print("Skipping quantization smoke: torch not available:", e)
        else:
            model = nn.Linear(8, 8)
            model.eval()
            model_path = tmpdir / "tiny_model.pt"
            torch.save(model, model_path)
            with model_path.open("rb") as f:
                r = await client.post(
                    "/api/quantization/quantize",
                    files={"model_file": ("tiny_model.pt", f, "application/octet-stream")},
                    data={"quantization_type": "dynamic", "model_format": "pytorch"},
                )
            r.raise_for_status()
            payload = r.json()
            assert payload.get("data", {}).get("success") is True, payload

    print("E2E hub smoke OK:", tmpdir)


if __name__ == "__main__":
    asyncio.run(main())

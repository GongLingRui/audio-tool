"""
Enhanced Speaker Diarization Service with pyannote.audio Integration
Advanced speaker diarization with professional model support
"""
import asyncio
import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import importlib.util
try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class DiarizationBackend(Enum):
    """Available diarization backends."""
    PYANNOTE = "pyannote"  # Professional, requires HuggingFace token
    SPEECHBRAIN = "speechbrain"  # SpeechBrain ECAPA
    MODELSCOPE = "modelscope"  # ModelScope/FunASR speaker diarization (real models)
    BASIC = "basic"  # Fallback basic implementation


@dataclass
class SpeakerSegment:
    """A segment with speaker information."""
    speaker: str  # Speaker ID (SPEAKER_00, SPEAKER_01, etc.)
    start_time: float  # Start time in seconds
    end_time: float  # End time in seconds
    confidence: float  # Confidence score (0-1)
    text: Optional[str] = None  # Transcribed text (if available)


@dataclass
class DiarizationResult:
    """Result of speaker diarization."""
    segments: List[SpeakerSegment]
    num_speakers: int
    duration: float
    processing_time: float
    backend: DiarizationBackend
    metadata: Dict[str, Any]
    confidence: float  # Overall confidence

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "segments": [
                {
                    "speaker": s.speaker,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "confidence": s.confidence,
                    "text": s.text,
                }
                for s in self.segments
            ],
            "num_speakers": self.num_speakers,
            "duration": self.duration,
            "processing_time": self.processing_time,
            "backend": self.backend.value,
            "metadata": self.metadata,
            "confidence": self.confidence,
        }


class EnhancedSpeakerDiarizationService:
    """
    Enhanced speaker diarization service with professional model support.

    Features:
    - pyannote.audio integration (professional grade)
    - SpeechBrain ECAPA support
    - Automatic speaker count detection
    - Speaker embedding extraction
    - Overlapping speech detection
    - Multi-language support
    """

    def __init__(
        self,
        preferred_backend: DiarizationBackend = DiarizationBackend.PYANNOTE,
        huggingface_token: Optional[str] = None,
        model_cache_dir: str = "./data/models/diarization",
    ):
        """
        Initialize enhanced diarization service.

        Args:
            preferred_backend: Preferred diarization backend
            huggingface_token: HuggingFace token for pyannote models
            model_cache_dir: Directory to cache models
        """
        self.preferred_backend = preferred_backend
        self.huggingface_token = huggingface_token
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized enhanced diarization service with backend: {preferred_backend.value}")

        # Lazy loaded models
        self._pyannote_model = None
        self._speechbrain_model = None
        self._modelscope_model = None

    def _torch_device(self) -> str:
        if not _TORCH_AVAILABLE:
            return "cpu"
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _load_mono_audio_16k(self, audio_path: str) -> Tuple[np.ndarray, int, float]:
        """Decode audio into mono float32 16kHz samples.

        Prefers system ffmpeg for broad format support; falls back to WAV-only.
        Returns (samples, sample_rate, duration_seconds).
        """
        path = Path(audio_path)

        target_sr = 16000
        if path.suffix.lower() != ".wav":
            from app.utils.audio_decode import ffmpeg_available
            if not ffmpeg_available():
                raise RuntimeError("ffmpeg not available for non-WAV diarization")

        from app.utils.audio_decode import ffmpeg_available, iter_audio_mono_float32, probe_audio
        from app.utils.wav_audio import read_wav_info, iter_wav_mono_samples

        if ffmpeg_available():
            # Stream decode and join (keeps memory moderate for typical short files).
            chunks: list[np.ndarray] = []
            for block in iter_audio_mono_float32(audio_path, sample_rate=target_sr, chunk_samples=target_sr * 5):
                if block.size:
                    chunks.append(block)
            samples = np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.float32)
            duration = float(samples.size) / float(target_sr) if samples.size else 0.0
            try:
                info = probe_audio(audio_path)
                duration = info.duration or duration
            except Exception:
                pass
            return samples, target_sr, float(duration)

        # WAV-only fallback (no resample).
        info = read_wav_info(audio_path)
        sr = info.sample_rate or target_sr
        chunks = [block for block in iter_wav_mono_samples(audio_path, block_frames=sr * 5) if block.size]
        samples = np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.float32)
        duration = info.duration or (float(samples.size) / float(sr) if samples.size else 0.0)
        return samples, sr, float(duration)

    async def _get_pyannote_model(self):
        """Lazy load pyannote.audio model."""
        if self._pyannote_model is None:
            try:
                from pyannote.audio import Pipeline

                # Load pipeline with token
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=self.huggingface_token,
                )

                # Prefer MPS on Apple Silicon when available.
                if not _TORCH_AVAILABLE:
                    raise RuntimeError("torch not available for pyannote runtime")
                pipeline.to(torch.device(self._torch_device()))

                self._pyannote_model = pipeline
                self.logger.info("pyannote.audio model loaded")

            except ImportError:
                self.logger.warning("pyannote.audio not installed")
                self._pyannote_model = False
            except Exception as e:
                self.logger.error(f"Error loading pyannote: {e}")
                self._pyannote_model = False

        return self._pyannote_model if self._pyannote_model is not False else None

    async def _get_speechbrain_model(self):
        """Lazy load SpeechBrain model."""
        if self._speechbrain_model is None:
            try:
                # SpeechBrain (as of 1.0.x) still expects legacy torchaudio backend helpers.
                # Newer torchaudio releases may not expose these at top-level, so we provide
                # minimal shims to keep model loading working.
                try:
                    import torchaudio  # type: ignore

                    if not hasattr(torchaudio, "list_audio_backends"):
                        torchaudio.list_audio_backends = lambda: []  # type: ignore[attr-defined]
                    if not hasattr(torchaudio, "get_audio_backend"):
                        torchaudio.get_audio_backend = lambda: None  # type: ignore[attr-defined]
                    if not hasattr(torchaudio, "set_audio_backend"):
                        torchaudio.set_audio_backend = lambda _backend=None: None  # type: ignore[attr-defined]
                except Exception:
                    pass

                # SpeechBrain may still call older huggingface_hub APIs.
                # Map legacy `use_auth_token=` to `token=` for newer hub versions.
                try:
                    import huggingface_hub  # type: ignore

                    _orig_hf_hub_download = getattr(huggingface_hub, "hf_hub_download", None)
                    if callable(_orig_hf_hub_download):
                        def _hf_hub_download_compat(*args, **kwargs):  # type: ignore
                            filename = kwargs.get("filename")
                            if filename is None and len(args) >= 2:
                                filename = args[1]
                            if "use_auth_token" in kwargs and "token" not in kwargs:
                                kwargs["token"] = kwargs.pop("use_auth_token")
                            try:
                                return _orig_hf_hub_download(*args, **kwargs)
                            except Exception as e:
                                # Some SpeechBrain model repos do not include `custom.py`, but SpeechBrain's
                                # downloader still tries to fetch it. Provide an empty placeholder so the
                                # rest of the model can load.
                                if filename == "custom.py" and "Entry Not Found" in str(e):
                                    placeholder = self.model_cache_dir / "speechbrain_custom.py"
                                    if not placeholder.exists():
                                        placeholder.write_text(
                                            "# Auto-generated placeholder for SpeechBrain model loading.\n",
                                            encoding="utf-8",
                                        )
                                    return str(placeholder)
                                raise

                        huggingface_hub.hf_hub_download = _hf_hub_download_compat  # type: ignore[attr-defined]
                except Exception:
                    pass

                from speechbrain.inference.speaker import SpeakerRecognition

                if not _TORCH_AVAILABLE:
                    raise RuntimeError("torch not available for SpeechBrain runtime")

                device = self._torch_device()
                verification = SpeakerRecognition.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=str(self.model_cache_dir / "speechbrain_spkrec_ecapa_voxceleb"),
                    run_opts={"device": device},
                )

                self._speechbrain_model = verification
                self.logger.info("SpeechBrain model loaded")

            except ImportError:
                self.logger.warning("speechbrain not installed")
                self._speechbrain_model = False
            except Exception as e:
                self.logger.error(f"Error loading SpeechBrain: {e}")
                self._speechbrain_model = False

        return self._speechbrain_model if self._speechbrain_model is not False else None

    async def _get_modelscope_model(self):
        """Lazy load ModelScope/FunASR diarization model bundle."""
        if self._modelscope_model is None:
            try:
                from funasr import AutoModel  # type: ignore

                device = self._torch_device()
                # FunASR uses torch device strings; keep CPU fallback for safety.
                # This bundle provides timestamps + punctuation + speaker clustering via CAM++.
                self._modelscope_model = AutoModel(
                    model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                    vad_model="fsmn-vad",
                    punc_model="ct-punc",
                    spk_model="cam++",
                    spk_mode="punc_segment",
                    hub="ms",
                    device=device,
                    disable_update=True,
                    log_level="WARNING",
                )
                self.logger.info("ModelScope/FunASR diarization bundle loaded")
            except ImportError:
                self.logger.warning("funasr/modelscope not installed")
                self._modelscope_model = False
            except Exception as e:
                self.logger.error(f"Error loading ModelScope/FunASR diarization bundle: {e}")
                self._modelscope_model = False

        return self._modelscope_model if self._modelscope_model is not False else None

    async def diarize(
        self,
        audio_path: str,
        min_speakers: int = 1,
        max_speakers: int = 10,
        backend: Optional[DiarizationBackend] = None,
    ) -> DiarizationResult:
        """
        Perform speaker diarization on audio file.

        Args:
            audio_path: Path to audio file
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            backend: Force specific backend

        Returns:
            DiarizationResult with segments
        """
        start_time = time.time()
        backend = backend or self.preferred_backend

        try:
            if backend == DiarizationBackend.PYANNOTE:
                result = await self._diarize_pyannote(
                    audio_path, min_speakers, max_speakers
                )
            elif backend == DiarizationBackend.SPEECHBRAIN:
                result = await self._diarize_speechbrain(
                    audio_path, min_speakers, max_speakers
                )
            elif backend == DiarizationBackend.MODELSCOPE:
                result = await self._diarize_modelscope(
                    audio_path, min_speakers, max_speakers
                )
            else:
                result = await self._diarize_basic(
                    audio_path, min_speakers, max_speakers
                )

            processing_time = time.time() - start_time
            result.processing_time = processing_time

            return result

        except Exception as e:
            self.logger.error(f"Diarization error: {e}")
            # Fallback to basic implementation
            self.logger.warning("Falling back to basic diarization")
            return await self._diarize_basic(
                audio_path, min_speakers, max_speakers
            )

    async def _diarize_pyannote(
        self,
        audio_path: str,
        min_speakers: int,
        max_speakers: int,
    ) -> DiarizationResult:
        """Diarize using pyannote.audio."""
        model = await self._get_pyannote_model()

        if model is None:
            raise RuntimeError("pyannote.audio not available")

        # Load audio
        from pyannote.audio import Audio
        audio = Audio(str(audio_path))

        # Apply diarization pipeline
        diarization = model(
            audio,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        # Convert to segments
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(SpeakerSegment(
                speaker=str(speaker),
                start_time=turn.start,
                end_time=turn.end,
                confidence=1.0,  # pyannote provides confidence
            ))

        return DiarizationResult(
            segments=segments,
            num_speakers=len(set(s.speaker for s in segments)),
            duration=audio.duration,
            processing_time=0.0,  # Will be set by caller
            backend=DiarizationBackend.PYANNOTE,
            metadata={"model": "pyannote/speaker-diarization-3.1"},
            confidence=1.0,
        )

    async def _diarize_speechbrain(
        self,
        audio_path: str,
        min_speakers: int,
        max_speakers: int,
    ) -> DiarizationResult:
        """Diarize using SpeechBrain speaker embeddings + clustering.

        Notes:
        - Uses a sliding-window embedding extractor (ECAPA) and agglomerative clustering.
        - This is not as accurate as a full pyannote pipeline, but it is a real,
          local, ungated model that works out-of-the-box with common audio formats
          when ffmpeg is installed.
        """
        model = await self._get_speechbrain_model()

        if model is None:
            raise RuntimeError("SpeechBrain not available")

        try:
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics import silhouette_score
        except Exception as e:
            raise RuntimeError(f"scikit-learn required for SpeechBrain diarization: {e}")

        samples, sample_rate, duration = self._load_mono_audio_16k(audio_path)
        if samples.size == 0:
            return DiarizationResult(
                segments=[SpeakerSegment(speaker="SPEAKER_00", start_time=0.0, end_time=0.0, confidence=0.0)],
                num_speakers=1,
                duration=0.0,
                processing_time=0.0,
                backend=DiarizationBackend.SPEECHBRAIN,
                metadata={"model": "speechbrain/spkrec-ecapa-voxceleb", "method": "empty_audio"},
                confidence=0.0,
            )

        # Sliding windows (seconds)
        window_sec = 1.5
        hop_sec = 0.75
        win = max(1, int(window_sec * sample_rate))
        hop = max(1, int(hop_sec * sample_rate))

        # Basic silence filter
        rms_floor = 0.008
        windows: list[tuple[float, float, np.ndarray]] = []
        for start in range(0, max(0, samples.size - win + 1), hop):
            chunk = samples[start:start + win]
            if chunk.size < win:
                break
            rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
            if rms < rms_floor:
                continue
            st = float(start) / float(sample_rate)
            et = float(start + win) / float(sample_rate)
            windows.append((st, et, chunk))

        if not windows:
            segments = [
                SpeakerSegment(
                    speaker="SPEAKER_00",
                    start_time=0.0,
                    end_time=float(duration),
                    confidence=0.4,
                    text="音频能量过低，无法进行可靠的说话人聚类",
                )
            ]
            return DiarizationResult(
                segments=segments,
                num_speakers=1,
                duration=float(duration),
                processing_time=0.0,
                backend=DiarizationBackend.SPEECHBRAIN,
                metadata={"model": "speechbrain/spkrec-ecapa-voxceleb", "method": "low_energy"},
                confidence=0.4,
            )

        # Compute embeddings
        device = self._torch_device()
        embeddings: list[np.ndarray] = []

        def encode_batch(wavs: np.ndarray) -> np.ndarray:
            if not _TORCH_AVAILABLE:
                raise RuntimeError("torch not available for SpeechBrain diarization")
            with torch.no_grad():  # type: ignore[union-attr]
                sig = torch.from_numpy(wavs).to(device=device)  # type: ignore[union-attr]
                if sig.dim() == 1:
                    sig = sig.unsqueeze(0)
                try:
                    emb = model.encode_batch(sig)  # type: ignore[attr-defined]
                except Exception:
                    # Fallback for older SpeechBrain inference objects
                    emb = model.model.encode_batch(sig)  # type: ignore[attr-defined]
                emb_np = emb.detach().to("cpu").numpy()
                # Normalize to 1D embedding for a single chunk.
                if emb_np.ndim == 3:
                    emb_np = emb_np[0, 0, :]
                elif emb_np.ndim == 2:
                    emb_np = emb_np[0, :]
                else:
                    emb_np = emb_np.reshape(-1)
                return emb_np.astype(np.float32, copy=False)

        for _, _, chunk in windows:
            try:
                emb = encode_batch(chunk.astype(np.float32, copy=False))
            except Exception:
                # If MPS fails, retry on CPU once.
                if device != "cpu":
                    try:
                        device = "cpu"
                        emb = encode_batch(chunk.astype(np.float32, copy=False))
                    except Exception as e:
                        raise RuntimeError(f"SpeechBrain embedding failed: {e}")
                else:
                    raise
            embeddings.append(np.asarray(emb, dtype=np.float32))

        X = np.stack(embeddings, axis=0) if embeddings else np.zeros((0, 192), dtype=np.float32)

        # Choose number of speakers.
        k_min = max(1, int(min_speakers))
        k_max = max(k_min, int(max_speakers))
        k_max = min(k_max, int(X.shape[0]))

        def cluster_labels(k: int) -> np.ndarray:
            try:
                clustering = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
            except TypeError:  # older sklearn
                clustering = AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage="average")
            return clustering.fit_predict(X)

        best_k = 1
        best_score = -1.0
        best_labels = np.zeros((X.shape[0],), dtype=np.int64)

        if k_min == 1 and k_max == 1:
            best_k = 1
            best_labels = best_labels
        else:
            for k in range(k_min, k_max + 1):
                if k <= 1 or k >= X.shape[0]:
                    continue
                labels = cluster_labels(k)
                # Silhouette with cosine metric.
                try:
                    score = float(silhouette_score(X, labels, metric="cosine"))
                except Exception:
                    score = -1.0
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels.astype(np.int64)

            # If silhouette couldn't decide, fall back to k_min.
            if best_score < 0.0 and k_min > 1:
                best_k = k_min
                best_labels = cluster_labels(best_k).astype(np.int64)

        # Merge contiguous windows by label.
        segments: list[SpeakerSegment] = []
        last_label: Optional[int] = None
        seg_start = 0.0
        seg_end = 0.0
        for (st, et, _), label in zip(windows, best_labels):
            if last_label is None:
                last_label = int(label)
                seg_start = st
                seg_end = et
                continue
            if int(label) == last_label and st <= seg_end + 0.01:
                seg_end = max(seg_end, et)
                continue
            segments.append(
                SpeakerSegment(
                    speaker=f"SPEAKER_{last_label:02d}",
                    start_time=round(seg_start, 3),
                    end_time=round(seg_end, 3),
                    confidence=0.8,
                )
            )
            last_label = int(label)
            seg_start = st
            seg_end = et

        if last_label is not None:
            segments.append(
                SpeakerSegment(
                    speaker=f"SPEAKER_{last_label:02d}",
                    start_time=round(seg_start, 3),
                    end_time=round(seg_end, 3),
                    confidence=0.8,
                )
            )

        return DiarizationResult(
            segments=segments,
            num_speakers=min(best_k, max_speakers) if best_k > 0 else 1,
            duration=float(duration),
            processing_time=0.0,
            backend=DiarizationBackend.SPEECHBRAIN,
            metadata={
                "model": "speechbrain/spkrec-ecapa-voxceleb",
                "sample_rate": sample_rate,
                "window_sec": window_sec,
                "hop_sec": hop_sec,
                "num_windows": len(windows),
                "device": device,
                "k_selected": best_k,
                "silhouette": best_score if best_score >= 0.0 else None,
            },
            confidence=0.8,
        )

    async def _diarize_modelscope(
        self,
        audio_path: str,
        min_speakers: int,
        max_speakers: int,
    ) -> DiarizationResult:
        """Diarize using ModelScope/FunASR bundle (ASR+timestamp+SPK clustering)."""
        model = await self._get_modelscope_model()
        if model is None:
            raise RuntimeError("ModelScope/FunASR diarization not available")

        # FunASR expects a local file path; prefer WAV for best compatibility.
        from app.utils.audio_decode import probe_audio

        info = None
        try:
            info = probe_audio(str(audio_path))
        except Exception:
            info = None

        # Run inference in a thread to avoid blocking the event loop (it is CPU/GPU heavy).
        loop = asyncio.get_running_loop()

        def _run():
            # return_spk_res ensures `sentence_info` with `spk` labels.
            return model.generate(input=str(audio_path), return_spk_res=True)

        res = await loop.run_in_executor(None, _run)
        if not isinstance(res, list) or not res:
            raise RuntimeError("ModelScope/FunASR returned empty result")
        r0 = res[0]

        sentence_info = r0.get("sentence_info") or []
        segments: list[SpeakerSegment] = []
        for s in sentence_info:
            try:
                st_ms = float(s.get("start", 0.0))
                et_ms = float(s.get("end", 0.0))
                spk = int(s.get("spk", 0))
                text = s.get("text") or None
            except Exception:
                continue
            if et_ms <= st_ms:
                continue
            segments.append(
                SpeakerSegment(
                    speaker=f"SPEAKER_{spk:02d}",
                    start_time=round(st_ms / 1000.0, 3),
                    end_time=round(et_ms / 1000.0, 3),
                    confidence=0.9,
                    text=text,
                )
            )

        # Fallback if sentence_info missing (should not happen when return_spk_res=True).
        if not segments:
            # Provide a single segment to keep UI usable.
            dur = float(info.duration) if info and info.duration else 0.0
            segments = [SpeakerSegment(speaker="SPEAKER_00", start_time=0.0, end_time=dur, confidence=0.4)]

        duration = 0.0
        if info and info.duration:
            duration = float(info.duration)
        else:
            duration = max((s.end_time for s in segments), default=0.0)

        return DiarizationResult(
            segments=segments,
            num_speakers=min(len(set(s.speaker for s in segments)), max_speakers),
            duration=duration,
            processing_time=0.0,
            backend=DiarizationBackend.MODELSCOPE,
            metadata={
                "bundle": "funasr+modelscope",
                "model": "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                "vad_model": "fsmn-vad",
                "punc_model": "ct-punc",
                "spk_model": "cam++",
            },
            confidence=0.9,
        )

    async def _diarize_basic(
        self,
        audio_path: str,
        min_speakers: int,
        max_speakers: int,
    ) -> DiarizationResult:
        """Basic diarization fallback.

        Avoids pydub/audioop. If system ffmpeg is available, supports common
        formats via streaming decode; otherwise WAV-only.
        """
        from pathlib import Path

        path = Path(audio_path)
        duration = 0.0
        sample_rate = 16000

        if path.suffix.lower() == ".wav":
            from app.utils.wav_audio import read_wav_info, iter_wav_mono_samples

            info = read_wav_info(str(path))
            duration = info.duration
            sample_rate = info.sample_rate or 16000

            def sample_iter():
                for block in iter_wav_mono_samples(str(path), block_frames=4096):
                    yield block

        else:
            from app.utils.audio_decode import ffmpeg_available, iter_audio_mono_float32, probe_audio

            if not ffmpeg_available():
                return DiarizationResult(
                    segments=[
                        SpeakerSegment(
                            speaker="SPEAKER_00",
                            start_time=0.0,
                            end_time=0.0,
                            confidence=0.0,
                            text="需要安装 ffmpeg 才能解码非 WAV 音频",
                        )
                    ],
                    num_speakers=1,
                    duration=0.0,
                    processing_time=0.0,
                    backend=DiarizationBackend.BASIC,
                    metadata={"method": "unavailable_non_wav"},
                    confidence=0.0,
                )

            try:
                info = probe_audio(str(path))
                duration = info.duration
            except Exception:
                duration = 0.0

            def sample_iter():
                for block in iter_audio_mono_float32(str(path), sample_rate=sample_rate, chunk_samples=8192):
                    yield block

        # Simple segmentation based on energy contour (same idea as ASR basic).
        window_ms = 200
        hop_ms = 100
        window = max(1, int(sample_rate * (window_ms / 1000.0)))
        hop = max(1, int(sample_rate * (hop_ms / 1000.0)))

        energies: list[float] = []
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

        if not energies:
            segments = [
                SpeakerSegment(speaker="SPEAKER_00", start_time=0.0, end_time=duration, confidence=0.3)
            ]
            return DiarizationResult(
                segments=segments,
                num_speakers=1,
                duration=duration,
                processing_time=0.0,
                backend=DiarizationBackend.BASIC,
                metadata={"method": "basic_empty"},
                confidence=0.3,
            )

        median = float(np.median(np.array(energies, dtype=np.float32)))
        threshold = max(0.01, median * 2.0)

        speech_windows: list[tuple[float, float]] = []
        in_speech = False
        start_t = 0.0

        def t_at(i: int) -> float:
            return i * (hop_ms / 1000.0)

        min_speech_sec = 0.5
        for i, e in enumerate(energies):
            if e >= threshold and not in_speech:
                in_speech = True
                start_t = t_at(i)
            elif e < threshold and in_speech:
                in_speech = False
                end_t = t_at(i)
                if end_t - start_t >= min_speech_sec:
                    speech_windows.append((start_t, end_t))

        if in_speech:
            end_t = t_at(len(energies))
            if end_t - start_t >= min_speech_sec:
                speech_windows.append((start_t, end_t))

        segments: list[SpeakerSegment] = []
        if not speech_windows:
            segments = [SpeakerSegment(speaker="SPEAKER_00", start_time=0.0, end_time=duration, confidence=0.3)]
        else:
            for i, (st, et) in enumerate(speech_windows):
                speaker_idx = i % max(1, min(max_speakers, 4))
                segments.append(
                    SpeakerSegment(
                        speaker=f"SPEAKER_{speaker_idx:02d}",
                        start_time=round(st, 3),
                        end_time=round(et, 3),
                        confidence=0.5,
                    )
                )

        return DiarizationResult(
            segments=segments,
            num_speakers=min(len(set(s.speaker for s in segments)), max_speakers),
            duration=duration,
            processing_time=0.0,
            backend=DiarizationBackend.BASIC,
            metadata={"method": "basic_wave_vad"},
            confidence=0.5 if speech_windows else 0.3,
        )

    async def extract_speaker_embeddings(
        self,
        audio_path: str,
        diarization_result: DiarizationResult,
    ) -> Dict[str, Any]:
        """
        Extract speaker embeddings for each detected speaker.

        Args:
            audio_path: Path to audio file
            diarization_result: Diarization result with segments

        Returns:
            Dictionary mapping speaker IDs to embeddings
        """
        model = await self._get_speechbrain_model()
        if model is None:
            raise RuntimeError("SpeechBrain not available for embeddings")

        samples, sample_rate, _ = self._load_mono_audio_16k(audio_path)
        if samples.size == 0:
            return {}

        device = self._torch_device()

        # Group segments by speaker
        speaker_segments: Dict[str, List[SpeakerSegment]] = {}
        for seg in diarization_result.segments:
            if seg.speaker not in speaker_segments:
                speaker_segments[seg.speaker] = []
            speaker_segments[seg.speaker].append(seg)

        embeddings: Dict[str, list[float]] = {}

        # Extract embedding for each speaker
        for speaker, segs in speaker_segments.items():
            speaker_audio: list[np.ndarray] = []
            for seg in segs:
                st = max(0, int(seg.start_time * sample_rate))
                et = max(st + 1, int(seg.end_time * sample_rate))
                chunk = samples[st:et]
                if chunk.size:
                    speaker_audio.append(chunk)

            if not speaker_audio:
                continue

            wav = np.concatenate(speaker_audio).astype(np.float32, copy=False)
            if not _TORCH_AVAILABLE:
                continue

            with torch.no_grad():  # type: ignore[union-attr]
                sig = torch.from_numpy(wav).to(device=device)  # type: ignore[union-attr]
                sig = sig.unsqueeze(0)
                try:
                    emb = model.encode_batch(sig)  # type: ignore[attr-defined]
                except Exception:
                    emb = model.model.encode_batch(sig)  # type: ignore[attr-defined]
                emb_np = emb.detach().to("cpu").numpy()
            if emb_np.ndim == 3:
                emb_np = emb_np[0, 0, :]
            elif emb_np.ndim == 2:
                emb_np = emb_np[0, :]
            else:
                emb_np = emb_np.reshape(-1)
            embeddings[speaker] = np.asarray(emb_np, dtype=np.float32).flatten().tolist()

        return embeddings

    async def compare_speakers(
        self,
        audio1_path: str,
        audio2_path: str,
    ) -> Dict[str, Any]:
        """
        Compare two audio files to determine if they're from the same speaker.

        Args:
            audio1_path: First audio file
            audio2_path: Second audio file

        Returns:
            Comparison result with similarity score
        """
        model = await self._get_speechbrain_model()
        if model is None:
            return {"similarity": 0.0, "same_speaker": False, "error": "SpeechBrain not available"}

        try:
            a1, sr1, _ = self._load_mono_audio_16k(audio1_path)
            a2, sr2, _ = self._load_mono_audio_16k(audio2_path)
            if a1.size == 0 or a2.size == 0:
                return {"similarity": 0.0, "same_speaker": False, "error": "Empty audio"}
            device = self._torch_device()

            def emb(w: np.ndarray) -> np.ndarray:
                with torch.no_grad():  # type: ignore[union-attr]
                    sig = torch.from_numpy(w.astype(np.float32, copy=False)).to(device=device)  # type: ignore[union-attr]
                    sig = sig.unsqueeze(0)
                    try:
                        e = model.encode_batch(sig)  # type: ignore[attr-defined]
                    except Exception:
                        e = model.model.encode_batch(sig)  # type: ignore[attr-defined]
                    e_np = e.detach().to("cpu").numpy()
                    if e_np.ndim == 3:
                        e_np = e_np[0, 0, :]
                    elif e_np.ndim == 2:
                        e_np = e_np[0, :]
                    else:
                        e_np = e_np.reshape(-1)
                    return np.asarray(e_np, dtype=np.float32).flatten()

            emb1 = emb(a1)
            emb2 = emb(a2)

            dot = float(np.dot(emb1, emb2))
            n1 = float(np.linalg.norm(emb1))
            n2 = float(np.linalg.norm(emb2))
            similarity = 0.0 if n1 == 0.0 or n2 == 0.0 else dot / (n1 * n2)
            same = similarity > 0.75
            return {
                "similarity": float(similarity),
                "same_speaker": bool(same),
                "confidence": float(min(1.0, abs(similarity - 0.5) * 2.0)),
                "sample_rate": 16000,
            }
        except Exception as e:
            self.logger.error(f"Speaker comparison error: {e}")
            return {"similarity": 0.0, "same_speaker": False, "error": str(e)}

    def get_supported_backends(self) -> List[Dict[str, Any]]:
        """Get list of supported backends."""
        try:
            has_pyannote = importlib.util.find_spec("pyannote.audio") is not None
        except ModuleNotFoundError:
            has_pyannote = False
        try:
            has_speechbrain = importlib.util.find_spec("speechbrain") is not None
        except ModuleNotFoundError:
            has_speechbrain = False
        try:
            has_funasr = importlib.util.find_spec("funasr") is not None
        except ModuleNotFoundError:
            has_funasr = False
        has_torch = _TORCH_AVAILABLE

        backends = [
            {
                "name": "pyannote",
                "display_name": "pyannote.audio",
                "description": "Professional-grade diarization",
                "requires": "HuggingFace token",
                "available": bool(has_pyannote and has_torch),
                "requires_token": True,
            },
            {
                "name": "speechbrain",
                "display_name": "SpeechBrain ECAPA",
                "description": "Research-grade diarization",
                "requires": "speechbrain package",
                "available": bool(has_speechbrain and has_torch),
                "requires_token": False,
            },
            {
                "name": "modelscope",
                "display_name": "ModelScope (FunASR)",
                "description": "Real local diarization via ModelScope models (large download on first use)",
                "requires": "pip install funasr modelscope",
                "available": bool(has_funasr and has_torch),
                "requires_token": False,
            },
            {
                "name": "basic",
                "display_name": "Basic",
                "description": "Energy-based detection (fallback)",
                "requires": "None",
                "available": True,
                "requires_token": False,
            },
        ]

        return backends

    async def get_installation_instructions(self, backend: str) -> str:
        """Get installation instructions for a backend."""
        instructions = {
            "pyannote": """
# Install pyannote.audio
pip install pyannote.audio

# You also need to accept the user conditions on hf.co
# Visit: https://hf.co/pyannote/speaker-diarization-3.1
# And accept the conditions to access the model

# Then set your HuggingFace token:
export HF_TOKEN="your_token_here"
""",
            "speechbrain": """
# Install SpeechBrain
pip install speechbrain

# The model will be downloaded automatically on first use
""",
            "basic": """
# Basic backend is always available
# No installation needed
""",
        }

        return instructions.get(backend, "Unknown backend")


# Global instance
_enhanced_diarization_service: Optional[EnhancedSpeakerDiarizationService] = None


def get_enhanced_diarization_service(
    backend: Optional[DiarizationBackend] = None,
    huggingface_token: Optional[str] = None,
) -> EnhancedSpeakerDiarizationService:
    """Get global enhanced diarization service instance."""
    if huggingface_token is None:
        import os
        huggingface_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    if backend is None:
        try:
            has_pyannote = importlib.util.find_spec("pyannote.audio") is not None
        except ModuleNotFoundError:
            has_pyannote = False
        try:
            has_speechbrain = importlib.util.find_spec("speechbrain") is not None
        except ModuleNotFoundError:
            has_speechbrain = False
        try:
            has_funasr = importlib.util.find_spec("funasr") is not None
        except ModuleNotFoundError:
            has_funasr = False

        if has_pyannote and _TORCH_AVAILABLE and huggingface_token:
            backend = DiarizationBackend.PYANNOTE
        elif has_funasr and _TORCH_AVAILABLE:
            backend = DiarizationBackend.MODELSCOPE
        elif has_speechbrain and _TORCH_AVAILABLE:
            backend = DiarizationBackend.SPEECHBRAIN
        else:
            backend = DiarizationBackend.BASIC

    global _enhanced_diarization_service
    if _enhanced_diarization_service is None:
        _enhanced_diarization_service = EnhancedSpeakerDiarizationService(
            preferred_backend=backend,
            huggingface_token=huggingface_token,
        )
    return _enhanced_diarization_service

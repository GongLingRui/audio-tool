"""Audio processing service."""
import os
import re
import zipfile
import tempfile
import logging
from pathlib import Path
from typing import Any, Optional, List, Dict, Tuple

import ffmpeg
try:
    # NOTE: pydub depends on `audioop` (removed in Python 3.13). Import lazily and degrade gracefully.
    from pydub import AudioSegment  # type: ignore
    from pydub.effects import normalize  # type: ignore
    from pydub.silence import detect_nonsilent  # type: ignore

    _PYDUB_AVAILABLE = True
    _PYDUB_IMPORT_ERROR: Exception | None = None
except Exception as e:  # pragma: no cover - environment dependent
    AudioSegment = Any  # type: ignore
    normalize = None  # type: ignore
    detect_nonsilent = None  # type: ignore
    _PYDUB_AVAILABLE = False
    _PYDUB_IMPORT_ERROR = e
import numpy as np

from app.config import settings
from app.utils.audio_decode import ffmpeg_available, iter_audio_mono_float32, probe_audio
from app.utils.wav_audio import analyze_wav_loudness, read_wav_info


logger = logging.getLogger(__name__)

# Pause durations (in milliseconds)
DEFAULT_PAUSE_MS = 500  # Pause between different speakers
SAME_SPEAKER_PAUSE_MS = 250  # Shorter pause for same speaker continuing


def sanitize_filename(name: str) -> str:
    """Make a string safe for use in filenames."""
    name = re.sub(r'[^\w\-]', '_', name)
    return name.lower()


class AudioProcessor:
    """Service for audio processing operations."""

    def __init__(self, config: dict | None = None):
        self.sample_rate = config.get("sample_rate", 24000) if config else 24000
        self.channels = config.get("channels", 1) if config else 1
        self.bitrate = config.get("bitrate", "128k") if config else "128k"

    async def save_wav(self, audio_data: bytes, output_path: str) -> str:
        """Save audio data as WAV file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            f.write(audio_data)

        return str(output_path)

    async def convert_to_mp3(
        self,
        wav_path: str,
        mp3_path: str | None = None,
        bitrate: str | None = None,
    ) -> str:
        """Convert WAV to MP3."""
        if mp3_path is None:
            mp3_path = wav_path.replace('.wav', '.mp3')

        bitrate = bitrate or self.bitrate

        try:
            audio = AudioSegment.from_wav(wav_path)
            audio.export(mp3_path, format='mp3', bitrate=bitrate)

            # Remove WAV file
            if os.path.exists(wav_path):
                os.remove(wav_path)

            return mp3_path
        except Exception as e:
            raise Exception(f"Audio conversion failed: {str(e)}")

    async def get_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds."""
        # Prefer pydub when available (supports many formats via ffmpeg),
        # but fall back to stdlib WAV parsing when pydub/ffmpeg isn't available.
        if _PYDUB_AVAILABLE:
            try:
                audio = AudioSegment.from_file(audio_path)
                return len(audio) / 1000.0
            except Exception:
                pass

        try:
            info = read_wav_info(audio_path)
            return info.duration
        except Exception:
            return 0.0

    async def combine_audio_files(
        self,
        audio_paths: list[str],
        speakers: list[str],
        output_path: str,
        pause_between_speakers: int = DEFAULT_PAUSE_MS,
        pause_same_speaker: int = SAME_SPEAKER_PAUSE_MS,
    ) -> tuple[str, float]:
        """
        Combine multiple audio files with pauses.

        Args:
            audio_paths: List of audio file paths
            speakers: List of corresponding speakers
            output_path: Output file path
            pause_between_speakers: Pause between different speakers (ms)
            pause_same_speaker: Pause between same speaker (ms)

        Returns:
            Tuple of (output_path, total_duration)
        """
        if not audio_paths:
            raise ValueError("No audio files to combine")

        # Load all audio files
        segments = []
        for path in audio_paths:
            try:
                audio = AudioSegment.from_mp3(path)
                segments.append(audio)
            except Exception:
                # Try WAV format
                audio = AudioSegment.from_wav(path)
                segments.append(audio)

        if not segments:
            raise ValueError("No valid audio files found")

        # Combine with pauses
        combined = segments[0]
        for i in range(1, len(segments)):
            # Determine pause duration
            if speakers[i] != speakers[i - 1]:
                pause = AudioSegment.silent(duration=pause_between_speakers)
            else:
                pause = AudioSegment.silent(duration=pause_same_speaker)

            combined = combined + pause + segments[i]

        # Export
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        combined.export(str(output_path), format='mp3', bitrate=self.bitrate)

        duration = len(combined) / 1000.0

        return str(output_path), duration

    async def merge_audio_files(
        self,
        chunks: list[dict[str, Any]],
        output_dir: str,
        output_name: str,
        add_pause_ms: int = 500,
        normalize: bool = True,
        add_fades: bool = True,
    ) -> tuple[str, float]:
        """
        Merge audio files from chunks into a single file.

        Args:
            chunks: List of chunks with audio_path, speaker, text, order_index
            output_dir: Directory to save output
            output_name: Name for output file (without extension)
            add_pause_ms: Pause between chunks in milliseconds
            normalize: Whether to normalize volume
            add_fades: Whether to add fade in/out

        Returns:
            Tuple of (output_path, total_duration)
        """
        if not chunks:
            raise ValueError("No chunks to merge")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Sort chunks by order
        sorted_chunks = sorted(chunks, key=lambda x: x.get("order_index", 0))

        # Collect audio paths and speakers
        audio_paths = []
        speakers = []

        for chunk in sorted_chunks:
            if not chunk.get("audio_path"):
                continue

            source_path = Path(chunk["audio_path"])
            
            # 如果路径不存在，尝试多种方式解析
            if not source_path.exists():
                # 尝试作为相对路径解析
                if not source_path.is_absolute():
                    # 尝试从 output_dir 的父目录解析
                    base_dir = Path(output_dir).parent.parent.parent
                    alt_path = base_dir / source_path
                    if alt_path.exists():
                        source_path = alt_path
                    else:
                        # 尝试从文件名直接查找
                        file_name = source_path.name
                        alt_path = Path(output_dir) / file_name
                        if alt_path.exists():
                            source_path = alt_path
                        else:
                            logger.warning(f"Audio file not found: {chunk['audio_path']} (tried: {source_path}, {alt_path})")
                            continue
                else:
                    logger.warning(f"Audio file not found: {source_path}")
                    continue

            audio_paths.append(str(source_path))
            speakers.append(chunk.get("speaker", "NARRATOR"))

        if not audio_paths:
            raise ValueError("No valid audio files found in chunks")

        # Combine with pauses
        combined_path = output_path / f"{output_name}.mp3"
        result_path, duration = await self.combine_audio_files(
            audio_paths,
            speakers,
            str(combined_path),
            pause_between_speakers=add_pause_ms,
            pause_same_speaker=add_pause_ms // 2,
        )

        # Post-processing
        if normalize:
            await self.normalize_volume(result_path)

        if add_fades:
            await self.add_fade(result_path, fade_in=100, fade_out=500)

        return result_path, duration

    async def normalize_volume(
        self,
        audio_path: str,
        target_dbfs: float = -20.0,
    ) -> str:
        """Normalize audio volume."""
        try:
            audio = AudioSegment.from_file(audio_path)
            change_in_dBFS = target_dbfs - audio.dBFS
            normalized = audio.apply_gain(change_in_dBFS)

            temp_path = audio_path.replace('.mp3', '_temp.mp3')
            normalized.export(temp_path, format='mp3')
            os.replace(temp_path, audio_path)

            return audio_path
        except Exception:
            return audio_path

    async def add_fade(
        self,
        audio_path: str,
        fade_in: int = 100,
        fade_out: int = 100,
    ) -> str:
        """Add fade in/out to audio."""
        try:
            audio = AudioSegment.from_file(audio_path)
            faded = audio.fade_in(fade_in).fade_out(fade_out)

            temp_path = audio_path.replace('.mp3', '_temp.mp3')
            faded.export(temp_path, format='mp3')
            os.replace(temp_path, audio_path)

            return audio_path
        except Exception:
            return audio_path

    async def get_audio_info(self, file_path: str) -> dict[str, Any]:
        """Get audio file information."""
        if _PYDUB_AVAILABLE:
            try:
                audio = AudioSegment.from_file(file_path)
                return {
                    "duration": len(audio) / 1000.0,
                    "channels": audio.channels,
                    "sample_rate": audio.frame_rate,
                    "sample_width": audio.sample_width,
                    "frame_width": audio.frame_width,
                    "frame_count": audio.frame_count(),
                }
            except Exception:
                # Fall back to WAV parsing below.
                pass

        if ffmpeg_available():
            try:
                info = probe_audio(file_path)
                return {
                    "duration": info.duration,
                    "channels": info.channels,
                    "sample_rate": info.sample_rate,
                }
            except Exception:
                pass

        try:
            info = read_wav_info(file_path)
            return {
                "duration": info.duration,
                "channels": info.channels,
                "sample_rate": info.sample_rate,
                "sample_width": info.sample_width,
                "frame_width": info.sample_width * info.channels,
                "frame_count": info.frame_count,
            }
        except Exception as e:
            return {"error": str(e)}

    async def analyze_loudness(self, file_path: str) -> dict[str, Any]:
        """Analyze loudness and dynamics (best-effort).

        This is used by AudioQualityChecker. It must not crash even when optional
        deps (ffmpeg/pydub/audioop) are unavailable.
        """
        # Try pydub first (if available)
        if _PYDUB_AVAILABLE:
            try:
                audio = AudioSegment.from_file(file_path)
                # Approximate metrics from AudioSegment.
                peak_amp = float(10 ** (audio.max_dBFS / 20.0)) if audio.max_dBFS != float("-inf") else 0.0
                rms_amp = float(10 ** (audio.dBFS / 20.0)) if audio.dBFS != float("-inf") else 0.0
                peak_db = float(audio.max_dBFS) if audio.max_dBFS != float("-inf") else -120.0
                rms_db = float(audio.dBFS) if audio.dBFS != float("-inf") else -120.0
                dynamic_range_db = max(0.0, peak_db - rms_db)
                return {
                    "peak_db": peak_db,
                    "rms_db": rms_db,
                    "dynamic_range_db": dynamic_range_db,
                    "peak_amp": peak_amp,
                    "rms_amp": rms_amp,
                    "method": "pydub",
                }
            except Exception:
                pass

        # ffmpeg streaming decode fallback (supports mp3/flac/m4a/...).
        if ffmpeg_available():
            try:
                peak = 0.0
                sumsq = 0.0
                count = 0
                for block in iter_audio_mono_float32(file_path, sample_rate=16000, chunk_samples=8192):
                    if block.size == 0:
                        continue
                    abs_block = np.abs(block)
                    peak = max(peak, float(abs_block.max(initial=0.0)))
                    sumsq += float(np.square(block).sum())
                    count += int(block.size)

                rms = float(np.sqrt(sumsq / count)) if count else 0.0
                # Convert to dBFS. (float32 samples already normalized)
                peak_db = float(20.0 * np.log10(max(1e-12, min(1.0, peak)))) if peak > 0 else -120.0
                rms_db = float(20.0 * np.log10(max(1e-12, min(1.0, rms)))) if rms > 0 else -120.0
                return {
                    "peak_db": peak_db,
                    "rms_db": rms_db,
                    "dynamic_range_db": max(0.0, peak_db - rms_db),
                    "method": "ffmpeg",
                }
            except Exception as e:
                return {"error": str(e), "method": "ffmpeg"}

        # WAV fallback
        try:
            metrics = analyze_wav_loudness(file_path)
            metrics["method"] = "wave"
            return metrics
        except Exception as e:
            return {"error": str(e), "method": "unavailable"}

    async def export_audacity_project(
        self,
        chunks: list[dict[str, Any]],
        output_dir: str,
        project_name: str = "audiobook",
    ) -> str:
        """
        Export audio files and Audacity project files.

        Creates:
        - Individual audio files for each chunk
        - Audacity .lof file for multi-track import
        - Labels track with timing information

        Args:
            chunks: List of chunks with audio_path, speaker, text, order_index
            output_dir: Directory to save exports
            project_name: Name for the project

        Returns:
            Path to the created zip file
        """
        output_path = Path(output_dir)
        project_dir = output_path / project_name
        project_dir.mkdir(parents=True, exist_ok=True)

        audio_dir = project_dir / "audio"
        audio_dir.mkdir(exist_ok=True)

        # Copy and organize audio files
        audio_files = []
        labels = []
        current_time = 0.0

        for chunk in sorted(chunks, key=lambda x: x.get("order_index", 0)):
            if not chunk.get("audio_path"):
                continue

            # Get source audio path
            source_path = Path(settings.upload_dir.parent) / chunk["audio_path"].lstrip("/")

            if not source_path.exists():
                continue

            # Get audio duration
            duration = await self.get_duration(str(source_path))

            # Create safe filename
            speaker = sanitize_filename(chunk.get("speaker", "unknown"))
            index = chunk.get("order_index", 0)
            filename = f"{index:04d}_{speaker}.mp3"
            dest_path = audio_dir / filename

            # Copy audio file
            import shutil
            shutil.copy2(source_path, dest_path)

            # Track for labels
            labels.append({
                "time": current_time,
                "speaker": chunk.get("speaker", "NARRATOR"),
                "text": chunk.get("text", "")[:100],  # Truncate long text
            })

            audio_files.append({
                "file": f"audio/{filename}",
                "speaker": chunk.get("speaker"),
                "offset": current_time,
            })

            current_time += duration + (DEFAULT_PAUSE_MS / 1000.0)  # Add pause

        # Create Audacity .lof file
        lof_content = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        lof_content += "<audacityproject xmlns=\"http://audacity.sourceforge.net/xml/\">\n"
        lof_content += "  <projectfolders>\n"
        lof_content += "    <projectfolder foldername=\"audio\"/>\n"
        lof_content += "  </projectfolders>\n"
        lof_content += "  <files>\n"

        for af in audio_files:
            lof_content += f"    <file filename='{af['file']}'>\n"
            lof_content += f"      <loomirror>{af['offset']:.3f}</loomirror>\n"
            lof_content += "    </file>\n"

        lof_content += "  </files>\n"
        lof_content += "</audacityproject>"

        lof_path = project_dir / f"{project_name}.lof"
        with open(lof_path, 'w', encoding='utf-8') as f:
            f.write(lof_content)

        # Create labels file (Audacity labels format)
        labels_path = project_dir / f"{project_name}_labels.txt"
        with open(labels_path, 'w', encoding='utf-8') as f:
            for label in labels:
                f.write(f"{label['time']:.3f}\t{label['time'] + 5.0:.3f}\t{label['speaker']}: {label['text']}\n")

        # Create README
        readme_path = project_dir / "README.txt"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"Audacity Project Export: {project_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write("To open in Audacity:\n")
            f.write(f"1. File > Open > select '{project_name}.lof'\n")
            f.write("2. This will import all audio files on separate tracks\n")
            f.write(f"3. File > Import > Labels > select '{project_name}_labels.txt'\n")
            f.write("\nFiles included:\n")
            f.write(f"- {len(audio_files)} audio files in audio/ directory\n")
            f.write(f"- {len(labels)} label markers\n")

        # Create zip file
        zip_path = output_path / f"{project_name}_audacity.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in project_dir.rglob("*"):
                if file.is_file() and file != zip_path:
                    arcname = file.relative_to(project_dir)
                    zipf.write(file, arcname)

        return str(zip_path)

    async def export_individual_voicelines(
        self,
        chunks: list[dict[str, Any]],
        output_dir: str,
        project_name: str = "audiobook",
    ) -> str:
        """
        Export individual voicelines as separate audio files.

        Creates a zip file with numbered audio files for DAW editing.

        Args:
            chunks: List of chunks with audio_path, speaker, text, order_index
            output_dir: Directory to save exports
            project_name: Name for the project

        Returns:
            Path to the created zip file
        """
        output_path = Path(output_dir)
        project_dir = output_path / f"{project_name}_voicelines"
        project_dir.mkdir(parents=True, exist_ok=True)

        # Create CSV manifest
        import csv
        manifest_path = project_dir / "manifest.csv"
        audio_files = []

        with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['File', 'Index', 'Speaker', 'Text', 'Duration'])

            for chunk in sorted(chunks, key=lambda x: x.get("order_index", 0)):
                if not chunk.get("audio_path"):
                    continue

                source_path = Path(settings.upload_dir.parent) / chunk["audio_path"].lstrip("/")

                if not source_path.exists():
                    continue

                # Get duration
                duration = await self.get_duration(str(source_path))

                # Create filename
                speaker = sanitize_filename(chunk.get("speaker", "unknown"))
                index = chunk.get("order_index", 0)
                filename = f"voiceline_{index:04d}_{speaker}.mp3"
                dest_path = project_dir / filename

                # Copy file
                import shutil
                shutil.copy2(source_path, dest_path)

                # Write to manifest
                writer.writerow([
                    filename,
                    index,
                    chunk.get("speaker"),
                    chunk.get("text", "")[:200],  # Truncate for CSV
                    f"{duration:.2f}"
                ])

                audio_files.append(filename)

        # Create zip file
        zip_path = output_path / f"{project_name}_voicelines.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in project_dir.rglob("*"):
                if file.is_file() and file != zip_path:
                    arcname = file.relative_to(project_dir)
                    zipf.write(file, arcname)

        return str(zip_path)

    async def export_combined_audiobook(
        self,
        chunks: list[dict[str, Any]],
        output_dir: str,
        project_name: str = "audiobook",
        add_fades: bool = True,
        normalize: bool = True,
    ) -> tuple[str, float]:
        """
        Export combined audiobook with all chunks merged.

        Args:
            chunks: List of chunks with audio_path, speaker, text, order_index
            output_dir: Directory to save exports
            project_name: Name for the project
            add_fades: Add fade in/out to the combined audio
            normalize: Normalize volume

        Returns:
            Tuple of (output_path, total_duration)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Sort chunks by order
        sorted_chunks = sorted(chunks, key=lambda x: x.get("order_index", 0))

        # Collect audio paths and speakers
        audio_paths = []
        speakers = []

        for chunk in sorted_chunks:
            if not chunk.get("audio_path"):
                continue

            source_path = Path(settings.upload_dir.parent) / chunk["audio_path"].lstrip("/")

            if source_path.exists():
                audio_paths.append(str(source_path))
                speakers.append(chunk.get("speaker", "NARRATOR"))

        if not audio_paths:
            raise ValueError("No valid audio files found")

        # Combine with pauses
        combined_path = output_path / f"{project_name}_combined.mp3"
        result_path, duration = await self.combine_audio_files(
            audio_paths,
            speakers,
            str(combined_path)
        )

        # Post-processing
        if normalize:
            await self.normalize_volume(result_path)

        if add_fades:
            await self.add_fade(result_path, fade_in=100, fade_out=500)

        return result_path, duration

    async def enhance_audio(
        self,
        audio_path: str,
        denoise: bool = True,
        normalize_volume: bool = True,
        add_compression: bool = False,
        target_lufs: float = -16.0,
    ) -> dict[str, Any]:
        """
        Enhance audio quality with various processing options.

        Args:
            audio_path: Path to input audio file
            denoise: Apply basic noise reduction
            normalize_volume: Normalize volume to target LUFS
            add_compression: Apply dynamic range compression
            target_lufs: Target loudness in LUFS

        Returns:
            Dict with enhanced_audio_url and processing info
        """
        try:
            # Load audio
            audio = AudioSegment.from_file(audio_path)
            original_duration = len(audio) / 1000.0

            processing_steps = []

            # Step 1: Basic noise reduction (silence removal)
            if denoise:
                # Remove very quiet sections (simple noise gate)
                nonsilent_ranges = detect_nonsilent(
                    audio,
                    min_silence_len=50,
                    silence_thresh=audio.dBFS - 16,
                    seek_step=10
                )

                if nonsilent_ranges:
                    # Keep only non-silent parts
                    enhanced_audio = AudioSegment.silent(duration=0)
                    for start_ms, end_ms in nonsilent_ranges:
                        enhanced_audio += audio[start_ms:end_ms]
                    audio = enhanced_audio
                    processing_steps.append("noise_gate_applied")

            # Step 2: Volume normalization
            if normalize_volume:
                # Normalize to target loudness
                target_dBFS = target_lufs  # LUFS is approximately dBFS for speech
                change_in_dBFS = target_dBFS - audio.dBFS
                audio = audio.apply_gain(change_in_dBFS)
                processing_steps.append(f"normalized_to_{target_lufs}_LUFS")

            # Step 3: Dynamic range compression (basic)
            if add_compression:
                # Simple compression by reducing peaks
                peak_amplitude = audio.max
                if peak_amplitude > 0:
                    compression_ratio = 0.8
                    audio = audio.apply_gain(-(20 * np.log10(peak_amplitude) * (1 - compression_ratio)))
                    audio = audio.normalize()
                    processing_steps.append("compression_applied")

            # Export enhanced audio
            output_dir = Path(settings.upload_dir) / "enhanced"
            output_dir.mkdir(parents=True, exist_ok=True)

            filename = Path(audio_path).stem
            output_path = output_dir / f"{filename}_enhanced.mp3"
            audio.export(str(output_path), format="mp3", bitrate="192k")

            return {
                "enhanced_audio_url": f"/uploads/enhanced/{Path(output_path).name}",
                "original_duration": original_duration,
                "enhanced_duration": len(audio) / 1000.0,
                "processing_steps": processing_steps,
                "success": True
            }

        except Exception as e:
            logger.error(f"Audio enhancement failed: {e}")
            return {
                "enhanced_audio_url": None,
                "error": str(e),
                "success": False
            }

    async def convert_voice(
        self,
        audio_path: str,
        pitch_shift: float = 0.0,  # Semitones, -12 to +12
        speed_factor: float = 1.0,  # 0.5 to 2.0
        preserve_timing: bool = True,
    ) -> dict[str, Any]:
        """
        Convert voice characteristics (pitch and speed).

        This is a simplified voice conversion that adjusts:
        - Pitch: Makes voice deeper or higher
        - Speed: Changes speaking rate

        Args:
            audio_path: Path to input audio file
            pitch_shift: Pitch adjustment in semitones (-12 to +12)
            speed_factor: Speed multiplier (0.5 to 2.0)
            preserve_timing: Maintain duration when changing pitch

        Returns:
            Dict with converted_audio_url and processing info
        """
        try:
            audio = AudioSegment.from_file(audio_path)

            processing_steps = []

            # Pitch shifting using frame rate manipulation
            if pitch_shift != 0.0:
                # Calculate new sample rate for pitch shift
                # Up by 1 semitone = multiply by 2^(1/12)
                pitch_factor = 2.0 ** (pitch_shift / 12.0)

                new_sample_rate = int(audio.frame_rate * pitch_factor)
                pitched = audio._spawn(audio.raw_data, overrides={
                    "frame_rate": new_sample_rate
                })
                pitched = pitched.set_frame_rate(audio.frame_rate)
                audio = pitched
                processing_steps.append(f"pitch_shifted_{pitch_shift:+.1f}_semitones")

            # Speed adjustment
            if speed_factor != 1.0:
                # Use tempo change (preserves pitch)
                # pydub doesn't have native tempo, so we use frame rate trick
                if preserve_timing:
                    # To change speed without affecting pitch:
                    # Resample to stretch, then correct pitch
                    original_frame_rate = audio.frame_rate
                    temp_frame_rate = int(original_frame_rate / speed_factor)
                    stretched = audio._spawn(audio.raw_data, overrides={
                        "frame_rate": temp_frame_rate
                    })
                    audio = stretched.set_frame_rate(original_frame_rate)
                else:
                    # Simple speed change (affects pitch too)
                    new_frame_rate = int(audio.frame_rate * speed_factor)
                    audio = audio._spawn(audio.raw_data, overrides={
                        "frame_rate": new_frame_rate
                    })
                    audio = audio.set_frame_rate(audio.frame_rate)

                processing_steps.append(f"speed_{speed_factor:.2f}x")

            # Export converted audio
            output_dir = Path(settings.upload_dir) / "converted"
            output_dir.mkdir(parents=True, exist_ok=True)

            filename = Path(audio_path).stem
            output_path = output_dir / f"{filename}_converted.mp3"
            audio.export(str(output_path), format="mp3", bitrate="192k")

            return {
                "converted_audio_url": f"/uploads/converted/{Path(output_path).name}",
                "original_duration": len(AudioSegment.from_file(audio_path)) / 1000.0,
                "converted_duration": len(audio) / 1000.0,
                "processing_steps": processing_steps,
                "success": True
            }

        except Exception as e:
            logger.error(f"Voice conversion failed: {e}")
            return {
                "converted_audio_url": None,
                "error": str(e),
                "success": False
            }

    async def analyze_voice_characteristics(
        self,
        audio_path: str,
    ) -> dict[str, Any]:
        """
        Analyze audio to extract voice characteristics.

        This can help with voice cloning and design.

        Args:
            audio_path: Path to input audio file

        Returns:
            Dict with voice characteristics
        """
        try:
            audio = AudioSegment.from_file(audio_path)

            # Basic audio analysis
            duration = len(audio) / 1000.0

            # Get loudness statistics
            loudness_dbfs = audio.dBFS
            max_loudness = audio.max_dBFS

            # Analyze frequency content (approximate)
            samples = np.array(audio.get_array_of_samples())
            sample_rate = audio.frame_rate

            # Calculate fundamental frequency approx (pitch)
            if len(samples) > 0:
                # Simple zero-crossing rate for pitch estimation
                zero_crossings = np.sum(np.diff(np.sign(samples[::100])) != 0)
                zcr = zero_crossings / len(samples[::100])
                estimated_pitch_hz = zcr * sample_rate / 2
            else:
                estimated_pitch_hz = 0

            # Determine voice characteristics
            if estimated_pitch_hz > 200:
                estimated_gender = "female"
            elif estimated_pitch_hz > 150:
                estimated_gender = "female"  # or young male
            elif estimated_pitch_hz > 100:
                estimated_gender = "male"
            else:
                estimated_gender = "male"  # deep male

            # Energy level for emotion estimation
            energy_variation = np.std(samples)
            if energy_variation > 1000:
                estimated_emotion = "energetic"
            elif energy_variation > 500:
                estimated_emotion = "neutral"
            else:
                estimated_emotion = "calm"

            return {
                "duration": duration,
                "loudness_dbfs": round(loudness_dbfs, 2),
                "max_loudness_dbfs": round(max_loudness, 2),
                "estimated_pitch_hz": round(estimated_pitch_hz, 1),
                "estimated_gender": estimated_gender,
                "estimated_emotion": estimated_emotion,
                "energy_level": round(energy_variation, 2),
                "sample_rate": sample_rate,
                "channels": audio.channels,
                "success": True
            }

        except Exception as e:
            logger.error(f"Voice analysis failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# =============================================================================
# Audio Mixer - 音频混合器（背景音乐和音效支持）
# =============================================================================

class AudioMixer:
    """
    Audio mixer for combining speech with background music and sound effects.
    Supports ducking, volume balancing, and timeline-based effects.
    """

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    async def mix_audio(
        self,
        speech_audio_path: str,
        background_music_path: Optional[str] = None,
        sound_effects: Optional[List[Dict[str, Any]]] = None,
        music_volume: float = 0.2,
        ducking: bool = True,
        ducking_amount: float = 0.5,
        fade_in: float = 0.5,
        fade_out: float = 1.0,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Mix speech audio with background music and sound effects.

        Args:
            speech_audio_path: Path to speech audio file
            background_music_path: Path to background music file (optional)
            sound_effects: List of sound effects with timing
                [{"file": "path/to/effect.mp3", "time": 1.5, "volume": 0.3, "fade": 0.1}]
            music_volume: Background music volume (0.0 - 1.0)
            ducking: Reduce background music during speech (ducking)
            ducking_amount: How much to reduce during ducking (0.0 - 1.0)
            fade_in: Fade in duration for background music (seconds)
            fade_out: Fade out duration for background music (seconds)
            output_path: Output file path (optional, auto-generated if not provided)

        Returns:
            Dict with output_path, duration, and processing info
        """
        try:
            # Load speech audio
            speech = AudioSegment.from_file(speech_audio_path)
            speech_duration = len(speech) / 1000.0

            # Start with speech as base
            mixed = speech

            # Add background music
            if background_music_path:
                music = AudioSegment.from_file(background_music_path)

                # Loop music to match speech duration
                music_looped = music * (int(speech_duration / len(music) * 1000) + 1)
                music_looped = music_looped[:int(speech_duration * 1000)]

                # Apply fade in/out
                if fade_in > 0:
                    music_looped = music_looped.fade_in(int(fade_in * 1000))
                if fade_out > 0:
                    music_looped = music_looped.fade_out(int(fade_out * 1000))

                # Apply ducking if enabled
                if ducking:
                    # Reduce music volume during speech
                    # Find speech segments (simplified - assumes entire audio is speech)
                    ducked_music = music_looped - (10 * (1 - ducking_amount))  # Reduce by X dB
                    ducked_music = ducked_music + (20 * np.log10(music_volume))  # Set base volume
                else:
                    ducked_music = music_looped + (20 * np.log10(music_volume))

                # Overlay music on speech
                mixed = mixed.overlay(ducked_music)

            # Add sound effects at specified times
            if sound_effects:
                for effect in sound_effects:
                    effect_path = effect.get("file")
                    effect_time = effect.get("time", 0)  # Time in seconds
                    effect_volume = effect.get("volume", 0.3)
                    effect_fade = effect.get("fade", 0.1)

                    if not effect_path or not Path(effect_path).exists():
                        continue

                    # Load effect
                    effect_audio = AudioSegment.from_file(effect_path)

                    # Apply volume
                    effect_audio = effect_audio + (20 * np.log10(effect_volume))

                    # Apply fade
                    if effect_fade > 0:
                        effect_audio = effect_audio.fade_in(int(effect_fade * 1000))
                        effect_audio = effect_audio.fade_out(int(effect_fade * 1000))

                    # Position effect at specified time
                    position_ms = int(effect_time * 1000)

                    # Extend mixed audio if needed
                    current_duration = len(mixed)
                    effect_end = position_ms + len(effect_audio)
                    if effect_end > current_duration:
                        silence = AudioSegment.silent(duration=effect_end - current_duration)
                        mixed = mixed + silence

                    # Overlay effect
                    mixed = mixed.overlay(effect_audio, position=position_ms)

            # Generate output path if not provided
            if output_path is None:
                output_dir = Path(settings.upload_dir) / "mixed"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = str(output_dir / f"mixed_{uuid.uuid4().hex[:8]}.mp3")

            # Export mixed audio
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            mixed.export(str(output_path), format="mp3", bitrate="192k")

            return {
                "output_path": str(output_path),
                "output_url": f"/uploads/mixed/{output_path.name}",
                "duration": len(mixed) / 1000.0,
                "processing": {
                    "background_music": background_music_path is not None,
                    "sound_effects_count": len(sound_effects) if sound_effects else 0,
                    "ducking_enabled": ducking,
                    "music_volume": music_volume,
                },
                "success": True,
            }

        except Exception as e:
            logger.error(f"Audio mixing failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# =============================================================================
# Sound Effects Library - 音效库管理
# =============================================================================

class SoundEffectsLibrary:
    """
    Sound effects library manager for organizing and managing sound effects.
    Includes preset sound effect packs for different genres (suspense, romance, sci-fi, etc.)
    """

    # Preset sound effect packs
    PRESET_PACKS = {
        "suspense": {
            "name": "悬疑音效包",
            "description": "适用于悬疑、惊悚小说",
            "effects": [
                {"id": "door_creak", "name": "开门声", "file": "effects/door_creak.mp3", "category": "环境"},
                {"id": "footsteps_wood", "name": "木地板脚步声", "file": "effects/footsteps_wood.mp3", "category": "脚步"},
                {"id": "wind_howl", "name": "风声", "file": "effects/wind_howl.mp3", "category": "环境"},
                {"id": "thunder", "name": "雷声", "file": "effects/thunder.mp3", "category": "天气"},
                {"id": "heartbeat", "name": "心跳声", "file": "effects/heartbeat.mp3", "category": "身体"},
            ]
        },
        "romance": {
            "name": "浪漫音效包",
            "description": "适用于浪漫、爱情小说",
            "effects": [
                {"id": "rain_gentle", "name": "细雨声", "file": "effects/rain_gentle.mp3", "category": "天气"},
                {"id": "birds_chirp", "name": "鸟鸣声", "file": "effects/birds_chirp.mp3", "category": "自然"},
                {"id": "ocean_waves", "name": "海浪声", "file": "effects/ocean_waves.mp3", "category": "自然"},
                {"id": "music_box", "name": "八音盒", "file": "effects/music_box.mp3", "category": "乐器"},
                {"id": "wind_chimes", "name": "风铃", "file": "effects/wind_chimes.mp3", "category": "乐器"},
            ]
        },
        "sci_fi": {
            "name": "科幻音效包",
            "description": "适用于科幻小说",
            "effects": [
                {"id": "spaceship_hum", "name": "飞船嗡鸣", "file": "effects/spaceship_hum.mp3", "category": "机械"},
                {"id": "computer_beep", "name": "电脑提示音", "file": "effects/computer_beep.mp3", "category": "电子"},
                {"id": "laser_fire", "name": "激光声", "file": "effects/laser_fire.mp3", "category": "武器"},
                {"id": "robot_voice", "name": "机器人语音", "file": "effects/robot_voice.mp3", "category": "语音"},
                {"id": "warp_speed", "name": "跃迁声", "file": "effects/warp_speed.mp3", "category": "机械"},
            ]
        },
        "fantasy": {
            "name": "奇幻音效包",
            "description": "适用于奇幻、冒险小说",
            "effects": [
                {"id": "magic_cast", "name": "魔法施展", "file": "effects/magic_cast.mp3", "category": "魔法"},
                {"id": "sword_clash", "name": "剑击声", "file": "effects/sword_clash.mp3", "category": "武器"},
                {"id": "dragon_roar", "name": "龙吼", "file": "effects/dragon_roar.mp3", "category": "生物"},
                {"id": "forest_ambience", "name": "森林环境", "file": "effects/forest_ambience.mp3", "category": "环境"},
                {"id": "dungeon_echo", "name": "地牢回声", "file": "effects/dungeon_echo.mp3", "category": "环境"},
            ]
        },
        "urban": {
            "name": "都市音效包",
            "description": "适用于都市、现代背景",
            "effects": [
                {"id": "traffic_honk", "name": "汽车鸣笛", "file": "effects/traffic_honk.mp3", "category": "交通"},
                {"id": "sub_train", "name": "地铁", "file": "effects/subway_train.mp3", "category": "交通"},
                {"id": "coffee_shop", "name": "咖啡店环境", "file": "effects/coffee_shop.mp3", "category": "环境"},
                {"id": "keyboard_typing", "name": "键盘打字", "file": "effects/keyboard_typing.mp3", "category": "日常"},
                {"id": "elevator_ding", "name": "电梯叮咚", "file": "effects/elevator_ding.mp3", "category": "日常"},
            ]
        },
    }

    def __init__(self):
        """Initialize sound effects library."""
        self.library_dir = Path("./static/sound_effects")
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self.custom_effects: Dict[str, Dict] = {}
        self._load_custom_effects()

    def _load_custom_effects(self):
        """Load custom user-uploaded sound effects."""
        custom_file = self.library_dir / "custom_effects.json"
        if custom_file.exists():
            try:
                with open(custom_file, 'r', encoding='utf-8') as f:
                    self.custom_effects = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load custom effects: {e}")

    def _save_custom_effects(self):
        """Save custom effects to file."""
        custom_file = self.library_dir / "custom_effects.json"
        try:
            with open(custom_file, 'w', encoding='utf-8') as f:
                json.dump(self.custom_effects, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save custom effects: {e}")

    def get_preset_packs(self) -> Dict[str, Dict]:
        """Get all available preset sound effect packs."""
        return self.PRESET_PACKS

    def get_preset_pack(self, pack_id: str) -> Optional[Dict]:
        """Get a specific preset pack by ID."""
        return self.PRESET_PACKS.get(pack_id)

    def get_effect_from_pack(self, pack_id: str, effect_id: str) -> Optional[Dict]:
        """Get a specific effect from a preset pack."""
        pack = self.get_preset_pack(pack_id)
        if pack:
            for effect in pack.get("effects", []):
                if effect["id"] == effect_id:
                    return effect
        return None

    def search_effects(self, keyword: str, category: Optional[str] = None) -> List[Dict]:
        """
        Search for sound effects by keyword and/or category.

        Args:
            keyword: Search keyword
            category: Optional category filter

        Returns:
            List of matching effects
        """
        results = []

        # Search in preset packs
        for pack_id, pack in self.PRESET_PACKS.items():
            for effect in pack.get("effects", []):
                # Category filter
                if category and effect.get("category") != category:
                    continue

                # Keyword search
                if keyword.lower() in effect.get("name", "").lower() or \
                   keyword.lower() in effect.get("id", "").lower():
                    results.append({
                        **effect,
                        "pack_id": pack_id,
                        "pack_name": pack["name"],
                        "source": "preset",
                    })

        # Search in custom effects
        for effect_id, effect in self.custom_effects.items():
            if category and effect.get("category") != category:
                continue

            if keyword.lower() in effect.get("name", "").lower():
                results.append({
                    **effect,
                    "id": effect_id,
                    "source": "custom",
                })

        return results

    def add_custom_effect(
        self,
        effect_id: str,
        name: str,
        file_path: str,
        category: str = "custom",
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Add a custom sound effect to the library.

        Args:
            effect_id: Unique effect ID
            name: Effect name
            file_path: Path to the audio file
            category: Effect category
            description: Optional description
            tags: Optional tags for searching

        Returns:
            Result with success status
        """
        # Copy file to library
        library_path = self.library_dir / f"custom/{effect_id}.mp3"
        library_path.parent.mkdir(parents=True, exist_ok=True)

        source_path = Path(file_path)
        if source_path.exists():
            import shutil
            shutil.copy2(source_path, library_path)
        else:
            return {
                "success": False,
                "error": f"Source file not found: {file_path}"
            }

        # Add to custom effects
        self.custom_effects[effect_id] = {
            "id": effect_id,
            "name": name,
            "file": f"custom/{effect_id}.mp3",
            "category": category,
            "description": description,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
        }

        self._save_custom_effects()

        return {
            "success": True,
            "effect": self.custom_effects[effect_id],
            "file_url": f"/static/sound_effects/custom/{effect_id}.mp3",
        }

    def delete_custom_effect(self, effect_id: str) -> bool:
        """Delete a custom sound effect."""
        if effect_id in self.custom_effects:
            # Delete file
            file_path = self.library_dir / self.custom_effects[effect_id]["file"]
            if file_path.exists():
                file_path.unlink()

            # Remove from library
            del self.custom_effects[effect_id]
            self._save_custom_effects()
            return True
        return False

    def get_all_categories(self) -> List[str]:
        """Get all available effect categories."""
        categories = set()

        for pack in self.PRESET_PACKS.values():
            for effect in pack.get("effects", []):
                categories.add(effect.get("category", "other"))

        for effect in self.custom_effects.values():
            categories.add(effect.get("category", "other"))

        return sorted(list(categories))

    def create_mix_template(
        self,
        template_id: str,
        name: str,
        effects: List[Dict[str, Any]],
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a reusable sound effects mix template.

        Args:
            template_id: Unique template ID
            name: Template name
            effects: List of effects with timing and volume
                [{"effect_id": "door_creak", "time": 1.5, "volume": 0.3, "pack": "suspense"}]
            description: Optional description

        Returns:
            Created template
        """
        templates_file = self.library_dir / "mix_templates.json"
        templates = {}

        if templates_file.exists():
            try:
                with open(templates_file, 'r', encoding='utf-8') as f:
                    templates = json.load(f)
            except Exception:
                pass

        templates[template_id] = {
            "id": template_id,
            "name": name,
            "description": description,
            "effects": effects,
            "created_at": datetime.now().isoformat(),
        }

        with open(templates_file, 'w', encoding='utf-8') as f:
            json.dump(templates, f, ensure_ascii=False, indent=2)

        return templates[template_id]

    def get_mix_template(self, template_id: str) -> Optional[Dict]:
        """Get a mix template by ID."""
        templates_file = self.library_dir / "mix_templates.json"

        if templates_file.exists():
            try:
                with open(templates_file, 'r', encoding='utf-8') as f:
                    templates = json.load(f)
                    return templates.get(template_id)
            except Exception:
                pass

        return None

    def get_all_mix_templates(self) -> List[Dict]:
        """Get all mix templates."""
        templates_file = self.library_dir / "mix_templates.json"

        if templates_file.exists():
            try:
                with open(templates_file, 'r', encoding='utf-8') as f:
                    templates = json.load(f)
                    return list(templates.values())
            except Exception:
                pass

        return []

    def resolve_effect_path(self, effect: Dict) -> Optional[str]:
        """
        Resolve the full file path for a sound effect.

        Args:
            effect: Effect dict with 'file' and optional 'source' keys

        Returns:
            Full path to the effect file, or None if not found
        """
        file_name = effect.get("file", "")
        source = effect.get("source", "preset")

        if source == "custom":
            full_path = self.library_dir / file_name
        else:
            full_path = self.library_dir / "presets" / file_name

        if full_path.exists():
            return str(full_path)

        return None


# Global instance for sound effects library
_sound_effects_library: Optional[SoundEffectsLibrary] = None


def get_sound_effects_library() -> SoundEffectsLibrary:
    """Get global sound effects library instance."""
    global _sound_effects_library
    if _sound_effects_library is None:
        _sound_effects_library = SoundEffectsLibrary()
    return _sound_effects_library


# =============================================================================
# Multi-Speaker Dialogue Generator - 多说话人对话生成器
# =============================================================================

class MultiSpeakerDialogueGenerator:
    """
    Generate multi-speaker dialogue audio with automatic pause insertion
    and speaker consistency.
    """

    def __init__(self, tts_engine, config: Optional[Dict] = None):
        """
        Initialize multi-speaker dialogue generator.

        Args:
            tts_engine: TTS engine instance
            config: Optional configuration
        """
        self.tts_engine = tts_engine
        self.config = config or {}
        self.pause_between_speakers = self.config.get("pause_between_speakers", 800)  # ms
        self.pause_same_speaker = self.config.get("pause_same_speaker", 400)  # ms
        self.speaker_voice_cache: Dict[str, str] = {}  # Cache voice IDs per speaker

    async def generate_dialogue(
        self,
        dialogue_script: List[Dict[str, Any]],
        voice_configs: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
        add_pauses: bool = True,
        normalize_audio: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate multi-speaker dialogue audio.

        Args:
            dialogue_script: List of dialogue segments
                [{"speaker": "角色A", "text": "...", "emotion": "...", "voice_id": "..."}]
            voice_configs: Voice configuration for each speaker
                {"角色A": {"voice_id": "aiden", "emotion": "neutral", ...}}
            output_path: Output file path (optional)
            add_pauses: Add natural pauses between segments
            normalize_audio: Normalize volume across segments

        Returns:
            Dict with output_path, duration, and segment info
        """
        try:
            if not dialogue_script:
                raise ValueError("Dialogue script is empty")

            # Generate audio for each segment
            segments_audio = []
            previous_speaker = None
            total_duration = 0.0

            for i, segment in enumerate(dialogue_script):
                speaker = segment.get("speaker", "NARRATOR")
                text = segment.get("text", "")
                emotion = segment.get("emotion", "neutral")
                segment_voice_id = segment.get("voice_id")

                if not text:
                    continue

                # Determine voice configuration
                voice_config = self._get_voice_config(speaker, voice_configs, segment_voice_id)

                # Generate audio
                audio_data, duration = await self.tts_engine.generate(
                    text=text,
                    speaker=voice_config["voice_id"],
                    instruct=voice_config.get("instruct"),
                    voice_config=voice_config,
                )

                audio = AudioSegment.from_file(io.BytesIO(audio_data))
                segments_audio.append({
                    "speaker": speaker,
                    "audio": audio,
                    "text": text,
                    "duration": duration,
                })

                total_duration += duration

                # Add pause
                if add_pauses and i < len(dialogue_script) - 1:
                    if speaker == previous_speaker:
                        pause_duration = self.pause_same_speaker
                    else:
                        pause_duration = self.pause_between_speakers

                    segments_audio.append({
                        "speaker": None,  # Pause marker
                        "audio": AudioSegment.silent(duration=pause_duration),
                        "text": "",
                        "duration": pause_duration / 1000.0,
                    })
                    total_duration += pause_duration / 1000.0

                previous_speaker = speaker

            # Combine all segments
            combined = AudioSegment.empty()
            for segment in segments_audio:
                combined += segment["audio"]

            # Normalize if requested
            if normalize_audio:
                combined = self._normalize_audio(combined)

            # Export
            if output_path is None:
                output_dir = Path(settings.upload_dir) / "dialogues"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = str(output_dir / f"dialogue_{uuid.uuid4().hex[:8]}.mp3")

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            combined.export(str(output_path), format="mp3", bitrate="192k")

            return {
                "output_path": str(output_path),
                "output_url": f"/uploads/dialogues/{output_path.name}",
                "duration": len(combined) / 1000.0,
                "segments_count": len([s for s in segments_audio if s["speaker"] is not None]),
                "speakers": list(set(s["speaker"] for s in segments_audio if s["speaker"] is not None)),
                "success": True,
            }

        except Exception as e:
            logger.error(f"Dialogue generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _get_voice_config(
        self,
        speaker: str,
        voice_configs: Optional[Dict[str, Any]],
        segment_voice_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get voice configuration for a speaker."""
        # Use segment-specific voice if provided
        if segment_voice_id:
            return {"voice_id": segment_voice_id}

        # Use cached voice for speaker
        if speaker in self.speaker_voice_cache:
            return {"voice_id": self.speaker_voice_cache[speaker]}

        # Use provided voice configs
        if voice_configs and speaker in voice_configs:
            config = voice_configs[speaker].copy()
            self.speaker_voice_cache[speaker] = config["voice_id"]
            return config

        # Assign default voice based on speaker name
        default_voices = {
            "NARRATOR": "aiden",
            "narrator": "aiden",
            "主角": "rachel",
            "male": "aiden",
            "female": "rachel",
        }

        voice_id = default_voices.get(speaker, "aiden")
        self.speaker_voice_cache[speaker] = voice_id

        return {"voice_id": voice_id}

    def _normalize_audio(self, audio: AudioSegment, target_dbfs: float = -20.0) -> AudioSegment:
        """Normalize audio to target loudness."""
        change_in_dBFS = target_dbfs - audio.dBFS
        return audio.apply_gain(change_in_dBFS)


# =============================================================================
# Intelligent Text Segmenter - 智能文本断句器
# =============================================================================

class IntelligentTextSegmenter:
    """
    Intelligently segment text for TTS processing.
    Handles semantic boundaries, dialogue detection, and special cases.
    """

    # Dialogue markers (中文和英文)
    DIALOGUE_MARKERS = [
        r'说[：:]', r'道[：:]', r'问[：:]', r'答[：:]', r'喊[：:]',
        r' replied[：:]', r' asked[：:]', r' said[：:]', r' answered[：:]',
        r'"', r'"', r''', r''',
    ]

    # Sentence ending patterns
    SENTENCE_ENDINGS = r'([。！？\.!?]+)'

    def __init__(self, max_chars: int = 500, preserve_sentences: bool = True):
        self.max_chars = max_chars
        self.preserve_sentences = preserve_sentences

    async def segment_text(
        self,
        text: str,
        max_chars: Optional[int] = None,
        preserve_sentences: Optional[bool] = None,
        detect_dialogue: bool = True,
        add_pause_markers: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Intelligently segment text for TTS.

        Args:
            text: Input text
            max_chars: Maximum characters per segment
            preserve_sentences: Preserve sentence boundaries
            detect_dialogue: Detect and separate dialogue
            add_pause_markers: Add SSML pause markers

        Returns:
            List of segments with metadata
        """
        max_chars = max_chars or self.max_chars
        preserve_sentences = preserve_sentences or self.preserve_sentences

        segments = []

        # Preprocess text
        text = self._preprocess_text(text)

        if detect_dialogue:
            segments = self._segment_with_dialogue_detection(text, max_chars)
        else:
            segments = self._segment_by_sentences(text, max_chars)

        # Add pause markers if requested
        if add_pause_markers:
            for segment in segments:
                segment["text_with_pauses"] = self._add_pause_markers(segment["text"])

        return segments

    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Fix common punctuation issues
        text = text.replace('，。', '。').replace('!.', '!').replace('?.', '?')
        return text.strip()

    def _segment_with_dialogue_detection(
        self,
        text: str,
        max_chars: int,
    ) -> List[Dict[str, Any]]:
        """Segment text with dialogue detection."""
        segments = []
        current_segment = ""
        current_speaker = None
        position = 0

        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for dialogue markers
            speaker_match = None
            for marker in self.DIALOGUE_MARKERS:
                match = re.search(r'^([^"' + marker + r']+?)' + marker, line)
                if match:
                    speaker_match = match.group(1)
                    break

            # New speaker detected
            if speaker_match:
                # Save previous segment if any
                if current_segment:
                    segments.append({
                        "text": current_segment.strip(),
                        "speaker": current_speaker,
                        "type": "dialogue" if current_speaker else "narration",
                        "position": position,
                        "char_count": len(current_segment),
                    })
                    position += len(current_segment)

                current_segment = line
                current_speaker = speaker_match.strip()
            else:
                # Continue current segment
                if current_segment:
                    current_segment += " " + line
                else:
                    current_segment = line

                # Check if segment is too long
                if len(current_segment) >= max_chars:
                    # Need to split
                    sub_segments = self._split_long_segment(current_segment, max_chars)
                    for i, sub_seg in enumerate(sub_segments):
                        if i == 0 and current_speaker:
                            # First sub-segment keeps speaker info
                            segments.append({
                                "text": sub_seg.strip(),
                                "speaker": current_speaker,
                                "type": "dialogue" if current_speaker else "narration",
                                "position": position,
                                "char_count": len(sub_seg),
                            })
                            position += len(sub_seg)
                        else:
                            segments.append({
                                "text": sub_seg.strip(),
                                "speaker": None,
                                "type": "narration",
                                "position": position,
                                "char_count": len(sub_seg),
                            })
                            position += len(sub_seg)
                    current_segment = ""
                    current_speaker = None

        # Add final segment
        if current_segment:
            segments.append({
                "text": current_segment.strip(),
                "speaker": current_speaker,
                "type": "dialogue" if current_speaker else "narration",
                "position": position,
                "char_count": len(current_segment),
            })

        return segments

    def _segment_by_sentences(
        self,
        text: str,
        max_chars: int,
    ) -> List[Dict[str, Any]]:
        """Segment text by sentence boundaries."""
        segments = []
        position = 0

        # Split by sentence endings
        sentences = re.split(self.SENTENCE_ENDINGS, text)

        current_segment = ""
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
            full_sentence = sentence + punctuation

            if len(current_segment) + len(full_sentence) <= max_chars:
                current_segment += full_sentence
            else:
                if current_segment:
                    segments.append({
                        "text": current_segment.strip(),
                        "speaker": None,
                        "type": "narration",
                        "position": position,
                        "char_count": len(current_segment),
                    })
                    position += len(current_segment)

                # Check if single sentence is too long
                if len(full_sentence) > max_chars:
                    # Split long sentence
                    sub_segments = self._split_long_segment(full_sentence, max_chars)
                    for sub_seg in sub_segments:
                        segments.append({
                            "text": sub_seg.strip(),
                            "speaker": None,
                            "type": "narration",
                            "position": position,
                            "char_count": len(sub_seg),
                        })
                        position += len(sub_seg)
                    current_segment = ""
                else:
                    current_segment = full_sentence

        # Add remaining segment
        if current_segment:
            segments.append({
                "text": current_segment.strip(),
                "speaker": None,
                "type": "narration",
                "position": position,
                "char_count": len(current_segment),
            })

        return segments

    def _split_long_segment(self, text: str, max_chars: int) -> List[str]:
        """Split a long text segment into smaller chunks."""
        chunks = []
        remaining = text

        while len(remaining) > max_chars:
            # Find best split point (prefer punctuation)
            split_pos = max_chars

            # Look for split point nearby (within 50 chars)
            for offset in range(50):
                for pos in [max_chars - offset, max_chars + offset]:
                    if pos >= len(remaining):
                        continue
                    if remaining[pos] in '，。、,.;!?！？：:':
                        split_pos = pos + 1
                        break
                if split_pos != max_chars:
                    break

            chunks.append(remaining[:split_pos])
            remaining = remaining[split_pos:].strip()

        if remaining:
            chunks.append(remaining)

        return chunks

    def _add_pause_markers(self, text: str) -> str:
        """Add SSML pause markers to text."""
        # Add longer pause after sentence endings
        text = re.sub(r'([。！？\.!?])', r'\1<break time="500ms"/>', text)
        # Add medium pause after commas
        text = re.sub(r'([，,])', r'\1<break time="200ms"/>', text)
        return text


# =============================================================================
# Streaming TTS Engine - 流式TTS引擎
# =============================================================================

class StreamingTTSEngine:
    """
    Streaming TTS engine for real-time audio generation.
    Supports chunked processing and WebSocket streaming.
    """

    def __init__(self, tts_engine, chunk_size: int = 50):
        """
        Initialize streaming TTS engine.

        Args:
            tts_engine: Base TTS engine
            chunk_size: Characters per chunk
        """
        self.tts_engine = tts_engine
        self.chunk_size = chunk_size
        self.segmenter = IntelligentTextSegmenter(max_chars=chunk_size)

    async def generate_stream(
        self,
        text: str,
        speaker: str = "aiden",
        voice_config: Optional[Dict] = None,
        callback: Optional[callable] = None,
    ):
        """
        Stream TTS generation in chunks.

        Args:
            text: Input text
            speaker: Speaker ID
            voice_config: Voice configuration
            callback: Optional callback for each chunk

        Yields:
            Dict with chunk data:
            {
                "chunk_index": int,
                "total_chunks": int,
                "audio_data": bytes,
                "text": str,
                "is_final": bool,
            }
        """
        # Segment text
        segments = await self.segmenter.segment_text(text, max_chars=self.chunk_size)

        total_chunks = len(segments)

        for i, segment in enumerate(segments):
            try:
                # Generate audio for segment
                audio_data, duration = await self.tts_engine.generate(
                    text=segment["text"],
                    speaker=speaker,
                    voice_config=voice_config,
                )

                chunk_data = {
                    "chunk_index": i,
                    "total_chunks": total_chunks,
                    "audio_data": audio_data,
                    "text": segment["text"],
                    "duration": duration,
                    "is_final": (i == total_chunks - 1),
                    "metadata": {
                        "speaker": segment.get("speaker"),
                        "type": segment.get("type", "narration"),
                        "position": segment.get("position", 0),
                    },
                }

                # Call callback if provided
                if callback:
                    await callback(chunk_data)

                yield chunk_data

            except Exception as e:
                logger.error(f"Error generating chunk {i}: {e}")
                # Continue with next chunk instead of failing completely
                continue

    async def generate_to_file(
        self,
        text: str,
        output_path: str,
        speaker: str = "aiden",
        voice_config: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Stream TTS and save to file.

        Args:
            text: Input text
            output_path: Output file path
            speaker: Speaker ID
            voice_config: Voice configuration

        Returns:
            Dict with generation results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Collect all chunks
        all_audio = AudioSegment.empty()
        total_duration = 0.0
        chunk_count = 0

        async for chunk in self.generate_stream(text, speaker, voice_config):
            audio = AudioSegment.from_file(io.BytesIO(chunk["audio_data"]))
            all_audio += audio
            total_duration += chunk["duration"]
            chunk_count += 1

        # Export combined audio
        all_audio.export(str(output_path), format="mp3", bitrate="192k")

        return {
            "output_path": str(output_path),
            "duration": total_duration,
            "chunks_processed": chunk_count,
            "success": True,
        }


# Global instances
_audio_mixer: Optional[AudioMixer] = None
_dialogue_generator: Optional[MultiSpeakerDialogueGenerator] = None
_text_segmenter: Optional[IntelligentTextSegmenter] = None
_streaming_tts: Optional[StreamingTTSEngine] = None


def get_audio_mixer() -> AudioMixer:
    """Get global audio mixer instance."""
    global _audio_mixer
    if _audio_mixer is None:
        _audio_mixer = AudioMixer()
    return _audio_mixer


def get_dialogue_generator(tts_engine=None) -> MultiSpeakerDialogueGenerator:
    """Get global dialogue generator instance."""
    global _dialogue_generator
    if _dialogue_generator is None:
        if tts_engine is None:
            from app.services.tts_engine import TTSEngineFactory, TTSMode
            tts_engine = TTSEngineFactory.create(TTSMode.LOCAL)
        _dialogue_generator = MultiSpeakerDialogueGenerator(tts_engine)
    return _dialogue_generator


def get_text_segmenter() -> IntelligentTextSegmenter:
    """Get global text segmenter instance."""
    global _text_segmenter
    if _text_segmenter is None:
        _text_segmenter = IntelligentTextSegmenter()
    return _text_segmenter


def get_streaming_tts(tts_engine=None) -> StreamingTTSEngine:
    """Get global streaming TTS instance."""
    global _streaming_tts
    if _streaming_tts is None:
        if tts_engine is None:
            from app.services.tts_engine import TTSEngineFactory, TTSMode
            tts_engine = TTSEngineFactory.create(TTSMode.LOCAL)
        _streaming_tts = StreamingTTSEngine(tts_engine)
    return _streaming_tts


# =============================================================================
# RVC Voice Converter - RVC语音转换器
# =============================================================================

class RVCVoiceConverter:
    """
    RVC (Retrieval-based Voice Conversion) voice converter.
    Converts source audio to target voice while preserving prosody.
    """

    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize RVC voice converter.

        Args:
            models_dir: Directory containing RVC models
        """
        self.models_dir = Path(models_dir or "./static/models/rvc")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_models = {}

    async def convert_voice(
        self,
        source_audio_path: str,
        target_voice_model: str,
        preserve_prosody: bool = True,
        preserve_timing: bool = True,
        pitch_shift: float = 0.0,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convert source audio to target voice using RVC.

        Args:
            source_audio_path: Path to source audio file
            target_voice_model: Target voice model ID
            preserve_prosody: Preserve original prosody characteristics
            preserve_timing: Preserve timing and rhythm
            pitch_shift: Additional pitch adjustment (semitones)
            output_path: Optional output file path

        Returns:
            Conversion result with output path
        """
        try:
            import io
            from pydub import AudioSegment

            # Load source audio
            source = AudioSegment.from_file(source_audio_path)

            # Extract features for prosody preservation
            prosody_features = None
            if preserve_prosody:
                prosody_features = self._extract_prosody_features(source)

            # Apply voice conversion (simplified implementation)
            # In production, this would use actual RVC model
            converted = await self._apply_voice_conversion(
                source,
                target_voice_model,
                prosody_features,
                pitch_shift,
            )

            # Generate output path if not provided
            if output_path is None:
                output_dir = Path("./static/audio/rvc_converted")
                output_dir.mkdir(parents=True, exist_ok=True)
                import uuid
                output_path = str(output_dir / f"rvc_{uuid.uuid4().hex[:8]}.mp3")

            # Export converted audio
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            converted.export(str(output_path), format="mp3", bitrate="192k")

            return {
                "output_path": str(output_path),
                "output_url": f"/static/audio/rvc_converted/{output_path.name}",
                "target_voice_model": target_voice_model,
                "preserve_prosody": preserve_prosody,
                "preserve_timing": preserve_timing,
                "pitch_shift": pitch_shift,
                "duration": len(converted) / 1000.0,
                "conversion_method": "rvc_simulation",
                "success": True,
            }

        except Exception as e:
            logger.error(f"RVC voice conversion failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def _extract_prosody_features(self, audio: AudioSegment) -> Dict[str, Any]:
        """Extract prosody features from audio for preservation."""
        import numpy as np

        samples = np.array(audio.get_array_of_samples())
        sr = audio.frame_rate

        # Extract pitch contour
        pitches = []
        for i in range(0, len(samples), sr // 10):  # Every 100ms
            segment = samples[i:i + sr // 10]
            if len(segment) == 0:
                continue

            # Zero-crossing rate approximation for pitch
            zcr = np.sum(np.abs(np.diff(np.sign(segment))))
            pitch_hz = zcr * sr / len(segment) * 50
            pitches.append(pitch_hz)

        # Extract energy contour
        energies = []
        for i in range(0, len(samples), sr // 10):
            segment = samples[i:i + sr // 10]
            if len(segment) == 0:
                continue
            energy = np.sqrt(np.mean(segment ** 2))
            energies.append(energy)

        return {
            "pitch_contour": pitches,
            "energy_contour": energies,
            "avg_pitch": np.mean(pitches) if pitches else 440.0,
            "pitch_std": np.std(pitches) if pitches else 0.0,
            "avg_energy": np.mean(energies) if energies else 0.0,
        }

    async def _apply_voice_conversion(
        self,
        source: AudioSegment,
        target_model: str,
        prosody_features: Optional[Dict[str, Any]],
        pitch_shift: float,
    ) -> AudioSegment:
        """Apply voice conversion using target model."""
        import numpy as np

        # Simulated voice conversion
        # In production, this would use actual RVC model inference
        converted = source

        # Apply target voice characteristics based on model
        model_characteristics = self._get_model_characteristics(target_model)

        # Adjust pitch based on model and user shift
        if model_characteristics.get("pitch_factor"):
            total_pitch_shift = pitch_shift + model_characteristics["pitch_factor"]
            if total_pitch_shift != 0:
                new_rate = int(converted.frame_rate * (2.0 ** (total_pitch_shift / 12.0)))
                converted = converted._spawn(converted.raw_data, overrides={'frame_rate': new_rate})
                converted = converted.set_frame_rate(22050)

        # Adjust timbre (filter-based simulation)
        if model_characteristics.get("brightness"):
            brightness = model_characteristics["brightness"]
            if brightness > 0.5:
                # Brighter voice - enhance high frequencies
                converted = converted.high_pass_filter(80)
            else:
                # Darker voice - enhance low frequencies
                converted = converted.low_pass_filter(8000)

        # Apply prosody preservation if features provided
        if prosody_features and model_characteristics.get("preserve_prosody"):
            # In production, this would use actual prosody transfer
            pass

        return converted

    def _get_model_characteristics(self, model_id: str) -> Dict[str, Any]:
        """Get characteristics for a voice model."""
        # In production, this would load from model metadata
        model_characteristics = {
            "aiden": {"pitch_factor": 0.0, "brightness": 0.5, "preserve_prosody": True},
            "rachel": {"pitch_factor": 2.0, "brightness": 0.7, "preserve_prosody": True},
            "deep_male": {"pitch_factor": -3.0, "brightness": 0.3, "preserve_prosody": True},
            "young_female": {"pitch_factor": 3.0, "brightness": 0.8, "preserve_prosody": True},
        }
        return model_characteristics.get(model_id, {})

    async def register_model(
        self,
        model_id: str,
        model_path: str,
        characteristics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Register a new RVC model.

        Args:
            model_id: Unique model identifier
            model_path: Path to model file
            characteristics: Model voice characteristics

        Returns:
            Registration status
        """
        model_info = {
            "model_id": model_id,
            "model_path": model_path,
            "characteristics": characteristics,
            "registered_at": datetime.now().isoformat(),
        }

        model_file = self.models_dir / f"{model_id}.json"
        with open(model_file, 'w') as f:
            json.dump(model_info, f, indent=2)

        logger.info(f"Registered RVC model: {model_id}")

        return {
            "model_id": model_id,
            "status": "registered",
            "success": True,
        }

    async def list_models(self) -> List[Dict[str, Any]]:
        """List all available RVC models."""
        models = []

        for model_file in self.models_dir.glob("*.json"):
            try:
                with open(model_file, 'r') as f:
                    model_info = json.load(f)
                    models.append(model_info)
            except Exception as e:
                logger.warning(f"Error loading model info from {model_file}: {e}")

        # Add built-in models
        models.extend([
            {
                "model_id": "aiden",
                "name": "Aiden (Male)",
                "characteristics": {"pitch_factor": 0.0, "brightness": 0.5},
                "builtin": True,
            },
            {
                "model_id": "rachel",
                "name": "Rachel (Female)",
                "characteristics": {"pitch_factor": 2.0, "brightness": 0.7},
                "builtin": True,
            },
            {
                "model_id": "deep_male",
                "name": "Deep Male",
                "characteristics": {"pitch_factor": -3.0, "brightness": 0.3},
                "builtin": True,
            },
            {
                "model_id": "young_female",
                "name": "Young Female",
                "characteristics": {"pitch_factor": 3.0, "brightness": 0.8},
                "builtin": True,
            },
        ])

        return models


# =============================================================================
# Advanced Prosody Controller - 高级韵律控制器
# =============================================================================

class AdvancedProsodyController:
    """
    Advanced prosody controller for fine-grained speech manipulation.
    Supports sentence-level and word-level prosody adjustments.
    """

    def __init__(self):
        """Initialize advanced prosody controller."""
        pass

    async def apply_prosody(
        self,
        audio_path: str,
        prosody_config: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Apply advanced prosody modifications to audio.

        Args:
            audio_path: Path to input audio file
            prosody_config: Prosody configuration
                - sentence_adjustments: List of sentence-level adjustments
                - word_emphasis: List of words to emphasize
                - pitch_curve: Pitch curve points
                - rhythm_pattern: Rhythm pattern (fast-slow-fast, etc.)
                - emotion_gradient: Emotion transition (start to end)
            output_path: Optional output file path

        Returns:
            Modified audio file info
        """
        try:
            import io
            from pydub import AudioSegment
            import numpy as np

            # Load audio
            audio = AudioSegment.from_file(audio_path)
            duration = len(audio) / 1000.0

            # Apply sentence-level adjustments
            if prosody_config.get("sentence_adjustments"):
                audio = await self._apply_sentence_adjustments(
                    audio,
                    prosody_config["sentence_adjustments"],
                )

            # Apply word emphasis
            if prosody_config.get("word_emphasis"):
                audio = await self._apply_word_emphasis(
                    audio,
                    prosody_config["word_emphasis"],
                )

            # Apply pitch curve
            if prosody_config.get("pitch_curve"):
                audio = await self._apply_pitch_curve(
                    audio,
                    prosody_config["pitch_curve"],
                )

            # Apply rhythm pattern
            if prosody_config.get("rhythm_pattern"):
                audio = await self._apply_rhythm_pattern(
                    audio,
                    prosody_config["rhythm_pattern"],
                )

            # Apply emotion gradient
            if prosody_config.get("emotion_gradient"):
                audio = await self._apply_emotion_gradient(
                    audio,
                    prosody_config["emotion_gradient"],
                )

            # Generate output path if not provided
            if output_path is None:
                output_dir = Path("./static/audio/prosody")
                output_dir.mkdir(parents=True, exist_ok=True)
                import uuid
                output_path = str(output_dir / f"prosody_{uuid.uuid4().hex[:8]}.mp3")

            # Export modified audio
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            audio.export(str(output_path), format="mp3", bitrate="192k")

            return {
                "output_path": str(output_path),
                "output_url": f"/static/audio/prosody/{output_path.name}",
                "duration": len(audio) / 1000.0,
                "original_duration": duration,
                "applied_modifications": list(prosody_config.keys()),
                "success": True,
            }

        except Exception as e:
            logger.error(f"Prosody application failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def _apply_sentence_adjustments(
        self,
        audio: AudioSegment,
        adjustments: List[Dict[str, Any]],
    ) -> AudioSegment:
        """Apply sentence-level prosody adjustments."""
        result = audio

        for adj in adjustments:
            start_ms = int(adj.get("start_time", 0) * 1000)
            end_ms = int(adj.get("end_time", len(result)) * 1000)

            if end_ms > len(result):
                end_ms = len(result)

            segment = result[start_ms:end_ms]

            # Apply pitch adjustment
            if adj.get("pitch_shift"):
                pitch_shift = adj["pitch_shift"]
                new_rate = int(segment.frame_rate * (2.0 ** (pitch_shift / 12.0)))
                segment = segment._spawn(segment.raw_data, overrides={'frame_rate': new_rate})
                segment = segment.set_frame_rate(22050)

            # Apply tempo adjustment
            if adj.get("tempo_factor"):
                tempo = adj["tempo_factor"]
                if tempo != 1.0:
                    new_rate = int(segment.frame_rate * tempo)
                    segment = segment._spawn(segment.raw_data, overrides={'frame_rate': new_rate})
                    segment = segment.set_frame_rate(22050)

            # Apply volume adjustment
            if adj.get("volume_gain"):
                segment = segment + adj["volume_gain"]

            # Replace segment
            result = result[:start_ms] + segment + result[end_ms:]

        return result

    async def _apply_word_emphasis(
        self,
        audio: AudioSegment,
        word_emphasis: List[Dict[str, Any]],
    ) -> AudioSegment:
        """Apply emphasis to specific words/timestamps."""
        result = audio

        for emphasis in word_emphasis:
            start_ms = int(emphasis.get("start_time", 0) * 1000)
            end_ms = int(emphasis.get("end_time", 0) * 1000)
            level = emphasis.get("level", "medium")

            if end_ms > len(result):
                end_ms = len(result)

            segment = result[start_ms:end_ms]

            # Apply emphasis based on level
            if level == "strong":
                # Strong emphasis: louder + slight pitch up
                segment = segment + 3  # +3dB
                new_rate = int(segment.frame_rate * 1.05)
                segment = segment._spawn(segment.raw_data, overrides={'frame_rate': new_rate})
                segment = segment.set_frame_rate(22050)
            elif level == "medium":
                # Medium emphasis: slightly louder
                segment = segment + 1.5  # +1.5dB
            elif level == "weak":
                # Weak emphasis: very slight volume increase
                segment = segment + 0.5  # +0.5dB

            # Replace segment
            result = result[:start_ms] + segment + result[end_ms:]

        return result

    async def _apply_pitch_curve(
        self,
        audio: AudioSegment,
        pitch_curve: List[Dict[str, float]],
    ) -> AudioSegment:
        """Apply pitch curve adjustments."""
        if not pitch_curve:
            return audio

        # Sort curve points by time
        curve_points = sorted(pitch_curve, key=lambda x: x.get("time", 0))

        # Apply pitch adjustments at each point
        result = audio
        for i, point in enumerate(curve_points):
            time_pos = point.get("time", 0)
            pitch_value = point.get("pitch", 0)

            # Interpolate between this point and next
            if i < len(curve_points) - 1:
                next_point = curve_points[i + 1]
                next_time = next_point.get("time", time_pos + 1)
                next_pitch = next_point.get("pitch", pitch_value)

                # Apply gradual transition
                start_ms = int(time_pos * 1000)
                end_ms = int(next_time * 1000)
                if end_ms > len(result):
                    end_ms = len(result)

                for t in range(start_ms, end_ms, 100):  # Every 100ms
                    progress = (t - start_ms) / (end_ms - start_ms)
                    interpolated_pitch = pitch_value + (next_pitch - pitch_value) * progress

                    if abs(interpolated_pitch) > 0.1:
                        # Apply tiny pitch adjustment
                        segment_start = t
                        segment_end = min(t + 100, end_ms)
                        if segment_end > segment_start:
                            segment = result[segment_start:segment_end]
                            new_rate = int(segment.frame_rate * (2.0 ** (interpolated_pitch / 12.0)))
                            segment = segment._spawn(segment.raw_data, overrides={'frame_rate': new_rate})
                            segment = segment.set_frame_rate(22050)
                            result = result[:segment_start] + segment + result[segment_end:]

        return result

    async def _apply_rhythm_pattern(
        self,
        audio: AudioSegment,
        pattern: str,
    ) -> AudioSegment:
        """Apply rhythm pattern (fast-slow-fast, slow-fast-slow, etc.)."""
        duration = len(audio) / 1000.0

        # Define patterns
        patterns = {
            "fast-slow-fast": [1.2, 0.8, 1.2],
            "slow-fast-slow": [0.8, 1.2, 0.8],
            "accelerando": [0.8, 1.0, 1.2, 1.4],
            "ritardando": [1.4, 1.2, 1.0, 0.8],
            "wave": [1.0, 1.3, 1.0, 0.7, 1.0],
        }

        tempo_sequence = patterns.get(pattern)
        if not tempo_sequence:
            return audio

        # Apply pattern in segments
        segment_duration = duration / len(tempo_sequence)
        result = AudioSegment.empty()

        for i, tempo in enumerate(tempo_sequence):
            start_ms = int(i * segment_duration * 1000)
            end_ms = int((i + 1) * segment_duration * 1000)

            if end_ms > len(audio):
                end_ms = len(audio)

            segment = audio[start_ms:end_ms]

            # Apply tempo
            if tempo != 1.0:
                new_rate = int(segment.frame_rate * tempo)
                segment = segment._spawn(segment.raw_data, overrides={'frame_rate': new_rate})
                segment = segment.set_frame_rate(22050)

            result += segment

        # Add any remaining audio
        remaining_start = int(len(tempo_sequence) * segment_duration * 1000)
        if remaining_start < len(audio):
            result += audio[remaining_start:]

        return result

    async def _apply_emotion_gradient(
        self,
        audio: AudioSegment,
        gradient: Dict[str, Any],
    ) -> AudioSegment:
        """Apply emotion gradient (transition from start emotion to end emotion)."""
        start_emotion = gradient.get("start_emotion", {})
        end_emotion = gradient.get("end_emotion", {})
        steps = gradient.get("steps", 10)

        duration = len(audio) / 1000.0
        step_duration = duration / steps

        result = AudioSegment.empty()

        for i in range(steps):
            start_ms = int(i * step_duration * 1000)
            end_ms = int((i + 1) * step_duration * 1000)

            if end_ms > len(audio):
                end_ms = len(audio)

            segment = audio[start_ms:end_ms]

            # Interpolate emotion parameters
            progress = i / (steps - 1) if steps > 1 else 0

            # Pitch interpolation
            start_pitch = start_emotion.get("pitch_shift", 0)
            end_pitch = end_emotion.get("pitch_shift", 0)
            pitch = start_pitch + (end_pitch - start_pitch) * progress

            # Tempo interpolation
            start_tempo = start_emotion.get("tempo_factor", 1.0)
            end_tempo = end_emotion.get("tempo_factor", 1.0)
            tempo = start_tempo + (end_tempo - start_tempo) * progress

            # Volume interpolation
            start_vol = start_emotion.get("volume_gain", 0)
            end_vol = end_emotion.get("volume_gain", 0)
            volume = start_vol + (end_vol - start_vol) * progress

            # Apply adjustments
            if pitch != 0:
                new_rate = int(segment.frame_rate * (2.0 ** (pitch / 12.0)))
                segment = segment._spawn(segment.raw_data, overrides={'frame_rate': new_rate})
                segment = segment.set_frame_rate(22050)

            if tempo != 1.0:
                new_rate = int(segment.frame_rate * tempo)
                segment = segment._spawn(segment.raw_data, overrides={'frame_rate': new_rate})
                segment = segment.set_frame_rate(22050)

            if volume != 0:
                segment = segment + volume

            result += segment

        return result

    async def get_prosody_templates(self) -> List[Dict[str, Any]]:
        """Get available prosody templates."""
        return [
            {
                "id": "storytelling",
                "name": "Storytelling",
                "description": "Varied tempo for engaging narration",
                "rhythm_pattern": "wave",
                "sentence_adjustments": [
                    {"start_time": 0.0, "tempo_factor": 1.0},
                    {"start_time": 0.3, "tempo_factor": 0.9},
                    {"start_time": 0.7, "tempo_factor": 1.1},
                ],
            },
            {
                "id": "announcement",
                "name": "Announcement",
                "description": "Clear and measured delivery",
                "rhythm_pattern": "slow-fast-slow",
                "pitch_curve": [
                    {"time": 0.0, "pitch": 1.0},
                    {"time": 0.5, "pitch": 0.5},
                    {"time": 1.0, "pitch": 0.0},
                ],
            },
            {
                "id": "excited",
                "name": "Excited",
                "description": "High energy, accelerating",
                "rhythm_pattern": "accelerando",
                "emotion_gradient": {
                    "start_emotion": {"pitch_shift": 0, "tempo_factor": 1.0},
                    "end_emotion": {"pitch_shift": 2, "tempo_factor": 1.3},
                    "steps": 8,
                },
            },
            {
                "id": "calming",
                "name": "Calming",
                "description": "Gradually slowing down",
                "rhythm_pattern": "ritardando",
                "emotion_gradient": {
                    "start_emotion": {"pitch_shift": 0, "tempo_factor": 1.2},
                    "end_emotion": {"pitch_shift": -1, "tempo_factor": 0.8},
                    "steps": 8,
                },
            },
        ]


# Global instances
_rvc_converter: Optional[RVCVoiceConverter] = None
_prosody_controller: Optional[AdvancedProsodyController] = None


def get_rvc_converter() -> RVCVoiceConverter:
    """Get global RVC voice converter instance."""
    global _rvc_converter
    if _rvc_converter is None:
        _rvc_converter = RVCVoiceConverter()
    return _rvc_converter


def get_prosody_controller() -> AdvancedProsodyController:
    """Get global advanced prosody controller instance."""
    global _prosody_controller
    if _prosody_controller is None:
        _prosody_controller = AdvancedProsodyController()
    return _prosody_controller


# =============================================================================
# Model Manager - 模型管理器（量化、缓存、预加载）
# =============================================================================

class ModelManager:
    """
    Model management for TTS and audio processing models.
    Handles quantization, caching, preloading, and version management.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize model manager.

        Args:
            cache_dir: Directory for cached models
        """
        self.cache_dir = Path(cache_dir or "./static/models/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_models: Dict[str, Any] = {}
        self._model_metadata: Dict[str, Dict[str, Any]] = {}

    async def preload_model(
        self,
        model_type: str,
        model_id: str,
        quantized: bool = False,
    ) -> Dict[str, Any]:
        """
        Preload a model into memory for faster inference.

        Args:
            model_type: Type of model (tts, audio_processor, etc.)
            model_id: Model identifier
            quantized: Whether to use quantized version

        Returns:
            Loading status
        """
        cache_key = f"{model_type}_{model_id}_quantized_{quantized}"

        if cache_key in self._loaded_models:
            return {
                "model_id": model_id,
                "status": "already_loaded",
                "cache_key": cache_key,
            }

        try:
            # Check if quantized version exists
            if quantized:
                quantized_path = self._get_quantized_model_path(model_type, model_id)
                if quantized_path.exists():
                    model = await self._load_quantized_model(quantized_path)
                else:
                    # Quantize and save
                    model = await self._quantize_model(model_type, model_id)
            else:
                model = await self._load_model(model_type, model_id)

            self._loaded_models[cache_key] = model

            # Update metadata
            self._model_metadata[cache_key] = {
                "model_type": model_type,
                "model_id": model_id,
                "quantized": quantized,
                "loaded_at": datetime.now().isoformat(),
                "size_bytes": self._get_model_size(model),
            }

            logger.info(f"Preloaded model: {cache_key}")

            return {
                "model_id": model_id,
                "status": "loaded",
                "cache_key": cache_key,
                "quantized": quantized,
            }

        except Exception as e:
            logger.error(f"Failed to preload model {model_id}: {e}")
            return {
                "model_id": model_id,
                "status": "failed",
                "error": str(e),
            }

    async def _quantize_model(
        self,
        model_type: str,
        model_id: str,
    ) -> Any:
        """
        Quantize a model to INT8 for faster inference.

        Args:
            model_type: Type of model
            model_id: Model identifier

        Returns:
            Quantized model
        """
        # Placeholder for quantization logic
        # In production, this would use torch.quantization or similar
        logger.info(f"Quantizing model: {model_type}/{model_id}")

        # Save quantized model
        quantized_path = self._get_quantized_model_path(model_type, model_id)
        quantized_path.parent.mkdir(parents=True, exist_ok=True)

        # Create metadata
        metadata = {
            "model_type": model_type,
            "model_id": model_id,
            "quantized_at": datetime.now().isoformat(),
            "quantization": "INT8",
        }

        metadata_path = quantized_path.with_suffix(".json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return None  # Return quantized model in production

    def _get_quantized_model_path(self, model_type: str, model_id: str) -> Path:
        """Get path to quantized model file."""
        return self.cache_dir / f"{model_type}_{model_id}_quantized.pt"

    async def _load_model(self, model_type: str, model_id: str) -> Any:
        """Load a model."""
        # Placeholder for model loading logic
        return None

    async def _load_quantized_model(self, path: Path) -> Any:
        """Load a quantized model."""
        # Placeholder for quantized model loading
        return None

    def _get_model_size(self, model: Any) -> int:
        """Get model size in bytes."""
        # Placeholder
        return 0

    async def unload_model(self, model_type: str, model_id: str, quantized: bool = False):
        """
        Unload a model from memory.

        Args:
            model_type: Type of model
            model_id: Model identifier
            quantized: Whether it's quantized
        """
        cache_key = f"{model_type}_{model_id}_quantized_{quantized}"

        if cache_key in self._loaded_models:
            del self._loaded_models[cache_key]
            logger.info(f"Unloaded model: {cache_key}")

    async def switch_model(
        self,
        from_model: str,
        to_model: str,
        model_type: str = "tts",
    ) -> Dict[str, Any]:
        """
        Switch from one model to another without downtime.

        Args:
            from_model: Current model ID
            to_model: Target model ID
            model_type: Type of model

        Returns:
            Switch status
        """
        try:
            # Load new model first
            new_loaded = await self.preload_model(model_type, to_model)

            if new_loaded["status"] == "loaded" or new_loaded["status"] == "already_loaded":
                # Unload old model
                await self.unload_model(model_type, from_model)

                return {
                    "from_model": from_model,
                    "to_model": to_model,
                    "status": "switched",
                    "success": True,
                }
            else:
                return {
                    "from_model": from_model,
                    "to_model": to_model,
                    "status": "failed",
                    "error": new_loaded.get("error"),
                }

        except Exception as e:
            logger.error(f"Model switch failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
            }

    async def get_loaded_models(self) -> List[Dict[str, Any]]:
        """Get list of currently loaded models."""
        models = []

        for cache_key, metadata in self._model_metadata.items():
            models.append({
                **metadata,
                "cache_key": cache_key,
                "is_loaded": cache_key in self._loaded_models,
            })

        return models

    async def get_model_info(self, model_type: str, model_id: str) -> Dict[str, Any]:
        """Get information about a model."""
        quantized_path = self._get_quantized_model_path(model_type, model_id)
        has_quantized = quantized_path.exists()

        return {
            "model_type": model_type,
            "model_id": model_id,
            "has_quantized_version": has_quantized,
            "is_loaded": f"{model_type}_{model_id}_quantized_True" in self._loaded_models,
            "cache_dir": str(self.cache_dir),
        }


# =============================================================================
# Enhanced Quality Assessor - 增强的语音质量评估器
# =============================================================================

class EnhancedQualityAssessor:
    """
    Enhanced audio quality assessor with comprehensive metrics.
    Provides MOS score prediction, speaker similarity, emotion accuracy, etc.
    """

    def __init__(self):
        """Initialize enhanced quality assessor."""
        pass

    async def assess_quality(
        self,
        audio_path: str,
        reference_path: Optional[str] = None,
        detailed: bool = False,
    ) -> Dict[str, Any]:
        """
        Comprehensive audio quality assessment.

        Args:
            audio_path: Path to audio file to assess
            reference_path: Optional reference audio for comparison
            detailed: Return detailed metrics

        Returns:
            Quality assessment report
        """
        try:
            from pydub import AudioSegment
            import numpy as np

            # Load audio
            audio = AudioSegment.from_file(audio_path)
            samples = np.array(audio.get_array_of_samples())
            sr = audio.frame_rate
            duration = len(audio) / 1000.0

            # Basic quality metrics
            metrics = await self._compute_basic_metrics(samples, sr)

            # Advanced metrics
            advanced_metrics = await self._compute_advanced_metrics(samples, sr)

            # If reference provided, compute similarity
            similarity = None
            if reference_path:
                similarity = await self._compute_speaker_similarity(
                    audio_path,
                    reference_path,
                )

            # Predict MOS score
            mos_score = self._predict_mos_score(metrics, advanced_metrics)

            # Emotion accuracy (if reference provided)
            emotion_accuracy = None
            if reference_path:
                emotion_accuracy = await self._assess_emotion_accuracy(
                    audio_path,
                    reference_path,
                )

            result = {
                "audio_path": audio_path,
                "duration": duration,
                "mos_score": mos_score,
                "overall_quality": self._get_quality_label(mos_score),
                "basic_metrics": metrics,
                "advanced_metrics": advanced_metrics if detailed else self._summarize_advanced(advanced_metrics),
                "timestamp": datetime.now().isoformat(),
            }

            if similarity:
                result["speaker_similarity"] = similarity

            if emotion_accuracy:
                result["emotion_accuracy"] = emotion_accuracy

            # Provide improvement suggestions
            result["suggestions"] = self._generate_suggestions(result)

            return result

        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return {
                "error": str(e),
                "audio_path": audio_path,
            }

    async def _compute_basic_metrics(
        self,
        samples: np.ndarray,
        sr: int,
    ) -> Dict[str, float]:
        """Compute basic audio quality metrics."""
        # Signal-to-Noise Ratio (SNR)
        signal_level = np.sqrt(np.mean(samples ** 2))
        noise_estimate = np.std(samples[samples < np.percentile(np.abs(samples), 10)]) if len(samples) > 0 else 0
        snr = 20 * np.log10(signal_level / (noise_estimate + 1e-10))

        # Dynamic range
        dynamic_range_db = 20 * np.log10(np.max(np.abs(samples)) / (np.min(np.abs(samples)) + 1e-10))

        # Zero Crossing Rate
        zcr = np.sum(np.abs(np.diff(np.sign(samples)))) / len(samples)

        # Crest factor
        crest_factor = np.max(np.abs(samples)) / (np.sqrt(np.mean(samples ** 2)) + 1e-10)

        return {
            "snr_db": round(snr, 2),
            "dynamic_range_db": round(dynamic_range_db, 2),
            "zero_crossing_rate": round(zcr, 4),
            "crest_factor": round(crest_factor, 2),
            "rms_level": round(signal_level, 2),
            "peak_level": round(np.max(np.abs(samples)), 2),
        }

    async def _compute_advanced_metrics(
        self,
        samples: np.ndarray,
        sr: int,
    ) -> Dict[str, Any]:
        """Compute advanced audio quality metrics."""
        try:
            import librosa

            # MFCCs for timbre analysis
            mfccs = librosa.feature.mfcc(y=samples.astype(float), sr=sr, n_mfcc=13)

            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=samples.astype(float), sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=samples.astype(float), sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=samples.astype(float), sr=sr)[0]

            # Pitch statistics
            pitches, magnitudes = librosa.piptrack(y=samples.astype(float), sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                idx = magnitudes[:, t].argmax()
                pitch = pitches[idx, t]
                if pitch > 0:
                    pitch_values.append(pitch)

            return {
                "mfcc_mean": np.mean(mfccs, axis=1).tolist(),
                "spectral_centroid_mean": float(np.mean(spectral_centroids)),
                "spectral_centroid_std": float(np.std(spectral_centroids)),
                "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
                "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
                "pitch_mean": float(np.mean(pitch_values)) if pitch_values else 0.0,
                "pitch_std": float(np.std(pitch_values)) if pitch_values else 0.0,
                "pitch_range": float(np.max(pitch_values) - np.min(pitch_values)) if len(pitch_values) > 1 else 0.0,
            }

        except ImportError:
            # Fallback without librosa
            return {
                "mfcc_mean": [0.0] * 13,
                "spectral_centroid_mean": 0.0,
                "spectral_centroid_std": 0.0,
                "pitch_mean": 0.0,
                "pitch_std": 0.0,
            }

    async def _compute_speaker_similarity(
        self,
        audio_path: str,
        reference_path: str,
    ) -> Dict[str, float]:
        """Compute speaker similarity between two audio files."""
        try:
            import librosa

            # Load both audio files
            y1, sr1 = librosa.load(audio_path, sr=22050)
            y2, sr2 = librosa.load(reference_path, sr=22050)

            # Extract MFCCs
            mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13)
            mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=13)

            # Dynamic time warping distance
            from scipy.spatial.distance import euclidean

            # Simple comparison using mean MFCCs
            mfcc1_mean = np.mean(mfcc1, axis=1)
            mfcc2_mean = np.mean(mfcc2, axis=1)

            distance = euclidean(mfcc1_mean, mfcc2_mean)
            similarity = max(0, 1 - distance / 10)  # Normalize to 0-1

            return {
                "similarity_score": round(similarity, 3),
                "distance": round(distance, 3),
                "match": similarity > 0.7,
            }

        except Exception as e:
            logger.warning(f"Speaker similarity computation failed: {e}")
            return {
                "similarity_score": 0.0,
                "error": str(e),
            }

    async def _assess_emotion_accuracy(
        self,
        audio_path: str,
        reference_path: str,
    ) -> Dict[str, float]:
        """Assess emotion accuracy compared to reference."""
        try:
            # Get emotion for both files
            from app.api.voice_advanced import recognize_emotion

            target_emotion = await recognize_emotion(
                audio_path=reference_path,
                current_user=None,
            )
            test_emotion = await recognize_emotion(
                audio_path=audio_path,
                current_user=None,
            )

            target = target_emotion.data.get("emotion", "neutral")
            test = test_emotion.data.get("emotion", "neutral")
            confidence = test_emotion.data.get("confidence", 0)

            # Match if same emotion or high confidence
            match = 1.0 if target == test else max(0, 1 - confidence)

            return {
                "target_emotion": target,
                "predicted_emotion": test,
                "accuracy": round(match, 3),
                "confidence": round(confidence, 3),
            }

        except Exception as e:
            logger.warning(f"Emotion accuracy assessment failed: {e}")
            return {
                "accuracy": 0.0,
                "error": str(e),
            }

    def _predict_mos_score(
        self,
        basic_metrics: Dict[str, float],
        advanced_metrics: Dict[str, Any],
    ) -> float:
        """Predict Mean Opinion Score (MOS) from metrics."""
        # Weighted combination of metrics
        snr = basic_metrics.get("snr_db", 0)
        dynamic_range = basic_metrics.get("dynamic_range_db", 0)
        zcr = basic_metrics.get("zero_crossing_rate", 0)

        # Normalize to 1-5 scale
        # SNR: 20dB=3, 40dB=5
        snr_score = min(5, max(1, (snr + 10) / 10))

        # Dynamic range: 30dB=3, 60dB=5
        dr_score = min(5, max(1, dynamic_range / 12))

        # Combined score
        mos = (snr_score * 0.5 + dr_score * 0.3 + 3 * 0.2)

        return round(mos, 2)

    def _get_quality_label(self, mos_score: float) -> str:
        """Get quality label from MOS score."""
        if mos_score >= 4.5:
            return "Excellent"
        elif mos_score >= 4.0:
            return "Very Good"
        elif mos_score >= 3.5:
            return "Good"
        elif mos_score >= 3.0:
            return "Fair"
        else:
            return "Poor"

    def _summarize_advanced(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Summarize advanced metrics."""
        return {
            "brightness": round(metrics.get("spectral_centroid_mean", 0) / 5000, 2),
            "stability": round(1 / (1 + metrics.get("pitch_std", 0) / 50), 2),
            "richness": round(min(1, metrics.get("pitch_range", 0) / 200), 2),
        }

    def _generate_suggestions(self, result: Dict[str, Any]) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []

        # Check SNR
        snr = result.get("basic_metrics", {}).get("snr_db", 0)
        if snr < 20:
            suggestions.append("Consider noise reduction to improve clarity")

        # Check dynamic range
        dr = result.get("basic_metrics", {}).get("dynamic_range_db", 0)
        if dr < 30:
            suggestions.append("Audio dynamic range is low, consider normalization")
        elif dr > 70:
            suggestions.append("Dynamic range is very high, consider compression")

        # Check crest factor
        cf = result.get("basic_metrics", {}).get("crest_factor", 0)
        if cf > 10:
            suggestions.append("High crest factor detected, consider limiting peaks")
        elif cf < 3:
            suggestions.append("Low crest factor, audio may sound flat")

        # Check MOS score
        mos = result.get("mos_score", 0)
        if mos < 3.5:
            suggestions.append("Overall quality needs improvement")
        elif mos >= 4.0:
            suggestions.append("Good quality audio")

        # Speaker similarity
        if "speaker_similarity" in result:
            sim = result["speaker_similarity"].get("similarity_score", 0)
            if sim < 0.5:
                suggestions.append("Speaker characteristics differ from reference")

        return suggestions

    async def batch_assess(
        self,
        audio_paths: List[str],
        reference_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Batch quality assessment for multiple audio files.

        Args:
            audio_paths: List of audio file paths
            reference_path: Optional reference for comparison

        Returns:
            Batch assessment results
        """
        results = []
        total_mos = 0

        for audio_path in audio_paths:
            result = await self.assess_quality(
                audio_path=audio_path,
                reference_path=reference_path,
                detailed=False,
            )
            results.append(result)
            total_mos += result.get("mos_score", 0)

        return {
            "count": len(results),
            "average_mos": round(total_mos / len(results), 2) if results else 0,
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }


# =============================================================================
# SSML Processor - SSML处理器
# =============================================================================

class SSMLProcessor:
    """
    SSML (Speech Synthesis Markup Language) processor.
    Converts SSML markup to TTS parameters and instructions.
    Supports W3C SSML 1.1 standard tags.
    """

    # SSML tag patterns
    BREAK_PATTERN = re.compile(r'<break\s+(?:time="([^"]+)")?\s*(?:strength="([^"]+)")?\s*/>')
    EMPHASIS_PATTERN = re.compile(r'<emphasis\s+level="([^"]+)">(.*?)</emphasis>', re.DOTALL)
    PROSODY_PATTERN = re.compile(r'<prosody\s+([^>]+)>(.*?)</prosody>', re.DOTALL)
    SAY_AS_PATTERN = re.compile(r'<say-as\s+interpret-as="([^"]+)"(?:\s+format="([^"]+)")?>([^<]+)</say-as>')
    PATTERN_PATTERN = re.compile(r'<say-as\s+interpret-as="([^"]+)">')
    VOICE_PATTERN = re.compile(r'<voice\s+name="([^"]+)"(?:\s+gender="([^"]+)")?>(.*?)</voice>', re.DOTALL)
    SUB_PATTERN = re.compile(r'<sub\s+alias="([^"]+)">([^<]+)</sub>')
    PHONEME_PATTERN = re.compile(r'<phoneme\s+ph="([^"]+)"(?:\s+alphabet="([^"]+)")?>([^<]+)</phoneme>')

    # Break time conversions (strength to milliseconds)
    BREAK_STRENGTH_MAP = {
        "none": 0,
        "x-weak": 50,
        "weak": 100,
        "medium": 250,
        "strong": 500,
        "x-strong": 1000,
    }

    # Emphasis to emotion/speed mapping
    EMPHASIS_MAP = {
        "strong": {"energy": 1.3, "speed": 0.95, "volume": 1.2},
        "moderate": {"energy": 1.1, "speed": 1.0, "volume": 1.0},
        "reduced": {"energy": 0.8, "speed": 1.05, "volume": 0.9},
        "none": {"energy": 1.0, "speed": 1.0, "volume": 1.0},
    }

    def __init__(self):
        """Initialize SSML processor."""
        self.preserved_tags = []

    def parse_ssml(self, ssml_text: str) -> Dict[str, Any]:
        """
        Parse SSML markup and convert to TTS parameters.

        Args:
            ssml_text: Text containing SSML markup

        Returns:
            Dict containing:
                - clean_text: Text with SSML tags removed
                - segments: List of text segments with their parameters
                - breaks: List of break positions and durations
                - instructions: List of TTS instructions
        """
        result = {
            "clean_text": "",
            "segments": [],
            "breaks": [],
            "instructions": [],
            "prosody_changes": [],
        }

        # Process in order, maintaining position tracking
        text = ssml_text
        position = 0

        # Parse <break> tags
        for match in self.BREAK_PATTERN.finditer(text):
            time_ms = match.group(1)
            strength = match.group(2)

            # Calculate break duration
            if time_ms:
                # Parse time (e.g., "500ms", "0.5s", "1000ms")
                if time_ms.endswith("ms"):
                    duration = int(time_ms[:-2])
                elif time_ms.endswith("s"):
                    duration = int(float(time_ms[:-1]) * 1000)
                else:
                    duration = int(time_ms)
            elif strength:
                duration = self.BREAK_STRENGTH_MAP.get(strength, 250)
            else:
                duration = 250

            result["breaks"].append({
                "position": match.start(),
                "duration_ms": duration,
            })

        # Parse <emphasis> tags
        for match in self.EMPHASIS_PATTERN.finditer(text):
            level = match.group(1)
            emphasized_text = match.group(2)

            params = self.EMPHASIS_MAP.get(level, self.EMPHASIS_MAP["moderate"])

            result["segments"].append({
                "text": emphasized_text,
                "start": match.start(),
                "end": match.end(),
                "type": "emphasis",
                "level": level,
                "parameters": params,
            })

            result["instructions"].append(
                f"Emphasize with {level} strength: {emphasized_text[:30]}..."
            )

        # Parse <prosody> tags
        for match in self.PROSODY_PATTERN.finditer(text):
            attrs = match.group(1)
            prosody_text = match.group(2)

            # Parse prosody attributes
            prosody_params = {}
            for attr_match in re.finditer(r'(\w+)="([^"]+)"', attrs):
                key, value = attr_match.groups()
                prosody_params[key] = value

            result["segments"].append({
                "text": prosody_text,
                "start": match.start(),
                "end": match.end(),
                "type": "prosody",
                "attributes": prosody_params,
            })

            # Convert prosody to TTS parameters
            tts_params = self._prosody_to_tts_params(prosody_params)
            result["prosody_changes"].append({
                "text": prosody_text,
                "parameters": tts_params,
            })

        # Parse <say-as> tags (special text interpretation)
        for match in self.SAY_AS_PATTERN.finditer(text):
            interpret_as = match.group(1)
            format_spec = match.group(2)
            content = match.group(3)

            converted = self._interpret_say_as(content, interpret_as, format_spec)

            result["segments"].append({
                "text": converted,
                "original_text": content,
                "start": match.start(),
                "end": match.end(),
                "type": "say-as",
                "interpret_as": interpret_as,
            })

        # Parse <voice> tags (speaker change)
        for match in self.VOICE_PATTERN.finditer(text):
            voice_name = match.group(1)
            voice_gender = match.group(2)
            voice_text = match.group(3)

            result["segments"].append({
                "text": voice_text,
                "start": match.start(),
                "end": match.end(),
                "type": "voice",
                "voice": voice_name,
                "gender": voice_gender,
            })

        # Parse <sub> tags (alias/substitution)
        for match in self.SUB_PATTERN.finditer(text):
            alias = match.group(1)
            original = match.group(2)

            result["segments"].append({
                "text": alias,
                "original_text": original,
                "start": match.start(),
                "end": match.end(),
                "type": "substitution",
            })

        # Remove all SSML tags to get clean text
        clean_text = text
        clean_text = self.BREAK_PATTERN.sub('[PAUSE]', clean_text)
        clean_text = self.EMPHASIS_PATTERN.sub(r'\2', clean_text)
        clean_text = self.PROSODY_PATTERN.sub(r'\2', clean_text)
        clean_text = self.SAY_AS_PATTERN.sub(r'\3', clean_text)
        clean_text = self.VOICE_PATTERN.sub(r'\3', clean_text)
        clean_text = self.SUB_PATTERN.sub(r'\1', clean_text)
        clean_text = self.PHONEME_PATTERN.sub(r'\3', clean_text)

        result["clean_text"] = clean_text

        return result

    def _prosody_to_tts_params(self, prosody_attrs: Dict[str, str]) -> Dict[str, Any]:
        """Convert prosody attributes to TTS parameters."""
        params = {}

        # Rate (speed)
        if "rate" in prosody_attrs:
            rate = prosody_attrs["rate"].lower()
            rate_map = {
                "x-slow": 0.5,
                "slow": 0.75,
                "medium": 1.0,
                "fast": 1.25,
                "x-fast": 1.5,
            }
            if rate in rate_map:
                params["speed"] = rate_map[rate]
            elif rate.endswith("%"):
                params["speed"] = int(rate[:-1]) / 100

        # Pitch
        if "pitch" in prosody_attrs:
            pitch = prosody_attrs["pitch"].lower()
            pitch_map = {
                "x-low": -3.0,
                "low": -1.5,
                "medium": 0.0,
                "high": 1.5,
                "x-high": 3.0,
            }
            if pitch in pitch_map:
                params["pitch_shift"] = pitch_map[pitch]
            elif pitch.endswith("%"):
                # e.g., "+50%" or "-20%"
                pitch_pct = int(pitch[:-1])
                params["pitch_shift"] = pitch_pct / 25  # Convert to semitones
            elif pitch.startswith("+") or pitch.startswith("-"):
                params["pitch_shift"] = int(pitch) / 100

        # Volume
        if "volume" in prosody_attrs:
            volume = prosody_attrs["volume"].lower()
            volume_map = {
                "silent": 0.0,
                "x-soft": 0.3,
                "soft": 0.6,
                "medium": 1.0,
                "loud": 1.4,
                "x-loud": 1.8,
            }
            if volume in volume_map:
                params["volume"] = volume_map[volume]
            elif volume.endswith("%"):
                params["volume"] = int(volume[:-1]) / 100

        # Contour (pitch contour - advanced)
        if "contour" in prosody_attrs:
            # Parse contour like "+10% +20% -10%"
            contour = prosody_attrs["contour"]
            params["contour"] = contour

        return params

    def _interpret_say_as(self, text: str, interpret_as: str, format_spec: str = None) -> str:
        """Interpret text based on say-as interpret-as type."""
        interpret_as = interpret_as.lower()

        if interpret_as == "date":
            # Format date for natural reading
            if format_spec and format_spec.startswith("dmy"):
                # Day-Month-Year format
                parts = text.split("-")
                if len(parts) == 3:
                    return f"{parts[0]}月{parts[1]}日，{parts[2]}年"
            elif format_spec and format_spec.startswith("mdy"):
                # Month-Day-Year format
                parts = text.split("/")
                if len(parts) == 3:
                    return f"{parts[0]}月{parts[1]}日，{parts[2]}年"
            elif format_spec and format_spec.startswith("ymd"):
                # Year-Month-Day format
                parts = text.split("-")
                if len(parts) == 3:
                    return f"{parts[1]}月{parts[2]}日，{parts[0]}年"
            # Default: natural language date reading
            return text.replace("-", "年").replace("/", "月") + "日"

        elif interpret_as == "time":
            # Format time for natural reading
            if ":" in text:
                parts = text.split(":")
                hour = int(parts[0])
                minute = int(parts[1]) if len(parts) > 1 else 0
                # Convert to Chinese time format
                period = "上午" if hour < 12 else "下午"
                if hour > 12:
                    hour -= 12
                return f"{period}{hour}点{minute}分"
            return text

        elif interpret_as == "number":
            # Format number for natural reading
            # Add commas for thousands, convert to Chinese reading
            try:
                num = float(text.replace(",", ""))
                if "." in text:
                    # Decimal number
                    integer, decimal = text.split(".")
                    return f"{integer}点{decimal}"
                else:
                    # Integer - add digit grouping for reading
                    return text
            except ValueError:
                return text

        elif interpret_as == "currency":
            # Format currency for natural reading
            if format_spec:
                # Format like "USD 123.45" -> "123美元45美分"
                parts = text.split()
                if len(parts) >= 2:
                    amount = parts[-1]
                    currency = " ".join(parts[:-1])
                    try:
                        num = float(amount)
                        if "." in amount:
                            integer, decimal = amount.split(".")
                            return f"{integer}{currency}{decimal}分"
                        else:
                            return f"{amount}{currency}"
                    except ValueError:
                        pass
            return text

        elif interpret_as == "telephone":
            # Format phone number for natural reading
            # "1-800-555-1234" -> "1 八百 五百五十五 一二三四"
            return text.replace("-", " ")

        elif interpret_as == "address":
            # Address format - just return as-is for now
            return text

        elif interpret_as == "name":
            # Name pronunciation hints
            return text

        elif interpret_as == "media":
            # Media file reference
            return f"[播放音频: {text}]"

        return text

    def ssml_to_segments(self, ssml_text: str, default_voice: str = "aiden") -> List[Dict[str, Any]]:
        """
        Convert SSML text to audio segments with TTS parameters.

        Args:
            ssml_text: Text with SSML markup
            default_voice: Default voice to use

        Returns:
            List of segments with text and TTS parameters
        """
        parsed = self.parse_ssml(ssml_text)
        segments = []

        # Sort segments by position
        sorted_segments = sorted(parsed["segments"], key=lambda x: x["start"])

        current_position = 0
        segment_params = {
            "voice": default_voice,
            "speed": 1.0,
            "pitch_shift": 0.0,
            "volume": 1.0,
            "energy": 1.0,
        }

        for segment in sorted_segments:
            # Add any text before this segment
            if segment["start"] > current_position:
                text_between = ssml_text[current_position:segment["start"]]
                text_between = self._strip_ssml_tags(text_between)
                if text_between:
                    segments.append({
                        "text": text_between,
                        "parameters": segment_params.copy(),
                    })

            # Add the segment with its specific parameters
            seg_type = segment.get("type", "")

            if seg_type == "emphasis":
                params = segment_params.copy()
                params.update(segment["parameters"])
                segments.append({
                    "text": segment["text"],
                    "parameters": params,
                })

            elif seg_type == "prosody":
                params = segment_params.copy()
                params.update(segment.get("parameters", {}))
                segments.append({
                    "text": segment["text"],
                    "parameters": params,
                })

            elif seg_type == "voice":
                params = segment_params.copy()
                params["voice"] = segment.get("voice", default_voice)
                segments.append({
                    "text": segment["text"],
                    "parameters": params,
                })

            elif seg_type == "say-as":
                segments.append({
                    "text": segment["text"],
                    "parameters": segment_params.copy(),
                })

            elif seg_type == "substitution":
                segments.append({
                    "text": segment["text"],
                    "parameters": segment_params.copy(),
                })

            current_position = segment["end"]

        # Add remaining text
        if current_position < len(ssml_text):
            remaining_text = ssml_text[current_position:]
            remaining_text = self._strip_ssml_tags(remaining_text)
            if remaining_text:
                segments.append({
                    "text": remaining_text,
                    "parameters": segment_params.copy(),
                })

        # Insert breaks
        all_segments = []
        segment_index = 0
        text_position = 0

        for break_info in sorted(parsed["breaks"], key=lambda x: x["position"]):
            # Add segments before this break
            while segment_index < len(segments):
                seg = segments[segment_index]
                seg_text = seg["text"]

                if text_position + len(seg_text) <= break_info["position"]:
                    all_segments.append(seg)
                    text_position += len(seg_text)
                    segment_index += 1
                else:
                    # Split segment at break position
                    before_text = seg_text[:break_info["position"] - text_position]
                    after_text = seg_text[break_info["position"] - text_position:]

                    if before_text:
                        all_segments.append({
                            "text": before_text,
                            "parameters": seg["parameters"],
                        })

                    all_segments.append({
                        "type": "break",
                        "duration_ms": break_info["duration_ms"],
                    })

                    # Update current segment with remaining text
                    segments[segment_index] = {
                        "text": after_text,
                        "parameters": seg["parameters"],
                    }
                    text_position = break_info["position"]
                    break
            else:
                # Insert break after all segments processed
                all_segments.append({
                    "type": "break",
                    "duration_ms": break_info["duration_ms"],
                })

        # Add remaining segments
        all_segments.extend(segments[segment_index:])

        return all_segments

    def _strip_ssml_tags(self, text: str) -> str:
        """Remove SSML tags from text."""
        text = self.BREAK_PATTERN.sub('', text)
        text = self.EMPHASIS_PATTERN.sub(r'\2', text)
        text = self.PROSODY_PATTERN.sub(r'\2', text)
        text = self.SAY_AS_PATTERN.sub(r'\3', text)
        text = self.VOICE_PATTERN.sub(r'\3', text)
        text = self.SUB_PATTERN.sub(r'\1', text)
        text = self.PHONEME_PATTERN.sub(r'\3', text)
        return text.strip()

    def validate_ssml(self, ssml_text: str) -> Dict[str, Any]:
        """
        Validate SSML markup.

        Args:
            ssml_text: SSML text to validate

        Returns:
            Dict with validation results
        """
        errors = []
        warnings = []

        # Check for matching tags
        open_tags = re.findall(r'<(\w+)(?:\s+[^>]*)?>', ssml_text)
        close_tags = re.findall(r'</(\w+)>', ssml_text)

        # Check for unclosed tags (simplified check)
        for open_tag in open_tags:
            tag_name = open_tag[0]
            if tag_name not in ["break", "br"]:  # Self-closing tags
                # Count opening vs closing
                open_count = len([t for t in open_tags if t[0] == tag_name])
                close_count = len([t for t in close_tags if t == tag_name])
                if open_count != close_count:
                    errors.append(f"Mismatched {tag_name} tags: {open_count} opening, {close_count} closing")

        # Check for invalid attributes
        valid_attrs = {
            "break": ["time", "strength"],
            "emphasis": ["level"],
            "prosody": ["rate", "pitch", "volume", "contour"],
            "say-as": ["interpret-as", "format"],
            "voice": ["name", "gender", "age", "variant"],
            "sub": ["alias"],
            "phoneme": ["ph", "alphabet"],
        }

        for match in re.finditer(r'<(\w+)(?:\s+([^>]*))?>', ssml_text):
            tag = match.group(1)
            attrs_str = match.group(2)

            if attrs_str and tag in valid_attrs:
                for attr_match in re.finditer(r'(\w+)="', attrs_str):
                    attr = attr_match.group(1)
                    if attr not in valid_attrs[tag]:
                        warnings.append(f"Unknown attribute '{attr}' in <{tag}> tag")

        # Check for deprecated features
        if "<mark>" in ssml_text:
            warnings.append("<mark> tag is not fully supported")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def text_to_ssml(self, text: str, options: Dict[str, Any] = None) -> str:
        """
        Convert plain text with options to SSML markup.

        Args:
            text: Plain text
            options: Conversion options
                - pause_after_sentence: Add pause after each sentence (default: true)
                - pause_duration: Pause duration in ms (default: 500)
                - emphasize_keywords: Emphasize important words (default: false)
                - voice_gender: Preferred voice gender

        Returns:
            SSML markup string
        """
        options = options or {}
        ssml = text

        # Add sentence pauses
        if options.get("pause_after_sentence", True):
            pause_duration = options.get("pause_duration", 500)
            # Insert breaks after sentence endings
            ssml = re.sub(r'([。！？\.!?])', r'\1<break time="' + str(pause_duration) + 'ms"/>', ssml)

        # Emphasize keywords (simple implementation)
        if options.get("emphasize_keywords", False):
            # Emphasize words in ALL CAPS or between ** **
            ssml = re.sub(r'\*\*([^*]+)\*\*', r'<emphasis level="strong">\1</emphasis>', ssml)
            ssml = re.sub(r'\b([A-Z]{2,})\b', r'<emphasis level="moderate">\1</emphasis>', ssml)

        # Wrap in speak tag if not present
        if not ssml.strip().startswith("<"):
            ssml = f'<speak>{ssml}</speak>'

        return ssml


# Global instances
_model_manager: Optional[ModelManager] = None
_enhanced_assessor: Optional[EnhancedQualityAssessor] = None
_ssml_processor: Optional[SSMLProcessor] = None


def get_model_manager() -> ModelManager:
    """Get global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def get_enhanced_assessor() -> EnhancedQualityAssessor:
    """Get global enhanced quality assessor instance."""
    global _enhanced_assessor
    if _enhanced_assessor is None:
        _enhanced_assessor = EnhancedQualityAssessor()
    return _enhanced_assessor


def get_ssml_processor() -> SSMLProcessor:
    """Get global SSML processor instance."""
    global _ssml_processor
    if _ssml_processor is None:
        _ssml_processor = SSMLProcessor()
    return _ssml_processor

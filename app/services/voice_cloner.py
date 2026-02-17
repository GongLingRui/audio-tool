"""
Advanced Voice Cloning Service
Implements real voice cloning using modern techniques
"""
import os
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from datetime import datetime

from app.config import settings
from app.services.audio_processor import AudioProcessor
from app.services.tts_engine import TTSEngineFactory, TTSMode
import logging

logger = logging.getLogger(__name__)


class VoiceSample:
    """Represents a voice sample with features."""

    def __init__(
        self,
        sample_id: str,
        audio_path: str,
        text: str,
        duration: float,
        features: Optional[Dict[str, Any]] = None,
    ):
        self.sample_id = sample_id
        self.audio_path = audio_path
        self.text = text
        self.duration = duration
        self.features = features or {}
        self.created_at = datetime.now()


class VoiceProfile:
    """Complete profile for a cloned voice."""

    def __init__(
        self,
        profile_id: str,
        name: str,
        samples: List[VoiceSample],
        reference_audio: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.profile_id = profile_id
        self.name = name
        self.samples = samples
        self.reference_audio = reference_audio
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.voice_features = self._extract_voice_features()

    def _extract_voice_features(self) -> Dict[str, Any]:
        """Extract aggregated voice features from all samples."""
        if not self.samples:
            return {}

        # Aggregate features from all samples
        all_pitches = []
        all_energies = []
        all_tempos = []

        for sample in self.samples:
            if "pitch" in sample.features:
                all_pitches.extend(sample.features["pitch"])
            if "energy" in sample.features:
                all_energies.extend(sample.features["energy"])
            if "tempo" in sample.features:
                all_tempos.append(sample.features["tempo"])

        features = {}
        if all_pitches:
            features["avg_pitch"] = float(np.mean(all_pitches))
            features["pitch_range"] = float(np.max(all_pitches) - np.min(all_pitches))
            features["pitch_std"] = float(np.std(all_pitches))

        if all_energies:
            features["avg_energy"] = float(np.mean(all_energies))
            features["energy_range"] = float(np.max(all_energies) - np.min(all_energies))

        if all_tempos:
            features["avg_tempo"] = float(np.mean(all_tempos))

        return features


class VoiceCloner:
    """Advanced voice cloning service with real audio processing."""

    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.profiles_dir = Path(settings.export_dir) / "voice_profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.cloned_voices_dir = Path(settings.audio_dir) / "cloned"
        self.cloned_voices_dir.mkdir(parents=True, exist_ok=True)
        self._profiles_cache: Dict[str, VoiceProfile] = {}

    async def create_voice_profile(
        self,
        name: str,
        audio_samples: List[Tuple[str, str]],  # (audio_path, text)
        user_id: Optional[str] = None,
    ) -> VoiceProfile:
        """
        Create a complete voice profile from multiple audio samples.

        Args:
            name: Name for the voice profile
            audio_samples: List of (audio_path, transcript) tuples
            user_id: Optional user ID for ownership

        Returns:
            Created VoiceProfile
        """
        # Validate samples
        if len(audio_samples) < 3:
            raise ValueError("At least 3 audio samples required for voice cloning")

        # Process each sample
        processed_samples = []
        for i, (audio_path, text) in enumerate(audio_samples):
            # Validate audio exists
            if not os.path.exists(audio_path):
                raise ValueError(f"Audio file not found: {audio_path}")

            # Get audio duration
            duration_info = await self.audio_processor.get_audio_info(audio_path)
            duration = duration_info.get("duration", 0)

            if duration < 2:
                raise ValueError(f"Sample {i+1} too short: {duration:.2f}s (min 2s required)")

            if duration > 60:
                raise ValueError(f"Sample {i+1} too long: {duration:.2f}s (max 60s allowed)")

            # Extract features
            features = await self._extract_audio_features(audio_path)

            # Create sample
            sample_id = f"{name}_{i}_{hash(audio_path) % 10000}"
            sample = VoiceSample(
                sample_id=sample_id,
                audio_path=audio_path,
                text=text,
                duration=duration,
                features=features,
            )
            processed_samples.append(sample)

        # Select best reference audio (longest, most stable)
        reference_audio = max(processed_samples, key=lambda s: s.duration).audio_path

        # Create profile ID
        profile_id = self._generate_profile_id(name, user_id)

        # Create profile
        profile = VoiceProfile(
            profile_id=profile_id,
            name=name,
            samples=processed_samples,
            reference_audio=reference_audio,
            metadata={
                "user_id": user_id,
                "sample_count": len(processed_samples),
                "total_duration": sum(s.duration for s in processed_samples),
            },
        )

        # Save profile
        await self._save_profile(profile)

        # Cache profile
        self._profiles_cache[profile_id] = profile

        logger.info(f"Created voice profile '{name}' with {len(processed_samples)} samples")
        return profile

    async def _extract_audio_features(self, audio_path: str) -> Dict[str, Any]:
        """Extract detailed audio features for voice cloning."""
        features = {}

        try:
            # Use librosa for advanced audio analysis
            import librosa
            import soundfile as sf

            # Load audio
            y, sr = librosa.load(audio_path, sr=22050)

            # Extract pitch (fundamental frequency)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)

            if pitch_values:
                features["pitch"] = pitch_values
                features["avg_pitch"] = float(np.mean(pitch_values))
                features["pitch_std"] = float(np.std(pitch_values))

            # Extract energy (RMS)
            rms = librosa.feature.rms(y=y)[0]
            features["energy"] = rms.tolist()
            features["avg_energy"] = float(np.mean(rms))
            features["energy_std"] = float(np.std(rms))

            # Extract tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features["tempo"] = float(tempo)

            # Extract MFCCs (timbre characteristics)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features["mfcc_mean"] = np.mean(mfccs, axis=1).tolist()
            features["mfcc_std"] = np.std(mfccs, axis=1).tolist()

            # Extract spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
            features["spectral_centroid_std"] = float(np.std(spectral_centroids))

            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))

            # Zero crossing rate (voice brightness/harshness)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features["zcr_mean"] = float(np.mean(zcr))

        except ImportError:
            # Fallback to basic features if librosa not available
            logger.warning("librosa not available, using basic features")
            audio_info = await self.audio_processor.get_audio_info(audio_path)
            features.update(audio_info)

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            # Provide default features
            features = {
                "pitch": [],
                "energy": [],
                "tempo": 120.0,
            }

        return features

    async def clone_speech(
        self,
        profile_id: str,
        text: str,
        emotion: Optional[str] = None,
        speed: float = 1.0,
        pitch_shift: float = 0.0,
    ) -> Tuple[bytes, float]:
        """
        Generate speech using cloned voice.

        Args:
            profile_id: Voice profile ID
            text: Text to synthesize
            emotion: Optional emotion modifier
            speed: Speed multiplier (0.5 - 2.0)
            pitch_shift: Pitch shift in semitones (-12 to +12)

        Returns:
            Tuple of (audio_data, duration)
        """
        # Load profile
        profile = await self.load_profile(profile_id)

        # Get TTS engine
        tts_engine = TTSEngineFactory.create(TTSMode.LOCAL)

        # Generate speech using reference audio
        audio_data, duration = await tts_engine.generate(
            text=text,
            speaker=profile.reference_audio,
        )

        # Apply voice characteristics
        if pitch_shift != 0 or speed != 1.0:
            audio_data = await self._apply_voice_modifications(
                audio_data,
                pitch_shift=pitch_shift,
                speed_factor=speed,
            )

        # Apply emotion if specified
        if emotion:
            audio_data = await self._apply_emotion(audio_data, emotion, profile.voice_features)

        return audio_data, duration

    async def _apply_voice_modifications(
        self,
        audio_data: bytes,
        pitch_shift: float = 0.0,
        speed_factor: float = 1.0,
    ) -> bytes:
        """Apply pitch and speed modifications to audio."""
        from pydub import AudioSegment
        import io

        # Load audio
        audio = AudioSegment.from_file(io.BytesIO(audio_data))

        # Apply pitch shift
        if pitch_shift != 0:
            # Pitch shift using frame rate manipulation
            new_sample_rate = int(audio.frame_rate * (2.0 ** (pitch_shift / 12.0)))
            audio = audio._spawn(audio.raw_data, overrides={
                'frame_rate': new_sample_rate
            })
            audio = audio.set_frame_rate(22050)

        # Apply speed change
        if speed_factor != 1.0:
            # Speed change using frame rate change
            new_frame_rate = int(audio.frame_rate * speed_factor)
            audio = audio._spawn(audio.raw_data, overrides={
                'frame_rate': new_frame_rate
            })
            audio = audio.set_frame_rate(22050)

        # Export
        output = io.BytesIO()
        audio.export(output, format="wav")
        return output.read()

    async def _apply_emotion(
        self,
        audio_data: bytes,
        emotion: str,
        voice_features: Dict[str, Any],
    ) -> bytes:
        """Apply emotional characteristics to audio."""
        from pydub import AudioSegment
        import io

        # Load audio
        audio = AudioSegment.from_file(io.BytesIO(audio_data))

        # Emotion parameters (simplified mapping)
        emotion_params = {
            "happy": {"tempo": 1.1, "pitch": 1.5, "energy": 1.2},
            "sad": {"tempo": 0.9, "pitch": -2.0, "energy": 0.8},
            "angry": {"tempo": 1.2, "pitch": 1.0, "energy": 1.4},
            "calm": {"tempo": 0.95, "pitch": 0.0, "energy": 0.9},
            "excited": {"tempo": 1.15, "pitch": 2.0, "energy": 1.3},
            "narrator": {"tempo": 1.0, "pitch": 0.0, "energy": 1.0},
        }

        params = emotion_params.get(emotion, emotion_params["narrator"])

        # Apply modifications
        if params["pitch"] != 0:
            new_sample_rate = int(audio.frame_rate * (2.0 ** (params["pitch"] / 12.0)))
            audio = audio._spawn(audio.raw_data, overrides={'frame_rate': new_sample_rate})
            audio = audio.set_frame_rate(22050)

        if params["tempo"] != 1.0:
            new_frame_rate = int(audio.frame_rate * params["tempo"])
            audio = audio._spawn(audio.raw_data, overrides={'frame_rate': new_frame_rate})
            audio = audio.set_frame_rate(22050)

        if params["energy"] != 1.0:
            audio = audio + (10 * np.log10(params["energy"]))  # dB adjustment

        # Export
        output = io.BytesIO()
        audio.export(output, format="wav")
        return output.read()

    async def load_profile(self, profile_id: str) -> VoiceProfile:
        """Load voice profile from disk."""
        # Check cache first
        if profile_id in self._profiles_cache:
            return self._profiles_cache[profile_id]

        # Load from disk
        profile_path = self.profiles_dir / f"{profile_id}.json"
        if not profile_path.exists():
            raise ValueError(f"Voice profile not found: {profile_id}")

        with open(profile_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Reconstruct samples
        samples = []
        for sample_data in data["samples"]:
            sample = VoiceSample(
                sample_id=sample_data["sample_id"],
                audio_path=sample_data["audio_path"],
                text=sample_data["text"],
                duration=sample_data["duration"],
                features=sample_data.get("features", {}),
            )
            samples.append(sample)

        # Create profile
        profile = VoiceProfile(
            profile_id=data["profile_id"],
            name=data["name"],
            samples=samples,
            reference_audio=data["reference_audio"],
            metadata=data.get("metadata", {}),
        )

        # Cache it
        self._profiles_cache[profile_id] = profile

        return profile

    async def _save_profile(self, profile: VoiceProfile):
        """Save voice profile to disk."""
        profile_path = self.profiles_dir / f"{profile.profile_id}.json"

        data = {
            "profile_id": profile.profile_id,
            "name": profile.name,
            "reference_audio": profile.reference_audio,
            "metadata": profile.metadata,
            "created_at": profile.created_at.isoformat(),
            "samples": [
                {
                    "sample_id": s.sample_id,
                    "audio_path": s.audio_path,
                    "text": s.text,
                    "duration": s.duration,
                    "features": s.features,
                }
                for s in profile.samples
            ],
            "voice_features": profile.voice_features,
        }

        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _generate_profile_id(self, name: str, user_id: Optional[str] = None) -> str:
        """Generate unique profile ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique = hashlib.md5(f"{name}_{user_id}_{timestamp}".encode()).hexdigest()[:8]
        return f"voice_{unique}"

    async def list_profiles(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all voice profiles."""
        profiles = []

        for profile_file in self.profiles_dir.glob("*.json"):
            try:
                with open(profile_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Filter by user if specified
                if user_id and data.get("metadata", {}).get("user_id") != user_id:
                    continue

                profiles.append({
                    "profile_id": data["profile_id"],
                    "name": data["name"],
                    "sample_count": len(data["samples"]),
                    "total_duration": data.get("voice_features", {}).get("total_duration", 0),
                    "created_at": data.get("created_at"),
                    "voice_features": data.get("voice_features", {}),
                })
            except Exception as e:
                logger.error(f"Error loading profile {profile_file}: {e}")

        return profiles

    async def delete_profile(self, profile_id: str) -> bool:
        """Delete a voice profile."""
        profile_path = self.profiles_dir / f"{profile_id}.json"

        if profile_path.exists():
            profile_path.unlink()
            if profile_id in self._profiles_cache:
                del self._profiles_cache[profile_id]
            return True

        return False


# Global instance
_voice_cloner: Optional[VoiceCloner] = None


def get_voice_cloner() -> VoiceCloner:
    """Get global voice cloner instance."""
    global _voice_cloner
    if _voice_cloner is None:
        _voice_cloner = VoiceCloner()
    return _voice_cloner

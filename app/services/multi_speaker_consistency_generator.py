"""
Multi-Speaker Consistency Generator - Voice consistency across speakers
Maintains consistent voice characteristics for multiple speakers in long-form content
"""
import asyncio
import hashlib
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import pickle

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


@dataclass
class SpeakerProfile:
    """Profile for a speaker with voice characteristics."""
    speaker_id: str
    name: str
    reference_audio: Optional[bytes] = None
    voice_embedding: Optional[np.ndarray] = None
    characteristics: Dict[str, Any] = field(default_factory=dict)
    average_pitch: float = 0.0
    average_speed: float = 1.0
    emotional_range: Dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'speaker_id': self.speaker_id,
            'name': self.name,
            'characteristics': self.characteristics,
            'average_pitch': self.average_pitch,
            'average_speed': self.average_speed,
            'emotional_range': self.emotional_range,
            'created_at': self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpeakerProfile':
        """Create from dictionary."""
        return cls(
            speaker_id=data['speaker_id'],
            name=data['name'],
            characteristics=data.get('characteristics', {}),
            average_pitch=data.get('average_pitch', 0.0),
            average_speed=data.get('average_speed', 1.0),
            emotional_range=data.get('emotional_range', {}),
            created_at=data.get('created_at', time.time()),
        )


@dataclass
class DialogueSegment:
    """A segment of dialogue with speaker info."""
    speaker_id: str
    text: str
    emotion: Optional[str] = None
    emphasis: List[str] = field(default_factory=list)
    pause_before: float = 0.0
    pause_after: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiSpeakerConsistencyGenerator:
    """
    Multi-speaker consistency generator for maintaining voice characteristics.

    Features:
    - Speaker profile management and learning
    - Voice consistency tracking across sessions
    - Automatic speaker characteristic extraction
    - Dialogue-based audio generation
    - Cross-session voice consistency
    - Emotional range tracking per speaker
    """

    def __init__(
        self,
        profile_dir: str = "./data/speaker_profiles",
        enable_learning: bool = True,
        consistency_threshold: float = 0.85,
    ):
        """
        Initialize multi-speaker consistency generator.

        Args:
            profile_dir: Directory to store speaker profiles
            enable_learning: Enable automatic profile learning
            consistency_threshold: Minimum similarity for consistency check
        """
        self.profile_dir = Path(profile_dir)
        self.profile_dir.mkdir(parents=True, exist_ok=True)

        self.enable_learning = enable_learning
        self.consistency_threshold = consistency_threshold

        # Speaker profiles storage
        self._profiles: Dict[str, SpeakerProfile] = {}
        self._load_profiles()

    def _load_profiles(self):
        """Load speaker profiles from disk."""
        try:
            for profile_file in self.profile_dir.glob("*.json"):
                with open(profile_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    profile = SpeakerProfile.from_dict(data)
                    self._profiles[profile.speaker_id] = profile

            logger.info(f"Loaded {len(self._profiles)} speaker profiles")
        except Exception as e:
            logger.warning(f"Error loading profiles: {e}")

    def _save_profile(self, profile: SpeakerProfile):
        """Save speaker profile to disk."""
        try:
            profile_file = self.profile_dir / f"{profile.speaker_id}.json"
            with open(profile_file, 'w', encoding='utf-8') as f:
                json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving profile {profile.speaker_id}: {e}")

    async def create_speaker_profile(
        self,
        speaker_id: str,
        name: str,
        reference_audio: Optional[bytes] = None,
        voice_config: Optional[Dict[str, Any]] = None,
    ) -> SpeakerProfile:
        """
        Create a new speaker profile.

        Args:
            speaker_id: Unique speaker identifier
            name: Speaker display name
            reference_audio: Reference audio sample (optional)
            voice_config: Voice configuration parameters (optional)

        Returns:
            SpeakerProfile object
        """
        profile = SpeakerProfile(
            speaker_id=speaker_id,
            name=name,
            reference_audio=reference_audio,
        )

        # Extract characteristics from reference audio
        if reference_audio:
            profile.characteristics = await self._extract_characteristics(reference_audio)
            profile.voice_embedding = await self._extract_embedding(reference_audio)

        # Add voice config if provided
        if voice_config:
            profile.characteristics.update(voice_config)

        self._profiles[speaker_id] = profile
        self._save_profile(profile)

        logger.info(f"Created speaker profile: {name} ({speaker_id})")
        return profile

    async def _extract_characteristics(self, audio: bytes) -> Dict[str, Any]:
        """Extract voice characteristics from audio."""
        try:
            audio_seg = AudioSegment.from_file(
                __import__('io').BytesIO(audio)
            )

            characteristics = {
                'duration': len(audio_seg) / 1000.0,
                'sample_rate': audio_seg.frame_rate,
                'channels': audio_seg.channels,
                'dbfs': audio_seg.dBFS,
                'max_dbfs': audio_seg.max_dBFS,
                'dynamic_range': audio_seg.max_dBFS - audio_seg.min_possible_dBFS,
            }

            return characteristics
        except Exception as e:
            logger.warning(f"Error extracting characteristics: {e}")
            return {}

    async def _extract_embedding(self, audio: bytes) -> Optional[np.ndarray]:
        """
        Extract voice embedding for similarity comparison.

        This is a simplified implementation.
        For production, integrate with:
        - Resemblyzer: https://github.com/resemble-ai/Resemblyzer
        - SpeechBrain ECAPA: https://github.com/speechbrain/speechbrain
        """
        try:
            audio_seg = AudioSegment.from_file(
                __import__('io').BytesIO(audio)
            )

            samples = np.array(audio_seg.get_array_of_samples())

            # Resample to 16kHz if needed
            if audio_seg.frame_rate != 16000:
                samples = samples[::max(1, audio_seg.frame_rate // 16000)]

            # Normalize
            if len(samples) > 0 and np.max(np.abs(samples)) > 0:
                samples = samples.astype(float)
                samples = samples / np.max(np.abs(samples))

            # Extract simple MFCC-like features
            features = []
            frame_size = 512
            for i in range(0, len(samples), frame_size):
                frame = samples[i:i + frame_size]
                if len(frame) == frame_size:
                    rms = np.sqrt(np.mean(frame ** 2))
                    features.append(rms)

            # Take first 256 features as embedding
            features = np.array(features[:256]) if len(features) >= 256 else np.array(features)

            # Pad if needed
            if len(features) < 256:
                features = np.pad(features, (0, 256 - len(features)))

            return features
        except Exception as e:
            logger.warning(f"Error extracting embedding: {e}")
            return None

    async def check_consistency(
        self,
        speaker_id: str,
        audio: bytes,
    ) -> Tuple[bool, float]:
        """
        Check if audio matches speaker profile.

        Args:
            speaker_id: Speaker to check against
            audio: Audio to check

        Returns:
            Tuple of (is_consistent, similarity_score)
        """
        if speaker_id not in self._profiles:
            return False, 0.0

        profile = self._profiles[speaker_id]

        if profile.voice_embedding is None:
            return True, 1.0  # No reference, accept

        # Extract embedding from new audio
        new_embedding = await self._extract_embedding(audio)
        if new_embedding is None:
            return True, 1.0

        # Calculate cosine similarity
        similarity = self._cosine_similarity(profile.voice_embedding, new_embedding)

        is_consistent = similarity >= self.consistency_threshold

        return is_consistent, similarity

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    async def generate_with_consistency(
        self,
        segments: List[DialogueSegment],
        generator: Callable,
        reference_profile: Optional[SpeakerProfile] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate audio for dialogue segments with consistency checks.

        Args:
            segments: List of dialogue segments
            generator: Audio generator function
            reference_profile: Reference profile for consistency

        Returns:
            List of generated audio segments with metadata
        """
        results = []

        for segment in segments:
            start_time = time.time()

            try:
                # Get speaker profile
                speaker_id = segment.speaker_id
                if speaker_id in self._profiles:
                    profile = self._profiles[speaker_id]
                    voice_config = profile.characteristics.copy()
                else:
                    voice_config = {}

                # Add emotion if specified
                if segment.emotion:
                    voice_config['emotion'] = segment.emotion

                # Generate audio
                audio = await generator(
                    segment.text,
                    voice_config,
                    segment.emotion,
                )

                generation_time = time.time() - start_time

                # Check consistency if profile exists
                is_consistent = True
                similarity = 1.0

                if speaker_id in self._profiles and reference_profile:
                    is_consistent, similarity = await self.check_consistency(
                        speaker_id, audio
                    )

                results.append({
                    'speaker_id': speaker_id,
                    'text': segment.text,
                    'audio': audio,
                    'is_consistent': is_consistent,
                    'similarity': similarity,
                    'generation_time': generation_time,
                    'metadata': segment.metadata,
                })

                # Learn from this generation if enabled
                if self.enable_learning and speaker_id in self._profiles:
                    await self._update_profile_from_audio(
                        speaker_id, audio
                    )

            except Exception as e:
                logger.error(f"Error generating segment for {segment.speaker_id}: {e}")
                results.append({
                    'speaker_id': segment.speaker_id,
                    'text': segment.text,
                    'error': str(e),
                })

        return results

    async def _update_profile_from_audio(
        self,
        speaker_id: str,
        audio: bytes,
    ):
        """Update speaker profile from generated audio."""
        if speaker_id not in self._profiles:
            return

        profile = self._profiles[speaker_id]

        # Update embedding (moving average)
        new_embedding = await self._extract_embedding(audio)
        if new_embedding is not None and profile.voice_embedding is not None:
            # Update with exponential moving average
            alpha = 0.1  # Learning rate
            profile.voice_embedding = (
                alpha * new_embedding +
                (1 - alpha) * profile.voice_embedding
            )

    async def generate_dialogue(
        self,
        dialogue_text: str,
        speaker_mapping: Dict[str, str],
        generator: Callable,
    ) -> bytes:
        """
        Generate full dialogue audio.

        Args:
            dialogue_text: Full dialogue text with speaker markers
            speaker_mapping: Mapping of speaker markers to speaker_ids
            generator: Audio generator function

        Returns:
            Complete dialogue audio bytes
        """
        # Parse dialogue into segments
        segments = await self._parse_dialogue(dialogue_text, speaker_mapping)

        # Generate with consistency
        results = await self.generate_with_consistency(segments, generator)

        # Combine audio segments
        combined = await self._combine_dialogue_audio(results)

        return combined

    async def _parse_dialogue(
        self,
        dialogue_text: str,
        speaker_mapping: Dict[str, str],
    ) -> List[DialogueSegment]:
        """Parse dialogue text into segments."""
        segments = []

        lines = dialogue_text.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for speaker marker format: "Speaker: Text"
            if ':' in line:
                parts = line.split(':', 1)
                speaker_marker = parts[0].strip()
                text = parts[1].strip()

                speaker_id = speaker_mapping.get(speaker_marker)

                if speaker_id:
                    segments.append(DialogueSegment(
                        speaker_id=speaker_id,
                        text=text,
                    ))

        return segments

    async def _combine_dialogue_audio(
        self,
        results: List[Dict[str, Any]],
    ) -> bytes:
        """Combine dialogue segments into single audio."""
        combined_segments = []

        for result in results:
            if 'audio' in result and result['audio']:
                audio_seg = AudioSegment.from_file(
                    __import__('io').BytesIO(result['audio'])
                )
                combined_segments.append(audio_seg)

                # Add pause after if specified
                pause_duration = result.get('metadata', {}).get('pause_after', 0)
                if pause_duration > 0:
                    combined_segments.append(
                        AudioSegment.silent(duration=int(pause_duration * 1000))
                    )

        if not combined_segments:
            return b""

        # Combine all segments
        result = sum(combined_segments)

        # Export
        output = __import__('io').BytesIO()
        result.export(output, format="mp3", bitrate="192k")
        return output.read()

    def get_speaker_profile(self, speaker_id: str) -> Optional[SpeakerProfile]:
        """Get speaker profile by ID."""
        return self._profiles.get(speaker_id)

    def list_speakers(self) -> List[Dict[str, Any]]:
        """List all speaker profiles."""
        return [
            {
                'speaker_id': p.speaker_id,
                'name': p.name,
                'has_reference': p.reference_audio is not None,
                'created_at': p.created_at,
            }
            for p in self._profiles.values()
        ]

    async def update_speaker_profile(
        self,
        speaker_id: str,
        reference_audio: Optional[bytes] = None,
        characteristics: Optional[Dict[str, Any]] = None,
    ) -> Optional[SpeakerProfile]:
        """Update existing speaker profile."""
        if speaker_id not in self._profiles:
            return None

        profile = self._profiles[speaker_id]

        if reference_audio:
            profile.reference_audio = reference_audio
            profile.voice_embedding = await self._extract_embedding(reference_audio)
            profile.characteristics = await self._extract_characteristics(reference_audio)

        if characteristics:
            profile.characteristics.update(characteristics)

        self._save_profile(profile)

        return profile

    async def clone_voice_characteristics(
        self,
        source_audio: bytes,
        target_speaker_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Clone voice characteristics from source audio to target speaker.

        Args:
            source_audio: Audio to extract characteristics from
            target_speaker_id: Target speaker to update

        Returns:
            Updated voice configuration
        """
        if target_speaker_id not in self._profiles:
            return None

        # Extract characteristics
        characteristics = await self._extract_characteristics(source_audio)

        # Update target profile
        profile = await self.update_speaker_profile(
            target_speaker_id,
            reference_audio=source_audio,
            characteristics=characteristics,
        )

        return profile.characteristics if profile else None

    async def compare_speakers(
        self,
        speaker_id1: str,
        speaker_id2: str,
    ) -> Dict[str, Any]:
        """
        Compare two speaker profiles.

        Returns:
            Comparison results with similarity metrics
        """
        if speaker_id1 not in self._profiles or speaker_id2 not in self._profiles:
            return {'error': 'One or both speakers not found'}

        profile1 = self._profiles[speaker_id1]
        profile2 = self._profiles[speaker_id2]

        result = {
            'speaker1': profile1.name,
            'speaker2': profile2.name,
            'characteristics_similarity': 0.0,
            'voice_similarity': 0.0,
        }

        # Compare characteristics
        if profile1.voice_embedding is not None and profile2.voice_embedding is not None:
            result['voice_similarity'] = self._cosine_similarity(
                profile1.voice_embedding,
                profile2.voice_embedding,
            )

        # Compare audio characteristics
        char1 = profile1.characteristics
        char2 = profile2.characteristics

        if char1 and char2:
            # Simple characteristic comparison
            similarities = []
            for key in ['dbfs', 'dynamic_range']:
                if key in char1 and key in char2:
                    val1 = char1[key]
                    val2 = char2[key]
                    if val1 != 0 and val2 != 0:
                        sim = 1.0 - abs(val1 - val2) / max(abs(val1), abs(val2))
                        similarities.append(sim)

            if similarities:
                result['characteristics_similarity'] = sum(similarities) / len(similarities)

        return result

    async def batch_check_consistency(
        self,
        speaker_id: str,
        audio_files: List[bytes],
    ) -> List[Dict[str, Any]]:
        """
        Check consistency for multiple audio files against a speaker.

        Args:
            speaker_id: Speaker to check against
            audio_files: List of audio files to check

        Returns:
            List of consistency results
        """
        results = []

        for i, audio in enumerate(audio_files):
            is_consistent, similarity = await self.check_consistency(
                speaker_id, audio
            )

            results.append({
                'index': i,
                'is_consistent': is_consistent,
                'similarity': similarity,
            })

        return results


# Global instance
_multi_speaker_generator: Optional[MultiSpeakerConsistencyGenerator] = None


def get_multi_speaker_generator(
    profile_dir: str = "./data/speaker_profiles",
    enable_learning: bool = True,
) -> MultiSpeakerConsistencyGenerator:
    """Get global multi-speaker consistency generator instance."""
    global _multi_speaker_generator
    if _multi_speaker_generator is None:
        _multi_speaker_generator = MultiSpeakerConsistencyGenerator(
            profile_dir=profile_dir,
            enable_learning=enable_learning,
        )
    return _multi_speaker_generator

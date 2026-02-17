"""
Emotion Gradient Service
Provides emotion interpolation and transition for smooth audio generation
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TransitionType(Enum):
    """Types of emotion transitions."""
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"


@dataclass
class EmotionKeyframe:
    """Emotion state at a specific point."""
    position: float  # 0.0 to 1.0 (relative position in text/chunk)
    emotion: Dict[str, float]
    transition: TransitionType = TransitionType.LINEAR


@dataclass
class EmotionGradient:
    """Defines emotion transition over a text segment."""
    start_emotion: Dict[str, float]
    end_emotion: Dict[str, float]
    transition_type: TransitionType = TransitionType.LINEAR
    duration_ratio: float = 1.0  # Portion of segment where transition occurs


class EmotionGradientService:
    """
    Service for creating smooth emotion transitions in audio generation.

    Provides:
    - Linear and eased interpolation between emotion states
    - Multi-keyframe emotion curves
    - Per-chunk emotion state calculation
    - Natural emotion progression across long passages
    """

    # Emotion parameter ranges for clamping
    EMOTION_RANGES = {
        "happiness": (0, 1),
        "sadness": (0, 1),
        "anger": (0, 1),
        "fear": (0, 1),
        "surprise": (0, 1),
        "neutral": (0, 1),
        "energy": (0.5, 2.0),
        "tempo": (0.5, 2.0),
        "pitch": (-6, 6),
        "volume": (0, 2.0),
    }

    @classmethod
    def clamp_emotion_value(cls, key: str, value: float) -> float:
        """Clamp emotion value to valid range."""
        if key in cls.EMOTION_RANGES:
            min_val, max_val = cls.EMOTION_RANGES[key]
            return max(min_val, min(max_val, value))
        return value

    @classmethod
    def interpolate_emotion(
        cls,
        start: Dict[str, float],
        end: Dict[str, float],
        t: float,
        transition: TransitionType = TransitionType.LINEAR,
    ) -> Dict[str, float]:
        """
        Interpolate between two emotion states.

        Args:
            start: Starting emotion state
            end: Ending emotion state
            t: Interpolation factor (0.0 to 1.0)
            transition: Type of transition/easing

        Returns:
            Interpolated emotion state
        """
        # Apply easing
        eased_t = cls._apply_easing(t, transition)

        result = {}

        # Get all emotion keys from both states
        all_keys = set(start.keys()) | set(end.keys())

        for key in all_keys:
            start_val = start.get(key, 0)
            end_val = end.get(key, 0)

            # Interpolate
            if key == "stability":
                # Stability should be handled separately (not part of TTS emotion)
                result[key] = start_val
            else:
                # Linear interpolation
                interpolated = start_val + (end_val - start_val) * eased_t
                result[key] = cls.clamp_emotion_value(key, interpolated)

        return result

    @classmethod
    def _apply_easing(cls, t: float, transition: TransitionType) -> float:
        """Apply easing function to interpolation factor."""
        t = max(0, min(1, t))  # Clamp to [0, 1]

        if transition == TransitionType.LINEAR:
            return t
        elif transition == TransitionType.EASE_IN:
            return t * t
        elif transition == TransitionType.EASE_OUT:
            return 1 - (1 - t) * (1 - t)
        elif transition == TransitionType.EASE_IN_OUT:
            return t * t * (3 - 2 * t)
        else:
            return t

    @classmethod
    def create_gradient(
        cls,
        start_emotion: Dict[str, float],
        end_emotion: Dict[str, float],
        steps: int,
        transition: TransitionType = TransitionType.EASE_IN_OUT,
    ) -> List[Dict[str, float]]:
        """
        Create a gradient of emotion states.

        Args:
            start_emotion: Starting emotion
            end_emotion: Ending emotion
            steps: Number of steps in gradient
            transition: Type of transition

        Returns:
            List of emotion states from start to end
        """
        gradient = []

        for i in range(steps):
            t = i / max(1, steps - 1) if steps > 1 else 0
            emotion = cls.interpolate_emotion(start_emotion, end_emotion, t, transition)
            gradient.append(emotion)

        return gradient

    @classmethod
    def apply_gradient_to_chunks(
        cls,
        chunk_ids: List[str],
        start_emotion: Dict[str, float],
        end_emotion: Dict[str, float],
        transition: TransitionType = TransitionType.EASE_IN_OUT,
    ) -> Dict[str, Dict[str, float]]:
        """
        Apply emotion gradient across multiple chunks.

        Args:
            chunk_ids: List of chunk IDs (in order)
            start_emotion: Emotion for first chunk
            end_emotion: Emotion for last chunk
            transition: Type of transition

        Returns:
            Dictionary mapping chunk_id to emotion state
        """
        chunk_emotions = {}

        if not chunk_ids:
            return chunk_emotions

        if len(chunk_ids) == 1:
            chunk_emotions[chunk_ids[0]] = start_emotion
        else:
            for i, chunk_id in enumerate(chunk_ids):
                t = i / (len(chunk_ids) - 1)
                emotion = cls.interpolate_emotion(
                    start_emotion,
                    end_emotion,
                    t,
                    transition
                )
                chunk_emotions[chunk_id] = emotion

        return chunk_emotions

    @classmethod
    def create_keyframe_curve(
        cls,
        keyframes: List[EmotionKeyframe],
        steps: int,
    ) -> List[Dict[str, float]]:
        """
        Create emotion curve from keyframes.

        Args:
            keyframes: List of emotion keyframes with positions
            steps: Total number of steps

        Returns:
            List of emotion states at each step
        """
        if not keyframes:
            return []

        # Sort keyframes by position
        sorted_keyframes = sorted(keyframes, key=lambda k: k.position)

        # Ensure first and last keyframes exist
        if sorted_keyframes[0].position > 0:
            sorted_keyframes.insert(
                0,
                EmotionKeyframe(position=0.0, emotion=sorted_keyframes[0].emotion.copy())
            )

        if sorted_keyframes[-1].position < 1.0:
            sorted_keyframes.append(
                EmotionKeyframe(position=1.0, emotion=sorted_keyframes[-1].emotion.copy())
            )

        curve = []

        for step in range(steps):
            t = step / max(1, steps - 1)

            # Find surrounding keyframes
            prev_kf = None
            next_kf = None

            for kf in sorted_keyframes:
                if kf.position <= t:
                    prev_kf = kf
                if kf.position >= t and next_kf is None:
                    next_kf = kf
                    break

            if prev_kf is None:
                prev_kf = sorted_keyframes[0]
            if next_kf is None:
                next_kf = sorted_keyframes[-1]

            # Calculate local interpolation factor
            if prev_kf.position == next_kf.position:
                local_t = 0
            else:
                local_t = (t - prev_kf.position) / (next_kf.position - prev_kf.position)

            # Interpolate
            emotion = cls.interpolate_emotion(
                prev_kf.emotion,
                next_kf.emotion,
                local_t,
                next_kf.transition,
            )
            curve.append(emotion)

        return curve

    @classmethod
    def detect_emotion_transition_points(
        cls,
        text: str,
        window_size: int = 3,
    ) -> List[Tuple[int, str]]:
        """
        Detect points where emotion should change based on text analysis.

        Args:
            text: Text to analyze
            window_size: Size of text window to analyze

        Returns:
            List of (position, emotion_type) tuples
        """
        transitions = []

        # Simple keyword-based detection
        # In production, use NLP model for better results

        sentences = text.split('。')
        position = 0

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            # Detect emotion shifts
            if any(word in sentence for word in ['突然', '忽然', '猛然']):
                transitions.append((position, 'sudden'))
            elif any(word in sentence for word in ['慢慢', '渐渐', '逐渐']):
                transitions.append((position, 'gradual'))
            elif any(word in sentence for word in ['但是', '然而', '可是']):
                transitions.append((position, 'contrast'))
            elif any(word in sentence for word in ['终于', '最终', '最后']):
                transitions.append((position, 'resolution'))

            position += len(sentence)

        return transitions

    @classmethod
    def suggest_emotion_curve(
        cls,
        text_segments: List[str],
        start_emotion: Dict[str, float],
        end_emotion: Dict[str, float],
    ) -> List[Dict[str, float]]:
        """
        Suggest emotion curve for text segments.

        Analyzes text and suggests appropriate emotion transitions.

        Args:
            text_segments: List of text segments in order
            start_emotion: Starting emotion
            end_emotion: Target ending emotion

        Returns:
            List of emotion states for each segment
        """
        if not text_segments:
            return []

        # Detect transition points
        full_text = ''.join(text_segments)
        transitions = cls.detect_emotion_transition_points(full_text)

        # If no transitions detected, use simple gradient
        if not transitions:
            return cls.create_gradient(
                start_emotion,
                end_emotion,
                len(text_segments),
                TransitionType.EASE_IN_OUT,
            )

        # Create keyframes from transitions
        keyframes = [
            EmotionKeyframe(position=0.0, emotion=start_emotion.copy())
        ]

        for pos, transition_type in transitions:
            # Calculate position as ratio
            position_ratio = pos / len(full_text)

            # Adjust emotion based on transition type
            emotion = start_emotion.copy()

            if transition_type == 'sudden':
                emotion['energy'] = emotion.get('energy', 1.0) * 1.3
                emotion['tempo'] = emotion.get('tempo', 1.0) * 1.2
            elif transition_type == 'gradual':
                emotion['energy'] = emotion.get('energy', 1.0) * 0.9
                emotion['tempo'] = emotion.get('tempo', 1.0) * 0.95
            elif transition_type == 'contrast':
                # Flip some emotions
                if emotion.get('happiness', 0) > 0.5:
                    emotion['happiness'] = 0.3
                    emotion['sadness'] = 0.4
            elif transition_type == 'resolution':
                emotion['neutral'] = 0.8
                emotion['energy'] = 1.0

            keyframes.append(
                EmotionKeyframe(position=position_ratio, emotion=emotion)
            )

        keyframes.append(
            EmotionKeyframe(position=1.0, emotion=end_emotion.copy())
        )

        # Generate curve from keyframes
        return cls.create_keyframe_curve(keyframes, len(text_segments))


# Singleton instance
_emotion_gradient_service: Optional[EmotionGradientService] = None


def get_emotion_gradient_service() -> EmotionGradientService:
    """Get the singleton emotion gradient service instance."""
    global _emotion_gradient_service
    if _emotion_gradient_service is None:
        _emotion_gradient_service = EmotionGradientService()
    return _emotion_gradient_service

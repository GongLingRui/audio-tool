"""
SSML (Speech Synthesis Markup Language) Processor
Advanced prosody control for TTS
"""
import re
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProsodyControl:
    """Prosody control parameters."""
    rate: float = 1.0  # Speech rate (0.1 - 10.0)
    pitch: float = 1.0  # Pitch multiplier (0.1 - 2.0)
    volume: float = 1.0  # Volume multiplier (0.0 - 2.0)
    emphasis: str = "none"  # none, strong, moderate, reduced
    contour: Optional[List[Tuple[float, float]]] = None  # Pitch contour points


@dataclass
class SSMLSegment:
    """A segment of text with prosody controls."""
    text: str
    prosody: ProsodyControl
    voice: Optional[str] = None
    breaks: List[float] = None  # Break durations in seconds


class SSMLProcessor:
    """Process SSML and extract prosody controls."""

    # SSML tag patterns
    SSML_TAGS = {
        'speak': True,
        'voice': True,
        'prosody': True,
        'break': True,
        'emphasis': True,
        'say-as': True,
        'sub': True,
        'mark': True,
        'p': True,
        's': True,
    }

    def __init__(self):
        self._default_prosody = ProsodyControl()

    def parse_ssml(self, ssml: str) -> List[SSMLSegment]:
        """
        Parse SSML and extract segments with prosody controls.

        Args:
            ssml: SSML string

        Returns:
            List of SSML segments
        """
        segments = []

        try:
            # Remove XML declaration if present
            ssml = re.sub(r'<\?xml[^>]*\?>', '', ssml)

            # Parse XML
            root = ET.fromstring(f"<root>{ssml}</root>")

            # Extract segments
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    prosody = self._extract_prosody(elem)
                    segments.append(SSMLSegment(
                        text=elem.text.strip(),
                        prosody=prosody,
                        voice=elem.get('voice'),
                        breaks=self._extract_breaks(elem),
                    ))

        except ET.ParseError as e:
            logger.warning(f"SSML parsing error: {e}, falling back to plain text")
            segments.append(SSMLSegment(
                text=ssml,
                prosody=self._default_prosody,
                breaks=[],
            ))

        return segments if segments else [SSMLSegment(
            text=ssml,
            prosody=self._default_prosody,
            breaks=[],
        )]

    def _extract_prosody(self, element: ET.Element) -> ProsodyControl:
        """Extract prosody attributes from an element."""
        prosody = ProsodyControl()

        # Check for prosody element
        if element.tag == 'prosody' or element.find('.//prosody') is not None:
            prosody_elem = element if element.tag == 'prosody' else element.find('.//prosody')

            # Rate (speed)
            rate = prosody_elem.get('rate')
            if rate:
                prosody.rate = self._parse_rate(rate)

            # Pitch
            pitch = prosody_elem.get('pitch')
            if pitch:
                prosody.pitch = self._parse_pitch(pitch)

            # Volume
            volume = prosody_elem.get('volume')
            if volume:
                prosody.volume = self._parse_volume(volume)

            # Contour (advanced pitch control)
            contour = prosody_elem.get('contour')
            if contour:
                prosody.contour = self._parse_contour(contour)

        # Check emphasis
        emphasis = element.get('emphasis')
        if emphasis:
            prosody.emphasis = emphasis

        return prosody

    def _extract_breaks(self, element: ET.Element) -> List[float]:
        """Extract break durations from an element."""
        breaks = []

        for break_elem in element.findall('.//break'):
            time_str = break_elem.get('time', '0s')
            strength = break_elem.get('strength', 'medium')

            # Convert strength to time if no time specified
            if time_str == '0s':
                strength_to_time = {
                    'none': 0.0,
                    'x-weak': 0.1,
                    'weak': 0.25,
                    'medium': 0.5,
                    'strong': 1.0,
                    'x-strong': 2.0,
                }
                break_time = strength_to_time.get(strength, 0.5)
            else:
                break_time = self._parse_time(time_str)

            breaks.append(break_time)

        return breaks

    def _parse_rate(self, rate: str) -> float:
        """Parse rate attribute."""
        # Percentage: "50%", "150%", "200%"
        if rate.endswith('%'):
            return float(rate.rstrip('%')) / 100.0

        # Presets: "x-slow", "slow", "medium", "fast", "x-fast"
        presets = {
            'x-slow': 0.5,
            'slow': 0.75,
            'medium': 1.0,
            'fast': 1.25,
            'x-fast': 1.5,
        }
        if rate in presets:
            return presets[rate]

        # Default
        return 1.0

    def _parse_pitch(self, pitch: str) -> float:
        """Parse pitch attribute."""
        # Percentage: "+50%", "-50%", "+50st"
        if pitch.endswith('%'):
            return float(pitch.rstrip('%')) / 100.0

        if pitch.endswith('st'):
            return 1.0 + (float(pitch.rstrip('st')) / 12.0)

        # Absolute values: "+100Hz", "-100Hz"
        if 'Hz' in pitch:
            # Convert to relative multiplier (simplified)
            hz = float(pitch.replace('Hz', '').replace('+', ''))
            return 1.0 + (hz / 200.0)  # Normalize around 200Hz

        # Presets: "x-low", "low", "medium", "high", "x-high"
        presets = {
            'x-low': 0.7,
            'low': 0.85,
            'medium': 1.0,
            'high': 1.15,
            'x-high': 1.3,
        }
        if pitch in presets:
            return presets[pitch]

        return 1.0

    def _parse_volume(self, volume: str) -> float:
        """Parse volume attribute."""
        # Percentage: "50%", "150%"
        if volume.endswith('%'):
            return float(volume.rstrip('%')) / 100.0

        # Presets: "silent", "x-soft", "soft", "medium", "loud", "x-loud", "default"
        presets = {
            'silent': 0.0,
            'x-soft': 0.3,
            'soft': 0.6,
            'medium': 1.0,
            'loud': 1.4,
            'x-loud': 1.8,
            'default': 1.0,
        }
        if volume in presets:
            return presets[volume]

        return 1.0

    def _parse_contour(self, contour: str) -> List[Tuple[float, float]]:
        """Parse pitch contour."""
        # Format: "0% +10st, 50% -5st, 100% +5st"
        points = []
        for point in contour.split(','):
            point = point.strip()
            if not point:
                continue

            try:
                # Parse position and pitch
                parts = point.split()
                if len(parts) == 2:
                    position = float(parts[0].rstrip('%')) / 100.0
                    pitch_str = parts[1]

                    if pitch_str.endswith('st'):
                        pitch = float(pitch_str.rstrip('st'))
                    else:
                        pitch = float(pitch_str)

                    points.append((position, pitch))
            except (ValueError, IndexError):
                continue

        return points

    def _parse_time(self, time_str: str) -> float:
        """Parse time duration."""
        if time_str.endswith('ms'):
            return float(time_str.rstrip('ms')) / 1000.0
        elif time_str.endswith('s'):
            return float(time_str.rstrip('s'))
        else:
            return float(time_str)

    def generate_ssml(
        self,
        text: str,
        rate: Optional[float] = None,
        pitch: Optional[float] = None,
        volume: Optional[float] = None,
        emphasis: Optional[str] = None,
        voice: Optional[str] = None,
    ) -> str:
        """
        Generate SSML from text and parameters.

        Args:
            text: Text to synthesize
            rate: Speech rate (0.1 - 10.0)
            pitch: Pitch multiplier (0.1 - 2.0)
            volume: Volume multiplier (0.0 - 2.0)
            emphasis: Emphasis level
            voice: Voice name

        Returns:
            SSML string
        """
        ssml_parts = ['<speak>']

        # Add voice if specified
        if voice:
            ssml_parts.append(f'<voice name="{voice}">')

        # Add prosody if any parameters specified
        if rate or pitch or volume:
            prosody_attrs = []
            if rate:
                prosody_attrs.append(f'rate="{rate * 100}%"')
            if pitch:
                prosody_attrs.append(f'pitch="{(pitch - 1) * 100}%"')
            if volume:
                prosody_attrs.append(f'volume="{volume * 100}%"')

            ssml_parts.append(f'<prosody {" ".join(prosody_attrs)}>')

        # Add emphasis if specified
        if emphasis:
            ssml_parts.append(f'<emphasis level="{emphasis}">{text}</emphasis>')
        else:
            ssml_parts.append(text)

        # Close tags
        if rate or pitch or volume:
            ssml_parts.append('</prosody>')
        if voice:
            ssml_parts.append('</voice>')
        ssml_parts.append('</speak>')

        return ''.join(ssml_parts)

    def convert_to_tts_params(
        self,
        ssml: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Convert SSML to plain text and TTS parameters.

        Args:
            ssml: SSML string

        Returns:
            Tuple of (plain_text, tts_params)
        """
        segments = self.parse_ssml(ssml)

        if not segments:
            return ssml, {}

        # Extract plain text
        plain_text = ' '.join(seg.text for seg in segments)

        # Use prosody from first segment for overall parameters
        prosody = segments[0].prosody

        tts_params = {
            'rate': prosody.rate,
            'pitch': prosody.pitch,
            'volume': prosody.volume,
            'emphasis': prosody.emphasis,
        }

        # Add breaks if any
        if segments[0].breaks:
            tts_params['breaks'] = segments[0].breaks

        return plain_text, tts_params


class ProsodyBuilder:
    """Builder for creating prosody controls programmatically."""

    def __init__(self):
        self._controls: List[Dict[str, Any]] = []
        self._current_prosody = ProsodyControl()

    def rate(self, value: float) -> 'ProsodyBuilder':
        """Set speech rate."""
        self._current_prosody.rate = value
        return self

    def pitch(self, value: float) -> 'ProsodyBuilder':
        """Set pitch."""
        self._current_prosody.pitch = value
        return self

    def volume(self, value: float) -> 'ProsodyBuilder':
        """Set volume."""
        self._current_prosody.volume = value
        return self

    def emphasis(self, level: str) -> 'ProsodyBuilder':
        """Set emphasis."""
        self._current_prosody.emphasis = level
        return self

    def build(self) -> ProsodyControl:
        """Build the prosody control."""
        return self._current_prosody


# Helper functions
def create_prosody(
    rate: float = 1.0,
    pitch: float = 1.0,
    volume: float = 1.0,
    emphasis: str = "none",
) -> ProsodyControl:
    """Create a prosody control object."""
    return ProsodyControl(
        rate=rate,
        pitch=pitch,
        volume=volume,
        emphasis=emphasis,
    )


# Preset prosodies for common use cases
PROSODY_PRESETS = {
    "narrator": ProsodyControl(rate=1.0, pitch=1.0, volume=1.0, emphasis="none"),
    "excited": ProsodyControl(rate=1.2, pitch=1.1, volume=1.2, emphasis="moderate"),
    "sad": ProsodyControl(rate=0.9, pitch=0.9, volume=0.8, emphasis="reduced"),
    "angry": ProsodyControl(rate=1.1, pitch=1.05, volume=1.3, emphasis="strong"),
    "whisper": ProsodyControl(rate=0.8, pitch=0.95, volume=0.5, emphasis="reduced"),
    "announcement": ProsodyControl(rate=0.95, pitch=1.0, volume=1.2, emphasis="moderate"),
    "question": ProsodyControl(rate=1.0, pitch=1.1, volume=1.0, emphasis="moderate"),
    "exclamation": ProsodyControl(rate=1.1, pitch=1.05, volume=1.3, emphasis="strong"),
}


def get_preset(name: str) -> Optional[ProsodyControl]:
    """Get a prosody preset by name."""
    return PROSODY_PRESETS.get(name)


# Global instance
_ssml_processor: Optional[SSMLProcessor] = None


def get_ssml_processor() -> SSMLProcessor:
    """Get global SSML processor instance."""
    global _ssml_processor
    if _ssml_processor is None:
        _ssml_processor = SSMLProcessor()
    return _ssml_processor

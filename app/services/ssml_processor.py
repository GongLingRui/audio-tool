"""
Complete SSML (Speech Synthesis Markup Language) Processor - W3C SSML 1.1
Advanced prosody control for TTS with full special text handling
"""
import re
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
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
    range: Optional[float] = None  # Pitch range (0.0 - 2.0)


@dataclass
class SSMLSegment:
    """A segment of text with prosody controls."""
    text: str
    original_text: str = ""
    prosody: 'ProsodyControl' = field(default_factory=lambda: ProsodyControl())
    voice: Optional[str] = None
    breaks: List[float] = field(default_factory=list)  # Break durations in seconds
    say_as: Optional[Dict[str, str]] = None  # say-as interpretation
    phoneme: Optional[str] = None  # IPA phoneme
    emphasis_level: Optional[str] = None  # emphasis level
    mark_name: Optional[str] = None  # mark name for synchronization
    audio_src: Optional[str] = None  # audio element source
    sub_alias: Optional[str] = None  # substitution alias


class CompleteSSMLProcessor:
    """
    Complete SSML processor implementing W3C SSML 1.1 standard.

    Supported tags:
    - speak: Root element
    - voice: Voice selection
    - prosody: Prosody control (rate, pitch, volume, contour, range)
    - break: Pauses (time, strength)
    - emphasis: Emphasis level (strong, moderate, reduced)
    - say-as: Special text interpretation (date, time, number, currency, telephone, characters)
    - phoneme: Phonemic pronunciation (IPA)
    - sub: Text substitution
    - mark: Timeline markers
    - audio: Embedded audio
    - p: Paragraphs
    - s: Sentences
    """

    # Rate presets to multiplier mapping
    RATE_PRESETS = {
        'x-slow': 0.5,
        'slow': 0.75,
        'medium': 1.0,
        'fast': 1.25,
        'x-fast': 1.5,
    }

    # Pitch presets to multiplier mapping
    PITCH_PRESETS = {
        'x-low': 0.7,
        'low': 0.85,
        'medium': 1.0,
        'high': 1.15,
        'x-high': 1.3,
    }

    # Volume presets to multiplier mapping
    VOLUME_PRESETS = {
        'silent': 0.0,
        'x-soft': 0.3,
        'soft': 0.6,
        'medium': 1.0,
        'loud': 1.4,
        'x-loud': 1.8,
        'default': 1.0,
    }

    # Strength to time mapping (in seconds)
    STRENGTH_TO_TIME = {
        'none': 0.0,
        'x-weak': 0.1,
        'weak': 0.25,
        'medium': 0.5,
        'strong': 1.0,
        'x-strong': 2.0,
    }

    def __init__(self):
        self._default_prosody = ProsodyControl()

    def parse_ssml(self, ssml: str) -> List[SSMLSegment]:
        """
        Parse SSML and extract segments with prosody controls.

        Args:
            ssml: SSML string

        Returns:
            List of SSML segments with all controls
        """
        segments = []

        try:
            # Remove XML declaration if present
            ssml = re.sub(r'<\?xml[^>]*\?>', '', ssml)

            # Wrap in root if needed
            if not ssml.strip().startswith('<'):
                ssml = f"<speak>{ssml}</speak>"
            elif not ssml.strip().startswith('<speak>'):
                ssml = f"<speak>{ssml}</speak>"

            # Parse XML
            root = ET.fromstring(ssml)

            # Process recursively
            segments = self._process_element(root, ProsodyControl())

        except ET.ParseError as e:
            logger.warning(f"SSML parsing error: {e}, falling back to plain text")
            # Try to extract text from malformed SSML
            text = self._strip_tags(ssml)
            segments = [SSMLSegment(
                text=text.strip(),
                original_text=ssml,
                prosody=self._default_prosody,
                breaks=[],
            )]

        return segments if segments else [SSMLSegment(
            text=self._strip_tags(ssml).strip(),
            original_text=ssml,
            prosody=self._default_prosody,
            breaks=[],
        )]

    def _process_element(
        self,
        element: ET.Element,
        parent_prosody: ProsodyControl,
        parent_voice: Optional[str] = None,
    ) -> List[SSMLSegment]:
        """Recursively process XML elements."""
        segments = []
        tag = element.tag.lower()

        # Get current prosody (inherit from parent)
        current_prosody = self._extract_prosody(element, parent_prosody)
        current_voice = element.get('name') if tag == 'voice' else parent_voice

        # Handle different element types
        if tag == 'break':
            # Break element - don't create a segment, just record break
            return segments

        elif tag == 'mark':
            # Mark element - for synchronization
            return [SSMLSegment(
                text="",
                prosody=current_prosody,
                voice=current_voice,
                mark_name=element.get('name'),
            )]

        elif tag == 'audio':
            # Audio element - embedded audio
            return [SSMLSegment(
                text="",
                prosody=current_prosody,
                voice=current_voice,
                audio_src=element.get('src'),
            )]

        # Process text content
        if element.text and element.text.strip():
            text_content = element.text.strip()

            # Handle say-as
            say_as = None
            if tag == 'say-as':
                say_as = self._parse_say_as(element, text_content)
                text_content = say_as.get('formatted', text_content)

            # Handle phoneme
            phoneme = None
            if tag == 'phoneme':
                phoneme = element.get('ph')
                # Use phoneme instead of text for pronunciation
                text_content = text_content  # Keep original for display

            # Handle sub (substitution)
            sub_alias = None
            if tag == 'sub':
                sub_alias = element.get('alias')

            # Handle emphasis
            emphasis_level = None
            if tag == 'emphasis':
                emphasis_level = element.get('level', 'moderate')

            # Check for breaks in children
            breaks = self._extract_breaks_from_children(element)

            segment = SSMLSegment(
                text=text_content,
                original_text=text_content,
                prosody=current_prosody,
                voice=current_voice,
                breaks=breaks,
                say_as=say_as,
                phoneme=phoneme,
                emphasis_level=emphasis_level,
                sub_alias=sub_alias,
            )
            segments.append(segment)

        # Process child elements
        for child in element:
            child_segments = self._process_element(child, current_prosody, current_voice)
            segments.extend(child_segments)

        # Handle tail text (text after child elements)
        if element.tail and element.tail.strip():
            segments.append(SSMLSegment(
                text=element.tail.strip(),
                original_text=element.tail.strip(),
                prosody=current_prosody,
                voice=current_voice,
                breaks=[],
            ))

        return segments

    def _extract_prosody(
        self,
        element: ET.Element,
        parent_prosody: ProsodyControl,
    ) -> ProsodyControl:
        """Extract prosody attributes from an element, inheriting from parent."""
        prosody = ProsodyControl(
            rate=parent_prosody.rate,
            pitch=parent_prosody.pitch,
            volume=parent_prosody.volume,
            emphasis=parent_prosody.emphasis,
            contour=parent_prosody.contour,
            range=parent_prosody.range,
        )

        # Check for prosody attributes directly on element
        for attr_name, attr_value in element.attrib.items():
            attr_name_lower = attr_name.lower()

            if attr_name_lower == 'rate':
                prosody.rate = self._parse_rate(attr_value)
            elif attr_name_lower == 'pitch':
                prosody.pitch = self._parse_pitch(attr_value)
            elif attr_name_lower == 'volume':
                prosody.volume = self._parse_volume(attr_value)
            elif attr_name_lower == 'range':
                prosody.range = self._parse_range(attr_value)
            elif attr_name_lower == 'contour':
                prosody.contour = self._parse_contour(attr_value)

        return prosody

    def _parse_say_as(self, element: ET.Element, text: str) -> Dict[str, str]:
        """Parse say-as element for special text interpretation."""
        interpret_as = element.get('interpret-as', 'characters')
        format_attr = element.get('format', '')
        detail = element.get('detail', '')

        result = {
            'interpret_as': interpret_as,
            'original_text': text,
            'format': format_attr,
            'detail': detail,
        }

        # Format the text according to interpret-as type
        formatted = self._format_special_text(text, interpret_as, format_attr, detail)
        result['formatted'] = formatted

        return result

    def _format_special_text(
        self,
        text: str,
        interpret_as: str,
        format: str,
        detail: str,
    ) -> str:
        """Format special text according to interpretation type."""
        try:
            if interpret_as == 'date':
                return self._format_date(text, format)
            elif interpret_as == 'time':
                return self._format_time(text, format)
            elif interpret_as == 'number':
                return self._format_number(text, format, detail)
            elif interpret_as == 'currency':
                return self._format_currency(text, format)
            elif interpret_as == 'telephone':
                return self._format_telephone(text)
            elif interpret_as == 'characters':
                # Read character by character
                return ' '.join(list(text.replace(' ', '')))
            elif interpret_as == 'ordinal':
                return self._number_to_ordinal(int(float(text)))
            elif interpret_as == 'cardinal':
                return self._number_to_cardinal(int(float(text)))
            elif interpret_as == 'digits':
                return ' '.join(list(text.replace(' ', '')))
            else:
                return text
        except (ValueError, AttributeError) as e:
            logger.warning(f"Error formatting text with interpret-as={interpret_as}: {e}")
            return text

    def _format_date(self, date_str: str, format: str) -> str:
        """Format date for natural language reading."""
        if not date_str:
            return date_str

        # Try different date formats
        date_formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%m-%d',
            '%m/%d',
            '%Y%m%d',
        ]

        dt = None
        for fmt in date_formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue

        if dt is None:
            return date_str

        # Format according to style
        if format == 'ymd':
            return f"{dt.year}年{dt.month}月{dt.day}日"
        elif format == 'md':
            return f"{dt.month}月{dt.day}日"
        elif format == 'dmy':
            return f"{dt.day}日{dt.month}月{dt.year}年"
        elif format == 'my':
            return f"{dt.month}月{dt.year}年"
        elif format == 'yw':
            # Week of year
            week_num = (dt - datetime(dt.year, 1, 1)).days // 7 + 1
            return f"{dt.year}年第{week_num}周"
        else:
            # Default: full date
            return f"{dt.year}年{dt.month}月{dt.day}日"

    def _format_time(self, time_str: str, format: str) -> str:
        """Format time for natural language reading."""
        if not time_str:
            return time_str

        time_formats = [
            '%H:%M',
            '%H:%M:%S',
            '%I:%M %p',
            '%I:%M:%S %p',
        ]

        dt = None
        for fmt in time_formats:
            try:
                dt = datetime.strptime(time_str, fmt)
                break
            except ValueError:
                continue

        if dt is None:
            return time_str

        if format == 'hms12':
            period = '上午' if dt.hour < 12 else '下午'
            hour_12 = dt.hour % 12 or 12
            return f"{period}{hour_12}点{dt.minute}分{dt.second}秒"
        elif format == 'hms24':
            return f"{dt.hour}点{dt.minute}分{dt.second}秒"
        elif format == 'hm12':
            period = '上午' if dt.hour < 12 else '下午'
            hour_12 = dt.hour % 12 or 12
            return f"{period}{hour_12}点{dt.minute}分"
        elif format == 'hm24':
            return f"{dt.hour}点{dt.minute}分"
        else:
            # Default: 24-hour format
            return f"{dt.hour}点{dt.minute}分"

    def _format_number(self, number_str: str, format: str, detail: str) -> str:
        """Format number for natural language reading."""
        try:
            num = float(number_str)
            is_integer = num == int(num)
            num_int = int(num)

            if format == 'ordinal':
                return self._number_to_ordinal(num_int)
            elif format == 'digits':
                # Read digit by digit
                return ' '.join(str(num_int))
            elif format == 'fraction':
                # Handle fractions like "1/2"
                if '/' in number_str:
                    numerator, denominator = number_str.split('/')
                    return f"{self._number_to_cardinal(int(numerator))}分之{self._number_to_cardinal(int(denominator))}"
                return number_str
            elif format == 'decimal':
                # Read decimal part digit by digit
                integer_part = int(num)
                decimal_part = number_str.split('.')[-1] if '.' in number_str else ''
                if decimal_part:
                    return f"{self._number_to_cardinal(integer_part)}点{' '.join(decimal_part)}"
                return self._number_to_cardinal(integer_part)
            elif format == 'scientific':
                # Scientific notation
                mantissa, exponent = re.split(r'[eE]', str(num))
                return f"{mantissa}乘以10的{exponent}次方"
            else:
                # Default: natural reading
                return self._number_to_natural(num, is_integer)

        except ValueError:
            return number_str

    def _number_to_natural(self, num: float, is_integer: bool) -> str:
        """Convert number to natural language."""
        if is_integer:
            return self._number_to_cardinal(int(num))
        else:
            # Handle decimal
            integer_part = int(num)
            decimal_part = round((num - integer_part) * 100)
            if decimal_part == 0:
                return self._number_to_cardinal(integer_part)
            else:
                return f"{self._number_to_cardinal(integer_part)}点{decimal_part:02d}"

    def _number_to_cardinal(self, num: int) -> str:
        """Convert integer to cardinal in Chinese."""
        if num == 0:
            return '零'

        # Define Chinese numerals
        digits = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
        units = ['', '十', '百', '千', '万']
        large_units = ['', '万', '亿', '兆']

        result = ''
        num_str = str(num)
        length = len(num_str)

        # Handle numbers >= 10000
        if num >= 10000:
            wan = num // 10000
            remainder = num % 10000
            if remainder == 0:
                return self._number_to_cardinal(wan) + '万'
            else:
                return self._number_to_cardinal(wan) + '万' + self._number_to_cardinal(remainder)

        # Handle numbers 1-9999
        for i, digit_char in enumerate(num_str):
            digit = int(digit_char)
            pos = length - i - 1

            if digit != 0:
                if digit == 1 and pos == 1 and length == 2:
                    # Special case: "十几"
                    result += units[pos]
                else:
                    result += digits[digit] + units[pos]
            elif result and result[-1] != '零':
                result += '零'

        # Clean up trailing zeros
        result = result.rstrip('零')
        return result if result else '零'

    def _number_to_ordinal(self, num: int) -> str:
        """Convert integer to ordinal in Chinese."""
        if num == 1:
            return '第一'
        elif num == 2:
            return '第二'
        elif num == 3:
            return '第三'
        else:
            return f'第{self._number_to_cardinal(num)}'

    def _format_currency(self, amount_str: str, currency: str) -> str:
        """Format currency for natural language reading."""
        try:
            amount = float(amount_str)
            currency = currency or 'CNY'

            if currency in ['CNY', 'RMB', '¥', 'yuan']:
                if amount >= 10000:
                    wan_amount = amount / 10000
                    return f"{wan_amount:.1f}万元"
                else:
                    return f"{amount:.2f}元"
            elif currency in ['USD', '$', 'dollar']:
                return f"{amount:.2f}美元"
            elif currency in ['EUR', '€', 'euro']:
                return f"{amount:.2f}欧元"
            elif currency in ['JPY', '¥', 'yen']:
                return f"{amount:.0f}日元"
            elif currency in ['GBP', '£', 'pound']:
                return f"{amount:.2f}英镑"
            else:
                return f"{amount:.2f}{currency}"

        except ValueError:
            return amount_str

    def _format_telephone(self, phone: str) -> str:
        """Format telephone number for reading."""
        # Remove common separators
        phone = re.sub(r'[-\s\(\)]', '', phone)

        # Read digit by digit with pauses
        if len(phone) == 11 and phone.startswith('1'):
            # Chinese mobile: 1XX XXXX XXXX
            return f"{phone[0]}，{phone[1:4]}，{phone[4:7]}，{phone[7:]}"
        elif len(phone) == 10:
            # 10-digit number: XXX XXX XXXX
            return f"{phone[0:3]}，{phone[3:6]}，{phone[6:]}"
        else:
            # Read digit by digit
            return '，'.join(list(phone))

    def _extract_breaks_from_children(self, element: ET.Element) -> List[float]:
        """Extract breaks from child elements."""
        breaks = []

        for break_elem in element.findall('.//break'):
            time_str = break_elem.get('time', '0s')
            strength = break_elem.get('strength', 'medium')

            if time_str and time_str != '0s':
                break_time = self._parse_time(time_str)
            else:
                break_time = self.STRENGTH_TO_TIME.get(strength, 0.5)

            breaks.append(break_time)

        return breaks

    def _parse_rate(self, rate: str) -> float:
        """Parse rate attribute."""
        # Percentage: "50%", "150%", "200%"
        if rate.endswith('%'):
            return float(rate.rstrip('%')) / 100.0

        # Presets: "x-slow", "slow", "medium", "fast", "x-fast"
        if rate in self.RATE_PRESETS:
            return self.RATE_PRESETS[rate]

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
            hz = float(pitch.replace('Hz', '').replace('+', ''))
            return 1.0 + (hz / 200.0)  # Normalize around 200Hz

        # Presets: "x-low", "low", "medium", "high", "x-high"
        if pitch in self.PITCH_PRESETS:
            return self.PITCH_PRESETS[pitch]

        return 1.0

    def _parse_volume(self, volume: str) -> float:
        """Parse volume attribute."""
        # Percentage: "50%", "150%"
        if volume.endswith('%'):
            return float(volume.rstrip('%')) / 100.0

        # Presets
        if volume in self.VOLUME_PRESETS:
            return self.VOLUME_PRESETS[volume]

        return 1.0

    def _parse_range(self, range_str: str) -> float:
        """Parse pitch range attribute."""
        # Percentage or preset
        if range_str.endswith('%'):
            return float(range_str.rstrip('%')) / 100.0

        presets = {
            'x-low': 0.5,
            'low': 0.7,
            'medium': 1.0,
            'high': 1.3,
            'x-high': 1.5,
        }
        return presets.get(range_str, 1.0)

    def _parse_contour(self, contour: str) -> List[Tuple[float, float]]:
        """Parse pitch contour.

        Format: "0% +10st, 50% -5st, 100% +5st"
        """
        points = []
        for point_str in contour.split(','):
            point_str = point_str.strip()
            if not point_str:
                continue

            try:
                match = re.match(r'(\d+)%\s*([+-]?\d+(?:\.\d+)?)(st|%)', point_str)
                if match:
                    position = float(match.group(1)) / 100.0
                    value = float(match.group(2))
                    unit = match.group(3)

                    if unit == 'st':
                        points.append((position, value))
                    else:
                        points.append((position, value / 100.0))
            except (ValueError, AttributeError):
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

    def _strip_tags(self, text: str) -> str:
        """Remove XML tags from text."""
        return re.sub(r'<[^>]+>', '', text)

    def generate_ssml(
        self,
        text: str,
        rate: Optional[float] = None,
        pitch: Optional[float] = None,
        volume: Optional[float] = None,
        emphasis: Optional[str] = None,
        voice: Optional[str] = None,
        say_as: Optional[str] = None,
        phoneme: Optional[str] = None,
        breaks: Optional[List[float]] = None,
    ) -> str:
        """
        Generate SSML from text and parameters.

        Args:
            text: Text to synthesize
            rate: Speech rate (0.1 - 10.0)
            pitch: Pitch multiplier (0.1 - 2.0)
            volume: Volume multiplier (0.0 - 2.0)
            emphasis: Emphasis level (strong, moderate, reduced, none)
            voice: Voice name
            say_as: Interpret-as type (date, time, number, currency, etc.)
            phoneme: IPA phoneme string
            breaks: List of break durations in seconds

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

        # Add say-as if specified
        if say_as:
            ssml_parts.append(f'<say-as interpret-as="{say_as}">{text}</say-as>')
        # Add phoneme if specified
        elif phoneme:
            ssml_parts.append(f'<phoneme ph="{phoneme}">{text}</phoneme>')
        # Add emphasis if specified
        elif emphasis:
            ssml_parts.append(f'<emphasis level="{emphasis}">{text}</emphasis>')
        else:
            ssml_parts.append(text)

        # Add breaks if specified
        if breaks:
            for break_duration in breaks:
                ssml_parts.append(f'<break time="{break_duration}s"/>')

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
            return self._strip_tags(ssml), {}

        # Extract plain text
        plain_text = ' '.join(seg.text for seg in segments if seg.text)

        # Use prosody from first segment for overall parameters
        first_segment = next((s for s in segments if s.text), None)
        if not first_segment:
            return plain_text, {}

        prosody = first_segment.prosody

        tts_params = {
            'rate': prosody.rate,
            'pitch': prosody.pitch,
            'volume': prosody.volume,
            'emphasis': prosody.emphasis,
        }

        # Add breaks if any
        if first_segment.breaks:
            tts_params['breaks'] = first_segment.breaks

        # Add voice if specified
        if first_segment.voice:
            tts_params['voice'] = first_segment.voice

        # Add say-as info
        if first_segment.say_as:
            tts_params['say_as'] = first_segment.say_as

        return plain_text, tts_params


class ProsodyBuilder:
    """Builder for creating prosody controls programmatically."""

    def __init__(self):
        self._prosody = ProsodyControl()

    def rate(self, value: float) -> 'ProsodyBuilder':
        """Set speech rate."""
        self._prosody.rate = value
        return self

    def pitch(self, value: float) -> 'ProsodyBuilder':
        """Set pitch."""
        self._prosody.pitch = value
        return self

    def volume(self, value: float) -> 'ProsodyBuilder':
        """Set volume."""
        self._prosody.volume = value
        return self

    def emphasis(self, level: str) -> 'ProsodyBuilder':
        """Set emphasis."""
        self._prosody.emphasis = level
        return self

    def contour(self, points: List[Tuple[float, float]]) -> 'ProsodyBuilder':
        """Set pitch contour."""
        self._prosody.contour = points
        return self

    def range(self, value: float) -> 'ProsodyBuilder':
        """Set pitch range."""
        self._prosody.range = value
        return self

    def build(self) -> ProsodyControl:
        """Build the prosody control."""
        return self._prosody


# Helper functions
def create_prosody(
    rate: float = 1.0,
    pitch: float = 1.0,
    volume: float = 1.0,
    emphasis: str = "none",
    contour: Optional[List[Tuple[float, float]]] = None,
    range: Optional[float] = None,
) -> ProsodyControl:
    """Create a prosody control object."""
    return ProsodyControl(
        rate=rate,
        pitch=pitch,
        volume=volume,
        emphasis=emphasis,
        contour=contour,
        range=range,
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
    "calm": ProsodyControl(rate=0.95, pitch=0.95, volume=0.9, emphasis="reduced"),
    "energetic": ProsodyControl(rate=1.3, pitch=1.15, volume=1.15, emphasis="strong"),
}


def get_preset(name: str) -> Optional[ProsodyControl]:
    """Get a prosody preset by name."""
    return PROSODY_PRESETS.get(name)


# Global instance
_ssml_processor: Optional[CompleteSSMLProcessor] = None


def get_ssml_processor() -> CompleteSSMLProcessor:
    """Get global SSML processor instance."""
    global _ssml_processor
    if _ssml_processor is None:
        _ssml_processor = CompleteSSMLProcessor()
    return _ssml_processor


# Backwards-compatible alias expected by older API modules.
SSMLProcessor = CompleteSSMLProcessor

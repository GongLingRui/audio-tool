"""
Dialect and Multi-Language Support Service
Provides support for Chinese dialects and mixed-language text processing
"""
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class LanguageCode(Enum):
    """Supported language codes."""
    MANDARIN = "zh-CN"
    CANTONESE = "zh-HK"
    HAKKA = "zh-HAK"
    MIN_NAN = "zh-MIN"
    WU = "zh-WU"
    XIANG = "zh-XIANG"
    GAN = "zh-GAN"
    ENGLISH = "en-US"
    JAPANESE = "ja-JP"
    KOREAN = "ko-KR"


class DialectConfig:
    """Configuration for Chinese dialects."""

    DIALECT_CONFIGS = {
        LanguageCode.CANTONESE: {
            "name": "粤语",
            "engine": "edge-tts",  # Edge TTS supports Cantonese
            "voice_id": "zh-HK-HiuMaanNeural",
            "character_replacements": {
                "这里": "呢度",
                "什么": "乜嘢",
                "的": "嘅",
            },
            "tone_shift": 2.0,  # Cantonese has different tones
            "speed_factor": 1.05,
        },
        LanguageCode.HAKKA: {
            "name": "客家话",
            "engine": "cosyvoice",  # CosyVoice for fine control
            "voice_id": "hakka_female",
            "tone_shift": 1.5,
            "speed_factor": 0.95,
        },
        LanguageCode.MIN_NAN: {
            "name": "闽南语/台语",
            "engine": "cosyvoice",
            "voice_id": "minnan_male",
            "tone_shift": 1.0,
            "speed_factor": 1.0,
        },
        LanguageCode.WU: {
            "name": "吴语/上海话",
            "engine": "cosyvoice",
            "voice_id": "wu_female",
            "tone_shift": 0.5,
            "speed_factor": 1.1,
        },
    }


@dataclass
class LanguageSegment:
    """A segment of text with detected language."""
    text: str
    language: LanguageCode
    confidence: float
    start_pos: int
    end_pos: int


class DialectMultiLanguageService:
    """
    Service for handling Chinese dialects and multi-language text.

    Features:
    - Automatic language/dialect detection
    - Character conversion for dialects
    - Mixed-language text processing
    - Appropriate voice selection
    - Dialect-specific prosody adjustments
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        # Language detection patterns
        self._init_detection_patterns()

    def _init_detection_patterns(self):
        """Initialize patterns for language detection."""
        # Cantonese patterns
        self.cantonese_patterns = [
            r'呢度', r'乜嘢', r'嘅', r'唔', r'冇',
            r'好喫', r'系', r'返', r'嚟',
        ]

        # Regional characters
        self.regional_chars = {
            LanguageCode.CANTONESE: set('佢佇佮併侎侐'),
            LanguageCode.HAKKA: set('�𠓻𠹺𠰺'),  # Simplified Hakka chars
            LanguageCode.MIN_NAN: set('佇佮' + '干你'),  # Hokkien
        }

        # Mixed language patterns (Chinese-English, Chinese-Japanese, etc.)
        self.mixed_lang_patterns = {
            LanguageCode.ENGLISH: r'[a-zA-Z\s]{3,}',
            LanguageCode.JAPANESE: r'[\u3040-\u309F\u30A0-\u30FF]{2,}',
            LanguageCode.KOREAN: r'[\uAC00-\uD7AF]{2,}',
        }

    async def detect_language(
        self,
        text: str,
    ) -> List[LanguageSegment]:
        """
        Detect language/dialect for text segments.

        Args:
            text: Input text

        Returns:
            List of language segments with detected languages
        """
        segments = []
        current_pos = 0

        # Split by sentence boundaries first
        sentences = self._split_sentences(text)

        for sentence in sentences:
            start_pos = text.find(sentence, current_pos)
            end_pos = start_pos + len(sentence)

            # Detect dialect/language
            detected_lang = self._detect_sentence_language(sentence)

            segments.append(LanguageSegment(
                text=sentence,
                language=detected_lang,
                confidence=self._calculate_confidence(sentence, detected_lang),
                start_pos=start_pos,
                end_pos=end_pos,
            ))

            current_pos = end_pos

        return segments

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Chinese sentence boundaries
        pattern = r'[^。！？.!?]*[。！？.!?]?'
        matches = re.findall(pattern, text)
        return [m.strip() for m in matches if m.strip()]

    def _detect_sentence_language(self, sentence: str) -> LanguageCode:
        """Detect language for a single sentence."""
        # Check for Cantonese patterns
        for pattern in self.cantonese_patterns:
            if re.search(pattern, sentence):
                return LanguageCode.CANTONESE

        # Check for regional characters
        for lang, chars in self.regional_chars.items():
            if any(char in sentence for char in chars):
                return lang

        # Check for mixed languages
        for lang, pattern in self.mixed_lang_patterns.items():
            matches = re.findall(pattern, sentence)
            if matches and len(''.join(matches)) / len(sentence) > 0.3:
                return lang

        # Default: Mandarin
        return LanguageCode.MANDARIN

    def _calculate_confidence(
        self,
        text: str,
        detected_lang: LanguageCode,
    ) -> float:
        """Calculate confidence score for language detection."""
        if detected_lang == LanguageCode.MANDARIN:
            # Default confidence for Mandarin
            return 0.7

        # Check if dialect-specific patterns exist
        if detected_lang == LanguageCode.CANTONESE:
            pattern_count = sum(
                1 for p in self.cantonese_patterns
                if re.search(p, text)
            )
            return min(0.95, 0.6 + pattern_count * 0.1)

        # For other languages, check character presence
        chars = self.regional_chars.get(detected_lang, set())
        char_count = sum(1 for c in text if c in chars)
        if char_count > 0:
            return min(0.9, 0.5 + char_count * 0.1)

        return 0.5  # Low confidence

    async def convert_dialect_text(
        self,
        text: str,
        target_dialect: LanguageCode,
    ) -> str:
        """
        Convert text to dialect-specific characters.

        Args:
            text: Original text
            target_dialect: Target dialect

        Returns:
            Converted text
        """
        if target_dialect not in DialectConfig.DIALECT_CONFIGS:
            return text

        config = DialectConfig.DIALECT_CONFIGS[target_dialect]
        replacements = config.get("character_replacements", {})

        result = text
        for original, dialect in replacements.items():
            result = result.replace(original, dialect)

        return result

    async def get_dialect_config(
        self,
        dialect: LanguageCode,
    ) -> Optional[Dict[str, Any]]:
        """
        Get TTS configuration for a dialect.

        Args:
            dialect: Target dialect

        Returns:
            Configuration dict with engine, voice_id, parameters
        """
        if dialect not in DialectConfig.DIALECT_CONFIGS:
            # Use default Mandarin config
            return {
                "engine": "cosyvoice",
                "voice_id": "zh-CN-female",
                "tone_shift": 0.0,
                "speed_factor": 1.0,
            }

        return DialectConfig.DIALECT_CONFIGS[dialect]

    async def process_mixed_language(
        self,
        text: str,
        voice_configs: Dict[LanguageCode, Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process mixed-language text with appropriate voices.

        Args:
            text: Input text (may contain multiple languages)
            voice_configs: Voice config for each language

        Returns:
            List of segments with voice configurations
        """
        # Detect languages
        segments = await self.detect_language(text)

        # Get voice configs
        if voice_configs is None:
            voice_configs = {}

        result = []
        for segment in segments:
            # Get config for this language
            if segment.language not in voice_configs:
                voice_configs[segment.language] = await self.get_dialect_config(
                    segment.language
                )

            result.append({
                "text": segment.text,
                "language": segment.language.value,
                "voice_config": voice_configs[segment.language],
                "confidence": segment.confidence,
                "position": (segment.start_pos, segment.end_pos),
            })

        return result

    async def detect_and_segment_dialogue(
        self,
        text: str,
        known_speakers: Dict[str, LanguageCode] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect dialogue segments with speaker language preferences.

        Useful for multi-speaker content where different speakers
        use different dialects/languages.

        Args:
            text: Input dialogue text
            known_speakers: Known speakers and their preferred languages

        Returns:
            List of dialogue segments with language info
        """
        if known_speakers is None:
            known_speakers = {}

        # Detect dialogue patterns
        dialogue_pattern = r'([^：""]+)[说讲道]："([^"]+)"'
        matches = re.finditer(dialogue_pattern, text)

        segments = []
        for match in matches:
            speaker = match.group(1).strip()
            dialogue_text = match.group(2)

            # Detect language for this segment
            lang_segments = await self.detect_language(dialogue_text)

            if lang_segments:
                primary_lang = lang_segments[0].language

                # Update known speakers
                if speaker not in known_speakers:
                    known_speakers[speaker] = primary_lang

                segments.append({
                    "speaker": speaker,
                    "text": dialogue_text,
                    "language": primary_lang.value,
                    "confidence": lang_segments[0].confidence,
                    "voice_config": await self.get_dialect_config(primary_lang),
                })

        return segments

    async def suggest_dialect_for_content(
        self,
        text: str,
        target_audience: str = "mainland",
    ) -> Dict[str, Any]:
        """
        Suggest appropriate dialect/language for content.

        Args:
            text: Content text
            target_audience: Target audience (mainland, hk, taiwan, overseas)

        Returns:
            Suggestion with dialect and reasoning
        """
        # Detect primary language
        segments = await self.detect_language(text)

        if not segments:
            return {
                "suggested_dialect": LanguageCode.MANDARIN,
                "confidence": 0.5,
                "reasoning": "No clear dialect detected",
            }

        # Count language distribution
        lang_counts = {}
        for seg in segments:
            lang = seg.language
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        # Get most common
        primary_lang = max(lang_counts, key=lang_counts.get)
        count = lang_counts[primary_lang]
        confidence = count / len(segments)

        # Adjust for target audience
        reasoning = f"Detected {primary_lang.value} in {count}/{len(segments)} segments"

        if target_audience == "hk" and primary_lang == LanguageCode.MANDARIN:
            return {
                "suggested_dialect": LanguageCode.CANTONESE,
                "confidence": 0.6,
                "reasoning": "Hong Kong audience, suggest Cantonese",
            }

        return {
            "suggested_dialect": primary_lang,
            "confidence": confidence,
            "reasoning": reasoning,
        }

    async def validate_dialect_support(
        self,
        dialect: LanguageCode,
    ) -> Dict[str, Any]:
        """
        Validate if a dialect is supported and return availability.

        Args:
            dialect: Dialect to check

        Returns:
            Support status with availability info
        """
        supported = dialect in DialectConfig.DIALECT_CONFIGS or dialect == LanguageCode.MANDARIN

        return {
            "dialect": dialect.value,
            "supported": supported,
            "config_available": dialect in DialectConfig.DIALECT_CONFIGS,
            "engines": ["cosyvoice", "edge-tts"] if supported else [],
            "note": "Mandarin is always supported" if dialect == LanguageCode.MANDARIN else None,
        }

    async def get_supported_dialects(self) -> List[Dict[str, str]]:
        """Get list of all supported dialects/languages."""
        dialects = [
            {"code": lang.value, "name": lang.value}
            for lang in LanguageCode
        ]

        # Add dialect names
        for lang, config in DialectConfig.DIALECT_CONFIGS.items():
            for d in dialects:
                if d["code"] == lang.value:
                    d["name"] = config["name"]
                    break

        return dialects


# Global instance
_dialect_service: Optional[DialectMultiLanguageService] = None


def get_dialect_service() -> DialectMultiLanguageService:
    """Get global dialect service instance."""
    global _dialect_service
    if _dialect_service is None:
        _dialect_service = DialectMultiLanguageService()
    return _dialect_service

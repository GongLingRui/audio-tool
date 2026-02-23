"""
Intelligent Text Segmenter - Semantic-aware text segmentation
Advanced text segmentation for TTS with dialogue detection and semantic analysis
"""
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class TextSegment:
    """A text segment with metadata."""
    text: str
    original_text: str = ""
    speaker: Optional[str] = None  # For dialogue
    segment_type: str = "normal"  # normal, dialogue, narration
    pause_after: float = 0.5  # Suggested pause duration in seconds
    emotion: Optional[str] = None  # Detected emotion
    keywords: List[str] = field(default_factory=list)  # Important keywords
    importance: float = 1.0  # Importance score (0.0 - 1.0)


class IntelligentTextSegmenter:
    """
    Intelligent text segmenter for TTS.

    Features:
    - Semantic-aware sentence boundary detection
    - Dialogue detection and speaker attribution
    - LLM-assisted semantic boundary analysis (optional)
    - Natural pause calculation
    - Emotion detection from text
    """

    # Dialogue patterns
    DIALOGUE_PATTERNS = [
        # Standard: xxx说："..."
        r'([^「『"\n]{1,10}?)[说讲道问喊叫哭笑低声轻声高声怒斥冷冷地热切地激动地]：[：""]([^"」『""\n]{10,})[""」』]',
        # Chinese quote: 「...」xxx说
        r'「([^」]{10,})」([^「"\n]{1,10}?)[说问道]',
        # Book title: 《...》
        r'《([^》]{3,50})》',
        # English quote: "...", xxx said
        r'"([^"]{10,})"\s*([^"\n]{1,10}?)[said asked replied]',
    ]

    # Sentence boundaries
    SENTENCE_BOUNDARIES = ['。', '！', '？', '；', '\n', '…', '…']

    # Clauses boundaries (within sentences)
    CLAUSE_BOUNDARIES = ['，', '、', '：', '—', '－', '·']

    # Emotion keywords
    EMOTION_KEYWORDS = {
        'joy': ['开心', '高兴', '快乐', '欢乐', '喜悦', '愉快', '兴奋', '欣喜'],
        'sad': ['难过', '悲伤', '痛苦', '伤心', '哀伤', '忧愁', '凄凉', '沮丧'],
        'anger': ['生气', '愤怒', '恼火', '气愤', '暴怒', '怒', '愤恨'],
        'fear': ['害怕', '恐惧', '惊恐', '害怕', '担心', '忧虑', '不安'],
        'surprise': ['惊讶', '惊奇', '震惊', '吃惊', '意外', '诧异'],
        'excited': ['兴奋', '激动', '振奋', '热情', '亢奋'],
        'calm': ['平静', '冷静', '淡定', '安宁', '从容'],
        'tender': ['温柔', '轻柔', '柔和', '温柔体贴'],
    }

    # Important words that should be in same segment
    IMPORTANT_WORDS = [
        '因为', '所以', '但是', '然而', '不过', '因此', '而且',
        '不仅', '并且', '或者', '还是', '虽然', '尽管',
        '如果', '要是', '假如', '只要', '只有', '无论',
    ]

    def __init__(self, use_llm: bool = False, llm_client=None):
        """
        Initialize the intelligent text segmenter.

        Args:
            use_llm: Whether to use LLM for semantic analysis
            llm_client: LLM client for semantic analysis (optional)
        """
        self.use_llm = use_llm
        self.llm_client = llm_client

    async def segment_text(
        self,
        text: str,
        max_chars: int = 500,
        preserve_sentences: bool = True,
        detect_dialogue: bool = True,
        target_segment_count: Optional[int] = None,
    ) -> List[TextSegment]:
        """
        Intelligently segment text for TTS.

        Args:
            text: Input text
            max_chars: Maximum characters per segment
            preserve_sentences: Try to preserve sentence boundaries
            detect_dialogue: Detect and tag dialogue segments
            target_segment_count: Target number of segments (optional)

        Returns:
            List of text segments with metadata
        """
        # Step 1: Detect dialogue first (if enabled)
        if detect_dialogue:
            dialogue_segments = await self._detect_dialogue(text)
            if dialogue_segments:
                # Merge dialogue with non-dialogue text
                return await self._merge_dialogue_segments(
                    text, dialogue_segments, max_chars
                )

        # Step 2: Split into sentences
        sentences = self._split_sentences(text)

        # Step 3: LLM semantic analysis (if enabled)
        semantic_boundaries = []
        if self.use_llm and self.llm_client:
            semantic_boundaries = await self._llm_analyze_boundaries(text)

        # Step 4: Intelligent merging
        segments = await self._intelligent_merge(
            sentences,
            max_chars,
            semantic_boundaries,
            target_segment_count,
        )

        return segments

    async def _detect_dialogue(self, text: str) -> List[Dict[str, Any]]:
        """Detect dialogue patterns in text."""
        dialogue_segments = []

        for pattern in self.DIALOGUE_PATTERNS:
            matches = re.finditer(pattern, text)
            for match in matches:
                groups = match.groups()

                if len(groups) >= 2:
                    # Check pattern type
                    if '说' in match.group() or '道' in match.group():
                        # Standard: xxx说："..."
                        speaker = groups[0].strip()
                        dialogue_text = groups[1].strip()
                        start_pos = match.start()
                        end_pos = match.end()

                        dialogue_segments.append({
                            'speaker': speaker,
                            'text': dialogue_text,
                            'start': start_pos,
                            'end': end_pos,
                            'type': 'dialogue',
                            'original': match.group(),
                        })

        return dialogue_segments

    async def _merge_dialogue_segments(
        self,
        text: str,
        dialogue_segments: List[Dict[str, Any]],
        max_chars: int,
    ) -> List[TextSegment]:
        """Merge dialogue segments with narration text."""
        result = []
        current_pos = 0

        # Sort by position
        dialogue_segments.sort(key=lambda x: x['start'])

        for dialogue in dialogue_segments:
            # Add narration before dialogue
            narration_text = text[current_pos:dialogue['start']].strip()
            if narration_text:
                # Split narration into smaller chunks if needed
                narration_chunks = await self._split_large_text(
                    narration_text, max_chars
                )
                for chunk in narration_chunks:
                    result.append(TextSegment(
                        text=chunk,
                        original_text=chunk,
                        segment_type='narration',
                        pause_after=self._calculate_pause(chunk, 'narration'),
                    ))

            # Add dialogue
            result.append(TextSegment(
                text=dialogue['text'],
                original_text=dialogue['original'],
                speaker=dialogue['speaker'],
                segment_type='dialogue',
                pause_after=self._calculate_pause(dialogue['text'], 'dialogue'),
                emotion=self._detect_emotion_from_text(dialogue['text']),
            ))

            current_pos = dialogue['end']

        # Add remaining narration
        remaining_text = text[current_pos:].strip()
        if remaining_text:
            narration_chunks = await self._split_large_text(
                remaining_text, max_chars
            )
            for chunk in narration_chunks:
                result.append(TextSegment(
                    text=chunk,
                    original_text=chunk,
                    segment_type='narration',
                    pause_after=self._calculate_pause(chunk, 'narration'),
                ))

        return result

    async def _split_large_text(
        self,
        text: str,
        max_chars: int,
    ) -> List[str]:
        """Split large text into smaller chunks."""
        if len(text) <= max_chars:
            return [text]

        chunks = []
        current_chunk = ""
        sentences = self._split_sentences(text)

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chars:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # Handle single sentence that's too long
                if len(sentence) > max_chars:
                    # Split at clause boundaries
                    clause_chunks = await self._split_by_clauses(sentence, max_chars)
                    chunks.extend(clause_chunks)
                    current_chunk = ""
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    async def _split_by_clauses(
        self,
        text: str,
        max_chars: int,
    ) -> List[str]:
        """Split text by clause boundaries."""
        if len(text) <= max_chars:
            return [text]

        chunks = []
        current_chunk = ""
        i = 0

        while i < len(text):
            # Find next clause boundary
            next_boundary = -1
            for boundary in self.CLAUSE_BOUNDARIES:
                pos = text.find(boundary, i, i + max_chars)
                if pos != -1 and (next_boundary == -1 or pos < next_boundary):
                    next_boundary = pos + 1

            if next_boundary != -1 and next_boundary - i <= max_chars:
                # Found a boundary
                chunk = text[i:next_boundary].strip()
                if chunk:
                    chunks.append(chunk)
                i = next_boundary
            else:
                # No boundary found, force split at max_chars
                chunk = text[i:i + max_chars].strip()
                if chunk:
                    chunks.append(chunk)
                i += max_chars

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Add period to end if missing
        if text and text[-1] not in self.SENTENCE_BOUNDARIES:
            text = text + '。'

        sentences = []
        current = ""
        i = 0

        while i < len(text):
            char = text[i]

            # Check for sentence boundary
            if char in self.SENTENCE_BOUNDARIES:
                current += char
                if current.strip():
                    sentences.append(current.strip())
                current = ""
            else:
                current += char

            i += 1

        if current.strip():
            sentences.append(current.strip())

        return sentences

    async def _llm_analyze_boundaries(self, text: str) -> List[int]:
        """Use LLM to analyze semantic boundaries."""
        if not self.llm_client:
            return []

        sentences = self._split_sentences(text)

        prompt = f"""分析以下文本，找出最佳的断句位置（句子索引）。
考虑语义连贯性和自然停顿位置。返回最适合断开的地方的句子索引（0-based）。

文本：
{text.split(chr(10))[0][:500]}...

返回JSON格式：{{"boundaries": [1, 3, 5]}}
只返回3-5个最重要的断句位置。"""

        try:
            response = await self.llm_client.complete(prompt)

            # Try to parse JSON from response
            import json
            result = json.loads(response)
            return result.get('boundaries', [])
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
            return []

    async def _intelligent_merge(
        self,
        sentences: List[str],
        max_chars: int,
        semantic_boundaries: List[int],
        target_segment_count: Optional[int],
    ) -> List[TextSegment]:
        """Intelligently merge sentences into segments."""
        segments = []
        current_chunk = ""
        current_sentences = []

        for i, sentence in enumerate(sentences):
            test_chunk = current_chunk + sentence

            # Check if we should break
            should_break = (
                len(test_chunk) > max_chars or
                i in semantic_boundaries or
                self._is_natural_break(sentence) or
                self._has_important_boundary(sentence)
            )

            if should_break and current_chunk:
                # Create segment
                segment_text = current_chunk.strip()
                segments.append(TextSegment(
                    text=segment_text,
                    original_text=segment_text,
                    segment_type='normal',
                    pause_after=self._calculate_pause(segment_text, 'normal'),
                    emotion=self._detect_emotion_from_text(segment_text),
                ))
                current_chunk = sentence
                current_sentences = [i]
            else:
                current_chunk += sentence
                current_sentences.append(i)

        # Add final chunk
        if current_chunk.strip():
            segment_text = current_chunk.strip()
            segments.append(TextSegment(
                text=segment_text,
                original_text=segment_text,
                segment_type='normal',
                pause_after=self._calculate_pause(segment_text, 'normal'),
                emotion=self._detect_emotion_from_text(segment_text),
            ))

        # Adjust to target segment count if specified
        if target_segment_count and len(segments) != target_segment_count:
            segments = self._adjust_segment_count(segments, target_segment_count)

        return segments

    def _is_natural_break(self, sentence: str) -> bool:
        """Check if this is a natural break point."""
        # Break after questions
        if sentence.strip().endswith('？'):
            return True

        # Break after exclamations
        if sentence.strip().endswith('！'):
            return True

        # Break before chapter headers
        if re.match(r'^第[一二三四五六七八九十百千\d]+[章节回]', sentence.strip()):
            return True

        return False

    def _has_important_boundary(self, sentence: str) -> bool:
        """Check if sentence has important boundary words."""
        for word in self.IMPORTANT_WORDS:
            if word in sentence:
                return True
        return False

    def _calculate_pause(self, text: str, segment_type: str) -> float:
        """Calculate natural pause duration after segment."""
        # Base pause
        base_pause = 0.3

        # Adjust for segment type
        if segment_type == 'dialogue':
            base_pause = 0.8  # Longer pause after dialogue
        elif segment_type == 'narration':
            base_pause = 0.5

        # Adjust for ending punctuation
        if text.endswith('！'):
            return 0.6  # Excitement - short pause
        elif text.endswith('？'):
            return 0.7  # Question - give listener time to think
        elif text.endswith('…'):
            return 0.4  # Trailing off - shorter pause
        elif text.endswith('。'):
            return base_pause
        elif text.endswith('，'):
            return 0.2  # Comma - very short pause
        else:
            return base_pause

    def _detect_emotion_from_text(self, text: str) -> Optional[str]:
        """Detect emotion from text content."""
        emotion_scores = {}

        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                emotion_scores[emotion] = score

        if emotion_scores:
            return max(emotion_scores.items(), key=lambda x: x[1])[0]

        return None

    def _adjust_segment_count(
        self,
        segments: List[TextSegment],
        target_count: int,
    ) -> List[TextSegment]:
        """Adjust segment count to match target."""
        current_count = len(segments)

        if current_count == target_count:
            return segments
        elif current_count < target_count:
            # Split some segments
            return self._split_more_segments(segments, target_count)
        else:
            # Merge some segments
            return self._merge_adjacent_segments(segments, target_count)

    def _split_more_segments(
        self,
        segments: List[TextSegment],
        target_count: int,
    ) -> List[TextSegment]:
        """Split segments to reach target count."""
        result = []
        needed = target_count - len(segments)

        for segment in segments:
            if len(result) >= target_count:
                break

            # If we need more segments and this one is long enough, split it
            if needed > 0 and len(segment.text) > 100:
                # Split at middle
                mid = len(segment.text) // 2
                # Try to find a good split point
                split_pos = self._find_best_split_position(segment.text, mid)

                part1 = TextSegment(
                    text=segment.text[:split_pos].strip(),
                    original_text=segment.original_text[:split_pos].strip(),
                    segment_type=segment.segment_type,
                    pause_after=0.3,
                    emotion=segment.emotion,
                )

                part2 = TextSegment(
                    text=segment.text[split_pos:].strip(),
                    original_text=segment.original_text[split_pos:].strip(),
                    segment_type=segment.segment_type,
                    pause_after=segment.pause_after,
                    emotion=segment.emotion,
                )

                result.extend([part1, part2])
                needed -= 1
            else:
                result.append(segment)

        return result

    def _find_best_split_position(self, text: str, target_pos: int) -> int:
        """Find best position to split text."""
        # Search around target position for sentence/clause boundary
        search_range = 50
        start = max(0, target_pos - search_range)
        end = min(len(text), target_pos + search_range)

        # Look for sentence boundaries first
        for boundary in self.SENTENCE_BOUNDARIES:
            pos = text.rfind(boundary, start, end)
            if pos != -1:
                return pos + 1

        # Look for clause boundaries
        for boundary in self.CLAUSE_BOUNDARIES:
            pos = text.rfind(boundary, start, end)
            if pos != -1:
                return pos + 1

        # Just split at target position
        return target_pos

    def _merge_adjacent_segments(
        self,
        segments: List[TextSegment],
        target_count: int,
    ) -> List[TextSegment]:
        """Merge adjacent segments to reach target count."""
        while len(segments) > target_count:
            # Find two adjacent segments that are best to merge
            best_merge_idx = 0
            best_score = float('inf')

            for i in range(len(segments) - 1):
                # Prefer merging segments of same type
                if segments[i].segment_type == segments[i + 1].segment_type:
                    score = 0
                else:
                    score = 1

                # Prefer segments with no emotion change
                if segments[i].emotion == segments[i + 1].emotion:
                    score += 0
                else:
                    score += 2

                # Prefer shorter combined length
                combined_length = len(segments[i].text) + len(segments[i + 1].text)
                score += combined_length / 1000.0

                if score < best_score:
                    best_score = score
                    best_merge_idx = i

            # Merge the best pair
            merged = TextSegment(
                text=segments[best_merge_idx].text + ' ' + segments[best_merge_idx + 1].text,
                original_text=(
                    segments[best_merge_idx].original_text + ' ' +
                    segments[best_merge_idx + 1].original_text
                ),
                segment_type=segments[best_merge_idx].segment_type,
                pause_after=segments[best_merge_idx + 1].pause_after,
                emotion=segments[best_merge_idx].emotion or segments[best_merge_idx + 1].emotion,
            )

            segments = segments[:best_merge_idx] + [merged] + segments[best_merge_idx + 2:]

        return segments

    async def segment_for_streaming(
        self,
        text: str,
        first_chunk_size: int = 30,
        subsequent_chunk_size: int = 80,
    ) -> List[Dict[str, Any]]:
        """
        Segment text specifically for streaming TTS (low first-byte latency).

        Args:
            text: Input text
            first_chunk_size: Size of first chunk (for fast response)
            subsequent_chunk_size: Size of subsequent chunks

        Returns:
            List of chunks optimized for streaming
        """
        # Detect if text has dialogue
        dialogue_segments = await self._detect_dialogue(text)

        if dialogue_segments:
            # For dialogue, return dialogue first
            return await self._create_streaming_chunks_with_dialogue(
                text, dialogue_segments, first_chunk_size, subsequent_chunk_size
            )

        # Split sentences
        sentences = self._split_sentences(text)

        if not sentences:
            return [{'text': text, 'is_first': True, 'is_final': True}]

        chunks = []

        # First chunk - small for fast response
        first_text = sentences[0]
        if len(first_text) > first_chunk_size:
            # Split first sentence
            split_pos = self._find_best_split_position(first_text, first_chunk_size)
            chunks.append({
                'text': first_text[:split_pos].strip(),
                'is_first': True,
                'is_final': False,
                'pause_after': 0.1,
            })
            chunks.append({
                'text': first_text[split_pos:].strip(),
                'is_first': False,
                'is_final': len(sentences) == 1,
                'pause_after': self._calculate_pause(first_text, 'normal'),
            })
        else:
            chunks.append({
                'text': first_text,
                'is_first': True,
                'is_final': len(sentences) == 1,
                'pause_after': self._calculate_pause(first_text, 'normal'),
            })

        # Remaining chunks
        current_chunk = ""
        for i, sentence in enumerate(sentences[1:], 1):
            is_last = i == len(sentences) - 1

            if len(current_chunk) + len(sentence) <= subsequent_chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'is_first': False,
                        'is_final': False,
                        'pause_after': 0.4,
                    })
                current_chunk = sentence

            if is_last and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'is_first': False,
                    'is_final': True,
                    'pause_after': 0.5,
                })

        return chunks

    async def _create_streaming_chunks_with_dialogue(
        self,
        text: str,
        dialogue_segments: List[Dict[str, Any]],
        first_chunk_size: int,
        subsequent_chunk_size: int,
    ) -> List[Dict[str, Any]]:
        """Create streaming chunks with dialogue preservation."""
        chunks = []
        current_pos = 0

        # Sort dialogue by position
        dialogue_segments.sort(key=lambda x: x['start'])

        for dialogue in dialogue_segments:
            # Add narration before dialogue
            narration_text = text[current_pos:dialogue['start']].strip()
            if narration_text:
                if len(chunks) == 0 and len(narration_text) > first_chunk_size:
                    split_pos = self._find_best_split_position(narration_text, first_chunk_size)
                    chunks.append({
                        'text': narration_text[:split_pos].strip(),
                        'is_first': True,
                        'is_final': False,
                        'type': 'narration',
                        'pause_after': 0.1,
                    })
                    chunks.append({
                        'text': narration_text[split_pos:].strip(),
                        'is_first': False,
                        'is_final': False,
                        'type': 'narration',
                        'pause_after': 0.3,
                    })
                else:
                    chunks.append({
                        'text': narration_text,
                        'is_first': len(chunks) == 0,
                        'is_final': False,
                        'type': 'narration',
                        'pause_after': 0.4,
                    })

            # Add dialogue
            chunks.append({
                'text': dialogue['text'],
                'speaker': dialogue['speaker'],
                'is_first': False,
                'is_final': False,
                'type': 'dialogue',
                'pause_after': 0.8,
                'emotion': self._detect_emotion_from_text(dialogue['text']),
            })

            current_pos = dialogue['end']

        # Add remaining narration
        remaining_text = text[current_pos:].strip()
        if remaining_text:
            chunks.append({
                'text': remaining_text,
                'is_first': len(chunks) == 0,
                'is_final': True,
                'type': 'narration',
                'pause_after': 0.5,
            })

        # Mark final chunk
        if chunks:
            chunks[-1]['is_final'] = True

        return chunks


# Global instance
_segmenter: Optional[IntelligentTextSegmenter] = None


def get_intelligent_segmenter(
    use_llm: bool = False,
    llm_client=None,
) -> IntelligentTextSegmenter:
    """Get global intelligent text segmenter instance."""
    global _segmenter
    if _segmenter is None or _segmenter.use_llm != use_llm:
        _segmenter = IntelligentTextSegmenter(use_llm=use_llm, llm_client=llm_client)
    return _segmenter

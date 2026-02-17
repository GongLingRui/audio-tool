"""Text utility functions."""
import re
from typing import Any


def normalize_text(text: str) -> str:
    """Normalize text by removing extra whitespace."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def count_characters(text: str, include_spaces: bool = True) -> int:
    """Count characters in text."""
    if include_spaces:
        return len(text)
    return len(text.replace(" ", ""))


def split_text_into_chunks(
    text: str,
    max_chars: int = 500,
    preserve_paragraphs: bool = True,
) -> list[str]:
    """Split text into chunks of maximum characters.

    Args:
        text: Text to split
        max_chars: Maximum characters per chunk
        preserve_paragraphs: Whether to preserve paragraph boundaries

    Returns:
        List of text chunks
    """
    if preserve_paragraphs:
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= max_chars:
                current_chunk += ("\n\n" + para if current_chunk else para)
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
    else:
        chunks = []
        for i in range(0, len(text), max_chars):
            chunks.append(text[i:i + max_chars])
        return chunks


def extract_chapters(text: str) -> list[dict[str, Any]]:
    """Extract chapters from text.

    Looks for common chapter patterns like:
    - Chapter 1
    - 第一章
    - CHAPTER I
    """
    chapters = []
    lines = text.split("\n")

    chapter_patterns = [
        r'^第[一二三四五六七八九十百千\d]+[章回節节]',
        r'^Chapter\s+\d+',
        r'^CHAPTER\s+[IVXLCDM\d]+',
        r'^\d+\.\s+[A-Z]',
    ]

    combined_pattern = '|'.join(f'({p})' for p in chapter_patterns)
    pattern = re.compile(combined_pattern, re.MULTILINE)

    current_offset = 0
    for line in lines:
        match = pattern.match(line.strip())
        if match:
            chapters.append({
                "title": line.strip(),
                "offset": current_offset,
            })
        current_offset += len(line) + 1  # +1 for newline

    return chapters


def clean_text(text: str) -> str:
    """Clean text by removing artifacts."""
    # Remove page numbers
    text = re.sub(r'\b\d+\s*\n', '', text)
    # Remove headers/footers (basic)
    text = re.sub(r'.*\n\s*\n', '\n\n', text)
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    return text


def detect_language(text: str) -> str:
    """Detect language of text (simplified)."""
    # Check for Chinese characters
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    if chinese_chars > len(text) * 0.3:
        return "zh-CN"

    # Check for Japanese characters
    japanese_chars = sum(1 for c in text if '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff')
    if japanese_chars > len(text) * 0.2:
        return "ja-JP"

    # Default to English
    return "en-US"


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

"""Chunk service for intelligent audio chunk grouping."""
import re
from typing import Any


MAX_CHUNK_CHARS = 500


def get_speaker(entry: dict[str, Any]) -> str:
    """Get speaker from entry, checking both 'speaker' and 'type' fields."""
    return entry.get("speaker") or entry.get("type") or ""


def _is_structural_text(text: str) -> bool:
    """Check if text is a title, chapter heading, dedication, or other structural fragment."""
    stripped = text.strip()
    if not stripped:
        return True
    # Very short and not a full sentence (no sentence-ending punctuation)
    if len(stripped) < 80 and not stripped[-1] in '.!?':
        return True
    return False


def _is_section_break(text: str) -> bool:
    """Check if text looks like a chapter heading or section title."""
    stripped = text.strip()
    # "CHAPTER ONE", "CHAPTER II", "Chapter Three", etc.
    if re.match(r'(?i)^chapter\b', stripped):
        return True
    # All-caps short text = likely a title
    if stripped == stripped.upper() and len(stripped) < 80 and stripped.isascii():
        return True
    return False


def group_into_chunks(
    script_entries: list[dict[str, Any]],
    max_chars: int = MAX_CHUNK_CHARS
) -> list[dict[str, Any]]:
    """
    Group consecutive entries by same speaker into chunks up to max_chars.

    This intelligently merges consecutive lines from the same speaker to:
    - Reduce API calls and improve generation speed
    - Maintain natural flow for continuous dialogue
    - Respect structural boundaries (chapter headings, etc.)
    - Keep chunks manageable for TTS processing

    Args:
        script_entries: List of script entries with speaker, text, instruct
        max_chars: Maximum characters per chunk (default 500)

    Returns:
        List of grouped chunks ready for TTS generation
    """
    if not script_entries:
        return []

    chunks = []
    current_speaker = get_speaker(script_entries[0])
    current_text = script_entries[0].get("text", "")
    current_instruct = script_entries[0].get("instruct", "")

    for entry in script_entries[1:]:
        speaker = get_speaker(entry)
        text = entry.get("text", "")
        instruct = entry.get("instruct", "")

        # Don't merge if:
        # - Different speaker
        # - Different instruction (emotion/tone change)
        # - Current text is structural (title, heading)
        # - Next text is structural
        # - Combined text would exceed max_chars
        should_merge = (
            speaker == current_speaker
            and instruct == current_instruct
            and not _is_structural_text(current_text)
            and not _is_structural_text(text)
        )

        if should_merge:
            combined = current_text + " " + text
            if len(combined) <= max_chars:
                current_text = combined
            else:
                # Combined too long, save current chunk and start new
                chunks.append({
                    "speaker": current_speaker,
                    "text": current_text,
                    "instruct": current_instruct,
                })
                current_text = text
                current_instruct = instruct
        else:
            # Can't merge, save current chunk
            chunks.append({
                "speaker": current_speaker,
                "text": current_text,
                "instruct": current_instruct,
            })
            current_speaker = speaker
            current_text = text
            current_instruct = instruct

    # Don't forget the last chunk
    chunks.append({
        "speaker": current_speaker,
        "text": current_text,
        "instruct": current_instruct,
    })

    return chunks


def merge_consecutive_narrators(
    entries: list[dict[str, Any]],
    max_merged_length: int = 800
) -> tuple[list[dict[str, Any]], int]:
    """
    Merge consecutive NARRATOR entries that share the same instruct value.

    Skips merging across section/chapter breaks. Caps merged text at
    max_merged_length characters to avoid creating overly long TTS entries.

    Args:
        entries: List of script entries
        max_merged_length: Maximum characters for merged text

    Returns:
        Tuple of (merged_entries, merge_count)
    """
    if not entries:
        return entries, 0

    merged = []
    merges = 0
    i = 0

    while i < len(entries):
        entry = entries[i]

        # Skip non-narrators and section breaks
        if entry.get("speaker") != "NARRATOR" or _is_section_break(entry.get("text", "")):
            merged.append(entry)
            i += 1
            continue

        # Start a narrator run
        combined_text = entry["text"]
        instruct = entry.get("instruct", "")
        run_count = 1
        j = i + 1

        # Accumulate consecutive NARRATORs with same instruct
        while j < len(entries):
            next_entry = entries[j]
            if next_entry.get("speaker") != "NARRATOR":
                break
            if next_entry.get("instruct", "") != instruct:
                break
            if _is_section_break(next_entry.get("text", "")):
                break

            candidate = combined_text + " " + next_entry["text"]
            if len(candidate) > max_merged_length:
                break

            combined_text = candidate
            run_count += 1
            j += 1

        # Add merged or single entry
        merged.append({
            "speaker": "NARRATOR",
            "text": combined_text,
            "instruct": instruct
        })

        if run_count > 1:
            merges += run_count - 1

        i = j

    return merged, merges


def split_script_to_chunks(
    script: list[dict[str, Any]],
    max_chars: int = MAX_CHUNK_CHARS,
    merge_narrators: bool = True,
    max_narrator_length: int = 800
) -> list[dict[str, Any]]:
    """
    Complete chunk processing pipeline.

    1. Optionally merge consecutive narrators
    2. Group into chunks by same speaker

    Args:
        script: Raw script entries from LLM
        max_chars: Max characters per TTS chunk
        merge_narrators: Whether to merge consecutive NARRATOR entries
        max_narrator_length: Max length for merged narrator segments

    Returns:
        List of processed chunks ready for audio generation
    """
    # Step 1: Merge consecutive narrators if requested
    if merge_narrators:
        script, merge_count = merge_consecutive_narrators(
            script,
            max_merged_length=max_narrator_length
        )

    # Step 2: Group into chunks by same speaker
    chunks = group_into_chunks(script, max_chars=max_chars)

    return chunks


def estimate_audio_duration(text: str, words_per_minute: int = 150) -> float:
    """
    Estimate audio duration for a text segment.

    Args:
        text: Text to estimate duration for
        words_per_minute: Average speaking rate (default 150 wpm)

    Returns:
        Estimated duration in seconds
    """
    # Count words (split by whitespace)
    words = len(text.split())

    # Calculate minutes and convert to seconds
    minutes = words / words_per_minute
    seconds = minutes * 60

    # Add 10% buffer for pauses and emphasis
    seconds *= 1.1

    return seconds


def calculate_chunk_timing(
    chunks: list[dict[str, Any]],
    pause_between_speakers: float = 0.5,
    pause_same_speaker: float = 0.25
) -> list[dict[str, Any]]:
    """
    Calculate timing information for chunks.

    Adds start_time and end_time to each chunk based on estimated durations.

    Args:
        chunks: List of chunks with speaker, text, instruct
        pause_between_speakers: Pause seconds between different speakers
        pause_same_speaker: Pause seconds between same speaker

    Returns:
        List of chunks with added timing information
    """
    current_time = 0.0
    prev_speaker = None

    for chunk in chunks:
        # Estimate duration
        chunk["estimated_duration"] = estimate_audio_duration(chunk["text"])

        # Add timing
        chunk["start_time"] = current_time
        chunk["end_time"] = current_time + chunk["estimated_duration"]

        # Calculate next start time with pause
        if prev_speaker and chunk["speaker"] != prev_speaker:
            current_time += pause_between_speakers
        else:
            current_time += pause_same_speaker

        current_time += chunk["estimated_duration"]
        prev_speaker = chunk["speaker"]

    return chunks

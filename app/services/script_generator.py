"""Script generation service using LLM."""
import httpx
import json
import re
import os
from typing import Any, Optional
from datetime import datetime
from pathlib import Path

from app.config import settings
from app.core.exceptions import LLMError


def load_default_prompts():
    """Read default_prompts.txt from disk and return (system_prompt, user_prompt).

    Re-reads on every call so edits are picked up without restarting the app.
    """
    prompts_file = Path(__file__).parent.parent.parent / "default_prompts.txt"

    try:
        with open(prompts_file, "r", encoding="utf-8") as f:
            raw = f.read()
    except FileNotFoundError:
        # Fall back to built-in prompts if file doesn't exist
        return _get_builtin_prompts()

    parts = raw.split("---SEPARATOR---", maxsplit=1)
    if len(parts) != 2:
        return _get_builtin_prompts()

    return parts[0].strip(), parts[1].strip()


def _get_builtin_prompts():
    """Fallback built-in prompts."""
    system_prompt = """You are a script writer converting books into audiobook scripts for an advanced TTS system. Output ONLY valid JSON arrays — no markdown, no explanations.

FORMAT:
[
  {"speaker": "NARRATOR", "text": "The room had gone cold.", "instruct": "Quiet, tense narration."},
  {"speaker": "CHARACTER", "text": "Tell me the truth.", "instruct": "Firm quiet authority."}
]

RULES:
1. NARRATOR = everything except spoken dialogue
2. PRESERVE the author's original text
3. Drop attribution tags ("he said") but extract actions as NARRATOR
4. Keep consecutive narrator text together unless tone shifts"""

    user_prompt = """{context}

Remember: if another character would hear the words, it is dialogue. Everything else is NARRATOR.

SOURCE TEXT:
{chunk}"""

    return system_prompt, user_prompt


def clean_json_string(text: str) -> Optional[str]:
    """Clean and extract valid JSON array from LLM response.

    Removes thinking tags, markdown code blocks, and extracts JSON array.
    """
    # Remove thinking tags (various formats used by different models)
    # GLM, DeepSeek, Qwen, etc. use different thinking tag formats
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'<thinking>[\s\S]*?</thinking>', '', text)
    text = re.sub(r'<reflection>[\s\S]*?</reflection>', '', text)
    text = re.sub(r'<reasoning>[\s\S]*?</reasoning>', '', text)
    # Handle unclosed thinking tags (model started thinking but didn't close)
    text = re.sub(r'```[\s\S]*$', '', text)
    text = re.sub(r'<thinking>[\s\S]*$', '', text)

    # Remove markdown code blocks
    if "```" in text:
        # Find content between ```json and ``` or just ``` and ```
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if match:
            text = match.group(1).strip()

    # Find the JSON array - match from first [ to its closing ]
    # Use a bracket counter to find the correct closing bracket
    start = text.find('[')
    if start == -1:
        return None

    bracket_count = 0
    end = -1
    in_string = False
    escape_next = False

    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
            if bracket_count == 0:
                end = i + 1
                break

    if end == -1:
        # No closing bracket found, try to salvage
        last_complete = text.rfind('},')
        if last_complete > start:
            return text[start:last_complete+1] + ']'
        return None

    json_text = text[start:end]

    # Clean control characters inside strings (common LLM issue)
    # Replace literal newlines/tabs inside JSON strings with escaped versions
    def fix_control_chars(match):
        s = match.group(0)
        # Replace unescaped control characters
        s = s.replace('\n', '\\n')
        s = s.replace('\r', '\\r')
        s = s.replace('\t', '\\t')
        return s

    # Fix control characters inside string values
    json_text = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_control_chars, json_text)

    return json_text


def repair_json_array(json_text: str) -> Optional[list]:
    """Attempt to repair common JSON array issues from LLM output."""
    if not json_text:
        return None

    # Try parsing as-is first
    try:
        result = json.loads(json_text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Fix 1: Add missing commas between objects (}\s*{" -> },\n{")
    fixed = re.sub(r'\}\s*\{', '},\n{', json_text)
    try:
        result = json.loads(fixed)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Fix 2: Remove trailing commas before ]
    fixed = re.sub(r',\s*\]', ']', fixed)
    try:
        result = json.loads(fixed)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Fix 3: Try to extract individual entries and rebuild
    entries = []
    # Match individual JSON objects
    pattern = r'\{\s*"speaker"\s*:\s*"[^"]*"\s*,\s*"text"\s*:\s*"(?:[^"\\]|\\.)*"\s*,\s*"instruct"\s*:\s*"(?:[^"\\]|\\.)*"\s*\}'
    matches = re.findall(pattern, json_text, re.DOTALL)

    for match in matches:
        try:
            entry = json.loads(match)
            entries.append(entry)
        except json.JSONDecodeError:
            continue

    if entries:
        return entries

    # Fix 4: Last resort - find last complete entry and truncate
    last_complete = json_text.rfind('},')
    if last_complete > 0:
        try:
            truncated = json_text[:last_complete+1] + ']'
            # Ensure it starts with [
            if not truncated.strip().startswith('['):
                truncated = '[' + truncated
            result = json.loads(truncated)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return None


def salvage_json_entries(json_text: str) -> list:
    """Last resort: extract individual valid entries with regex."""
    entries = []
    # Match individual JSON objects with speaker, text, instruct fields
    pattern = r'\{\s*"speaker"\s*:\s*"([^"]*)"\s*,\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"instruct"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}'

    for match in re.finditer(pattern, json_text, re.DOTALL):
        try:
            speaker = match.group(1).replace('\\"', '"').replace('\\n', '\n')
            text = match.group(2).replace('\\"', '"').replace('\\n', '\n')
            instruct = match.group(3).replace('\\"', '"').replace('\\n', '\n')

            entries.append({
                "speaker": speaker,
                "text": text,
                "instruct": instruct
            })
        except Exception:
            continue

    return entries


def _is_section_break(text: str) -> bool:
    """Check if text looks like a chapter heading or section title."""
    stripped = text.strip()
    # "CHAPTER ONE", "CHAPTER II", "Chapter Three", etc.
    if re.match(r'(?i)^chapter\b', stripped):
        return True
    # All-caps short text = likely a title ("A SCANDAL IN BOHEMIA", "THE RED-HEADED LEAGUE")
    if stripped == stripped.upper() and len(stripped) < 80 and stripped.isascii():
        return True
    return False


def merge_consecutive_narrators(entries: list, max_merged_length: int = 800) -> tuple[list, int]:
    """Merge consecutive NARRATOR entries that share the same instruct value.

    Skips merging across section/chapter breaks. Caps merged text at
    max_merged_length characters to avoid creating overly long TTS entries.
    """
    if not entries:
        return entries, 0

    merged = []
    merges = 0
    i = 0
    while i < len(entries):
        entry = entries[i]

        if entry.get("speaker") != "NARRATOR" or _is_section_break(entry.get("text", "")):
            merged.append(entry)
            i += 1
            continue

        # Start a narrator run — accumulate consecutive NARRATORs with same instruct
        combined_text = entry["text"]
        instruct = entry.get("instruct", "")
        run_count = 1
        j = i + 1

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

        merged.append({
            "speaker": "NARRATOR",
            "text": combined_text,
            "instruct": instruct
        })
        if run_count > 1:
            merges += run_count - 1
        i = j

    return merged, merges


class ScriptGenerator:
    """Service for generating audiobook scripts using LLM."""

    def __init__(self, config: dict | None = None):
        self.base_url = config.get("base_url", settings.llm_base_url) if config else settings.llm_base_url
        self.api_key = config.get("api_key", settings.llm_api_key) if config else settings.llm_api_key
        self.model_name = config.get("model_name", settings.llm_model) if config else settings.llm_model
        self.timeout = config.get("timeout", 300) if config else 300
        self.log_dir = os.path.join(settings.upload_dir.parent, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

    async def generate_script(
        self,
        text: str,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        character_roster: str | None = None,
        previous_context: list | None = None,
        max_tokens: int = 16000,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0,
        presence_penalty: float = 0.0,
    ) -> list[dict]:
        """
        Generate script from text using LLM.

        Args:
            text: Book text to convert to script
            system_prompt: Custom system prompt
            user_prompt: Custom user prompt
            character_roster: Known characters for context preservation
            previous_context: Last few script entries for continuity
            max_tokens: Maximum tokens in LLM response
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            min_p: Minimum p sampling parameter
            presence_penalty: Presence penalty for repetition

        Returns:
            List of script entries

        Raises:
            LLMError: If LLM request fails
        """
        # Load default prompts if not provided
        if system_prompt is None or user_prompt is None:
            default_system, default_user = load_default_prompts()
            system_prompt = system_prompt or default_system
            user_prompt = user_prompt or default_user

        # Add context
        context_parts = []
        if character_roster:
            context_parts.append(f"Known characters:\n{character_roster}")
        if previous_context:
            context_parts.append("Previous script context:")
            context_parts.extend(json.dumps(e, ensure_ascii=False) for e in previous_context[-3:])

        context = "\n".join(context_parts) if context_parts else ""
        full_user_prompt = user_prompt.format(context=context, chunk=text)

        # Generate
        entries = await self._generate_chunk(
            "",
            system_prompt,
            full_user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
        )

        return entries

    async def generate_script_batches(
        self,
        text: str,
        batch_size: int = 5000,
        overlap: int = 200,
        **kwargs
    ) -> list[dict]:
        """Generate script in batches for long texts with context preservation."""
        # Split text into overlapping batches
        batches = []
        start = 0
        while start < len(text):
            end = start + batch_size
            batch_text = text[start:end]
            batches.append((batch_text, start))
            start = end - overlap

        all_entries = []
        previous_tail = []

        for i, (batch_text, batch_start) in enumerate(batches):
            # Pass last few entries as context
            kwargs["previous_context"] = previous_tail

            entries = await self.generate_script(batch_text, **kwargs)
            all_entries.extend(entries)

            # Keep last few entries for next batch context
            previous_tail = entries[-3:] if len(entries) > 3 else entries

        return all_entries

    async def _generate_chunk(
        self,
        text: str,
        system_prompt: str,
        user_prompt: str,
        start_index: int = 0,
        max_tokens: int = 16000,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0,
        presence_penalty: float = 0.0,
        max_retries: int = 2,
    ) -> list[dict]:
        """Generate script entries for a text chunk."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    # Build extra body for parameters
                    extra_body = {
                        k: v for k, v in {
                            "top_k": top_k,
                            "min_p": min_p,
                        }.items() if v is not None
                    }

                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self.model_name,
                            "messages": messages,
                            "temperature": temperature,
                            "top_p": top_p,
                            "presence_penalty": presence_penalty,
                            "max_tokens": max_tokens,
                            **extra_body,
                        },
                    )
                    response.raise_for_status()
            except httpx.HTTPError as e:
                if attempt < max_retries:
                    continue
                raise LLMError(f"LLM request failed after {max_retries + 1} attempts: {str(e)}")

        # Parse response
        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            finish_reason = data["choices"][0].get("finish_reason", "unknown")
            usage = data.get("usage", {})
        except (KeyError, IndexError) as e:
            raise LLMError(f"Invalid LLM response format: {str(e)}")

        # Log response
        self._log_response(content, finish_reason, usage, attempt)

        # Extract JSON from response
        json_content = clean_json_string(content)

        if not json_content:
            # Try salvage as last resort
            entries = salvage_json_entries(content)
            if entries:
                return self._add_indices(entries, start_index)
            raise LLMError("Could not extract valid JSON from LLM response")

        # Try to parse JSON
        entries = None
        try:
            entries = json.loads(json_content)
        except json.JSONDecodeError:
            entries = repair_json_array(json_content)

        if not entries or not isinstance(entries, list):
            # Try salvage
            entries = salvage_json_entries(content)

        if not entries:
            raise LLMError("Could not parse valid script entries from LLM response")

        return self._add_indices(entries, start_index)

    def _add_indices(self, entries: list, start_index: int) -> list:
        """Add index field to entries."""
        for i, entry in enumerate(entries):
            entry["index"] = start_index + i
            # Ensure required fields
            if "emotion" not in entry:
                entry["emotion"] = None
            if "section" not in entry:
                entry["section"] = None
        return entries

    def _log_response(self, content: str, finish_reason: str, usage: dict, attempt: int):
        """Log LLM response for debugging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(self.log_dir, f"script_gen_{timestamp}.log")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Attempt {attempt + 1} | finish_reason={finish_reason}\n")
            if usage:
                f.write(f"Tokens: prompt={usage.get('prompt_tokens', '?')} "
                       f"completion={usage.get('completion_tokens', '?')} "
                       f"total={usage.get('total_tokens', '?')}\n")
            f.write(f"{'─'*80}\n")
            f.write(content[:2000])  # Truncate long responses
            if len(content) > 2000:
                f.write(f"\n... ({len(content) - 2000} more chars)")
            f.write(f"\n{'='*80}\n")

    async def review_script(
        self,
        script: list[dict],
        source_text: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ) -> tuple[list[dict], list[str]]:
        """
        Review script and fix issues.

        Args:
            script: Script entries to review
            source_text: Original source text for reference
            system_prompt: Custom review system prompt
            user_prompt: Custom review user prompt

        Returns:
            Tuple of (fixed_script, issues_found)
        """
        issues = []

        # Check for speaker consistency
        speakers = set(entry.get("speaker", "") for entry in script)
        if "NARRATOR" not in speakers:
            issues.append("No narrator found - consider adding NARRATOR segments")

        # Check for empty text
        for i, entry in enumerate(script):
            if not entry.get("text", "").strip():
                issues.append(f"Entry {i} has empty text")
            if not entry.get("speaker", "").strip():
                issues.append(f"Entry {i} missing speaker")

        # Merge consecutive narrators
        merged_script, merge_count = merge_consecutive_narrators(script)
        if merge_count > 0:
            issues.append(f"Merged {merge_count} consecutive narrator entries")

        return merged_script, issues

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt."""
        return """You are a professional audiobook script annotator. Your task is to analyze the given text and create a structured script with speaker labels, dialogue, and TTS instructions.

Guidelines:
1. Identify the narrator (NARRATOR) and all characters
2. Mark dialogue with the speaker's name in ALL CAPS
3. Include TTS instructions for emotion, tone, and pacing using the Voice Reference lexicon
4. Keep narrative segments together - don't over-fragment
5. Output must be valid JSON array

Output format:
[
  {
    "index": 0,
    "speaker": "NARRATOR",
    "text": "The narrative text here",
    "instruct": "Calm, objective narration",
    "emotion": "neutral"
  },
  {
    "index": 1,
    "speaker": "CHARACTER_NAME",
    "text": "Character dialogue here",
    "instruct": "Emotional, questioning tone",
    "emotion": "curious"
  }
]

TTS Instruction Guidelines:
- Use emotion/attitude terms from Section II (sad, joyful, anxious, calm, etc.)
- Use delivery/pacing terms from Section III (staccato, legato, rapid-fire, measured, etc.)
- Keep instructions concise and actionable
- Focus on how the line should be delivered, not the voice quality"""

    def _get_default_user_prompt(self) -> str:
        """Get default user prompt."""
        return """Please convert the following text into an audiobook script format.

Instructions:
1. Identify all speakers (narrator and characters)
2. Mark dialogue with character names in ALL CAPS
3. Add appropriate TTS instructions for each segment
4. Keep narrative passages together when appropriate
5. Output as a JSON array"""

    def _get_default_review_system_prompt(self) -> str:
        """Get default review system prompt."""
        return """You are an expert audiobook script reviewer. Your task is to review and fix issues in the script.

Common issues to fix:
1. Split dialogue that should be merged (consecutive same-speaker entries)
2. Missing or incorrect speaker labels
3. Inappropriate or missing TTS instructions
4. Over-fragmented narrative passages
5. Incorrect attribution tags in dialogue

Review the script and return a corrected version."""

    def _get_default_review_user_prompt(self) -> str:
        """Get default review user prompt."""
        return """Please review the following script and fix any issues.

Return the corrected script as a JSON array."""

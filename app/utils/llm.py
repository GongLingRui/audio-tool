"""LLM utility functions."""
import httpx
from typing import Any

from app.config import settings


async def call_llm(
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    """Call LLM API and return response.

    Args:
        messages: List of message dicts with 'role' and 'content'
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Generated text response

    Raises:
        Exception: If API call fails
    """
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{settings.llm_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.llm_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.llm_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            response.raise_for_status()

            data = response.json()
            return data["choices"][0]["message"]["content"]

    except httpx.HTTPError as e:
        raise Exception(f"LLM API call failed: {str(e)}")


async def call_llm_with_structured_output(
    messages: list[dict[str, str]],
    output_format: dict[str, Any],
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Call LLM API and parse structured output.

    Args:
        messages: List of message dicts
        output_format: Expected output format description
        temperature: Sampling temperature

    Returns:
        Parsed structured output

    Raises:
        Exception: If API call fails or parsing fails
    """
    # Add format instruction to system message
    format_instruction = f"\n\nOutput must be in JSON format: {output_format}"
    messages[0]["content"] += format_instruction

    response_text = await call_llm(messages, temperature)

    # Parse JSON response
    import json

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        import re

        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        raise Exception(f"Failed to parse LLM response as JSON: {response_text[:200]}...")


async def translate_text(
    text: str,
    source_lang: str = "zh",
    target_lang: str = "en",
) -> str:
    """Translate text using LLM.

    Args:
        text: Text to translate
        source_lang: Source language code (zh, en, ja, ko, etc.)
        target_lang: Target language code

    Returns:
        Translated text

    Raises:
        Exception: If translation fails
    """
    # Language name mapping
    lang_names = {
        "zh": "Chinese",
        "en": "English",
        "ja": "Japanese",
        "ko": "Korean",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "ru": "Russian",
        "ar": "Arabic",
        "pt": "Portuguese",
        "it": "Italian",
    }

    source_name = lang_names.get(source_lang, source_lang)
    target_name = lang_names.get(target_lang, target_lang)

    messages = [
        {
            "role": "system",
            "content": f"You are a professional translator. Translate the given text from {source_name} to {target_name}. Only return the translated text, no explanations."
        },
        {
            "role": "user",
            "content": text
        }
    ]

    try:
        translated = await call_llm(messages, temperature=0.3)
        return translated.strip()
    except Exception as e:
        raise Exception(f"Translation failed: {str(e)}")

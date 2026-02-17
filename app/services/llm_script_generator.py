"""
LLM Script Generation Service
Integrates with LLM APIs for audiobook script generation
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    QWEN = "qwen"
    CUSTOM = "custom"


@dataclass
class LLMResponse:
    """Response from LLM API."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    error: Optional[str] = None


@dataclass
class ScriptGenerationRequest:
    """Request for script generation."""
    book_content: str
    project_id: str
    options: Dict[str, Any]
    custom_prompt: Optional[str] = None


class LLMScriptGenerator:
    """
    Service for generating audiobook scripts using LLM APIs.

    Supports:
    - Multiple LLM providers (OpenAI, Anthropic, Qwen, Custom)
    - Custom prompts and system messages
    - Retry logic with exponential backoff
    - Structured output parsing
    - Error handling and validation
    """

    # Default prompts for script generation
    DEFAULT_SYSTEM_PROMPT = """你是一个专业的有声书脚本创作助手。你的任务是将小说文本转换为适合有声书制作的脚本。

脚本格式要求：
1. 将文本按对话和叙述分块
2. 标识每个块的说话人（NARRATOR 表示旁白）
3. 为每个块添加适当的情感和语气指导
4. 保持原作的故事节奏和情感

输出格式为 JSON 数组，每个元素包含：
- index: 序号
- speaker: 说话人（NARRATOR 或角色名）
- text: 文本内容
- instruct: 情感/语气指导

示例：
[
  {"index": 0, "speaker": "NARRATOR", "text": "这是一个关于勇气的故事", "instruct": "平静叙述"},
  {"index": 1, "speaker": "张三", "text": "你好！", "instruct": "友好、热情"}
]"""

    DEFAULT_USER_PROMPT_TEMPLATE = """请将以下小说文本转换为有声书脚本：

{text}

要求：
1. 识别所有对话和叙述部分
2. 为对话部分识别角色名称
3. 添加适当的情感和语气指导
4. 输出为 JSON 格式"""

    # Model configurations
    MODEL_CONFIGS = {
        LLMProvider.OPENAI: {
            "default_model": "gpt-4o-mini",
            "api_base": "https://api.openai.com/v1",
            "timeout": 60,
        },
        LLMProvider.ANTHROPIC: {
            "default_model": "claude-3-haiku-20240307",
            "api_base": "https://api.anthropic.com",
            "timeout": 60,
        },
        LLMProvider.QWEN: {
            "default_model": "qwen-turbo",
            "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "timeout": 120,
        },
    }

    def __init__(
        self,
        provider: LLMProvider = LLMProvider.QWEN,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize LLM script generator.

        Args:
            provider: LLM provider to use
            api_key: API key for the provider (defaults to settings)
            model: Model name (defaults to provider's default)
            base_url: Custom API base URL
        """
        self.provider = provider
        self.api_key = api_key or self._get_default_api_key()
        self.model = model or self.MODEL_CONFIGS[provider]["default_model"]
        self.base_url = base_url or self.MODEL_CONFIGS[provider].get("api_base")
        self.timeout = self.MODEL_CONFIGS[provider].get("timeout", 60)

    def _get_default_api_key(self) -> Optional[str]:
        """Get default API key from settings."""
        key_map = {
            LLMProvider.OPENAI: settings.OPENAI_API_KEY if hasattr(settings, 'OPENAI_API_KEY') else None,
            LLMProvider.ANTHROPIC: settings.ANTHROPIC_API_KEY if hasattr(settings, 'ANTHROPIC_API_KEY') else None,
            LLMProvider.QWEN: settings.QWEN_API_KEY if hasattr(settings, 'QWEN_API_KEY') else None,
        }
        return key_map.get(self.provider)

    async def generate_script(
        self,
        request: ScriptGenerationRequest,
        max_retries: int = 3,
    ) -> LLMResponse:
        """
        Generate audiobook script from book content.

        Args:
            request: Script generation request
            max_retries: Maximum number of retry attempts

        Returns:
            LLM response with generated script
        """
        # Prepare prompts
        system_prompt = request.custom_prompt or self.DEFAULT_SYSTEM_PROMPT
        user_prompt = self.DEFAULT_USER_PROMPT_TEMPLATE.format(
            text=request.book_content[:8000]  # Limit content length
        )

        # Add custom options
        if request.options.get("style"):
            system_prompt += f"\n\n风格要求：{request.options['style']}"
        if request.options.get("speakers"):
            system_prompt += f"\n\n主要角色：{', '.join(request.options['speakers'])}"

        # Call LLM API
        try:
            response = await self._call_llm_api(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_retries=max_retries,
            )
            return response

        except Exception as e:
            logger.error(f"LLM script generation failed: {e}")
            return LLMResponse(
                content="",
                model=self.model,
                error=str(e),
            )

    async def _call_llm_api(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 3,
    ) -> LLMResponse:
        """Call LLM API with retry logic."""

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    if self.provider in [LLMProvider.OPENAI, LLMProvider.QWEN]:
                        return await self._call_openai_compat(
                            client, system_prompt, user_prompt
                        )
                    elif self.provider == LLMProvider.ANTHROPIC:
                        return await self._call_anthropic(
                            client, system_prompt, user_prompt
                        )
                    else:
                        return await self._call_custom(
                            client, system_prompt, user_prompt
                        )

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    # Rate limited, wait and retry
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                    import asyncio
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying in {wait_time}s")
                    import asyncio
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise

        return LLMResponse(
            content="",
            model=self.model,
            error="Max retries exceeded",
        )

    async def _call_openai_compat(
        self,
        client: httpx.AsyncClient,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        """Call OpenAI-compatible API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 4000,
        }

        response = await client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            model=self.model,
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        )

    async def _call_anthropic(
        self,
        client: httpx.AsyncClient,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        """Call Anthropic API."""
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        payload = {
            "model": self.model,
            "max_tokens": 4000,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
        }

        response = await client.post(
            f"{self.base_url}/v1/messages",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()

        data = response.json()
        content = data["content"][0]["text"]
        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            model=self.model,
            usage={
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            },
        )

    async def _call_custom(
        self,
        client: httpx.AsyncClient,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        """Call custom API endpoint."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        response = await client.post(
            self.base_url,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        return LLMResponse(
            content=content,
            model=self.model,
        )

    def parse_script_response(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse LLM response into script entries.

        Args:
            content: LLM response content

        Returns:
            List of script entries
        """
        try:
            # Try to parse as JSON directly
            if content.strip().startswith('['):
                return json.loads(content)

            # Try to extract JSON from markdown code blocks
            if '```' in content:
                start = content.find('[')
                end = content.rfind(']') + 1
                if start >= 0 and end > start:
                    return json.loads(content[start:end])

            # Fallback: try to parse line by line
            entries = []
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Simple heuristic parsing
                    if '|' in line:
                        parts = line.split('|')
                        if len(parts) >= 3:
                            entries.append({
                                "index": len(entries),
                                "speaker": parts[0].strip(),
                                "text": parts[1].strip(),
                                "instruct": parts[2].strip() if len(parts) > 2 else "",
                            })

            return entries

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse script response: {e}")
            return []

    def validate_script(self, entries: List[Dict[str, Any]]) -> tuple[bool, List[str]]:
        """
        Validate generated script.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if not entries:
            errors.append("脚本为空")
            return False, errors

        # Check required fields
        required_fields = ["index", "speaker", "text"]
        for i, entry in enumerate(entries):
            for field in required_fields:
                if field not in entry:
                    errors.append(f"条目 {i} 缺少字段: {field}")

        # Check text content
        for i, entry in enumerate(entries):
            if not entry.get("text") or len(entry.get("text", "").strip()) == 0:
                errors.append(f"条目 {i} 的文本为空")

        # Check for duplicate speakers (might indicate inconsistency)
        speakers = [entry.get("speaker") for entry in entries]
        narrator_count = speakers.count("NARRATOR")
        if narrator_count == 0:
            errors.append("缺少旁白 (NARRATOR)")

        return len(errors) == 0, errors


# Singleton instances
_llm_generators: Dict[LLMProvider, LLMScriptGenerator] = {}


def get_llm_generator(
    provider: LLMProvider = LLMProvider.QWEN,
    **kwargs
) -> LLMScriptGenerator:
    """Get or create LLM generator instance."""
    if provider not in _llm_generators:
        _llm_generators[provider] = LLMScriptGenerator(provider, **kwargs)
    return _llm_generators[provider]

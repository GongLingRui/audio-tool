"""
TTS Retry Service
Handles retry logic for failed TTS generation with exponential backoff
"""

import asyncio
import logging
from typing import Optional, Callable, Any, Dict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class RetryPolicy(Enum):
    """Retry policy types."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"


class TTSRetryService:
    """Service for retrying failed TTS generation with configurable policies."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        backoff_multiplier: float = 2.0,
        policy: RetryPolicy = RetryPolicy.EXPONENTIAL_BACKOFF,
    ):
        """
        Initialize retry service.

        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay between retries
            backoff_multiplier: Multiplier for exponential backoff
            policy: Retry policy to use
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.policy = policy

    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        error_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            error_context: Context information for logging
            **kwargs: Keyword arguments for func

        Returns:
            Result of func

        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        delay = self.initial_delay

        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt}/{self.max_retries} after {delay:.1f}s delay")
                    await asyncio.sleep(delay)

                result = await func(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt} succeeded")
                return result

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {str(e)}",
                    extra=error_context or {}
                )

                # Don't delay after last attempt
                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)

        # All retries failed
        error_message = f"Failed after {self.max_retries + 1} attempts"
        if error_context:
            error_message += f" | Context: {error_context}"

        logger.error(error_message)
        raise last_exception

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay before next retry."""
        if self.policy == RetryPolicy.EXPONENTIAL_BACKOFF:
            delay = self.initial_delay * (self.backoff_multiplier ** attempt)
        elif self.policy == RetryPolicy.LINEAR_BACKOFF:
            delay = self.initial_delay + (attempt * self.initial_delay)
        else:  # FIXED_DELAY
            delay = self.initial_delay

        return min(delay, self.max_delay)

    async def retry_chunk_generation(
        self,
        chunk_id: str,
        generate_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Retry TTS generation for a specific chunk.

        Args:
            chunk_id: ID of the chunk being generated
            generate_func: Function to generate audio
            *args, **kwargs: Arguments for generate_func

        Returns:
            Generated audio data

        Raises:
            Exception if all retries fail
        """
        error_context = {
            "chunk_id": chunk_id,
            "timestamp": datetime.now().isoformat(),
        }

        return await self.execute_with_retry(
            generate_func,
            *args,
            error_context=error_context,
            **kwargs
        )

    def get_retry_info(self) -> Dict[str, Any]:
        """Get retry service configuration."""
        return {
            "max_retries": self.max_retries,
            "initial_delay": self.initial_delay,
            "max_delay": self.max_delay,
            "backoff_multiplier": self.backoff_multiplier,
            "policy": self.policy.value,
        }


# Global retry service instance
_tts_retry_service: Optional[TTSRetryService] = None


def get_tts_retry_service() -> TTSRetryService:
    """Get or create TTS retry service instance."""
    global _tts_retry_service
    if _tts_retry_service is None:
        _tts_retry_service = TTSRetryService()
    return _tts_retry_service

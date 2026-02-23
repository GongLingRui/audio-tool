"""
Streaming TTS Optimizer - Low first-byte latency streaming
Optimized streaming TTS with intelligent chunking for <150ms first-byte latency
"""
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, AsyncIterator, Callable
from dataclasses import dataclass

from .intelligent_text_segmenter import (
    get_intelligent_segmenter,
    TextSegment,
)


logger = logging.getLogger(__name__)


@dataclass
class StreamingChunk:
    """A chunk of audio for streaming."""
    index: int
    text: str
    audio_data: Optional[bytes] = None
    is_first: bool = False
    is_final: bool = False
    latency_ms: float = 0.0
    pause_after: float = 0.0
    metadata: Dict[str, Any] = None


class StreamingTTSOptimizer:
    """
    Streaming TTS optimizer for ultra-low first-byte latency.

    Target first-byte latency: <150ms (CosyVoice2-0.5B standard)

    Features:
    - Intelligent text chunking for streaming
    - First chunk optimization (small, fast)
    - Parallel chunk generation
    - Incremental streaming
    - Latency monitoring
    """

    def __init__(
        self,
        first_chunk_size: int = 30,
        subsequent_chunk_size: int = 80,
        target_first_byte_latency_ms: float = 150.0,
        enable_parallel_generation: bool = True,
    ):
        """
        Initialize streaming TTS optimizer.

        Args:
            first_chunk_size: Size of first chunk (for fast response)
            subsequent_chunk_size: Size of subsequent chunks
            target_first_byte_latency_ms: Target first-byte latency
            enable_parallel_generation: Enable parallel chunk generation
        """
        self.first_chunk_size = first_chunk_size
        self.subsequent_chunk_size = subsequent_chunk_size
        self.target_latency = target_first_byte_latency_ms
        self.enable_parallel = enable_parallel_generation
        self._segmenter = get_intelligent_segmenter()

    async def generate_streaming(
        self,
        text: str,
        generator: Callable,
        voice_config: Dict[str, Any],
        emotion: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[StreamingChunk]:
        """
        Generate streaming audio with low first-byte latency.

        Args:
            text: Text to synthesize
            generator: Async TTS generator function
            voice_config: Voice configuration
            emotion: Emotion parameters

        Yields:
            StreamingChunk with audio data
        """
        start_time = time.time()

        # Step 1: Fast segmentation for streaming
        chunks = await self._segmenter.segment_for_streaming(
            text,
            self.first_chunk_size,
            self.subsequent_chunk_size,
        )

        logger.info(f"Segmented into {len(chunks)} chunks for streaming")

        # Step 2: Generate chunks (optimized for speed)
        if self.enable_parallel and len(chunks) > 2:
            async for chunk in self._generate_with_parallel(
                chunks, generator, voice_config, emotion, start_time
            ):
                yield chunk
        else:
            async for chunk in self._generate_sequential(
                chunks, generator, voice_config, emotion, start_time
            ):
                yield chunk

    async def _generate_with_parallel(
        self,
        chunks: List[Dict[str, Any]],
        generator: Callable,
        voice_config: Dict[str, Any],
        emotion: Dict[str, Any],
        start_time: float,
    ) -> AsyncIterator[StreamingChunk]:
        """Generate chunks with parallel processing (after first chunk)."""
        # Process first chunk immediately for low latency
        first_chunk_data = chunks[0]

        first_latency = (time.time() - start_time) * 1000

        try:
            audio = await generator(
                first_chunk_data['text'],
                voice_config,
                emotion,
            )

            yield StreamingChunk(
                index=0,
                text=first_chunk_data['text'],
                audio_data=audio,
                is_first=True,
                is_final=len(chunks) == 1,
                latency_ms=first_latency,
                pause_after=first_chunk_data.get('pause_after', 0.0),
                metadata=first_chunk_data,
            )
        except Exception as e:
            logger.error(f"Error generating first chunk: {e}")
            yield StreamingChunk(
                index=0,
                text=first_chunk_data['text'],
                is_first=True,
                is_final=len(chunks) == 1,
                latency_ms=first_latency,
                error=str(e),
            )

        # Generate remaining chunks in parallel
        if len(chunks) > 1:
            remaining_chunks = chunks[1:]

            # Create generation tasks
            semaphore = asyncio.Semaphore(3)  # Limit concurrent generation

            async def generate_chunk(index: int, chunk_data: Dict):
                async with semaphore:
                    gen_start = time.time()
                    try:
                        audio = await generator(
                            chunk_data['text'],
                            voice_config,
                            emotion,
                        )
                        return StreamingChunk(
                            index=index,
                            text=chunk_data['text'],
                            audio_data=audio,
                            is_first=False,
                            is_final=False,
                            latency_ms=(time.time() - gen_start) * 1000,
                            pause_after=chunk_data.get('pause_after', 0.0),
                            metadata=chunk_data,
                        )
                    except Exception as e:
                        logger.error(f"Error generating chunk {index}: {e}")
                        return StreamingChunk(
                            index=index,
                            text=chunk_data['text'],
                            is_first=False,
                            is_final=False,
                            error=str(e),
                        )

            # Generate in parallel and yield as they complete
            tasks = [
                generate_chunk(i + 1, chunk_data)
                for i, chunk_data in enumerate(remaining_chunks)
            ]

            for task in asyncio.as_completed(tasks):
                chunk = await task
                if len(chunks) - 1 == chunk.index or chunk.index == len(chunks) - 1:
                    chunk.is_final = True
                yield chunk

    async def _generate_sequential(
        self,
        chunks: List[Dict[str, Any]],
        generator: Callable,
        voice_config: Dict[str, Any],
        emotion: Dict[str, Any],
        start_time: float,
    ) -> AsyncIterator[StreamingChunk]:
        """Generate chunks sequentially (simpler, more predictable)."""
        for i, chunk_data in enumerate(chunks):
            gen_start = time.time()

            try:
                audio = await generator(
                    chunk_data['text'],
                    voice_config,
                    emotion,
                )

                latency = (time.time() - start_time) * 1000 if i == 0 else (time.time() - gen_start) * 1000

                yield StreamingChunk(
                    index=i,
                    text=chunk_data['text'],
                    audio_data=audio,
                    is_first=(i == 0),
                    is_final=(i == len(chunks) - 1),
                    latency_ms=latency,
                    pause_after=chunk_data.get('pause_after', 0.0),
                    metadata=chunk_data,
                )
            except Exception as e:
                logger.error(f"Error generating chunk {i}: {e}")
                yield StreamingChunk(
                    index=i,
                    text=chunk_data['text'],
                    is_first=(i == 0),
                    is_final=(i == len(chunks) - 1),
                    error=str(e),
                )

    def _calculate_optimal_chunk_size(self, text: str) -> int:
        """Calculate optimal chunk size based on text characteristics."""
        # For dialogue, use smaller chunks
        if '"' in text or '"' in text or '说' in text:
            return min(self.first_chunk_size, 20)

        # For long sentences without breaks, use larger chunks
        if '。' not in text and '，' not in text:
            if len(text) > 200:
                return 100

        return self.subsequent_chunk_size

    async def generate_streaming_with_checkpoints(
        self,
        text: str,
        generator: Callable,
        voice_config: Dict[str, Any],
        emotion: Optional[Dict[str, Any]] = None,
        checkpoint_interval: int = 10,
    ) -> AsyncIterator[StreamingChunk]:
        """
        Generate streaming audio with checkpoint support for long texts.

        Args:
            text: Text to synthesize
            generator: Async TTS generator function
            voice_config: Voice configuration
            emotion: Emotion parameters
            checkpoint_interval: Save checkpoint every N chunks

        Yields:
            StreamingChunk with audio data and checkpoint info
        """
        chunks = await self._segmenter.segment_for_streaming(
            text,
            self.first_chunk_size,
            self.subsequent_chunk_size,
        )

        for i, chunk_data in enumerate(chunks):
            is_checkpoint = (i + 1) % checkpoint_interval == 0

            try:
                audio = await generator(
                    chunk_data['text'],
                    voice_config,
                    emotion,
                )

                yield StreamingChunk(
                    index=i,
                    text=chunk_data['text'],
                    audio_data=audio,
                    is_first=(i == 0),
                    is_final=(i == len(chunks) - 1),
                    checkpoint=is_checkpoint,
                    metadata=chunk_data,
                )
            except Exception as e:
                logger.error(f"Error generating chunk {i}: {e}")
                yield StreamingChunk(
                    index=i,
                    text=chunk_data['text'],
                    is_first=(i == 0),
                    is_final=(i == len(chunks) - 1),
                    error=str(e),
                )

    async def measure_latency(
        self,
        text: str,
        generator: Callable,
        voice_config: Dict[str, Any],
        trials: int = 5,
    ) -> Dict[str, float]:
        """
        Measure actual streaming latency.

        Args:
            text: Test text
            generator: TTS generator function
            voice_config: Voice configuration
            trials: Number of trials to run

        Returns:
            Dictionary with latency metrics
        """
        first_byte_latencies = []
        stream_latencies = []

        for _ in range(trials):
            start_time = time.time()

            # Time to first chunk
            first_chunk_time = None
            chunk_count = 0

            async for chunk in self.generate_streaming(text, generator, voice_config):
                if chunk.is_first and chunk.audio_data:
                    first_chunk_time = time.time() - start_time
                    first_byte_latencies.append(first_chunk_time * 1000)  # ms

                chunk_count += 1

                if chunk.is_final:
                    total_time = time.time() - start_time
                    stream_latencies.append(total_time * 1000)
                    break

        return {
            "first_byte_avg": sum(first_byte_latencies) / len(first_byte_latencies),
            "first_byte_min": min(first_byte_latencies),
            "first_byte_max": max(first_byte_latencies),
            "stream_avg": sum(stream_latencies) / len(stream_latencies),
            "target_first_byte": self.target_latency,
            "meets_target": sum(first_byte_latencies) / len(first_byte_latencies) <= self.target_latency,
        }


# Global instance
_streaming_optimizer: Optional[StreamingTTSOptimizer] = None


def get_streaming_optimizer(
    first_chunk_size: int = 30,
    subsequent_chunk_size: int = 80,
    target_latency_ms: float = 150.0,
) -> StreamingTTSOptimizer:
    """Get global streaming TTS optimizer instance."""
    global _streaming_optimizer
    if _streaming_optimizer is None:
        _streaming_optimizer = StreamingTTSOptimizer(
            first_chunk_size=first_chunk_size,
            subsequent_chunk_size=subsequent_chunk_size,
            target_first_byte_latency_ms=target_latency_ms,
        )
    return _streaming_optimizer

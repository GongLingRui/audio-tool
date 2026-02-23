"""
Optimized Batch Processor for TTS
Intelligent batch processing with parallel execution and smart batching
"""
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    """A single item in a batch."""
    id: str
    text: str
    voice_config: Dict[str, Any]
    emotion: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    duration: Optional[float] = None


@dataclass
class BatchResult:
    """Result of batch processing."""
    items: List[BatchItem]
    total_duration: float
    processing_time: float
    success_count: int
    failure_count: int
    items_per_second: float


class OptimizedBatchProcessor:
    """
    Optimized batch processor for TTS with intelligent batching and parallel execution.

    Features:
    - Intelligent batching by text length and complexity
    - Parallel processing with configurable workers
    - Result caching to avoid duplicate processing
    - Progress tracking and checkpointing
    - Automatic retry on failure
    """

    def __init__(
        self,
        max_workers: int = 4,
        batch_size: int = None,
        enable_caching: bool = True,
        max_retries: int = 2,
    ):
        """
        Initialize batch processor.

        Args:
            max_workers: Maximum parallel workers
            batch_size: Target batch size (None = auto)
            enable_caching: Enable result caching
            max_retries: Maximum retry attempts on failure
        """
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.enable_caching = enable_caching
        self.max_retries = max_retries
        self._cache = {} if enable_caching else None

    async def process_batch(
        self,
        items: List[Dict[str, Any]],
        generator: Callable,
        show_progress: bool = False,
    ) -> BatchResult:
        """
        Process a batch of TTS items.

        Args:
            items: List of items to process (each with text, voice_config, etc.)
            generator: Async function to generate audio
            show_progress: Show progress during processing

        Returns:
            BatchResult with all processed items
        """
        start_time = time.time()

        # Convert to BatchItem objects
        batch_items = []
        for item in items:
            batch_items.append(BatchItem(
                id=item.get('id', str(id(item))),
                text=item.get('text', ''),
                voice_config=item.get('voice_config', {}),
                emotion=item.get('emotion'),
                metadata=item.get('metadata', {}),
            ))

        # Check cache
        if self.enable_caching:
            batch_items = await self._check_cache(batch_items)

        # Process uncached items
        uncached_items = [item for item in batch_items if item.result is None]

        if uncached_items:
            # Intelligent batching
            batches = self._intelligent_batch(uncached_items)

            # Process batches in parallel
            processed = await self._process_batches_parallel(
                batches,
                generator,
                show_progress,
            )

            # Update batch items with results
            processed_map = {item.id: item for item in processed}
            for item in batch_items:
                if item.result is None and item.id in processed_map:
                    result_item = processed_map[item.id]
                    item.result = result_item.result
                    item.error = result_item.error
                    item.duration = result_item.duration

        # Cache results
        if self.enable_caching:
            await self._update_cache(batch_items)

        processing_time = time.time() - start_time

        # Calculate stats
        success_count = sum(1 for item in batch_items if item.result is not None)
        failure_count = len(batch_items) - success_count
        total_duration = sum(item.duration or 0 for item in batch_items)

        return BatchResult(
            items=batch_items,
            total_duration=total_duration,
            processing_time=processing_time,
            success_count=success_count,
            failure_count=failure_count,
            items_per_second=len(batch_items) / processing_time if processing_time > 0 else 0,
        )

    async def _check_cache(self, items: List[BatchItem]) -> List[BatchItem]:
        """Check cache for existing results."""
        if not self.enable_caching:
            return items

        for item in items:
            cache_key = self._generate_cache_key(item)
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                item.result = cached['result']
                item.duration = cached.get('duration')

        return items

    async def _update_cache(self, items: List[BatchItem]):
        """Update cache with new results."""
        if not self.enable_caching:
            return

        for item in items:
            if item.result is not None:
                cache_key = self._generate_cache_key(item)
                self._cache[cache_key] = {
                    'result': item.result,
                    'duration': item.duration,
                    'timestamp': time.time(),
                }

    def _generate_cache_key(self, item: BatchItem) -> str:
        """Generate cache key for item."""
        import hashlib
        import json

        key_data = {
            'text': item.text,
            'voice': item.voice_config,
            'emotion': item.emotion,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _intelligent_batch(self, items: List[BatchItem]) -> List[List[BatchItem]]:
        """Intelligently batch items by length and complexity."""
        if self.batch_size:
            target_size = self.batch_size
        else:
            # Auto-calculate optimal batch size
            target_size = self._calculate_optimal_batch_size(items)

        # Sort by text length for better load balancing
        sorted_items = sorted(items, key=lambda x: len(x.text))

        batches = []
        current_batch = []
        current_size = 0

        for item in sorted_items:
            item_size = len(item.text)

            # Check if adding this item would exceed target size
            if current_batch and current_size + item_size > target_size:
                batches.append(current_batch)
                current_batch = [item]
                current_size = item_size
            else:
                current_batch.append(item)
                current_size += item_size

        if current_batch:
            batches.append(current_batch)

        return batches

    def _calculate_optimal_batch_size(self, items: List[BatchItem]) -> int:
        """Calculate optimal batch size based on items."""
        if not items:
            return 10

        # Get text lengths
        lengths = [len(item.text) for item in items]

        avg_length = sum(lengths) / len(lengths)
        max_length = max(lengths)

        # Consider text complexity
        # Longer texts need more processing time, so smaller batches
        if avg_length > 500:
            return 5
        elif avg_length > 200:
            return 8
        elif avg_length > 100:
            return 12
        else:
            # Short texts can be batched larger
            return 20

    async def _process_batches_parallel(
        self,
        batches: List[List[BatchItem]],
        generator: Callable,
        show_progress: bool,
    ) -> List[BatchItem]:
        """Process batches in parallel with semaphore limiting."""
        semaphore = asyncio.Semaphore(self.max_workers)

        async def process_single_batch(batch: List[BatchItem]) -> List[BatchItem]:
            async with semaphore:
                return await self._process_batch(batch, generator)

        # Process all batches concurrently
        tasks = [
            process_single_batch(batch)
            for batch in batches
        ]

        results = await asyncio.gather(*tasks)

        # Flatten results
        all_items = []
        for batch_result in results:
            all_items.extend(batch_result)

        return all_items

    async def _process_batch(
        self,
        batch: List[BatchItem],
        generator: Callable,
    ) -> List[BatchItem]:
        """Process a single batch with retry logic."""
        results = []

        for item in batch:
            for attempt in range(self.max_retries + 1):
                try:
                    result = await generator(
                        item.text,
                        item.voice_config,
                        item.emotion,
                    )

                    item.result = result
                    # Estimate duration (simplified)
                    item.duration = len(item.text) / 10.0  # Rough estimate
                    item.error = None

                    break
                except Exception as e:
                    logger.warning(
                        f"Error processing item {item.id} (attempt {attempt + 1}): {e}"
                    )

                    if attempt == self.max_retries:
                        item.error = str(e)

                    # Retry after short delay
                    if attempt < self.max_retries:
                        await asyncio.sleep(0.5 * (attempt + 1))

            results.append(item)

        return results

    async def process_streaming_batch(
        self,
        items: List[Dict[str, Any]],
        generator: Callable,
        callback: Callable[[str, Any], None],
    ):
        """
        Process batch with streaming results via callback.

        Args:
            items: List of items to process
            generator: Async function to generate audio
            callback: Callback function (item_id, result)
        """
        for item_dict in items:
            item = BatchItem(
                id=item_dict.get('id', str(id(item_dict))),
                text=item_dict.get('text', ''),
                voice_config=item_dict.get('voice_config', {}),
                emotion=item_dict.get('emotion'),
                metadata=item_dict.get('metadata', {}),
            )

            try:
                result = await generator(
                    item.text,
                    item.voice_config,
                    item.emotion,
                )
                item.result = result
                item.duration = len(item.text) / 10.0
            except Exception as e:
                item.error = str(e)

            # Send callback
            await callback(item.id, item)

    def estimate_processing_time(
        self,
        items: List[Dict[str, Any]],
        avg_time_per_char: float = 0.05,
    ) -> float:
        """
        Estimate processing time for a batch.

        Args:
            items: List of items to process
            avg_time_per_char: Average processing time per character (seconds)

        Returns:
            Estimated processing time in seconds
        """
        total_chars = sum(len(item.get('text', '')) for item in items)
        total_time = total_chars * avg_time_per_char

        # Adjust for parallelism
        adjusted_time = total_time / self.max_workers

        return adjusted_time


# Global instance
_batch_processor: Optional[OptimizedBatchProcessor] = None


def get_batch_processor(
    max_workers: int = 4,
    batch_size: int = None,
    enable_caching: bool = True,
) -> OptimizedBatchProcessor:
    """Get global batch processor instance."""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = OptimizedBatchProcessor(
            max_workers=max_workers,
            batch_size=batch_size,
            enable_caching=enable_caching,
        )
    return _batch_processor

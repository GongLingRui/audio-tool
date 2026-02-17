"""Chunk processing service."""
import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Dict, List, Callable

from app.services.audio_processor import AudioProcessor
from app.services.audio_cache import get_audio_cache
from app.services.tts_engine import TTSEngine

logger = logging.getLogger(__name__)


class OptimizedBatchProcessor:
    """
    Optimized batch processor with intelligent batching, caching, and progress tracking.
    """

    def __init__(
        self,
        tts_engine: TTSEngine,
        audio_processor: Optional[AudioProcessor] = None,
        max_workers: int = 4,
        use_cache: bool = True,
    ):
        """
        Initialize optimized batch processor.

        Args:
            tts_engine: TTS engine instance
            audio_processor: Audio processor instance
            max_workers: Maximum parallel workers
            use_cache: Enable result caching
        """
        self.tts_engine = tts_engine
        self.audio_processor = audio_processor or AudioProcessor()
        self.max_workers = max_workers
        self.use_cache = use_cache
        self.cache = get_audio_cache() if use_cache else None

        # Progress tracking
        self._progress_callbacks: Dict[str, List[Callable]] = {}

    async def process_batch(
        self,
        items: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
        max_workers: Optional[int] = None,
        use_cache: Optional[bool] = None,
        progress_callback: Optional[Callable] = None,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process batch of items with intelligent optimization.

        Args:
            items: List of items to process
            batch_size: Items per batch (auto-calculated if None)
            max_workers: Maximum parallel workers
            use_cache: Enable caching for this batch
            progress_callback: Optional callback for progress updates
            job_id: Optional job ID for progress tracking

        Returns:
            Dict with results and statistics
        """
        job_id = job_id or f"job_{uuid.uuid4().hex[:8]}"
        max_workers = max_workers or self.max_workers
        use_cache = use_cache if use_cache is not None else self.use_cache

        # Register progress callback
        if progress_callback:
            if job_id not in self._progress_callbacks:
                self._progress_callbacks[job_id] = []
            self._progress_callbacks[job_id].append(progress_callback)

        start_time = datetime.now()

        try:
            # Filter and categorize items
            cached_items = []
            pending_items = []

            for item in items:
                item_id = item.get("id", uuid.uuid4().hex[:8])

                # Check cache if enabled
                if use_cache and self.cache:
                    cache_key = self._generate_cache_key(item)
                    cached_audio = self.cache.get(
                        text=item.get("text", ""),
                        voice_config=item.get("voice_id", "default"),
                        emotion=item.get("emotion"),
                    )
                    if cached_audio:
                        cached_items.append({
                            "item_id": item_id,
                            "cached": True,
                            "audio_data": cached_audio,
                        })
                        continue

                pending_items.append({**item, "item_id": item_id})

            logger.info(f"Job {job_id}: {len(cached_items)} cached, {len(pending_items)} pending")

            # Report cached items
            if cached_items:
                await self._report_progress(
                    job_id,
                    len(cached_items),
                    len(items),
                    {"cached": len(cached_items)},
                )

            # Calculate optimal batch size
            batch_size = batch_size or self._calculate_optimal_batch_size(pending_items)

            # Process pending items in parallel batches
            results = []

            if pending_items:
                # Split into batches
                batches = [
                    pending_items[i:i + batch_size]
                    for i in range(0, len(pending_items), batch_size)
                ]

                logger.info(f"Job {job_id}: Processing {len(pending_items)} items in {len(batches)} batches")

                # Process batches
                for batch_idx, batch in enumerate(batches):
                    batch_results = await self._process_batch_parallel(
                        batch,
                        max_workers=max_workers,
                        job_id=job_id,
                    )
                    results.extend(batch_results)

                    # Report progress
                    total_processed = len(cached_items) + len(results)
                    await self._report_progress(
                        job_id,
                        total_processed,
                        len(items),
                        {"batch": batch_idx + 1, "total_batches": len(batches)},
                    )

            # Add cached items to results
            for cached in cached_items:
                results.append({
                    "item_id": cached["item_id"],
                    "status": "completed",
                    "cached": True,
                    "audio_path": await self._save_cached_audio(cached["audio_data"], cached["item_id"]),
                })

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Calculate statistics
            successful = sum(1 for r in results if r.get("status") == "completed")
            failed = sum(1 for r in results if r.get("status") == "failed")
            cached_count = sum(1 for r in results if r.get("cached"))

            stats = {
                "job_id": job_id,
                "total_items": len(items),
                "successful": successful,
                "failed": failed,
                "cached": cached_count,
                "duration_seconds": duration,
                "items_per_second": len(items) / duration if duration > 0 else 0,
                "cache_hit_rate": cached_count / len(items) if items else 0,
            }

            logger.info(f"Job {job_id} completed: {stats}")

            return {
                "results": results,
                "stats": stats,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            return {
                "results": [],
                "stats": {"error": str(e)},
                "success": False,
            }
        finally:
            # Cleanup progress callbacks
            if job_id in self._progress_callbacks:
                del self._progress_callbacks[job_id]

    async def _process_batch_parallel(
        self,
        items: List[Dict[str, Any]],
        max_workers: int,
        job_id: str,
    ) -> List[Dict[str, Any]]:
        """Process a batch of items in parallel."""
        semaphore = asyncio.Semaphore(max_workers)

        async def bounded_process(item: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                try:
                    return await self._process_single_item(item, job_id)
                except Exception as e:
                    logger.error(f"Error processing item {item.get('item_id')}: {e}")
                    return {
                        "item_id": item.get("item_id"),
                        "status": "failed",
                        "error": str(e),
                    }

        tasks = [bounded_process(item) for item in items]
        return await asyncio.gather(*tasks)

    async def _process_single_item(
        self,
        item: Dict[str, Any],
        job_id: str,
    ) -> Dict[str, Any]:
        """Process a single item."""
        item_id = item.get("item_id")
        text = item.get("text", "")
        speaker = item.get("speaker", "NARRATOR")
        voice_id = item.get("voice_id", "default")
        emotion = item.get("emotion")
        instruct = item.get("instruct")

        # Build voice config
        voice_config = {
            "voice_id": voice_id,
            "voice_type": item.get("voice_type", "custom"),
        }
        if emotion:
            voice_config["emotion"] = emotion

        # Generate audio
        audio_data, duration = await self.tts_engine.generate(
            text=text,
            speaker=speaker,
            instruct=instruct,
            voice_config=voice_config,
        )

        # Save audio
        output_dir = Path("./static/audio/chunks")
        output_dir.mkdir(parents=True, exist_ok=True)

        wav_path = output_dir / f"{item_id}.wav"
        wav_path_str = str(wav_path)

        await self.audio_processor.save_wav(audio_data, wav_path_str)

        # Convert to MP3
        mp3_path = await self.audio_processor.convert_to_mp3(wav_path_str)

        # Cache result if enabled
        if self.cache:
            self.cache.set(
                text=text,
                voice_config=voice_id,
                audio_data=audio_data,
                emotion=emotion,
                duration=duration,
            )

        return {
            "item_id": item_id,
            "status": "completed",
            "cached": False,
            "audio_path": mp3_path,
            "duration": duration,
        }

    def _calculate_optimal_batch_size(self, items: List[Dict[str, Any]]) -> int:
        """Calculate optimal batch size based on item characteristics."""
        if not items:
            return 1

        # Calculate average text length
        avg_length = sum(len(i.get("text", "")) for i in items) / len(items)

        # Determine batch size based on text length
        if avg_length < 100:
            return max(10, self.max_workers * 2)
        elif avg_length < 300:
            return max(5, self.max_workers)
        elif avg_length < 500:
            return max(3, self.max_workers // 2)
        else:
            return max(1, self.max_workers // 4)

    def _generate_cache_key(self, item: Dict[str, Any]) -> str:
        """Generate cache key for item."""
        key_data = {
            "text": item.get("text", ""),
            "voice_id": item.get("voice_id", "default"),
            "emotion": item.get("emotion"),
            "speaker": item.get("speaker"),
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()

    async def _save_cached_audio(self, audio_data: bytes, item_id: str) -> str:
        """Save cached audio to file."""
        import io
        from pydub import AudioSegment

        output_dir = Path("./static/audio/chunks")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{item_id}.mp3"

        audio = AudioSegment.from_file(io.BytesIO(audio_data))
        audio.export(str(output_path), format="mp3", bitrate="192k")

        return str(output_path)

    async def _report_progress(
        self,
        job_id: str,
        current: int,
        total: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Report progress to registered callbacks."""
        if job_id not in self._progress_callbacks:
            return

        progress_data = {
            "job_id": job_id,
            "current": current,
            "total": total,
            "progress": current / total if total > 0 else 0,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        for callback in self._progress_callbacks.get(job_id, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(progress_data)
                else:
                    callback(progress_data)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")

    async def get_job_progress(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get current progress for a job."""
        # This would be implemented with persistent storage for production
        return None


class ChunkProcessor:
    """Service for processing audio chunks (legacy, optimized version available)."""

    def __init__(self, tts_engine: TTSEngine, audio_processor: AudioProcessor | None = None):
        self.tts_engine = tts_engine
        self.audio_processor = audio_processor or AudioProcessor()
        self.max_workers = 2

        # Use optimized processor
        self._optimized_processor: Optional[OptimizedBatchProcessor] = None

    def _get_optimized_processor(self) -> OptimizedBatchProcessor:
        """Get or create optimized batch processor."""
        if self._optimized_processor is None:
            self._optimized_processor = OptimizedBatchProcessor(
                tts_engine=self.tts_engine,
                audio_processor=self.audio_processor,
                max_workers=self.max_workers,
            )
        return self._optimized_processor

    async def process_chunk(
        self,
        chunk: dict[str, Any],
        voice_configs: list[dict],
    ) -> dict[str, Any]:
        """
        Process a single chunk.

        Args:
            chunk: Chunk data
            voice_configs: List of voice configurations

        Returns:
            Processing result
        """
        speaker = chunk.get("speaker", "NARRATOR")
        text = chunk.get("text", "")
        instruct = chunk.get("instruct")

        # Find voice config for speaker
        speaker_config = self._get_speaker_config(speaker, voice_configs)

        # Generate audio
        audio_data, duration = await self.tts_engine.generate(
            text=text,
            speaker=speaker,
            instruct=instruct,
            voice_config=speaker_config,
        )

        # Save audio
        chunk_id = chunk.get("id")
        wav_path = f"./static/audio/chunks/{chunk_id}.wav"
        await self.audio_processor.save_wav(audio_data, wav_path)

        # Convert to MP3
        mp3_path = await self.audio_processor.convert_to_mp3(wav_path)

        return {
            "chunk_id": chunk_id,
            "audio_path": mp3_path,
            "duration": duration,
            "status": "completed",
        }

    async def process_chunks_batch(
        self,
        chunks: list[dict[str, Any]],
        voice_configs: list[dict],
        max_workers: int | None = None,
        use_cache: bool = True,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Process multiple chunks with optimization.

        Args:
            chunks: List of chunk data
            voice_configs: List of voice configurations
            max_workers: Maximum parallel workers
            use_cache: Enable result caching
            job_id: Optional job ID for tracking

        Returns:
            Processing results with statistics
        """
        processor = self._get_optimized_processor()

        # Prepare items for optimized processor
        items = []
        for chunk in chunks:
            if chunk.get("status") == "completed":
                continue

            # Find voice config
            speaker = chunk.get("speaker", "NARRATOR")
            speaker_config = self._get_speaker_config(speaker, voice_configs)

            items.append({
                "id": chunk.get("id"),
                "text": chunk.get("text", ""),
                "speaker": speaker,
                "voice_id": speaker_config.get("voice_name", "ryan"),
                "voice_type": speaker_config.get("voice_type", "custom"),
                "emotion": chunk.get("emotion"),
                "instruct": chunk.get("instruct"),
            })

        if not items:
            return {
                "results": [],
                "stats": {"total_items": 0, "cached": 0},
                "success": True,
            }

        # Process with optimized processor
        return await processor.process_batch(
            items=items,
            max_workers=max_workers,
            use_cache=use_cache,
            job_id=job_id,
        )

    def _get_speaker_config(
        self,
        speaker: str,
        voice_configs: list[dict],
    ) -> dict[str, Any]:
        """Get voice configuration for a speaker."""
        for config in voice_configs:
            if config.get("speaker") == speaker:
                return {
                    "voice_type": config.get("voice_type", "custom"),
                    "voice_name": config.get("voice_name"),
                    "style": config.get("style"),
                    "ref_audio_path": config.get("ref_audio_path"),
                    "lora_model_path": config.get("lora_model_path"),
                    "description": config.get("description"),
                }

        # Default configuration
        return {"voice_type": "custom", "voice_name": "ryan"}


# Global instance
_optimized_processor: Optional[OptimizedBatchProcessor] = None


def get_optimized_processor(tts_engine: TTSEngine, max_workers: int = 4) -> OptimizedBatchProcessor:
    """Get or create global optimized batch processor."""
    global _optimized_processor
    if _optimized_processor is None:
        _optimized_processor = OptimizedBatchProcessor(
            tts_engine=tts_engine,
            max_workers=max_workers,
        )
    return _optimized_processor

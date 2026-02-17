"""
Test new features: Smart Cache, Streaming TTS, Multi-Speaker Dialogue
"""
import asyncio
import pytest
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.smart_cache import (
    SmartCacheManager,
    MemoryCache,
    CacheEntry,
    get_smart_cache,
)


class TestMemoryCache:
    """Test in-memory cache implementation."""

    @pytest.mark.asyncio
    async def test_cache_set_get(self):
        """Test basic set and get operations."""
        cache = MemoryCache(max_size=10, max_memory_mb=1)

        # Set a value
        await cache.set("test_key", b"test_value", ttl=60)

        # Get it back
        value = await cache.get("test_key")
        assert value == b"test_value"

    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        """Test cache entry expiration."""
        cache = MemoryCache(max_size=10, max_memory_mb=1)

        # Set with short TTL
        await cache.set("expiring_key", b"value", ttl=0.1)  # 100ms

        # Should be available immediately
        assert await cache.get("expiring_key") == b"value"

        # Wait for expiration
        await asyncio.sleep(0.15)

        # Should be expired now
        assert await cache.get("expiring_key") is None

    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = MemoryCache(max_size=3, max_memory_mb=1)

        # Fill cache
        await cache.set("key1", b"value1")
        await cache.set("key2", b"value2")
        await cache.set("key3", b"value3")

        # Access key1 to make it more recently used
        await cache.get("key1")

        # Add new key, should evict key2 (least recently used)
        await cache.set("key4", b"value4")

        # key1 should still exist
        assert await cache.get("key1") == b"value1"

        # key2 should have been evicted
        assert await cache.get("key2") is None

        # key3 and key4 should exist
        assert await cache.get("key3") == b"value3"
        assert await cache.get("key4") == b"value4"

    @pytest.mark.asyncio
    async def test_cache_tags(self):
        """Test tag-based cache operations."""
        cache = MemoryCache(max_size=10, max_memory_mb=1)

        # Set values with tags
        await cache.set("key1", b"value1", tags=["tts", "voice1"])
        await cache.set("key2", b"value2", tags=["tts", "voice2"])
        await cache.set("key3", b"value3", tags=["other"])

        # Clear by tag
        await cache.clear(tag="tts")

        # tts tagged items should be gone
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None

        # Other tag should remain
        assert await cache.get("key3") == b"value3"

    @pytest.mark.asyncio
    async def test_cache_stats(self):
        """Test cache statistics."""
        cache = MemoryCache(max_size=10, max_memory_mb=1)

        # Populate cache
        await cache.set("key1", b"value1")
        await cache.set("key2", b"value2")

        # Access one entry
        await cache.get("key1")
        await cache.get("key1")  # Access twice

        # Get stats
        stats = cache.get_stats()

        assert stats["entries"] == 2
        assert stats["max_entries"] == 10
        assert stats["total_accesses"] == 2  # key1 accessed twice
        assert stats["type"] == "memory"


class TestSmartCacheManager:
    """Test smart cache manager with multi-level caching."""

    @pytest.mark.asyncio
    async def test_multi_level_cache(self):
        """Test caching with memory level."""
        manager = SmartCacheManager(
            enable_memory=True,
            enable_redis=False,  # Disable Redis for testing
        )

        # Set value
        await manager.set(
            text="Hello world",
            voice_config="aiden",
            audio_data=b"audio_data_here",
        )

        # Get from cache
        cached = await manager.get(
            text="Hello world",
            voice_config="aiden",
        )

        assert cached == b"audio_data_here"

    @pytest.mark.asyncio
    async def test_cache_key_generation(self):
        """Test that different parameters generate different keys."""
        manager = SmartCacheManager(enable_memory=True, enable_redis=False)

        # Different texts should have different keys
        key1 = manager._generate_key("text1", "voice1")
        key2 = manager._generate_key("text2", "voice1")
        assert key1 != key2

        # Different voices should have different keys
        key3 = manager._generate_key("text1", "voice2")
        assert key1 != key3

        # Same parameters should generate same key
        key4 = manager._generate_key("text1", "voice1")
        assert key1 == key4

    @pytest.mark.asyncio
    async def test_cache_with_emotion(self):
        """Test caching with emotion parameters."""
        manager = SmartCacheManager(enable_memory=True, enable_redis=False)

        # Set with emotion
        await manager.set(
            text="Hello",
            voice_config="aiden",
            audio_data=b"happy_audio",
            emotion={"happy": 0.8},
        )

        # Get with same emotion
        cached = await manager.get(
            text="Hello",
            voice_config="aiden",
            emotion={"happy": 0.8},
        )
        assert cached == b"happy_audio"

        # Different emotion should miss
        cached_sad = await manager.get(
            text="Hello",
            voice_config="aiden",
            emotion={"sad": 0.8},
        )
        assert cached_sad is None

    @pytest.mark.asyncio
    async def test_cache_hit_rate_tracking(self):
        """Test cache hit rate statistics."""
        manager = SmartCacheManager(enable_memory=True, enable_redis=False)

        # Set a value
        await manager.set(
            text="test",
            voice_config="voice",
            audio_data=b"data",
        )

        # Hit
        await manager.get(text="test", voice_config="voice")

        # Miss
        await manager.get(text="other", voice_config="voice")

        # Check stats
        stats = manager.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["total_requests"] == 2
        assert stats["hit_rate"] == 50.0


class TestCacheEntry:
    """Test CacheEntry class."""

    def test_entry_expiration(self):
        """Test entry expiration check."""
        entry = CacheEntry("key", "value", ttl=1)

        # Should not be expired immediately
        assert not entry.is_expired()

        # Should be expired after TTL
        import time
        time.sleep(1.1)
        assert entry.is_expired()

    def test_entry_touch(self):
        """Test access time tracking."""
        entry = CacheEntry("key", "value", ttl=60)

        initial_access_count = entry.access_count

        # Touch
        import time
        time.sleep(0.01)  # Small delay to ensure time difference
        entry.touch()

        assert entry.access_count == initial_access_count + 1


class TestIntegrationFeatures:
    """Integration tests for new features."""

    @pytest.mark.asyncio
    async def test_cache_with_multiple_voices(self):
        """Test caching with different voice configurations."""
        manager = SmartCacheManager(enable_memory=True, enable_redis=False)

        text = "Hello world"

        # Cache for different voices
        await manager.set(
            text=text,
            voice_config="voice1",
            audio_data=b"audio1",
        )
        await manager.set(
            text=text,
            voice_config="voice2",
            audio_data=b"audio2",
        )

        # Verify different voices get different cached audio
        audio1 = await manager.get(text=text, voice_config="voice1")
        audio2 = await manager.get(text=text, voice_config="voice2")

        assert audio1 == b"audio1"
        assert audio2 == b"audio2"
        assert audio1 != audio2

    @pytest.mark.asyncio
    async def test_cache_with_different_speeds(self):
        """Test caching with different speech speeds."""
        manager = SmartCacheManager(enable_memory=True, enable_redis=False)

        text = "Test text"
        voice_config = "aiden"

        # Cache at different speeds
        await manager.set(
            text=text,
            voice_config=voice_config,
            audio_data=b"normal_speed",
            speed=1.0,
        )
        await manager.set(
            text=text,
            voice_config=voice_config,
            audio_data=b"fast_speed",
            speed=1.5,
        )

        # Verify different speeds produce different cache entries
        normal_audio = await manager.get(text=text, voice_config=voice_config, speed=1.0)
        fast_audio = await manager.get(text=text, voice_config=voice_config, speed=1.5)

        assert normal_audio == b"normal_speed"
        assert fast_audio == b"fast_speed"

    @pytest.mark.asyncio
    async def test_cache_deletion(self):
        """Test cache deletion functionality."""
        manager = SmartCacheManager(enable_memory=True, enable_redis=False)

        # Set cache entry
        await manager.set(
            text="delete me",
            voice_config="voice",
            audio_data=b"to_be_deleted",
        )

        # Verify it exists
        assert await manager.get(text="delete me", voice_config="voice") == b"to_be_deleted"

        # Delete it
        await manager.delete(text="delete me", voice_config="voice")

        # Verify it's gone
        assert await manager.get(text="delete me", voice_config="voice") is None

    @pytest.mark.asyncio
    async def test_cache_clear_by_tag(self):
        """Test clearing cache by tag."""
        manager = SmartCacheManager(enable_memory=True, enable_redis=False)

        # Set entries with different tags
        await manager.set(
            text="text1",
            voice_config="voice1",
            audio_data=b"audio1",
            tags=["tts", "voice1"],
        )
        await manager.set(
            text="text2",
            voice_config="voice2",
            audio_data=b"audio2",
            tags=["tts", "voice2"],
        )
        await manager.set(
            text="text3",
            voice_config="voice3",
            audio_data=b"audio3",
            tags=["other"],
        )

        # Clear all TTS tagged entries
        await manager.clear(tag="tts")

        # TTS entries should be gone
        assert await manager.get(text="text1", voice_config="voice1") is None
        assert await manager.get(text="text2", voice_config="voice2") is None

        # Other tag should remain
        assert await manager.get(text="text3", voice_config="voice3") == b"audio3"

    @pytest.mark.asyncio
    async def test_cache_clear_all(self):
        """Test clearing all cache entries."""
        manager = SmartCacheManager(enable_memory=True, enable_redis=False)

        # Set multiple entries
        await manager.set(
            text="text1",
            voice_config="voice1",
            audio_data=b"audio1",
        )
        await manager.set(
            text="text2",
            voice_config="voice2",
            audio_data=b"audio2",
        )

        # Clear all
        await manager.clear()

        # All should be gone
        assert await manager.get(text="text1", voice_config="voice1") is None
        assert await manager.get(text="text2", voice_config="voice2") is None


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

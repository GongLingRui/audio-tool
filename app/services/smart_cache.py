"""
Smart Cache Manager with Redis Support
Multi-level caching with LRU eviction and distributed support
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
import asyncio
from functools import wraps

logger = logging.getLogger(__name__)


class CacheEntry:
    """Represents a cache entry with metadata."""

    def __init__(
        self,
        key: str,
        value: Any,
        ttl: int = 3600,
        size_bytes: int = 0,
    ):
        self.key = key
        self.value = value
        self.ttl = ttl
        self.created_at = time.time()
        self.accessed_at = self.created_at
        self.access_count = 0
        self.size_bytes = size_bytes
        self.tags: List[str] = []

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() - self.created_at > self.ttl

    def touch(self):
        """Update access time and count."""
        self.accessed_at = time.time()
        self.access_count += 1

    def get_age(self) -> float:
        """Get entry age in seconds."""
        return time.time() - self.created_at

    def get_idle_time(self) -> float:
        """Get idle time since last access."""
        return time.time() - self.accessed_at


class MemoryCache:
    """In-memory LRU cache."""

    def __init__(self, max_size: int = 1000, max_memory_mb: int = 500):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()

    def _get_current_memory(self) -> int:
        """Get current memory usage."""
        return sum(entry.size_bytes for entry in self._cache.values())

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None

            if entry.is_expired():
                del self._cache[key]
                return None

            entry.touch()
            return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = 3600,
        tags: List[str] = None,
    ) -> bool:
        """Set value in cache."""
        async with self._lock:
            # Calculate size (estimate for bytes, exact for others)
            if isinstance(value, bytes):
                size_bytes = len(value)
            elif isinstance(value, str):
                size_bytes = len(value.encode())
            elif isinstance(value, (dict, list)):
                size_bytes = len(json.dumps(value).encode())
            else:
                size_bytes = 1024  # Default estimate

            entry = CacheEntry(key, value, ttl, size_bytes)
            if tags:
                entry.tags = tags

            # Check memory limit
            current_memory = self._get_current_memory()
            if current_memory + size_bytes > self.max_memory_bytes:
                await self._evict_for_memory(size_bytes)

            # Check size limit
            if len(self._cache) >= self.max_size:
                await self._evict_lru()

            self._cache[key] = entry
            return True

    async def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self, tag: Optional[str] = None):
        """Clear cache entries."""
        async with self._lock:
            if tag:
                keys_to_delete = [
                    k for k, v in self._cache.items()
                    if tag in v.tags
                ]
                for key in keys_to_delete:
                    del self._cache[key]
            else:
                self._cache.clear()

    async def _evict_lru(self, count: int = 1):
        """Evict least recently used entries."""
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].accessed_at
        )
        for i in range(min(count, len(sorted_entries))):
            key = sorted_entries[i][0]
            del self._cache[key]

    async def _evict_for_memory(self, required_bytes: int):
        """Evict entries to free memory."""
        freed = 0
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].accessed_at
        )
        for key, entry in sorted_entries:
            if freed >= required_bytes:
                break
            del self._cache[key]
            freed += entry.size_bytes

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_memory = self._get_current_memory()
        entries = list(self._cache.values())

        return {
            "type": "memory",
            "entries": len(self._cache),
            "max_entries": self.max_size,
            "memory_mb": round(total_memory / (1024 * 1024), 2),
            "max_memory_mb": round(self.max_memory_bytes / (1024 * 1024), 2),
            "hit_rate": 0.0,  # Would need to track hits/misses
            "total_accesses": sum(e.access_count for e in entries),
            "oldest_entry_sec": min((e.get_age() for e in entries), default=0),
        }


class RedisCache:
    """Redis-backed cache."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "tts_cache:",
        default_ttl: int = 3600,
    ):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl
        self._redis = None

    async def _get_redis(self):
        """Get Redis connection."""
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                self._redis = await aioredis.from_url(self.redis_url)
            except ImportError:
                logger.warning("redis not installed, Redis cache disabled")
                self._redis = False
        return self._redis if self._redis is not False else None

    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        redis = await self._get_redis()
        if redis is None:
            return None

        try:
            full_key = self.key_prefix + key
            value = await redis.get(full_key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: List[str] = None,
    ) -> bool:
        """Set value in Redis."""
        redis = await self._get_redis()
        if redis is None:
            return False

        try:
            full_key = self.key_prefix + key
            ttl = ttl or self.default_ttl

            # Serialize value
            if isinstance(value, bytes):
                serialized_value = value
            else:
                serialized_value = json.dumps(value).encode()

            # Store with tags
            await redis.setex(full_key, ttl, serialized_value)

            # Add to tag sets if tags provided
            if tags:
                for tag in tags:
                    tag_key = f"{self.key_prefix}tag:{tag}"
                    await redis.sadd(tag_key, key)
                    await redis.expire(tag_key, ttl)

            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete entry from Redis."""
        redis = await self._get_redis()
        if redis is None:
            return False

        try:
            full_key = self.key_prefix + key
            await redis.delete(full_key)
            return True
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False

    async def clear(self, tag: Optional[str] = None):
        """Clear cache entries."""
        redis = await self._get_redis()
        if redis is None:
            return

        try:
            if tag:
                tag_key = f"{self.key_prefix}tag:{tag}"
                keys = await redis.smembers(tag_key)
                if keys:
                    full_keys = [self.key_prefix + k.decode() for k in keys]
                    await redis.delete(*full_keys)
                await redis.delete(tag_key)
            else:
                # Clear all with prefix
                pattern = self.key_prefix + "*"
                keys = []
                async for key in redis.scan_iter(match=pattern):
                    keys.append(key)
                if keys:
                    await redis.delete(*keys)
        except Exception as e:
            logger.error(f"Redis clear error: {e}")

    async def close(self):
        """Close Redis connection."""
        if self._redis and hasattr(self._redis, "close"):
            await self._redis.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "type": "redis",
            "url": self.redis_url,
            "key_prefix": self.key_prefix,
            "default_ttl": self.default_ttl,
        }


class SmartCacheManager:
    """
    Multi-level smart cache manager.

    Features:
    - LRU eviction for memory cache
    - Redis support for distributed caching
    - Tag-based cache invalidation
    - Automatic fallback
    - Cache statistics
    """

    def __init__(
        self,
        enable_memory: bool = True,
        enable_redis: bool = False,
        redis_url: str = "redis://localhost:6379/0",
        max_memory_mb: int = 500,
        max_entries: int = 1000,
        default_ttl: int = 3600,
    ):
        """
        Initialize smart cache manager.

        Args:
            enable_memory: Enable in-memory cache
            enable_redis: Enable Redis cache
            redis_url: Redis connection URL
            max_memory_mb: Maximum memory cache size
            max_entries: Maximum entries in memory cache
            default_ttl: Default TTL for cache entries
        """
        self.memory_cache: Optional[MemoryCache] = None
        self.redis_cache: Optional[RedisCache] = None
        self.default_ttl = default_ttl

        if enable_memory:
            self.memory_cache = MemoryCache(
                max_size=max_entries,
                max_memory_mb=max_memory_mb,
            )

        if enable_redis:
            self.redis_cache = RedisCache(
                redis_url=redis_url,
                default_ttl=default_ttl,
            )

        self._hits = 0
        self._misses = 0

    def _generate_key(
        self,
        text: str,
        voice_config: str,
        emotion: Optional[Dict[str, float]] = None,
        speed: float = 1.0,
        **kwargs
    ) -> str:
        """Generate cache key from parameters."""
        key_data = {
            "text": text,
            "voice_config": voice_config,
            "emotion": emotion or {},
            "speed": speed,
            **kwargs,
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()

    async def get(
        self,
        text: str,
        voice_config: str,
        emotion: Optional[Dict[str, float]] = None,
        speed: float = 1.0,
        **kwargs
    ) -> Optional[bytes]:
        """Get cached audio."""
        key = self._generate_key(text, voice_config, emotion, speed, **kwargs)

        # Try memory cache first
        if self.memory_cache:
            value = await self.memory_cache.get(key)
            if value is not None:
                self._hits += 1
                return value

        # Try Redis cache
        if self.redis_cache:
            value = await self.redis_cache.get(key)
            if value is not None:
                self._hits += 1
                # Populate memory cache
                if self.memory_cache:
                    await self.memory_cache.set(key, value, ttl=self.default_ttl)
                return value

        self._misses += 1
        return None

    async def set(
        self,
        text: str,
        voice_config: str,
        audio_data: bytes,
        emotion: Optional[Dict[str, float]] = None,
        speed: float = 1.0,
        ttl: Optional[int] = None,
        tags: List[str] = None,
        **kwargs
    ) -> bool:
        """Cache audio data."""
        key = self._generate_key(text, voice_config, emotion, speed, **kwargs)
        ttl = ttl or self.default_ttl

        success = True

        # Store in memory cache
        if self.memory_cache:
            if not await self.memory_cache.set(key, audio_data, ttl=ttl, tags=tags):
                success = False

        # Store in Redis cache
        if self.redis_cache:
            if not await self.redis_cache.set(key, audio_data, ttl=ttl, tags=tags):
                success = False

        return success

    async def delete(
        self,
        text: str,
        voice_config: str,
        emotion: Optional[Dict[str, float]] = None,
        speed: float = 1.0,
        **kwargs
    ) -> bool:
        """Delete cache entry."""
        key = self._generate_key(text, voice_config, emotion, speed, **kwargs)

        success = True

        if self.memory_cache:
            if not await self.memory_cache.delete(key):
                success = False

        if self.redis_cache:
            if not await self.redis_cache.delete(key):
                success = False

        return success

    async def clear(self, tag: Optional[str] = None):
        """Clear cache entries."""
        if self.memory_cache:
            await self.memory_cache.clear(tag)

        if self.redis_cache:
            await self.redis_cache.clear(tag)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        stats = {
            "hit_rate": round(hit_rate * 100, 2),
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total_requests,
            "default_ttl": self.default_ttl,
        }

        if self.memory_cache:
            stats["memory"] = self.memory_cache.get_stats()

        if self.redis_cache:
            stats["redis"] = self.redis_cache.get_stats()

        return stats

    async def close(self):
        """Close cache connections."""
        if self.redis_cache:
            await self.redis_cache.close()


def cache_result(
    cache_manager: SmartCacheManager,
    ttl: int = 3600,
    tags: List[str] = None,
):
    """
    Decorator to cache function results.

    Usage:
        @cache_result(cache_manager, ttl=1800, tags=["tts"])
        async def generate_tts(text, voice, emotion):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Try to get from cache
            text = kwargs.get("text", args[0] if args else "")
            voice_config = kwargs.get("voice", args[1] if len(args) > 1 else "")
            emotion = kwargs.get("emotion")
            speed = kwargs.get("speed", 1.0)

            cached = await cache_manager.get(
                text=text,
                voice_config=voice_config,
                emotion=emotion,
                speed=speed,
            )
            if cached is not None:
                return cached

            # Call function
            result = await func(*args, **kwargs)

            # Cache result
            if isinstance(result, tuple) and len(result) == 2:
                audio_data, duration = result
                await cache_manager.set(
                    text=text,
                    voice_config=voice_config,
                    audio_data=audio_data,
                    emotion=emotion,
                    speed=speed,
                    ttl=ttl,
                    tags=tags,
                )
            elif isinstance(result, bytes):
                await cache_manager.set(
                    text=text,
                    voice_config=voice_config,
                    audio_data=result,
                    emotion=emotion,
                    speed=speed,
                    ttl=ttl,
                    tags=tags,
                )

            return result

        return wrapper
    return decorator


# Global instance
_smart_cache: Optional[SmartCacheManager] = None


def get_smart_cache() -> SmartCacheManager:
    """Get global smart cache manager instance."""
    global _smart_cache
    if _smart_cache is None:
        from app.config import settings

        # Try to get Redis URL from settings
        redis_url = getattr(settings, "REDIS_URL", "redis://localhost:6379/0")
        enable_redis = getattr(settings, "ENABLE_REDIS_CACHE", False)

        _smart_cache = SmartCacheManager(
            enable_memory=True,
            enable_redis=enable_redis,
            redis_url=redis_url,
            max_memory_mb=500,
            max_entries=1000,
            default_ttl=3600,
        )
    return _smart_cache

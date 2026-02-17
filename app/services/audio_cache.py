"""
Audio Cache Service
Caches generated audio to avoid redundant TTS calls
"""

import hashlib
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import json

logger = logging.getLogger(__name__)


class AudioCache:
    """Cache for TTS-generated audio."""

    def __init__(self, cache_dir: Path, max_age_hours: int = 24, max_size_mb: int = 500):
        """
        Initialize audio cache.

        Args:
            cache_dir: Directory to store cached audio
            max_age_hours: Maximum age of cache entries in hours
            max_size_mb: Maximum total cache size in MB
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age = timedelta(hours=max_age_hours)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.index_file = self.cache_dir / "cache_index.json"
        self.index: Dict[str, Dict[str, Any]] = {}
        self._load_index()

    def _load_index(self):
        """Load cache index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    self.index = json.load(f)
                logger.info(f"Loaded {len(self.index)} cache entries")
            except Exception as e:
                logger.error(f"Error loading cache index: {e}")
                self.index = {}

    def _save_index(self):
        """Save cache index to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache index: {e}")

    def _generate_cache_key(
        self,
        text: str,
        voice_config: str,
        emotion: Optional[Dict[str, float]] = None,
        speed: float = 1.0,
    ) -> str:
        """Generate cache key from parameters."""
        key_data = {
            "text": text,
            "voice_config": voice_config,
            "emotion": emotion or {},
            "speed": speed,
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get(self, text: str, voice_config: str, emotion: Optional[Dict[str, float]] = None, speed: float = 1.0) -> Optional[bytes]:
        """
        Get cached audio if available and not expired.

        Returns:
            Cached audio data or None
        """
        cache_key = self._generate_cache_key(text, voice_config, emotion, speed)

        if cache_key not in self.index:
            return None

        entry = self.index[cache_key]

        # Check if expired
        created_at = datetime.fromisoformat(entry["created_at"])
        if datetime.now() - created_at > self.max_age:
            # Expired, remove from cache
            self._remove_entry(cache_key)
            return None

        # Check if file exists
        audio_path = self.cache_dir / f"{cache_key}.wav"
        if not audio_path.exists():
            # File missing, remove from index
            del self.index[cache_key]
            return None

        logger.debug(f"Cache hit for key {cache_key}")
        try:
            with open(audio_path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading cached audio: {e}")
            self._remove_entry(cache_key)
            return None

    def set(
        self,
        text: str,
        voice_config: str,
        audio_data: bytes,
        emotion: Optional[Dict[str, float]] = None,
        speed: float = 1.0,
        duration: float = 0.0,
    ) -> str:
        """
        Cache audio data.

        Returns:
            Cache key
        """
        cache_key = self._generate_cache_key(text, voice_config, emotion, speed)

        # Save audio file
        audio_path = self.cache_dir / f"{cache_key}.wav"
        try:
            with open(audio_path, 'wb') as f:
                f.write(audio_data)
        except Exception as e:
            logger.error(f"Error saving cached audio: {e}")
            return cache_key

        # Update index
        self.index[cache_key] = {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "voice_config": voice_config,
            "emotion": emotion,
            "speed": speed,
            "duration": duration,
            "size_bytes": len(audio_data),
            "created_at": datetime.now().isoformat(),
            "access_count": 0,
        }

        self._save_index()
        logger.info(f"Cached audio for key {cache_key} ({len(audio_data)} bytes)")

        # Check cache size limit
        self._cleanup_if_needed()

        return cache_key

    def _remove_entry(self, cache_key: str):
        """Remove cache entry and file."""
        if cache_key in self.index:
            del self.index[cache_key]

        audio_path = self.cache_dir / f"{cache_key}.wav"
        if audio_path.exists():
            try:
                audio_path.unlink()
            except Exception as e:
                logger.error(f"Error deleting cache file: {e}")

        self._save_index()

    def _cleanup_if_needed(self):
        """Clean up old cache entries if size limit exceeded."""
        total_size = sum(entry.get("size_bytes", 0) for entry in self.index.values())

        if total_size <= self.max_size_bytes:
            return

        logger.info(f"Cache size limit exceeded ({total_size} > {self.max_size_bytes}), cleaning up...")

        # Sort by creation time (oldest first)
        entries_by_age = sorted(
            self.index.items(),
            key=lambda x: x[1]["created_at"]
        )

        # Remove oldest entries until under limit
        for cache_key, entry in entries_by_age:
            if total_size <= self.max_size_bytes * 0.8:  # Target 80% of max
                break

            entry_size = entry.get("size_bytes", 0)
            self._remove_entry(cache_key)
            total_size -= entry_size
            logger.info(f"Removed cache entry {cache_key} ({entry_size} bytes)")

    def clear(self, voice_config: Optional[str] = None):
        """
        Clear cache entries.

        Args:
            voice_config: If specified, only clear entries for this voice config
        """
        if voice_config:
            # Clear only specific voice config
            keys_to_remove = [
                key for key, entry in self.index.items()
                if entry.get("voice_config") == voice_config
            ]
            for key in keys_to_remove:
                self._remove_entry(key)
        else:
            # Clear all
            for key in list(self.index.keys()):
                self._remove_entry(key)

        logger.info(f"Cleared cache for voice_config={voice_config}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(entry.get("size_bytes", 0) for entry in self.index.values())
        total_access = sum(entry.get("access_count", 0) for entry in self.index.values())

        return {
            "total_entries": len(self.index),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "total_access_count": total_access,
            "max_size_mb": round(self.max_size_bytes / (1024 * 1024), 2),
            "max_age_hours": self.max_age.total_seconds() / 3600,
            "cache_dir": str(self.cache_dir),
        }


# Global cache instance
_audio_cache: Optional[AudioCache] = None


def get_audio_cache() -> AudioCache:
    """Get or create audio cache instance."""
    global _audio_cache
    if _audio_cache is None:
        from app.config import settings
        cache_dir = settings.upload_dir / "audio_cache"
        _audio_cache = AudioCache(cache_dir)
    return _audio_cache

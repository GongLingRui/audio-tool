"""Security utilities for authentication and password hashing."""
import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Optional, Dict
from pathlib import Path

import bcrypt
from fastapi import HTTPException, status
from jose import JWTError, jwt

from app.config import settings

logger = logging.getLogger(__name__)


def _truncate_password(password: str) -> bytes:
    """Truncate password to 72 bytes (bcrypt limit)."""
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > 72:
        password_bytes = password_bytes[:72]
    return password_bytes


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash.

    Args:
        plain_password: Plain text password
        hashed_password: Hashed password

    Returns:
        bool: True if password matches
    """
    # Truncate to 72 bytes (bcrypt limit)
    password_bytes = _truncate_password(plain_password)
    hashed_bytes = hashed_password.encode('utf-8') if isinstance(hashed_password, str) else hashed_password

    try:
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False


def get_password_hash(password: str) -> str:
    """Hash a password.

    Args:
        password: Plain text password

    Returns:
        str: Hashed password
    """
    # Truncate to 72 bytes (bcrypt limit)
    password_bytes = _truncate_password(password)

    # Generate salt and hash
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def create_access_token(
    subject: str | Any,
    expires_delta: timedelta | None = None,
) -> str:
    """Create JWT access token.

    Args:
        subject: Token subject (usually user ID)
        expires_delta: Token expiration time

    Returns:
        str: JWT token
    """
    if expires_delta:
        delta_seconds = int(expires_delta.total_seconds())
    else:
        delta_seconds = settings.jwt_access_token_expire_minutes * 60
    # JWT exp 需为 Unix 时间戳（整数），避免 python-jose 编码异常
    exp_ts = int(time.time()) + delta_seconds
    to_encode = {"exp": exp_ts, "sub": str(subject)}
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )
    # python-jose 可能返回 bytes，统一转为 str
    if isinstance(encoded_jwt, bytes):
        return encoded_jwt.decode("utf-8")
    return encoded_jwt


def decode_access_token(token: str) -> dict | None:
    """Decode JWT access token.

    Args:
        token: JWT token

    Returns:
        dict | None: Decoded payload or None if invalid
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except JWTError:
        return None


# =============================================================================
# API Rate Limiter - API限流器
# =============================================================================

class RateLimiter:
    """
    Token bucket rate limiter for API endpoints.
    Supports per-user and per-endpoint limits.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        requests_per_day: int = 10000,
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Requests allowed per minute
            requests_per_hour: Requests allowed per hour
            requests_per_day: Requests allowed per day
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests_per_day = requests_per_day
        self.storage_dir = Path("./static/rate_limits")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._locks: Dict[str, asyncio.Lock] = {}

    def _get_lock(self, key: str) -> asyncio.Lock:
        """Get or create lock for key."""
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    def _get_storage_path(self, key: str) -> Path:
        """Get storage path for rate limit data."""
        return self.storage_dir / f"{key}.json"

    def _load_limits(self, key: str) -> Dict[str, Any]:
        """Load rate limit data for key."""
        path = self._get_storage_path(key)
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading rate limit data: {e}")
        return {
            "minute": [],
            "hour": [],
            "day": [],
        }

    def _save_limits(self, key: str, data: Dict[str, Any]):
        """Save rate limit data for key."""
        path = self._get_storage_path(key)
        try:
            with open(path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Error saving rate limit data: {e}")

    async def check_rate_limit(
        self,
        user_id: str,
        endpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Check if request is within rate limits.

        Args:
            user_id: User ID
            endpoint: Optional endpoint path

        Returns:
            Dict with remaining requests and status

        Raises:
            HTTPException: If rate limit exceeded
        """
        key = f"{user_id}:{endpoint}" if endpoint else user_id
        lock = self._get_lock(key)

        async with lock:
            now = datetime.now()
            limits = self._load_limits(key)

            # Clean old requests outside windows
            minute_ago = now - timedelta(minutes=1)
            hour_ago = now - timedelta(hours=1)
            day_ago = now - timedelta(days=1)

            limits["minute"] = [t for t in limits["minute"] if datetime.fromisoformat(t) > minute_ago]
            limits["hour"] = [t for t in limits["hour"] if datetime.fromisoformat(t) > hour_ago]
            limits["day"] = [t for t in limits["day"] if datetime.fromisoformat(t) > day_ago]

            # Check limits
            if len(limits["minute"]) >= self.requests_per_minute:
                retry_after = 60 - int((now - datetime.fromisoformat(limits["minute"][0])).total_seconds())
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "Rate limit exceeded",
                        "limit": "per_minute",
                        "retry_after": retry_after,
                    },
                    headers={"Retry-After": str(retry_after)},
                )

            if len(limits["hour"]) >= self.requests_per_hour:
                retry_after = 3600 - int((now - datetime.fromisoformat(limits["hour"][0])).total_seconds())
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "Rate limit exceeded",
                        "limit": "per_hour",
                        "retry_after": retry_after,
                    },
                    headers={"Retry-After": str(retry_after)},
                )

            if len(limits["day"]) >= self.requests_per_day:
                retry_after = 86400 - int((now - datetime.fromisoformat(limits["day"][0])).total_seconds())
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "Rate limit exceeded",
                        "limit": "per_day",
                        "retry_after": retry_after,
                    },
                    headers={"Retry-After": str(retry_after)},
                )

            # Add current request
            timestamp = now.isoformat()
            limits["minute"].append(timestamp)
            limits["hour"].append(timestamp)
            limits["day"].append(timestamp)

            # Save updated limits
            self._save_limits(key, limits)

            # Return remaining quota
            return {
                "remaining_minute": self.requests_per_minute - len(limits["minute"]),
                "remaining_hour": self.requests_per_hour - len(limits["hour"]),
                "remaining_day": self.requests_per_day - len(limits["day"]),
                "limit_minute": self.requests_per_minute,
                "limit_hour": self.requests_per_hour,
                "limit_day": self.requests_per_day,
            }

    def reset_user_limits(self, user_id: str, endpoint: Optional[str] = None):
        """Reset rate limits for a user."""
        key = f"{user_id}:{endpoint}" if endpoint else user_id
        path = self._get_storage_path(key)
        if path.exists():
            path.unlink()
            logger.info(f"Reset rate limits for {key}")

    def get_user_stats(self, user_id: str, endpoint: Optional[str] = None) -> Dict[str, Any]:
        """Get rate limit statistics for a user."""
        key = f"{user_id}:{endpoint}" if endpoint else user_id
        limits = self._load_limits(key)

        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)

        recent_minute = sum(1 for t in limits["minute"] if datetime.fromisoformat(t) > minute_ago)
        recent_hour = sum(1 for t in limits["hour"] if datetime.fromisoformat(t) > hour_ago)
        recent_day = sum(1 for t in limits["day"] if datetime.fromisoformat(t) > day_ago)

        return {
            "user_id": user_id,
            "endpoint": endpoint,
            "requests_last_minute": recent_minute,
            "requests_last_hour": recent_hour,
            "requests_last_day": recent_day,
            "limits": {
                "per_minute": self.requests_per_minute,
                "per_hour": self.requests_per_hour,
                "per_day": self.requests_per_day,
            },
        }


# =============================================================================
# Quota Manager - 配额管理系统
# =============================================================================

class QuotaManager:
    """
    User quota management system.
    Manages daily/monthly quotas for different user tiers.
    """

    # User tiers and their quotas
    TIER_QUOTAS = {
        "free": {
            "daily_tts_requests": 100,
            "daily_tts_characters": 10000,
            "monthly_tts_requests": 2000,
            "voice_clone_quota": 1,
            "lora_training_quota": 0,
            "max_project_count": 5,
        },
        "pro": {
            "daily_tts_requests": 500,
            "daily_tts_characters": 50000,
            "monthly_tts_requests": 10000,
            "voice_clone_quota": 10,
            "lora_training_quota": 2,
            "max_project_count": 50,
        },
        "enterprise": {
            "daily_tts_requests": -1,  # Unlimited
            "daily_tts_characters": -1,
            "monthly_tts_requests": -1,
            "voice_clone_quota": -1,
            "lora_training_quota": -1,
            "max_project_count": -1,
        },
    }

    def __init__(self):
        """Initialize quota manager."""
        self.storage_dir = Path("./static/quotas")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_user_quota_file(self, user_id: str) -> Path:
        """Get quota file path for user."""
        return self.storage_dir / f"{user_id}.json"

    def _load_user_quota(self, user_id: str) -> Dict[str, Any]:
        """Load user quota data."""
        path = self._get_user_quota_file(user_id)
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading quota data: {e}")

        # Return default quota structure
        return {
            "user_id": user_id,
            "tier": "free",
            "daily": {
                "date": datetime.now().date().isoformat(),
                "tts_requests": 0,
                "tts_characters": 0,
            },
            "monthly": {
                "year_month": datetime.now().strftime("%Y-%m"),
                "tts_requests": 0,
            },
            "voice_clones_used": 0,
            "lora_training_used": 0,
        }

    def _save_user_quota(self, user_id: str, quota_data: Dict[str, Any]):
        """Save user quota data."""
        path = self._get_user_quota_file(user_id)
        try:
            with open(path, 'w') as f:
                json.dump(quota_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving quota data: {e}")

    def _reset_if_needed(self, quota_data: Dict[str, Any]) -> Dict[str, Any]:
        """Reset daily/monthly counters if period changed."""
        today = datetime.now().date().isoformat()
        this_month = datetime.now().strftime("%Y-%m")

        # Reset daily if new day
        if quota_data["daily"]["date"] != today:
            quota_data["daily"] = {
                "date": today,
                "tts_requests": 0,
                "tts_characters": 0,
            }

        # Reset monthly if new month
        if quota_data["monthly"]["year_month"] != this_month:
            quota_data["monthly"] = {
                "year_month": this_month,
                "tts_requests": 0,
            }

        return quota_data

    async def check_and_consume_quota(
        self,
        user_id: str,
        resource: str,
        amount: int = 1,
        characters: int = 0,
    ) -> Dict[str, Any]:
        """
        Check and consume quota for a resource.

        Args:
            user_id: User ID
            resource: Resource type (tts_request, voice_clone, lora_training)
            amount: Amount to consume
            characters: Character count for TTS requests

        Returns:
            Dict with quota status

        Raises:
            HTTPException: If quota exceeded
        """
        quota_data = self._load_user_quota(user_id)
        quota_data = self._reset_if_needed(quota_data)

        tier = quota_data.get("tier", "free")
        tier_quotas = self.TIER_QUOTAS.get(tier, self.TIER_QUOTAS["free"])

        # Check quota based on resource type
        if resource == "tts_request":
            daily_quota = tier_quotas["daily_tts_requests"]
            monthly_quota = tier_quotas["monthly_tts_requests"]

            # Check daily quota
            if daily_quota >= 0 and quota_data["daily"]["tts_requests"] + amount > daily_quota:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "Daily TTS request quota exceeded",
                        "tier": tier,
                        "quota": daily_quota,
                        "used": quota_data["daily"]["tts_requests"],
                    },
                )

            # Check monthly quota
            if monthly_quota >= 0 and quota_data["monthly"]["tts_requests"] + amount > monthly_quota:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "Monthly TTS request quota exceeded",
                        "tier": tier,
                        "quota": monthly_quota,
                        "used": quota_data["monthly"]["tts_requests"],
                    },
                )

            # Check character quota
            if characters > 0:
                char_quota = tier_quotas["daily_tts_characters"]
                if char_quota >= 0 and quota_data["daily"]["tts_characters"] + characters > char_quota:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail={
                            "error": "Daily character quota exceeded",
                            "tier": tier,
                            "quota": char_quota,
                            "used": quota_data["daily"]["tts_characters"],
                        },
                    )

            # Consume quota
            quota_data["daily"]["tts_requests"] += amount
            quota_data["monthly"]["tts_requests"] += amount
            quota_data["daily"]["tts_characters"] += characters

        elif resource == "voice_clone":
            clone_quota = tier_quotas["voice_clone_quota"]
            if clone_quota >= 0 and quota_data["voice_clones_used"] + amount > clone_quota:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "Voice clone quota exceeded",
                        "tier": tier,
                        "quota": clone_quota,
                        "used": quota_data["voice_clones_used"],
                    },
                )

            quota_data["voice_clones_used"] += amount

        elif resource == "lora_training":
            lora_quota = tier_quotas["lora_training_quota"]
            if lora_quota >= 0 and quota_data["lora_training_used"] + amount > lora_quota:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "LoRA training quota exceeded",
                        "tier": tier,
                        "quota": lora_quota,
                        "used": quota_data["lora_training_used"],
                    },
                )

            quota_data["lora_training_used"] += amount

        # Save updated quota
        self._save_user_quota(user_id, quota_data)

        # Return quota status
        return {
            "user_id": user_id,
            "tier": tier,
            "resource": resource,
            "consumed": amount,
            "remaining": self._get_remaining_quota(quota_data, tier_quotas, resource),
        }

    def _get_remaining_quota(
        self,
        quota_data: Dict[str, Any],
        tier_quotas: Dict[str, int],
        resource: str,
    ) -> Dict[str, int]:
        """Get remaining quota for resource."""
        if resource == "tts_request":
            daily = tier_quotas["daily_tts_requests"]
            monthly = tier_quotas["monthly_tts_requests"]
            return {
                "daily": max(0, daily - quota_data["daily"]["tts_requests"]) if daily >= 0 else -1,
                "monthly": max(0, monthly - quota_data["monthly"]["tts_requests"]) if monthly >= 0 else -1,
            }
        elif resource == "voice_clone":
            quota = tier_quotas["voice_clone_quota"]
            return {"total": max(0, quota - quota_data["voice_clones_used"]) if quota >= 0 else -1}
        elif resource == "lora_training":
            quota = tier_quotas["lora_training_quota"]
            return {"total": max(0, quota - quota_data["lora_training_used"]) if quota >= 0 else -1}
        return {}

    def get_user_quota_status(self, user_id: str) -> Dict[str, Any]:
        """Get complete quota status for user."""
        quota_data = self._load_user_quota(user_id)
        quota_data = self._reset_if_needed(quota_data)

        tier = quota_data.get("tier", "free")
        tier_quotas = self.TIER_QUOTAS.get(tier, self.TIER_QUOTAS["free"])

        return {
            "user_id": user_id,
            "tier": tier,
            "daily": {
                "tts_requests": {
                    "used": quota_data["daily"]["tts_requests"],
                    "quota": tier_quotas["daily_tts_requests"],
                    "remaining": max(0, tier_quotas["daily_tts_requests"] - quota_data["daily"]["tts_requests"])
                                  if tier_quotas["daily_tts_requests"] >= 0 else -1,
                },
                "tts_characters": {
                    "used": quota_data["daily"]["tts_characters"],
                    "quota": tier_quotas["daily_tts_characters"],
                    "remaining": max(0, tier_quotas["daily_tts_characters"] - quota_data["daily"]["tts_characters"])
                                  if tier_quotas["daily_tts_characters"] >= 0 else -1,
                },
            },
            "monthly": {
                "tts_requests": {
                    "used": quota_data["monthly"]["tts_requests"],
                    "quota": tier_quotas["monthly_tts_requests"],
                    "remaining": max(0, tier_quotas["monthly_tts_requests"] - quota_data["monthly"]["tts_requests"])
                                  if tier_quotas["monthly_tts_requests"] >= 0 else -1,
                },
            },
            "voice_clones": {
                "used": quota_data["voice_clones_used"],
                "quota": tier_quotas["voice_clone_quota"],
                "remaining": max(0, tier_quotas["voice_clone_quota"] - quota_data["voice_clones_used"])
                              if tier_quotas["voice_clone_quota"] >= 0 else -1,
            },
            "lora_training": {
                "used": quota_data["lora_training_used"],
                "quota": tier_quotas["lora_training_quota"],
                "remaining": max(0, tier_quotas["lora_training_quota"] - quota_data["lora_training_used"])
                              if tier_quotas["lora_training_quota"] >= 0 else -1,
            },
        }

    def set_user_tier(self, user_id: str, tier: str):
        """Set user tier."""
        if tier not in self.TIER_QUOTAS:
            raise ValueError(f"Invalid tier: {tier}")

        quota_data = self._load_user_quota(user_id)
        quota_data["tier"] = tier
        self._save_user_quota(user_id, quota_data)
        logger.info(f"Set user {user_id} tier to {tier}")


# Global instances
_rate_limiter: Optional[RateLimiter] = None
_quota_manager: Optional[QuotaManager] = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def get_quota_manager() -> QuotaManager:
    """Get global quota manager instance."""
    global _quota_manager
    if _quota_manager is None:
        _quota_manager = QuotaManager()
    return _quota_manager

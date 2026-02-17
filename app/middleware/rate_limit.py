"""
Rate Limiter Middleware
API rate limiting and quota management
"""
import time
import asyncio
from typing import Dict, Optional, Callable
from collections import defaultdict, deque
from fastapi import Request, HTTPException, Response
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter.

    Features:
    - User-based rate limiting
    - Endpoint-based rate limiting
    - Sliding window counter
    - Burst allowance
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_size: int = 10,
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Requests allowed per minute
            requests_per_hour: Requests allowed per hour
            burst_size: Burst size allowance
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size

        # Token buckets: user_id -> tokens
        self.minute_tokens: Dict[str, float] = defaultdict(lambda: requests_per_minute)
        self.hour_tokens: Dict[str, float] = defaultdict(lambda: requests_per_hour)

        # Last update time: user_id -> timestamp
        self.last_minute_update: Dict[str, float] = {}
        self.last_hour_update: Dict[str, float] = {}

        # Request history for sliding window
        self.minute_history: Dict[str, deque] = defaultdict(deque)
        self.hour_history: Dict[str, deque] = {}

    async def check_rate_limit(
        self,
        user_id: str,
        endpoint: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if request is allowed.

        Returns:
            Tuple of (allowed, error_message)
        """
        now = time.time()

        # Check minute limit with sliding window
        minute_key = f"{user_id}:{endpoint or 'all'}:minute"
        self._clean_old_requests(self.minute_history[minute_key], now - 60)

        if len(self.minute_history[minute_key]) >= self.requests_per_minute:
            return False, f"Rate limit exceeded: {self.requests_per_minute} requests per minute"

        # Check hour limit with sliding window
        hour_key = f"{user_id}:{endpoint or 'all'}:hour"
        self._clean_old_requests(self.hour_history[hour_key], now - 3600)

        if len(self.hour_history[hour_key]) >= self.requests_per_hour:
            return False, f"Rate limit exceeded: {self.requests_per_hour} requests per hour"

        # Record request
        self.minute_history[minute_key].append(now)
        self.hour_history[hour_key].append(now)

        return True, None

    def _clean_old_requests(self, history: deque, cutoff_time: float):
        """Remove requests older than cutoff time."""
        while history and history[0] < cutoff_time:
            history.popleft()

    def get_remaining_quota(
        self,
        user_id: str,
        endpoint: Optional[str] = None,
    ) -> Dict[str, int]:
        """Get remaining quota for user."""
        now = time.time()

        minute_key = f"{user_id}:{endpoint or 'all'}:minute"
        hour_key = f"{user_id}:{endpoint or 'all'}:hour"

        self._clean_old_requests(self.minute_history[minute_key], now - 60)
        self._clean_old_requests(self.hour_history[hour_key], now - 3600)

        return {
            "minute_remaining": self.requests_per_minute - len(self.minute_history[minute_key]),
            "hour_remaining": self.requests_per_hour - len(self.hour_history[hour_key]),
            "minute_limit": self.requests_per_minute,
            "hour_limit": self.requests_per_hour,
        }

    def reset_user(self, user_id: str):
        """Reset rate limit for user (admin use)."""
        # Clear all history for this user
        keys_to_delete = [
            k for k in self.minute_history.keys()
            if k.startswith(f"{user_id}:")
        ]
        for key in keys_to_delete:
            del self.minute_history[key]

        keys_to_delete = [
            k for k in self.hour_history.keys()
            if k.startswith(f"{user_id}:")
        ]
        for key in keys_to_delete:
            del self.hour_history[key]


class QuotaManager:
    """
    Quota management for different user tiers.

    Tiers:
    - Free: 100 requests/day, 1000 requests/month
    - Pro: 1000 requests/day, 20000 requests/month
    - Enterprise: Unlimited
    """

    QUOTA_TIERS = {
        "free": {
            "daily": 100,
            "monthly": 1000,
        },
        "pro": {
            "daily": 1000,
            "monthly": 20000,
        },
        "enterprise": {
            "daily": float("inf"),
            "monthly": float("inf"),
        },
    }

    def __init__(self):
        # Usage tracking: user_id -> {date: count, month: count}
        self.daily_usage: Dict[str, Dict[str, int]] = defaultdict(lambda: {"count": 0, "date": ""})
        self.monthly_usage: Dict[str, Dict[str, int]] = defaultdict(lambda: {"count": 0, "month": ""})

    def get_user_tier(self, user_id: str) -> str:
        """Get user quota tier (to be implemented with actual user tier logic)."""
        # TODO: Implement actual tier lookup from database
        return "free"

    async def check_quota(
        self,
        user_id: str,
        cost: int = 1,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if user has quota for request.

        Returns:
            Tuple of (allowed, error_message)
        """
        tier = self.get_user_tier(user_id)
        quotas = self.QUOTA_TIERS[tier]

        from datetime import datetime
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_month = now.strftime("%Y-%m")

        # Reset counters if new day/month
        if self.daily_usage[user_id]["date"] != current_date:
            self.daily_usage[user_id] = {"count": 0, "date": current_date}

        if self.monthly_usage[user_id]["month"] != current_month:
            self.monthly_usage[user_id] = {"count": 0, "month": current_month}

        # Check daily quota
        if self.daily_usage[user_id]["count"] + cost > quotas["daily"]:
            return False, f"Daily quota exceeded: {quotas['daily']} requests per day"

        # Check monthly quota
        if self.monthly_usage[user_id]["count"] + cost > quotas["monthly"]:
            return False, f"Monthly quota exceeded: {quotas['monthly']} requests per month"

        # Deduct quota
        self.daily_usage[user_id]["count"] += cost
        self.monthly_usage[user_id]["count"] += cost

        return True, None

    def get_remaining_quota(self, user_id: str) -> Dict[str, any]:
        """Get remaining quota for user."""
        tier = self.get_user_tier(user_id)
        quotas = self.QUOTA_TIERS[tier]

        daily_used = self.daily_usage[user_id]["count"]
        monthly_used = self.monthly_usage[user_id]["count"]

        return {
            "tier": tier,
            "daily_used": daily_used,
            "daily_limit": quotas["daily"],
            "daily_remaining": quotas["daily"] - daily_used,
            "monthly_used": monthly_used,
            "monthly_limit": quotas["monthly"],
            "monthly_remaining": quotas["monthly"] - monthly_used,
        }

    def reset_user_quota(self, user_id: str):
        """Reset quota for user (admin use)."""
        if user_id in self.daily_usage:
            del self.daily_usage[user_id]
        if user_id in self.monthly_usage:
            del self.monthly_usage[user_id]


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


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware for FastAPI.

    Applies rate limiting to all API requests.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for auth endpoints
        if request.url.path.startswith("/api/auth"):
            return await call_next(request)

        # Get user ID from request (if authenticated)
        user_id = self._get_user_id(request)

        if user_id:
            # Check rate limit
            rate_limiter = get_rate_limiter()
            allowed, error = await rate_limiter.check_rate_limit(
                user_id=user_id,
                endpoint=request.url.path,
            )

            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Rate limit exceeded",
                        "message": error,
                        "retry_after": 60,
                    }
                )

            # Check quota
            quota_manager = get_quota_manager()
            allowed, error = await quota_manager.check_quota(user_id)

            if not allowed:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": "Quota exceeded",
                        "message": error,
                    }
                )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        if user_id:
            rate_limiter = get_rate_limiter()
            quota = rate_limiter.get_remaining_quota(user_id, request.url.path)
            response.headers["X-RateLimit-Limit"] = str(quota["minute_limit"])
            response.headers["X-RateLimit-Remaining"] = str(quota["minute_remaining"])
            response.headers["X-RateLimit-Reset"] = str(int(60))

        return response

    def _get_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request."""
        # Try to get user from JWT token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # In production, decode JWT and extract user_id
            # For now, return a placeholder
            return "anonymous_user"

        # Try to get from API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key_{api_key[:8]}"

        # Fallback to IP-based limiting
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip_{forwarded.split(',')[0].strip()}"

        return "anonymous"

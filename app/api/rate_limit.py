"""
Rate Limit and Quota Management API
"""
from fastapi import APIRouter, Depends, HTTPException
from app.schemas.common import ApiResponse
from app.middleware.rate_limit import get_rate_limiter, get_quota_manager
from typing import Dict, Any

router = APIRouter()


@router.get("/rate-limit/status", response_model=ApiResponse[Dict[str, Any]])
async def get_rate_limit_status(user_id: str = "anonymous"):
    """
    Get current rate limit status for user.

    Returns remaining quota and reset times.
    """
    rate_limiter = get_rate_limiter()
    quota_manager = get_quota_manager()

    rate_limit_info = rate_limiter.get_remaining_quota(user_id)
    quota_info = quota_manager.get_remaining_quota(user_id)

    return ApiResponse(data={
        "user_id": user_id,
        "rate_limit": rate_limit_info,
        "quota": quota_info,
    })


@router.get("/rate-limit/stats", response_model=ApiResponse[Dict[str, Any]])
async def get_rate_limit_stats():
    """
    Get global rate limit statistics (admin only).

    Returns overall system usage statistics.
    """
    rate_limiter = get_rate_limiter()
    quota_manager = get_quota_manager()

    # Calculate total tracked users
    total_users = len(set(
        list(rate_limiter.minute_history.keys()) +
        list(rate_limiter.hour_history.keys())
    ))

    return ApiResponse(data={
        "total_tracked_users": total_users,
        "rate_limit_config": {
            "requests_per_minute": rate_limiter.requests_per_minute,
            "requests_per_hour": rate_limiter.requests_per_hour,
            "burst_size": rate_limiter.burst_size,
        },
        "quota_tiers": quota_manager.QUOTA_TIERS,
    })


@router.post("/rate-limit/reset/{user_id}", response_model=ApiResponse[Dict[str, str]])
async def reset_user_rate_limit(user_id: str):
    """
    Reset rate limit for specific user (admin only).

    Clears all rate limit history for the user.
    """
    rate_limiter = get_rate_limiter()
    quota_manager = get_quota_manager()

    rate_limiter.reset_user(user_id)
    quota_manager.reset_user_quota(user_id)

    return ApiResponse(data={
        "message": f"Rate limit reset for user {user_id}",
        "user_id": user_id,
    })


@router.get("/quota/tiers", response_model=ApiResponse[Dict[str, Any]])
async def get_quota_tiers():
    """
    Get available quota tiers.

    Returns all available subscription tiers and their quotas.
    """
    quota_manager = get_quota_manager()

    return ApiResponse(data={
        "tiers": quota_manager.QUOTA_TIERS,
        "current_tier_counts": {
            tier: len([k for k in quota_manager.daily_usage.keys()
                       if quota_manager.get_user_tier(k.split(":")[0] if ":" in k else k) == tier])
            for tier in quota_manager.QUOTA_TIERS.keys()
        }
    })

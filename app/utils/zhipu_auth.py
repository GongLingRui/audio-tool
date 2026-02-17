"""Zhipu AI authentication utilities."""
import time
import jwt
from typing import Tuple


def parse_api_key(api_key: str) -> Tuple[str, str]:
    """Parse Zhipu AI API key into id and secret.

    Args:
        api_key: API key in format "id.secret"

    Returns:
        Tuple of (id, secret)

    Raises:
        ValueError: If API key format is invalid
    """
    parts = api_key.split(".")
    if len(parts) != 2:
        raise ValueError(f"Invalid Zhipu AI API key format. Expected 'id.secret', got: {api_key}")
    return parts[0], parts[1]


def generate_zhipu_token(api_key: str, exp_seconds: int = 3600) -> str:
    """Generate JWT token for Zhipu AI API authentication.

    According to Zhipu AI documentation, the JWT should contain:
    - api_key: the API key ID
    - exp: expiration timestamp
    - timestamp: current timestamp

    Args:
        api_key: Zhipu AI API key in format "id.secret"
        exp_seconds: Token expiration time in seconds (default: 3600)

    Returns:
        JWT token string

    Raises:
        ValueError: If API key format is invalid
    """
    api_id, api_secret = parse_api_key(api_key)

    # Current timestamp in seconds
    now = int(time.time())

    # JWT payload according to Zhipu AI spec
    payload = {
        "api_key": api_id,
        "exp": now + exp_seconds,
        "timestamp": now,
    }

    # Generate JWT token
    # Note: Zhipu AI requires HS256 algorithm
    token = jwt.encode(
        payload,
        api_secret,
        algorithm="HS256"
    )

    return token


def is_zhipu_config(base_url: str, api_key: str) -> bool:
    """Check if the configuration is for Zhipu AI.

    Args:
        base_url: API base URL
        api_key: API key

    Returns:
        True if configuration appears to be for Zhipu AI
    """
    return (
        "bigmodel.cn" in base_url or
        "zhipu" in base_url.lower() or
        ("." in api_key and len(api_key.split(".")) == 2)
    )

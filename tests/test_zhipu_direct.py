"""
Direct test of Zhipu AI API
"""
import asyncio
import httpx
import jwt
import time


async def test_zhipu_direct():
    """Test Zhipu AI API directly."""

    api_key = "your-zhipu-api-key-here"  # 替换为真实 API Key
    base_url = "https://open.bigmodel.cn/api/paas/v4"

    # Parse API key
    api_id, api_secret = api_key.split(".")

    # Generate JWT token
    now = int(time.time())
    payload = {
        "api_key": api_id,
        "exp": now + 3600,
        "timestamp": now,
    }
    token = jwt.encode(payload, api_secret, algorithm="HS256")

    print(f"Token: {token[:100]}...")
    print(f"Token length: {len(token)}")

    # Test API call
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'Hello, Zhipu AI!' in JSON format: {\"message\": \"...\"}"},
    ]

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "glm-4-flash",
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 100,
                },
            )

            print(f"\nStatus Code: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            print(f"\nResponse Body:\n{response.text[:1000]}")

            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                print(f"\nGenerated Content:\n{content}")
                return True
            else:
                print(f"\nRequest failed!")
                return False

    except Exception as e:
        print(f"\nException: {str(e)}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_zhipu_direct())
    print(f"\n{'='*60}")
    print(f"Test {'PASSED' if success else 'FAILED'}")
    print(f"{'='*60}")

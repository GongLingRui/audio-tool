"""
Test Zhipu AI API with different authentication methods
"""
import asyncio
import httpx


async def test_with_api_key():
    """Test using API key directly."""

    api_key = "your-zhipu-api-key-here"  # 替换为真实 API Key
    base_url = "https://open.bigmodel.cn/api/paas/v4"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'Hello, Zhipu AI!'"},
    ]

    # Method 1: Try with API key directly
    print("Testing Method 1: API Key as Bearer token")
    print(f"API Key: {api_key[:20]}...")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "glm-4-flash",
                    "messages": messages,
                },
            )

            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text[:500]}")

            if response.status_code == 200:
                print("\n✓ Method 1 SUCCESS!")
                return True

    except Exception as e:
        print(f"Method 1 Exception: {str(e)}")

    # Method 2: Try with API key in Authorization header without Bearer
    print("\n" + "="*60)
    print("Testing Method 2: API Key without Bearer prefix")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "model": "glm-4-flash",
                    "messages": messages,
                },
            )

            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text[:500]}")

            if response.status_code == 200:
                print("\n✓ Method 2 SUCCESS!")
                return True

    except Exception as e:
        print(f"Method 2 Exception: {str(e)}")

    return False


if __name__ == "__main__":
    success = asyncio.run(test_with_api_key())
    print(f"\n{'='*60}")
    print(f"Overall: {'PASSED' if success else 'FAILED'}")
    print(f"{'='*60}")

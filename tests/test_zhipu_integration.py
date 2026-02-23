"""
Test Zhipu AI GLM-4-Flash integration
"""
import asyncio
import os
import sys
from pathlib import Path
import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.script_generator import ScriptGenerator
from app.config import settings


async def test_zhipu_generation():
    """Test script generation with Zhipu AI GLM-4-Flash."""
    if os.getenv("RUN_ZHIPU_TESTS") != "1":
        pytest.skip("RUN_ZHIPU_TESTS!=1; skipping external integration test")

    print("=" * 60)
    print("Zhipu AI GLM-4-Flash Integration Test")
    print("=" * 60)

    # Display current configuration
    print(f"\nCurrent Configuration:")
    print(f"  Base URL: {settings.llm_base_url}")
    print(f"  Model: {settings.llm_model}")
    print(f"  API Key: {settings.llm_api_key[:20]}...")

    # Create script generator
    generator = ScriptGenerator()

    # Test text
    test_text = """
    第一章  开始

    这是一个关于勇气与成长的故事，发生在一个遥远的小镇。

    "你好，我是新来的。"年轻人微笑着说。

    "欢迎来到这里。"镇长热情地回答。
    """

    print(f"\nTest Text:")
    print(test_text)

    print("\n" + "=" * 60)
    print("Generating script...")
    print("=" * 60 + "\n")

    try:
        # Generate script
        entries = await generator.generate_script(
            text=test_text,
            max_tokens=2000,
            temperature=0.7,
        )

        print(f"\n{'=' * 60}")
        print(f"Success! Generated {len(entries)} script entries:")
        print(f"{'=' * 60}\n")

        for i, entry in enumerate(entries):
            print(f"\n[{i}] Speaker: {entry.get('speaker', 'NARRATOR')}")
            print(f"    Text: {entry.get('text', '')[:100]}...")
            print(f"    Instruct: {entry.get('instruct', 'N/A')}")

        print(f"\n{'=' * 60}")
        print("Test passed!")
        print(f"{'=' * 60}\n")

        return True

    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"Test failed with error:")
        print(f"  {str(e)}")
        print(f"{'=' * 60}\n")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_zhipu_generation())
    sys.exit(0 if success else 1)

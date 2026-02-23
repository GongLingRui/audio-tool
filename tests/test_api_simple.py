"""
Hub API Test Suite
Tests the endpoints used by the voice-studio-hub frontend.
"""

import pytest
from httpx import AsyncClient, ASGITransport
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app


@pytest.fixture
async def client():
    """Create test client."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac


class TestAudioToolsAPI:
    """Test audio tools endpoints."""

    @pytest.mark.asyncio
    async def test_audio_quality_guidelines(self, client: AsyncClient):
        """Test getting audio quality guidelines."""
        response = await client.get("/api/audio-quality/guidelines")
        print(f"Audio guidelines response: {response.status_code}")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        print(f"Guidelines keys: {list(data['data'].keys())}")

    @pytest.mark.asyncio
    async def test_asr_backends(self, client: AsyncClient):
        response = await client.get("/api/asr/backends")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "backends" in data["data"]

    @pytest.mark.asyncio
    async def test_diarization_backends(self, client: AsyncClient):
        response = await client.get("/api/diarization/backends")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "backends" in data["data"]

    @pytest.mark.asyncio
    async def test_dialect_supported(self, client: AsyncClient):
        response = await client.get("/api/dialect/supported")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "dialects" in data["data"]

    @pytest.mark.asyncio
    async def test_rvc_models_list(self, client: AsyncClient):
        response = await client.get("/api/rvc/models")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "models" in data["data"]

    @pytest.mark.asyncio
    async def test_quantization_models_list(self, client: AsyncClient):
        response = await client.get("/api/quantization/models")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "models" in data["data"]


class TestDashboardAPI:
    @pytest.mark.asyncio
    async def test_dashboard_tasks(self, client: AsyncClient):
        response = await client.get("/api/dashboard/tasks?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "recent_tasks" in data["data"]


# Run all tests
if __name__ == "__main__":
    print("=" * 60)
    print("Hub API Test Suite")
    print("=" * 60)

    pytest.main([__file__, "-v", "-s", "--tb=short"])

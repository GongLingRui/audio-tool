"""
Read-Rhyme API Test Suite (Simplified)
Tests all backend API endpoints without database dependency
"""

import pytest
import asyncio
from httpx import AsyncClient, ASGITransport
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app


@pytest.fixture
async def client():
    """Create test client without database."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac


class TestVoicesAPI:
    """Test voices management endpoints."""

    @pytest.mark.asyncio
    async def test_list_voices(self, client: AsyncClient):
        """Test listing available voices."""
        response = await client.get("/api/voices")
        print(f"List voices response: {response.status_code}")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data

    @pytest.mark.asyncio
    async def test_voice_reference(self, client: AsyncClient):
        """Test getting voice reference vocabulary."""
        response = await client.get("/api/voices/reference")
        print(f"Voice reference response: {response.status_code}")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data

    @pytest.mark.asyncio
    async def test_design_voice(self, client: AsyncClient):
        """Test voice design from description."""
        response = await client.post(
            "/api/voices/design",
            json={
                "description": "一个温柔的中年女性声音，语速适中，带有南方口音",
                "gender": "female",
                "age_range": "middle-aged"
            }
        )
        print(f"Design voice response: {response.status_code}")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "voice_id" in data["data"]
        print(f"Generated voice_id: {data['data']['voice_id']}")


class TestVoiceStylingAPI:
    """Test voice styling endpoints."""

    @pytest.mark.asyncio
    async def test_list_emotion_presets(self, client: AsyncClient):
        """Test listing emotion presets."""
        response = await client.get("/api/voice-styling/presets")
        print(f"Emotion presets response: {response.status_code}")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0
        print(f"Found {len(data['data'])} emotion presets")

    @pytest.mark.asyncio
    async def test_list_languages(self, client: AsyncClient):
        """Test listing supported languages."""
        response = await client.get("/api/voice-styling/languages")
        print(f"Languages response: {response.status_code}")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        print(f"Found {len(data['data'])} languages")


class TestRAGAPI:
    """Test RAG (Retrieval-Augmented Generation) endpoints."""

    @pytest.mark.asyncio
    async def test_ingest_document(self, client: AsyncClient):
        """Test document ingestion."""
        response = await client.post(
            "/api/rag/ingest",
            json={
                "text": "This is a test document for the RAG system. It contains sample text for testing.",
                "doc_id": "test_doc_001",
                "metadata": {"source": "test"}
            }
        )
        print(f"Ingest document response: {response.status_code}")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "chunk_count" in data["data"]
        print(f"Created {data['data']['chunk_count']} chunks")

    @pytest.mark.asyncio
    async def test_query_rag(self, client: AsyncClient):
        """Test RAG query."""
        response = await client.post(
            "/api/rag/query",
            json={
                "question": "What is this document about?",
                "use_web_search": False,
                "generate_answer": False,
                "top_k": 5
            }
        )
        print(f"Query RAG response: {response.status_code}")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data

    @pytest.mark.asyncio
    async def test_rag_stats(self, client: AsyncClient):
        """Test RAG statistics."""
        response = await client.get("/api/rag/stats")
        print(f"RAG stats response: {response.status_code}")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        print(f"RAG stats: {data['data']}")


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


class TestEmotionPresetsAPI:
    """Test emotion presets endpoints."""

    @pytest.mark.asyncio
    async def test_list_emotion_presets_full(self, client: AsyncClient):
        """Test listing all emotion presets."""
        response = await client.get("/api/emotion-presets")
        print(f"Emotion presets (full) response: {response.status_code}")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        print(f"Found {len(data['data'])} emotion presets")

    @pytest.mark.asyncio
    async def test_get_preset_by_category(self, client: AsyncClient):
        """Test getting presets by category."""
        response = await client.get("/api/emotion-presets/category/narration")
        print(f"Emotion presets (category) response: {response.status_code}")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        print(f"Narration presets: {data['data']}")


class TestLoRATrainingAPI:
    """Test LoRA training endpoints."""

    @pytest.mark.asyncio
    async def test_training_requirements(self, client: AsyncClient):
        """Test getting training requirements."""
        response = await client.get("/api/lora/requirements")
        print(f"Training requirements response: {response.status_code}")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        print(f"Requirements: {data['data']}")

    @pytest.mark.asyncio
    async def test_config_template(self, client: AsyncClient):
        """Test getting config template."""
        response = await client.get("/api/lora/config-template")
        print(f"Config template response: {response.status_code}")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data


# Run all tests
if __name__ == "__main__":
    print("=" * 60)
    print("Read-Rhyme API Test Suite")
    print("=" * 60)

    pytest.main([__file__, "-v", "-s", "--tb=short"])

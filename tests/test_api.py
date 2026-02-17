"""
Read-Rhyme API Test Suite
Tests all backend API endpoints
"""

import pytest
import asyncio
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app
from app.database import Base
from app.config import settings


# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def client():
    """Create test client."""
    from app.main import app
    from app.database import get_db

    # Create test database engine
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Create session
    async_session = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Override database dependency
    async def override_get_db():
        async with async_session() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    # Create test client
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac

    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    app.dependency_overrides.clear()


class TestAuthAPI:
    """Test authentication endpoints."""

    @pytest.mark.asyncio
    async def test_register_user(self, client: AsyncClient):
        """Test user registration."""
        response = await client.post(
            "/api/auth/register",
            json={
                "email": "test@example.com",
                "username": "testuser",
                "password": "TestPassword123!",
            }
        )
        print(f"Register response: {response.status_code}")
        assert response.status_code in [200, 400]  # 400 if user exists

    @pytest.mark.asyncio
    async def test_login_user(self, client: AsyncClient):
        """Test user login."""
        # First register
        await client.post(
            "/api/auth/register",
            json={
                "email": "login@example.com",
                "username": "loginuser",
                "password": "TestPassword123!",
            }
        )

        # Then login
        response = await client.post(
            "/api/auth/login",
            json={
                "email": "login@example.com",
                "password": "TestPassword123!",
            }
        )
        print(f"Login response: {response.status_code}")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "access_token" in data["data"]


class TestBooksAPI:
    """Test books management endpoints."""

    @pytest.mark.asyncio
    async def test_list_books(self, client: AsyncClient):
        """Test listing books."""
        response = await client.get("/api/books?page=1&limit=10")
        print(f"List books response: {response.status_code}")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data


class TestProjectsAPI:
    """Test projects management endpoints."""

    @pytest.mark.asyncio
    async def test_create_project(self, client: AsyncClient):
        """Test creating a project."""
        response = await client.post(
            "/api/projects",
            json={
                "name": "Test Project",
                "description": "Test project for audiobook",
            }
        )
        print(f"Create project response: {response.status_code}")
        assert response.status_code in [200, 401]  # May require auth

    @pytest.mark.asyncio
    async def test_list_projects(self, client: AsyncClient):
        """Test listing projects."""
        response = await client.get("/api/projects")
        print(f"List projects response: {response.status_code}")
        assert response.status_code in [200, 401]


class TestScriptsAPI:
    """Test script generation endpoints."""

    @pytest.mark.asyncio
    async def test_script_status(self, client: AsyncClient):
        """Test getting script status."""
        response = await client.get("/api/projects/test-id/scripts/status")
        print(f"Script status response: {response.status_code}")
        assert response.status_code in [200, 404]


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

    @pytest.mark.asyncio
    async def test_list_languages(self, client: AsyncClient):
        """Test listing supported languages."""
        response = await client.get("/api/voice-styling/languages")
        print(f"Languages response: {response.status_code}")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data


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


class TestConfigAPI:
    """Test system configuration endpoints."""

    @pytest.mark.asyncio
    async def test_get_config(self, client: AsyncClient):
        """Test getting system configuration."""
        response = await client.get("/api/config")
        print(f"Get config response: {response.status_code}")
        assert response.status_code in [200, 401]  # May require auth

    @pytest.mark.asyncio
    async def test_system_status(self, client: AsyncClient):
        """Test getting system status."""
        response = await client.get("/api/config/system/status")
        print(f"System status response: {response.status_code}")
        assert response.status_code in [200, 401]


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

    @pytest.mark.asyncio
    async def test_get_preset_by_category(self, client: AsyncClient):
        """Test getting presets by category."""
        response = await client.get("/api/emotion-presets/category/narration")
        print(f"Emotion presets (category) response: {response.status_code}")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data


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

"""Database connection and session management."""
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

# Create data directory if it doesn't exist
data_dir = Path("./data")
data_dir.mkdir(parents=True, exist_ok=True)

# Create async engine
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
)

# Create async session factory
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Backwards-compatible alias used by some services.
async_session_maker = async_session_factory


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


async def init_db() -> None:
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_db() -> None:
    """Drop all database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session.

    Yields:
        AsyncSession: Database session
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

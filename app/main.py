"""Main FastAPI application."""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.hub import create_hub_router
from app.config import settings
from app.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager."""
    # Startup
    await init_db()

    yield

    # Shutdown
    # Cleanup if needed


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
    )

    # Configure CORS
    # In debug mode, allow all origins for easier development
    cors_origins = settings.cors_origins
    if settings.debug:
        # Add wildcard for development
        cors_origins = ["*"] + settings.cors_origins

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    if settings.app_mode.lower() == "hub":
        app.include_router(create_hub_router())
    else:
        from app.api import api_router
        app.include_router(api_router)

    # Mount static files
    static_dir = settings.upload_dir.parent
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "version": settings.app_version}

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "message": "Read-Rhyme API",
            "version": settings.app_version,
            "docs": "/docs",
        }

    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )

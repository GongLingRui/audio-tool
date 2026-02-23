"""Hub API router.

This project has multiple backend modules, but the shipped frontend
(`voice-studio-hub`) only uses a subset of endpoints. This router exposes the
minimal surface area required by the UI.
"""

from fastapi import APIRouter

from app.api import audio_tools, dashboard


def create_hub_router() -> APIRouter:
    router = APIRouter(prefix="/api")

    router.include_router(audio_tools.router, tags=["Audio Tools"])
    router.include_router(dashboard.router, prefix="/dashboard", tags=["Dashboard"])

    return router


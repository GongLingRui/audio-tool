"""Database models."""
from app.models.book import Book
from app.models.chunk import Chunk
from app.models.highlight import Highlight
from app.models.note import Note
from app.models.project import Project
from app.models.script import Script
from app.models.user import User
from app.models.voice_config import VoiceConfig

__all__ = [
    "User",
    "Book",
    "Project",
    "Script",
    "Chunk",
    "VoiceConfig",
    "Highlight",
    "Note",
]

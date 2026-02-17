"""Pydantic schemas."""
from app.schemas.audio import (
    Chunk,
    ChunkGenerateBatch,
    ChunkGenerateFast,
    ChunkUpdate,
    MergeAudioOptions,
)
from app.schemas.book import Book, BookCreate, BookUpdate, BookUploadResponse
from app.schemas.highlight import (
    Highlight,
    HighlightCreate,
    Note,
    NoteCreate,
    NoteUpdate,
)
from app.schemas.project import (
    Project,
    ProjectConfig,
    ProjectCreate,
    ProjectProgress,
    ProjectUpdate,
)
from app.schemas.script import (
    Script,
    ScriptEntry,
    ScriptGenerateOptions,
    ScriptReviewOptions,
    ScriptUpdate,
)
from app.schemas.user import Token, User, UserCreate, UserLogin
from app.schemas.voice import (
    Voice,
    VoiceConfig,
    VoiceConfigSet,
    VoiceDesignRequest,
    VoicePreviewRequest,
)

__all__ = [
    "User",
    "UserCreate",
    "UserLogin",
    "Token",
    "Book",
    "BookCreate",
    "BookUpdate",
    "BookUploadResponse",
    "Project",
    "ProjectCreate",
    "ProjectUpdate",
    "ProjectConfig",
    "ProjectProgress",
    "Script",
    "ScriptEntry",
    "ScriptUpdate",
    "ScriptGenerateOptions",
    "ScriptReviewOptions",
    "Chunk",
    "ChunkUpdate",
    "ChunkGenerateBatch",
    "ChunkGenerateFast",
    "MergeAudioOptions",
    "Voice",
    "VoiceConfig",
    "VoiceConfigSet",
    "VoicePreviewRequest",
    "VoiceDesignRequest",
    "Highlight",
    "HighlightCreate",
    "Note",
    "NoteCreate",
    "NoteUpdate",
]

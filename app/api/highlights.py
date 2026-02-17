"""Highlights and notes API routes."""
from fastapi import APIRouter, Depends, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.deps import CurrentUserDep, DbDep
from app.core.exceptions import NotFoundException
from app.models.book import Book
from app.models.chunk import Chunk
from app.models.highlight import Highlight as HighlightModel
from app.models.note import Note as NoteModel
from app.schemas.highlight import (
    Highlight,
    HighlightCreate,
    HighlightUpdate,
    Note,
    NoteCreate,
    NoteUpdate,
)
from app.schemas.common import ApiResponse

router = APIRouter()


@router.get("", response_model=ApiResponse[list[Highlight]])
async def get_all_highlights(
    current_user: CurrentUserDep,
    db: DbDep,
    book_id: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=200),
):
    """Get all user's highlights, optionally filtered by book."""
    # Build query with eager loading
    query = select(HighlightModel).where(HighlightModel.user_id == current_user.id)

    if book_id:
        query = query.where(HighlightModel.book_id == book_id)

    # Order by most recent first
    query = query.order_by(HighlightModel.created_at.desc())

    # Pagination
    query = query.offset((page - 1) * page_size).limit(page_size)

    # Eager load book data and note
    query = query.options(selectinload(HighlightModel.book), selectinload(HighlightModel.note))

    result = await db.execute(query)
    highlights = result.scalars().all()

    return ApiResponse(
        data=[
            Highlight(
                id=h.id,
                user_id=h.user_id,
                book_id=h.book_id,
                chunk_id=h.chunk_id,
                text=h.text,
                color=h.color,
                start_offset=h.start_offset,
                end_offset=h.end_offset,
                chapter=h.chapter,
                note=Note.model_validate(h.note) if h.note else None,
                created_at=h.created_at,
            )
            for h in highlights
        ]
    )


@router.get("/{book_id}", response_model=ApiResponse[list])
async def get_highlights(
    book_id: str,
    current_user: CurrentUserDep,
    db: DbDep,
    color: str | None = None,
    chapter: str | None = None,
):
    """Get book highlights."""
    # Verify book ownership
    result = await db.execute(
        select(Book).where(Book.id == book_id, Book.user_id == current_user.id)
    )
    if not result.scalar_one_or_none():
        raise NotFoundException("Book not found")

    # Build query with eager loading
    query = select(HighlightModel).where(HighlightModel.book_id == book_id)

    if color:
        query = query.where(HighlightModel.color == color)
    if chapter:
        query = query.where(HighlightModel.chapter == chapter)

    # Eager load note
    query = query.options(selectinload(HighlightModel.note))

    result = await db.execute(query)
    highlights = result.scalars().all()

    return ApiResponse(
        data=[
            Highlight(
                id=h.id,
                user_id=h.user_id,
                book_id=h.book_id,
                chunk_id=h.chunk_id,
                text=h.text,
                color=h.color,
                start_offset=h.start_offset,
                end_offset=h.end_offset,
                chapter=h.chapter,
                note=Note.model_validate(h.note) if h.note else None,
                created_at=h.created_at,
            )
            for h in highlights
        ]
    )


@router.post("/{book_id}", response_model=ApiResponse[Highlight])
async def create_highlight(
    book_id: str,
    highlight_data: HighlightCreate,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Create a new highlight."""
    # Verify book ownership
    result = await db.execute(
        select(Book).where(Book.id == book_id, Book.user_id == current_user.id)
    )
    if not result.scalar_one_or_none():
        raise NotFoundException("Book not found")

    # Create highlight
    highlight = HighlightModel(
        user_id=current_user.id,
        book_id=book_id,
        chunk_id=highlight_data.chunk_id,
        text=highlight_data.text,
        color=highlight_data.color,
        start_offset=highlight_data.start_offset,
        end_offset=highlight_data.end_offset,
        chapter=highlight_data.chapter,
    )
    db.add(highlight)
    await db.commit()
    await db.refresh(highlight)

    # Create note if provided
    note_obj = None
    if highlight_data.note:
        note_obj = NoteModel(
            highlight_id=highlight.id,
            content=highlight_data.note,
        )
        db.add(note_obj)
        await db.commit()
        await db.refresh(note_obj)

    # Re-query highlight with eager loaded note
    result = await db.execute(
        select(HighlightModel)
        .options(selectinload(HighlightModel.note))
        .where(HighlightModel.id == highlight.id)
    )
    highlight = result.scalar_one()

    return ApiResponse(
        data=Highlight(
            id=highlight.id,
            user_id=highlight.user_id,
            book_id=highlight.book_id,
            chunk_id=highlight.chunk_id,
            text=highlight.text,
            color=highlight.color,
            start_offset=highlight.start_offset,
            end_offset=highlight.end_offset,
            chapter=highlight.chapter,
            note=Note.model_validate(highlight.note) if highlight.note else None,
            created_at=highlight.created_at,
        )
    )


@router.patch("/highlights/{highlight_id}", response_model=ApiResponse[Highlight])
async def update_highlight(
    highlight_id: str,
    highlight_update: HighlightUpdate,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Update highlight."""
    result = await db.execute(
        select(HighlightModel).where(
            HighlightModel.id == highlight_id,
            HighlightModel.user_id == current_user.id,
        )
    )
    highlight = result.scalar_one_or_none()

    if not highlight:
        raise NotFoundException("Highlight not found")

    if highlight_update.color:
        highlight.color = highlight_update.color

    await db.commit()

    return ApiResponse(data=Highlight.model_validate(highlight))


@router.delete("/highlights/{highlight_id}", response_model=ApiResponse[dict])
async def delete_highlight(
    highlight_id: str,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Delete highlight."""
    result = await db.execute(
        select(HighlightModel).where(
            HighlightModel.id == highlight_id,
            HighlightModel.user_id == current_user.id,
        )
    )
    highlight = result.scalar_one_or_none()

    if not highlight:
        raise NotFoundException("Highlight not found")

    await db.delete(highlight)
    await db.commit()

    return ApiResponse(data={"deleted": True})


@router.put("/highlights/{highlight_id}/note", response_model=ApiResponse[Note])
async def set_note(
    highlight_id: str,
    note_data: NoteCreate,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Set or update note for highlight."""
    result = await db.execute(
        select(HighlightModel).where(
            HighlightModel.id == highlight_id,
            HighlightModel.user_id == current_user.id,
        )
    )
    highlight = result.scalar_one_or_none()

    if not highlight:
        raise NotFoundException("Highlight not found")

    # Check if note exists
    result = await db.execute(
        select(NoteModel).where(NoteModel.highlight_id == highlight_id)
    )
    note = result.scalar_one_or_none()

    if note:
        note.content = note_data.content
    else:
        note = NoteModel(
            highlight_id=highlight_id,
            content=note_data.content,
        )
        db.add(note)

    await db.commit()
    await db.refresh(note)

    return ApiResponse(data=Note.model_validate(note))


@router.delete("/highlights/{highlight_id}/note", response_model=ApiResponse[dict])
async def delete_note(
    highlight_id: str,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Delete note from highlight."""
    # Verify ownership
    result = await db.execute(
        select(HighlightModel).where(
            HighlightModel.id == highlight_id,
            HighlightModel.user_id == current_user.id,
        )
    )
    if not result.scalar_one_or_none():
        raise NotFoundException("Highlight not found")

    result = await db.execute(
        select(NoteModel).where(NoteModel.highlight_id == highlight_id)
    )
    note = result.scalar_one_or_none()

    if note:
        await db.delete(note)
        await db.commit()

    return ApiResponse(data={"deleted": True})


@router.get("/{book_id}/notes/export", response_model=ApiResponse[dict])
async def export_notes(
    book_id: str,
    current_user: CurrentUserDep,
    db: DbDep,
    format: str = "markdown",
):
    """Export notes."""
    # Verify book ownership
    result = await db.execute(
        select(Book).where(Book.id == book_id, Book.user_id == current_user.id)
    )
    if not result.scalar_one_or_none():
        raise NotFoundException("Book not found")

    # Get all highlights with notes
    result = await db.execute(
        select(HighlightModel)
        .where(HighlightModel.book_id == book_id)
        .join(NoteModel)
    )
    highlights = result.scalars().all()

    if format == "markdown":
        content = "# Notes Export\n\n"
        for h in highlights:
            content += f"## {h.text[:50]}...\n\n"
            content += f"- Chapter: {h.chapter or 'N/A'}\n"
            content += f"- Color: {h.color}\n"
            if h.note:
                content += f"- Note: {h.note.content}\n"
            content += "\n---\n\n"

    return ApiResponse(data={"content": content, "format": format})

"""Thoughts API routes."""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from sqlalchemy.orm import selectinload

from app.core.deps import CurrentUserDep, DbDep
from app.core.exceptions import NotFoundException, BadRequestException
from app.models.book import Book
from app.models.thought import Thought as ThoughtModel
from app.schemas.thought import Thought, ThoughtCreate, ThoughtUpdate
from app.schemas.common import ApiResponse

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()


@router.get("", response_model=ApiResponse[list[Thought]])
async def get_thoughts(
    current_user: CurrentUserDep,
    db: DbDep,
    book_id: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
):
    """Get user's thoughts, optionally filtered by book."""
    # Build query
    query = select(ThoughtModel).where(ThoughtModel.user_id == current_user.id)

    if book_id:
        # Verify book ownership
        result = await db.execute(
            select(Book).where(Book.id == book_id, Book.user_id == current_user.id)
        )
        if not result.scalar_one_or_none():
            raise NotFoundException("Book not found")
        query = query.where(ThoughtModel.book_id == book_id)

    # Order by most recent first
    query = query.order_by(ThoughtModel.created_at.desc())

    # Pagination
    query = query.offset((page - 1) * page_size).limit(page_size)

    # Eager load book data
    query = query.options(selectinload(ThoughtModel.book))

    result = await db.execute(query)
    thoughts = result.scalars().all()

    return ApiResponse(
        data=[
            Thought(
                id=t.id,
                user_id=t.user_id,
                book_id=t.book_id,
                content=t.content,
                created_at=t.created_at,
                updated_at=t.updated_at,
                book_title=t.book.title if t.book else None,
                book_author=t.book.author if t.book else None,
                book_cover_url=t.book.cover_url if t.book else None,
            )
            for t in thoughts
        ]
    )


@router.get("/{thought_id}", response_model=ApiResponse[Thought])
async def get_thought(
    thought_id: str,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Get a specific thought."""
    result = await db.execute(
        select(ThoughtModel)
        .where(ThoughtModel.id == thought_id, ThoughtModel.user_id == current_user.id)
        .options(selectinload(ThoughtModel.book))
    )
    thought = result.scalar_one_or_none()

    if not thought:
        raise NotFoundException("Thought not found")

    return ApiResponse(
        data=Thought(
            id=thought.id,
            user_id=thought.user_id,
            book_id=thought.book_id,
            content=thought.content,
            created_at=thought.created_at,
            updated_at=thought.updated_at,
            book_title=thought.book.title if thought.book else None,
            book_author=thought.book.author if thought.book else None,
            book_cover_url=thought.book.cover_url if thought.book else None,
        )
    )


@router.post("", response_model=ApiResponse[Thought])
async def create_thought(
    thought_data: ThoughtCreate,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Create a new thought."""
    # Verify book ownership
    result = await db.execute(
        select(Book).where(
            Book.id == thought_data.book_id,
            Book.user_id == current_user.id,
        )
    )
    book = result.scalar_one_or_none()
    if not book:
        raise NotFoundException("Book not found")

    # Create thought
    thought = ThoughtModel(
        user_id=current_user.id,
        book_id=thought_data.book_id,
        content=thought_data.content,
    )
    db.add(thought)
    await db.commit()
    await db.refresh(thought)

    # Re-fetch with book data
    result = await db.execute(
        select(ThoughtModel)
        .where(ThoughtModel.id == thought.id)
        .options(selectinload(ThoughtModel.book))
    )
    thought = result.scalar_one()

    return ApiResponse(
        data=Thought(
            id=thought.id,
            user_id=thought.user_id,
            book_id=thought.book_id,
            content=thought.content,
            created_at=thought.created_at,
            updated_at=thought.updated_at,
            book_title=thought.book.title,
            book_author=thought.book.author,
            book_cover_url=thought.book.cover_url,
        )
    )


@router.patch("/{thought_id}", response_model=ApiResponse[Thought])
async def update_thought(
    thought_id: str,
    thought_update: ThoughtUpdate,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Update a thought."""
    result = await db.execute(
        select(ThoughtModel).where(
            ThoughtModel.id == thought_id,
            ThoughtModel.user_id == current_user.id,
        )
    )
    thought = result.scalar_one_or_none()

    if not thought:
        raise NotFoundException("Thought not found")

    thought.content = thought_update.content
    await db.commit()
    await db.refresh(thought)

    # Re-fetch with book data
    result = await db.execute(
        select(ThoughtModel)
        .where(ThoughtModel.id == thought.id)
        .options(selectinload(ThoughtModel.book))
    )
    thought = result.scalar_one()

    return ApiResponse(
        data=Thought(
            id=thought.id,
            user_id=thought.user_id,
            book_id=thought.book_id,
            content=thought.content,
            created_at=thought.created_at,
            updated_at=thought.updated_at,
            book_title=thought.book.title,
            book_author=thought.book.author,
            book_cover_url=thought.book.cover_url,
        )
    )


@router.delete("/{thought_id}", response_model=ApiResponse[dict])
async def delete_thought(
    thought_id: str,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Delete a thought."""
    result = await db.execute(
        select(ThoughtModel).where(
            ThoughtModel.id == thought_id,
            ThoughtModel.user_id == current_user.id,
        )
    )
    thought = result.scalar_one_or_none()

    if not thought:
        raise NotFoundException("Thought not found")

    await db.delete(thought)
    await db.commit()

    return ApiResponse(data={"deleted": True})


@router.get("/book/{book_id}/all", response_model=ApiResponse[list[Thought]])
async def get_book_thoughts(
    book_id: str,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Get all thoughts for a specific book."""
    # Verify book ownership
    result = await db.execute(
        select(Book).where(Book.id == book_id, Book.user_id == current_user.id)
    )
    if not result.scalar_one_or_none():
        raise NotFoundException("Book not found")

    result = await db.execute(
        select(ThoughtModel)
        .where(ThoughtModel.book_id == book_id)
        .order_by(ThoughtModel.created_at.desc())
        .options(selectinload(ThoughtModel.book))
    )
    thoughts = result.scalars().all()

    return ApiResponse(
        data=[
            Thought(
                id=t.id,
                user_id=t.user_id,
                book_id=t.book_id,
                content=t.content,
                created_at=t.created_at,
                updated_at=t.updated_at,
                book_title=t.book.title if t.book else None,
                book_author=t.book.author if t.book else None,
                book_cover_url=t.book.cover_url if t.book else None,
            )
            for t in thoughts
        ]
    )

"""Books API routes."""
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete

from app.core.deps import CurrentUserDep, DbDep
from app.core.exceptions import BadRequestException, NotFoundException
from app.config import settings
from app.models.book import Book as BookModel
from app.models.chunk import Chunk
from app.models.highlight import Highlight
from app.models.project import Project
from app.schemas.book import Book, BookCreate, BookUpdate, BookUploadResponse
from app.schemas.common import ApiResponse, PaginatedResponse

# PDF/EPUB parsing
import PyPDF2
from ebooklib import epub

router = APIRouter()


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file."""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            return "\n\n".join(text_parts)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract text from PDF: {str(e)}",
        )


def extract_text_from_epub(file_path: str) -> str:
    """Extract text from EPUB file."""
    try:
        book = epub.read_epub(file_path)
        text_parts = []
        for item in book.get_items():
            if item.get_type() == epub.ITEM_DOCUMENT:
                content = item.get_content().decode("utf-8")
                text_parts.append(content)
        return "\n\n".join(text_parts)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract text from EPUB: {str(e)}",
        )


async def get_user_books(
    user_id: str,
    db: AsyncSession,
    search: str | None = None,
    sort: str = "created_at",
    order: str = "desc",
    page: int = 1,
    page_size: int = 20,
) -> PaginatedResponse[Book]:
    """Get user's books with pagination."""
    # Build query
    query = select(BookModel).where(BookModel.user_id == user_id)

    # Apply search filter
    if search:
        query = query.where(BookModel.title.contains(search))

    # Apply sorting
    order_column = getattr(BookModel, sort, BookModel.created_at)
    if order == "desc":
        query = query.order_by(order_column.desc())
    else:
        query = query.order_by(order_column.asc())

    # Count total
    count_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = count_result.scalar() or 0

    # Apply pagination
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    books = result.scalars().all()

    total_pages = (total + page_size - 1) // page_size

    return PaginatedResponse(
        items=[Book.model_validate(book) for book in books],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("", response_model=ApiResponse[PaginatedResponse[Book]])
async def list_books(
    current_user: CurrentUserDep,
    db: DbDep,
    search: str | None = None,
    sort: str = "created_at",
    order: str = "desc",
    page: int = 1,
    page_size: int = 20,
):
    """Get user's books."""
    result = await get_user_books(
        user_id=current_user.id,
        db=db,
        search=search,
        sort=sort,
        order=order,
        page=page,
        page_size=page_size,
    )
    return ApiResponse(data=result)


@router.get("/{book_id}", response_model=ApiResponse[Book])
async def get_book(
    book_id: str,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Get book details."""
    result = await db.execute(
        select(BookModel).where(BookModel.id == book_id, BookModel.user_id == current_user.id)
    )
    book = result.scalar_one_or_none()

    if not book:
        raise NotFoundException("Book not found")

    return ApiResponse(data=Book.model_validate(book))


@router.post("/upload", response_model=ApiResponse[BookUploadResponse])
async def upload_book(
    current_user: CurrentUserDep,
    db: DbDep,
    file: UploadFile = File(...),
    title: str = Form(None),
    author: str = Form(None),
):
    """Upload a new book."""
    # Debug logging
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Upload request - file: {file}, title: {title}, author: {author}")
    logger.info(f"File details - filename: {file.filename}, content_type: {file.content_type}")

    # Validate file
    if not file.filename:
        raise BadRequestException("No file provided")

    file_ext = Path(file.filename).suffix.lower().lstrip(".")
    if file_ext not in ["txt", "pdf", "epub"]:
        raise BadRequestException(f"Unsupported file type: {file_ext}")

    # Check file size
    content = await file.read()
    if len(content) > settings.max_upload_size:
        raise BadRequestException(f"File too large (max {settings.max_upload_size} bytes)")

    # Generate file path
    book_id = str(uuid.uuid4())
    user_dir = settings.upload_dir / current_user.id
    user_dir.mkdir(parents=True, exist_ok=True)
    file_path = user_dir / f"{book_id}.{file_ext}"

    # Save file
    with open(file_path, "wb") as f:
        f.write(content)

    # Extract title if not provided
    if not title:
        title = Path(file.filename).stem

    # Count characters for text files
    total_chars = None
    if file_ext == "txt":
        try:
            total_chars = len(content.decode("utf-8"))
        except UnicodeDecodeError:
            total_chars = len(content.decode("utf-8", errors="ignore"))

    # Create book record
    book = BookModel(
        id=book_id,
        user_id=current_user.id,
        title=title,
        author=author,
        file_path=str(file_path),
        file_type=file_ext,
        total_chars=total_chars,
    )
    db.add(book)
    await db.commit()
    await db.refresh(book)

    return ApiResponse(
        data=BookUploadResponse(
            id=book.id,
            title=book.title,
            author=book.author,
            file_type=book.file_type,
            total_chars=book.total_chars,
        )
    )


@router.get("/{book_id}/content", response_model=ApiResponse[dict])
async def get_book_content(
    book_id: str,
    current_user: CurrentUserDep,
    db: DbDep,
    format: str = "plain",
    chapter: int | None = None,
):
    """Get book content."""
    result = await db.execute(
        select(BookModel).where(BookModel.id == book_id, BookModel.user_id == current_user.id)
    )
    book = result.scalar_one_or_none()

    if not book:
        raise NotFoundException("Book not found")

    # Read file content based on file type
    try:
        if book.file_type == "txt":
            with open(book.file_path, "r", encoding="utf-8") as f:
                content = f.read()
        elif book.file_type == "pdf":
            content = extract_text_from_pdf(book.file_path)
        elif book.file_type == "epub":
            content = extract_text_from_epub(book.file_path)
        else:
            raise BadRequestException(f"Unsupported file type: {book.file_type}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read file: {str(e)}",
        )

    return ApiResponse(
        data={
            "content": content,
            "chapters": [],
            "metadata": {
                "title": book.title,
                "author": book.author,
                "total_chars": book.total_chars,
            },
        }
    )


@router.patch("/{book_id}", response_model=ApiResponse[Book])
async def update_book(
    book_id: str,
    book_update: BookUpdate,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Update book."""
    result = await db.execute(
        select(BookModel).where(BookModel.id == book_id, BookModel.user_id == current_user.id)
    )
    book = result.scalar_one_or_none()

    if not book:
        raise NotFoundException("Book not found")

    # Update fields
    if book_update.title is not None:
        book.title = book_update.title
    if book_update.author is not None:
        book.author = book_update.author
    if book_update.cover_url is not None:
        book.cover_url = book_update.cover_url
    if book_update.progress is not None:
        book.progress = book_update.progress

    await db.commit()
    await db.refresh(book)

    return ApiResponse(data=Book.model_validate(book))


@router.delete("/{book_id}", response_model=ApiResponse[dict])
async def delete_book(
    book_id: str,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Delete book."""
    result = await db.execute(
        select(BookModel).where(BookModel.id == book_id, BookModel.user_id == current_user.id)
    )
    book = result.scalar_one_or_none()

    if not book:
        raise NotFoundException("Book not found")

    # Delete file
    try:
        if os.path.exists(book.file_path):
            os.remove(book.file_path)
    except Exception:
        pass  # Ignore file deletion errors

    # Delete from database (cascade will handle related records)
    await db.delete(book)
    await db.commit()

    return ApiResponse(data={"deleted": True})

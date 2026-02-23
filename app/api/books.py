"""Books API routes."""

from fastapi import APIRouter
from sqlalchemy import func, select

from app.core.deps import DbDep
from app.models.book import Book as BookModel
from app.schemas.book import Book
from app.schemas.common import ApiResponse, PaginatedResponse

router = APIRouter()


@router.get("", response_model=ApiResponse[PaginatedResponse[Book]])
async def list_books(
    db: DbDep,
    page: int = 1,
    limit: int = 20,
):
    """List books (public; returns empty if none)."""
    page = max(page, 1)
    limit = max(1, min(limit, 100))

    count_result = await db.execute(select(func.count()).select_from(BookModel))
    total = int(count_result.scalar() or 0)

    query = select(BookModel).order_by(BookModel.created_at.desc()).offset((page - 1) * limit).limit(limit)
    result = await db.execute(query)
    books = result.scalars().all()

    items = [Book.model_validate(b) for b in books]
    total_pages = (total + limit - 1) // limit if total else 0

    return ApiResponse(
        data=PaginatedResponse(
            items=items,
            total=total,
            page=page,
            page_size=limit,
            total_pages=total_pages,
        )
    )


"""Projects API routes."""
import uuid
from typing import Any

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.deps import CurrentUserDep, DbDep
from app.core.exceptions import BadRequestException, NotFoundException
from app.models.book import Book
from app.models.chunk import Chunk
from app.models.project import Project as ProjectModel
from app.models.script import Script
from app.models.voice_config import VoiceConfig
from app.schemas.project import (
    Project,
    ProjectCreate,
    ProjectUpdate,
    ProjectProgress,
)
from app.schemas.common import ApiResponse, PaginatedResponse

router = APIRouter()


@router.get("", response_model=ApiResponse[PaginatedResponse[Project]])
async def list_projects(
    current_user: CurrentUserDep,
    db: DbDep,
    book_id: str | None = None,
    status_filter: str | None = None,
    page: int = 1,
    page_size: int = 20,
):
    """Get user's projects."""
    # Build query
    query = (
        select(ProjectModel)
        .join(Book)
        .where(Book.user_id == current_user.id)
    )

    if book_id:
        query = query.where(ProjectModel.book_id == book_id)
    if status_filter:
        query = query.where(ProjectModel.status == status_filter)

    # Count total
    count_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = count_result.scalar() or 0

    # Apply pagination
    query = query.order_by(ProjectModel.created_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    projects = result.scalars().all()

    # Get book titles
    project_data = []
    for project in projects:
        book_result = await db.execute(select(Book).where(Book.id == project.book_id))
        book = book_result.scalar_one_or_none()

        # Calculate progress
        chunk_result = await db.execute(
            select(func.count()).where(Chunk.project_id == project.id)
        )
        total_chunks = chunk_result.scalar() or 0

        completed_result = await db.execute(
            select(func.count()).where(
                Chunk.project_id == project.id,
                Chunk.status == "completed",
            )
        )
        completed_chunks = completed_result.scalar() or 0

        percentage = (completed_chunks / total_chunks * 100) if total_chunks > 0 else 0

        project_dict = {
            **Project.model_validate(project).model_dump(),
            "book_title": book.title if book else None,
            "progress": ProjectProgress(
                total_chunks=total_chunks,
                completed_chunks=completed_chunks,
                percentage=percentage,
            ) if total_chunks > 0 else None,
        }
        project_data.append(Project(**project_dict))

    total_pages = (total + page_size - 1) // page_size

    return ApiResponse(
        data=PaginatedResponse(
            items=project_data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )
    )


@router.get("/{project_id}", response_model=ApiResponse[Project])
async def get_project(
    project_id: str,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Get project details."""
    result = await db.execute(
        select(ProjectModel)
        .join(Book)
        .where(ProjectModel.id == project_id, Book.user_id == current_user.id)
    )
    project = result.scalar_one_or_none()

    if not project:
        raise NotFoundException("Project not found")

    # Get book title
    book_result = await db.execute(select(Book).where(Book.id == project.book_id))
    book = book_result.scalar_one_or_none()

    # Calculate progress
    chunk_result = await db.execute(
        select(func.count()).where(Chunk.project_id == project.id)
    )
    total_chunks = chunk_result.scalar() or 0

    completed_result = await db.execute(
        select(func.count()).where(
            Chunk.project_id == project.id,
            Chunk.status == "completed",
        )
    )
    completed_chunks = completed_result.scalar() or 0

    percentage = (completed_chunks / total_chunks * 100) if total_chunks > 0 else 0

    project_dict = {
        **Project.model_validate(project).model_dump(),
        "book_title": book.title if book else None,
        "progress": ProjectProgress(
            total_chunks=total_chunks,
            completed_chunks=completed_chunks,
            percentage=percentage,
        ) if total_chunks > 0 else None,
    }

    return ApiResponse(data=Project(**project_dict))


@router.post("", response_model=ApiResponse[Project])
async def create_project(
    project_data: ProjectCreate,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Create a new project."""
    # Verify book ownership
    book_result = await db.execute(
        select(Book).where(Book.id == project_data.book_id, Book.user_id == current_user.id)
    )
    book = book_result.scalar_one_or_none()

    if not book:
        raise NotFoundException("Book not found")

    # Create project
    project = ProjectModel(
        book_id=project_data.book_id,
        name=project_data.name,
        description=project_data.description,
        config=project_data.config.model_dump() if project_data.config else {},
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)

    # Build response dict
    project_dict = Project.model_validate(project).model_dump()
    project_dict["book_title"] = book.title

    return ApiResponse(data=Project(**project_dict))


@router.patch("/{project_id}", response_model=ApiResponse[Project])
async def update_project(
    project_id: str,
    project_update: ProjectUpdate,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Update project."""
    result = await db.execute(
        select(ProjectModel)
        .join(Book)
        .where(ProjectModel.id == project_id, Book.user_id == current_user.id)
    )
    project = result.scalar_one_or_none()

    if not project:
        raise NotFoundException("Project not found")

    # Update fields
    if project_update.name is not None:
        project.name = project_update.name
    if project_update.description is not None:
        project.description = project_update.description
    if project_update.config is not None:
        project.config = project_update.config.model_dump()

    await db.commit()
    await db.refresh(project)

    # Get book title
    book_result = await db.execute(select(Book).where(Book.id == project.book_id))
    book = book_result.scalar_one_or_none()

    # Build response dict
    project_dict = Project.model_validate(project).model_dump()
    project_dict["book_title"] = book.title if book else None

    return ApiResponse(data=Project(**project_dict))


@router.delete("/{project_id}", response_model=ApiResponse[dict])
async def delete_project(
    project_id: str,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Delete project."""
    result = await db.execute(
        select(ProjectModel)
        .join(Book)
        .where(ProjectModel.id == project_id, Book.user_id == current_user.id)
    )
    project = result.scalar_one_or_none()

    if not project:
        raise NotFoundException("Project not found")

    await db.delete(project)
    await db.commit()

    return ApiResponse(data={"deleted": True})


@router.get("/{project_id}/chunks/progress", response_model=ApiResponse[ProjectProgress])
async def get_project_progress(
    project_id: str,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Get project audio generation progress."""
    # Verify ownership
    result = await db.execute(
        select(ProjectModel)
        .join(Book)
        .where(ProjectModel.id == project_id, Book.user_id == current_user.id)
    )
    project = result.scalar_one_or_none()

    if not project:
        raise NotFoundException("Project not found")

    # Get chunk counts
    total_result = await db.execute(
        select(func.count()).where(Chunk.project_id == project_id)
    )
    total = total_result.scalar() or 0

    completed_result = await db.execute(
        select(func.count()).where(
            Chunk.project_id == project_id,
            Chunk.status == "completed",
        )
    )
    completed = completed_result.scalar() or 0

    processing_result = await db.execute(
        select(func.count()).where(
            Chunk.project_id == project_id,
            Chunk.status == "processing",
        )
    )
    processing = processing_result.scalar() or 0

    pending = total - completed - processing
    percentage = (completed / total * 100) if total > 0 else 0

    return ApiResponse(
        data=ProjectProgress(
            total_chunks=total,
            completed_chunks=completed,
            percentage=percentage,
        )
    )

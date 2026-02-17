"""Scripts API routes."""
import json
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.core.deps import CurrentUserDep, DbDep
from app.core.exceptions import NotFoundException
from app.models.project import Project as ProjectModel
from app.models.script import Script as ScriptModel
from app.models.chunk import Chunk
from app.schemas.script import (
    Script,
    ScriptEntry,
    ScriptGenerateOptions,
    ScriptReviewOptions,
    ScriptStatusResponse,
    ScriptUpdate,
)
from app.schemas.common import ApiResponse
from app.services.script_generator import ScriptGenerator, load_default_prompts
from app.services.chunk_service import group_into_chunks, split_script_to_chunks

logger = logging.getLogger(__name__)
router = APIRouter()


async def generate_script_task(project_id: str):
    """Background task for script generation."""
    from app.services.script_generator import ScriptGenerator
    from app.services.chunk_service import group_into_chunks
    from app.models.book import Book
    from app.database import async_session_factory

    # 在后台任务中创建新的数据库会话（不能直接传递依赖注入的会话）
    async with async_session_factory() as db:
        try:
            # Get project with book
            result = await db.execute(
                select(ProjectModel).where(ProjectModel.id == project_id)
            )
            project = result.scalar_one_or_none()

            if not project:
                logger.error(f"Project not found: {project_id}")
                return

            # Get the associated book
            # IMPORTANT: Use explicit query to avoid any relationship confusion
            result = await db.execute(
                select(Book).where(Book.id == project.book_id)
            )
            book = result.scalar_one_or_none()

            if not book:
                logger.error(f"Book not found for project {project_id}, book_id: {project.book_id}")
                # Update script status if it exists
                script_result = await db.execute(select(ScriptModel).where(ScriptModel.project_id == project_id))
                script = script_result.scalar_one_or_none()
                if script:
                    script.status = "failed"
                    script.error_message = f"Book not found (book_id: {project.book_id})"
                    await db.commit()
                return
            
            # CRITICAL SAFETY CHECK: Ensure book is not accidentally project
            if isinstance(book, ProjectModel):
                logger.error(f"ERROR: book variable is actually a Project object! Project ID: {project.id}, Book ID should be: {project.book_id}")
                script_result = await db.execute(select(ScriptModel).where(ScriptModel.project_id == project_id))
                script = script_result.scalar_one_or_none()
                if script:
                    script.status = "failed"
                    script.error_message = f"Database query returned Project instead of Book. This is a critical error."
                    await db.commit()
                return

            # Get script (should exist as it's created before background task starts)
            result = await db.execute(select(ScriptModel).where(ScriptModel.project_id == project_id))
            script = result.scalar_one_or_none()

            if not script:
                logger.error(f"Script record not found for project {project_id}")
                return

            try:
                script.status = "processing"
                await db.commit()

                # Generate script using ScriptGenerator
                generator = ScriptGenerator()

                # Get book content from file based on file type
                # CRITICAL: Ensure we're using book, not project
                # Double-check book is actually a Book instance, not Project
                if not isinstance(book, Book):
                    script.status = "failed"
                    script.error_message = f"Invalid book object type: {type(book).__name__}. Expected Book, got {type(book)}"
                    await db.commit()
                    logger.error(f"Invalid book type: {type(book)}, Value: {book}, Project type: {type(project)}")
                    return
                
                # Verify book object has file_path attribute
                if not hasattr(book, 'file_path'):
                    script.status = "failed"
                    script.error_message = f"Book object missing file_path attribute. Book type: {type(book).__name__}, Book ID: {getattr(book, 'id', 'N/A')}"
                    await db.commit()
                    logger.error(f"Book object missing file_path. Book: {book}, Type: {type(book)}, Project: {project}, Project type: {type(project)}")
                    return
                
                # CRITICAL: Store book.file_path in local variable IMMEDIATELY to prevent any confusion
                # DO NOT use project.file_path anywhere - Project does not have file_path
                try:
                    book_file_path = book.file_path
                    logger.info(f"Retrieved file_path from book: {book_file_path[:50] if book_file_path else 'None'}...")
                except AttributeError as e:
                    script.status = "failed"
                    script.error_message = f"Failed to access book.file_path: {str(e)}. Book type: {type(book).__name__}"
                    await db.commit()
                    logger.error(f"AttributeError accessing book.file_path: {e}, Book: {book}, Type: {type(book)}")
                    return
                
                content = None
                if not book_file_path:
                    script.status = "failed"
                    script.error_message = f"Book file_path is empty or None. Book ID: {book.id}"
                    await db.commit()
                    logger.error(f"Book file_path is empty. Book: {book.id}")
                    return
                
                if book_file_path:
                    try:
                        if book.file_type == "txt":
                            with open(book_file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                        elif book.file_type == "pdf":
                            from app.api.books import extract_text_from_pdf
                            content = extract_text_from_pdf(book_file_path)
                        elif book.file_type == "epub":
                            from app.api.books import extract_text_from_epub
                            content = extract_text_from_epub(book_file_path)
                        else:
                            script.status = "failed"
                            script.error_message = f"Unsupported file type: {book.file_type}"
                            await db.commit()
                            return
                    except Exception as e:
                        script.status = "failed"
                        script.error_message = f"Failed to read book file: {str(e)}"
                        await db.commit()
                        logger.error(f"Failed to read book file {book_file_path}: {e}")
                        return
                
                if not content:
                    script.status = "failed"
                    script.error_message = "No content to generate script from"
                    await db.commit()
                    return

                if not content or not content.strip():
                    script.status = "failed"
                    script.error_message = "Book content is empty"
                    await db.commit()
                    return

                # Generate script entries
                entries = await generator.generate_script(
                    text=content,
                    temperature=0.7,
                    max_tokens=16000
                )

                # Store generated script
                script.content = entries
                script.status = "approved"
                script.entries_count = len(entries)

                # Extract speakers
                speakers = list(set(e.get("speaker", "NARRATOR") for e in entries))
                script.speakers = speakers

                await db.commit()

            except Exception as e:
                script.status = "failed"
                error_msg = str(e)
                script.error_message = error_msg
                await db.commit()
                # Log detailed error information
                logger.error(
                    f"Script generation failed for project {project_id}: {error_msg}\n"
                    f"Project: {project.id if project else 'None'}, "
                    f"Book: {book.id if book else 'None'}, "
                    f"Book has file_path: {hasattr(book, 'file_path') if book else 'N/A'}",
                    exc_info=True
                )
        except Exception as e:
            logger.error(f"Critical error in generate_script_task for project {project_id}: {e}", exc_info=True)


@router.get("/{project_id}/scripts/status", response_model=ApiResponse[ScriptStatusResponse])
async def get_script_status(
    project_id: str,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Get script generation status."""
    result = await db.execute(
        select(ScriptModel).where(ScriptModel.project_id == project_id)
    )
    script = result.scalar_one_or_none()

    if not script:
        return ApiResponse(
            data=ScriptStatusResponse(
                id="",
                project_id=project_id,
                status="not_started",
                entries_count=0,
                speakers=[],
                error_message=None,
                created_at="",
            )
        )

    # Extract speakers
    speakers = list(set(
        entry.get("speaker", "NARRATOR")
        for entry in script.content
    ))

    return ApiResponse(
        data=ScriptStatusResponse(
            id=script.id,
            project_id=script.project_id,
            status=script.status,
            entries_count=len(script.content),
            speakers=speakers,
            error_message=script.error_message,
            created_at=script.created_at.isoformat(),
        )
    )


@router.get("/{project_id}/scripts", response_model=ApiResponse[Script])
async def get_script(
    project_id: str,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Get script content."""
    result = await db.execute(
        select(ScriptModel).where(ScriptModel.project_id == project_id)
    )
    script = result.scalar_one_or_none()

    if not script:
        raise NotFoundException("Script not found. Generate a script first.")

    return ApiResponse(
        data=Script(
            id=script.id,
            project_id=script.project_id,
            content=[ScriptEntry(**entry) for entry in script.content],
            status=script.status,
            error_message=script.error_message,
            created_at=script.created_at.isoformat(),
            updated_at=script.updated_at.isoformat(),
        )
    )


@router.post("/{project_id}/scripts/generate", response_model=ApiResponse[dict])
async def generate_script(
    project_id: str,
    options: ScriptGenerateOptions,
    background_tasks: BackgroundTasks,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Generate script from book content."""
    # Verify project exists
    result = await db.execute(select(ProjectModel).where(ProjectModel.id == project_id))
    project = result.scalar_one_or_none()

    if not project:
        raise NotFoundException("Project not found")

    # Check if script already exists
    existing = await db.execute(select(ScriptModel).where(ScriptModel.project_id == project_id))
    if existing.scalar_one_or_none():
        return ApiResponse(data={"message": "Script already exists"})

    # Create script record
    script = ScriptModel(
        project_id=project_id,
        content=[],
        status="pending",
    )
    db.add(script)
    await db.commit()

    # Start background generation（不传递 db，在任务内部创建新会话）
    background_tasks.add_task(generate_script_task, project_id)

    return ApiResponse(
        data={
            "script_id": script.id,
            "status": "generating",
            "message": "Script generation started",
        }
    )


@router.patch("/{project_id}/scripts", response_model=ApiResponse[dict])
async def update_script(
    project_id: str,
    script_update: ScriptUpdate,
    current_user: CurrentUserDep,
    db: DbDep,
    regenerate_chunks: bool = True,
    merge_narrators: bool = True,
    max_chunk_chars: int = 500,
):
    """Update script content and optionally regenerate chunks."""
    result = await db.execute(
        select(ScriptModel).where(ScriptModel.project_id == project_id)
    )
    script = result.scalar_one_or_none()

    if not script:
        raise NotFoundException("Script not found")

    # Update content
    script.content = [entry.model_dump() for entry in script_update.content]
    await db.commit()

    # Regenerate chunks if requested
    if regenerate_chunks:
        # Delete existing chunks
        await db.execute(
            delete(Chunk).where(Chunk.script_id == script.id)
        )

        # Use intelligent chunking to create new chunks
        script_entries = script.content
        chunks_data = split_script_to_chunks(
            script_entries,
            max_chars=max_chunk_chars,
            merge_narrators=merge_narrators,
        )

        # Create chunk records
        chunks = []
        for i, chunk_data in enumerate(chunks_data):
            chunk = Chunk(
                project_id=project_id,
                script_id=script.id,
                speaker=chunk_data.get("speaker", "NARRATOR"),
                text=chunk_data.get("text", ""),
                instruct=chunk_data.get("instruct"),
                emotion=chunk_data.get("instruct"),  # Use instruct as emotion for now
                section=None,
                order_index=i,
                status="pending",  # Reset to pending for regeneration
            )
            db.add(chunk)
            chunks.append(chunk)

        await db.commit()

        return ApiResponse(
            data={
                "updated": True,
                "chunks_regenerated": len(chunks),
                "original_entries": len(script_entries),
                "message": f"Script updated and {len(chunks)} chunks regenerated",
            }
        )

    return ApiResponse(
        data={
            "updated": True,
            "chunks_regenerated": 0,
            "message": "Script updated (chunks not regenerated)",
        }
    )


@router.post("/{project_id}/scripts/review", response_model=ApiResponse[dict])
async def review_script(
    project_id: str,
    options: ScriptReviewOptions,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Review and fix script errors using AI."""
    result = await db.execute(
        select(ScriptModel).where(ScriptModel.project_id == project_id)
    )
    script = result.scalar_one_or_none()

    if not script:
        raise NotFoundException("Script not found")

    # Get script entries
    entries = script.content
    if not entries or len(entries) == 0:
        return ApiResponse(
            data={
                "status": "reviewed",
                "fixed_count": 0,
                "issues": [],
            }
        )

    # Collect issues found
    issues_found = []
    fixed_count = 0

    # Use LLM to review the script
    from app.services.script_generator import ScriptGenerator
    from app.config import settings

    generator = ScriptGenerator()

    # Build review prompt
    review_prompt = f"""You are a script quality checker. Review the following audiobook script for errors and inconsistencies.

Script entries:
{json.dumps(entries[:20], ensure_ascii=False, indent=2)}

Check for:
1. Speaker consistency - same character should have same speaker label
2. Dialogue attribution - ensure quotes are attributed to correct speakers
3. Over-split entries - entries that should be combined
4. Missing emotion/context where needed
5. Obvious formatting errors

Respond in JSON format:
{{
    "issues": [
        {{"entry_index": 0, "issue_type": "speaker_mismatch", "description": "Speaker label inconsistent", "suggested_fix": {{"speaker": "NARRATOR"}}}},
        ...
    ],
    "summary": "Brief summary of issues found"
}}

If no issues found, return: {{"issues": [], "summary": "No issues found"}}
"""

    try:
        reviewed_entries = await generator.generate_script(
            text=review_prompt,
            system_prompt="You are a meticulous script reviewer. Always respond with valid JSON.",
            temperature=0.3,
            max_tokens=2000
        )

        # Parse review results
        if reviewed_entries and len(reviewed_entries) > 0:
            review_text = reviewed_entries[0].get("text", "")

            # Try to parse JSON from the review
            import re
            json_match = re.search(r'\{[^{}]*\{.*\}[^{}]*\}', review_text, re.DOTALL)
            if json_match:
                try:
                    review_data = json.loads(json_match.group())
                    issues = review_data.get("issues", [])

                    if options.auto_fix and len(issues) > 0:
                        # Apply fixes
                        for issue in issues:
                            entry_idx = issue.get("entry_index")
                            if entry_idx is not None and 0 <= entry_idx < len(entries):
                                fix = issue.get("suggested_fix")
                                if fix:
                                    # Apply the fix
                                    for key, value in fix.items():
                                        if value:  # Only apply non-empty values
                                            entries[entry_idx][key] = value
                                    fixed_count += 1

                        # Update script with fixes
                        script.content = entries
                        script.status = "reviewed"
                        await db.commit()

                        # Collect issue descriptions
                        for issue in issues:
                            issues_found.append(issue.get("description", f"Entry {issue.get('entry_index')}: {issue.get('issue_type')}"))

                    return ApiResponse(
                        data={
                            "status": "reviewed",
                            "fixed_count": fixed_count,
                            "issues": issues_found[:10],  # Return first 10 issues
                        }
                    )
                except json.JSONDecodeError:
                    pass

        # Fallback: Basic rule-based checks
        issues_found = []
        fixed_count = 0

        if options.check_rules and options.check_rules.get("speaker_consistency", True):
            # Check speaker consistency
            speaker_map = {}
            for i, entry in enumerate(entries):
                speaker = entry.get("speaker", "")
                text = entry.get("text", "")

                # Simple heuristic: if entry contains quotes, it's likely dialogue
                if '"' in text or '"' in text:
                    # Extract potential character name from context
                    words = text.split()
                    for word in words[:5]:  # Check first few words
                        if word[0].isupper() and len(word) > 2:
                            if word not in speaker_map:
                                speaker_map[word] = speaker

            # Report inconsistencies
            if len(speaker_map) > 1:
                issues_found.append(f"Found multiple speaker patterns: {list(speaker_map.keys())[:5]}")

        # Update status
        script.status = "reviewed"
        await db.commit()

        return ApiResponse(
            data={
                "status": "reviewed",
                "fixed_count": fixed_count,
                "issues": issues_found[:10] if issues_found else ["No major issues detected"],
            }
        )

    except Exception as e:
        # Still mark as reviewed even if review fails
        script.status = "reviewed"
        await db.commit()

        return ApiResponse(
            data={
                "status": "reviewed",
                "fixed_count": 0,
                "issues": [f"Review completed with warnings: {str(e)[:100]}"],
            }
        )


@router.post("/{project_id}/scripts/approve", response_model=ApiResponse[dict])
async def approve_script(
    project_id: str,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Approve script for audio generation."""
    result = await db.execute(
        select(ScriptModel).where(ScriptModel.project_id == project_id)
    )
    script = result.scalar_one_or_none()

    if not script:
        raise NotFoundException("Script not found")

    script.status = "approved"
    await db.commit()

    return ApiResponse(data={"status": "approved"})


@router.post("/{project_id}/scripts/chunks", response_model=ApiResponse[dict])
async def create_chunks_from_script(
    project_id: str,
    current_user: CurrentUserDep,
    db: DbDep,
    merge_narrators: bool = True,
    max_chunk_chars: int = 500,
):
    """Create audio chunks from script using intelligent chunking."""
    result = await db.execute(
        select(ScriptModel).where(ScriptModel.project_id == project_id)
    )
    script = result.scalar_one_or_none()

    if not script:
        raise NotFoundException("Script not found")

    # Delete existing chunks
    await db.execute(
        delete(Chunk).where(Chunk.script_id == script.id)
    )

    # Use intelligent chunking to create chunks
    script_entries = script.content
    chunks_data = split_script_to_chunks(
        script_entries,
        max_chars=max_chunk_chars,
        merge_narrators=merge_narrators,
    )

    # Create chunk records
    chunks = []
    for i, chunk_data in enumerate(chunks_data):
        chunk = Chunk(
            project_id=project_id,
            script_id=script.id,
            speaker=chunk_data.get("speaker", "NARRATOR"),
            text=chunk_data.get("text", ""),
            instruct=chunk_data.get("instruct"),
            emotion=chunk_data.get("instruct"),  # Use instruct as emotion for now
            section=None,
            order_index=i,
        )
        db.add(chunk)
        chunks.append(chunk)

    await db.commit()

    return ApiResponse(
        data={
            "created": len(chunks),
            "original_entries": len(script_entries),
            "message": f"Created {len(chunks)} audio chunks from {len(script_entries)} script entries",
        }
    )

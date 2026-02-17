"""Audio API routes."""
from fastapi import APIRouter, BackgroundTasks, Depends, status, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from pathlib import Path
from typing import Literal
import logging

from app.core.deps import CurrentUserDep, DbDep
from app.core.exceptions import NotFoundException
from app.models.chunk import Chunk as ChunkModel
from app.models.project import Project as ProjectModel
from app.schemas.audio import (
    AudioResponse,
    Chunk,
    ChunkGenerateBatch,
    ChunkGenerateFast,
    ChunkProgressResponse,
    ChunkUpdate,
    MergeAudioOptions,
)
from app.schemas.common import ApiResponse, PaginatedResponse
from app.config import settings

logger = logging.getLogger(__name__)

# Lazy import to avoid missing dependencies
def _get_audio_processor():
    from app.services.audio_processor import AudioProcessor
    return AudioProcessor

def _get_websocket_manager():
    """
    获取全局 WebSocket 连接管理器。

    注意：app.services.__getattr__ 中已经将 `get_websocket_manager`
    映射为单例 ConnectionManager 实例本身，而不是一个可调用函数，
    所以这里不能再加括号调用，否则会出现
    TypeError: 'ConnectionManager' object is not callable。
    """
    from app.services import get_websocket_manager

    # 直接返回实例本身
    return get_websocket_manager

router = APIRouter()

# 音频实际存储在 upload_dir/audio/projects/ 即 ./static/uploads/audio/projects/
# 静态资源挂载在 /static -> ./static，正确 URL 应为 /static/uploads/audio/projects/{id}/{file}
def _audio_public_url(project_id: str, filename: str) -> str:
    """生成正确的音频公开访问 URL（与实际存储路径一致）"""
    return f"/static/uploads/audio/projects/{project_id}/{filename}"


def _audio_path_to_physical(project_id: str, url_or_filename: str) -> Path:
    """将 URL 或文件名映射为物理路径"""
    name = Path(url_or_filename).name
    return Path(settings.upload_dir) / "audio" / "projects" / project_id / name


def resolve_chunk_audio_path(chunk: ChunkModel, project_id: str) -> Path | None:
    """
    解析 chunk 的 audio_path，尝试多种路径方式。
    
    Args:
        chunk: Chunk 模型对象
        project_id: 项目 ID
        
    Returns:
        找到的音频文件路径，如果找不到则返回 None
    """
    if not chunk.audio_path:
        return None
    
    audio_dir = Path(settings.upload_dir) / "audio" / "projects" / project_id
    
    # 尝试多种路径方式
    candidate_paths = []
    
    # 1. 直接使用 chunk.audio_path（可能是绝对路径）
    candidate_paths.append(Path(chunk.audio_path))
    
    # 2. 从 chunk.id 构造预期路径
    candidate_paths.append(audio_dir / f"{chunk.id}.mp3")
    
    # 3. 尝试从 audio_path 中提取文件名
    if chunk.audio_path:
        candidate_paths.append(audio_dir / Path(chunk.audio_path).name)
    
    # 4. 如果 audio_path 是 URL 格式 /static/uploads/... 或相对路径
    if chunk.audio_path:
        clean_path = chunk.audio_path.lstrip("/")
        if clean_path.startswith("static/"):
            rest = clean_path[7:]  # 移除 "static/"
            candidate_paths.append(Path(settings.upload_dir.parent) / rest)
        elif not Path(chunk.audio_path).is_absolute():
            candidate_paths.append(Path(settings.upload_dir.parent) / clean_path)
    
    # 5. 如果 audio_path 包含项目路径，尝试直接解析
    if chunk.audio_path and project_id in chunk.audio_path:
        file_name = Path(chunk.audio_path).name
        candidate_paths.append(audio_dir / file_name)

    # 6. ChunkProcessor/OptimizedBatchProcessor 保存到 static/audio/chunks/
    if chunk.audio_path:
        name = Path(chunk.audio_path).name
        chunks_dir = Path(settings.upload_dir.parent) / "audio" / "chunks"
        candidate_paths.append(chunks_dir / name)

    # 检查每个候选路径
    for candidate in candidate_paths:
        if candidate.exists() and candidate.is_file():
            return candidate
    
    return None


async def generate_chunk_task(chunk_id: str, project_id: str, db: AsyncSession):
    """Background task for single chunk generation with WebSocket progress."""
    from app.services.tts_engine import TTSEngineFactory, TTSMode

    manager = _get_websocket_manager()
    result = await db.execute(select(ChunkModel).where(ChunkModel.id == chunk_id))
    chunk = result.scalar_one_or_none()

    if chunk:
        try:
            # Send progress update
            await manager.send_progress_update(
                project_id=project_id,
                current=0,
                total=1,
                status=f"Generating audio for chunk: {chunk.text[:30]}..."
            )

            # Generate audio using TTS engine（不再把 "NARRATOR" 当作具体语音 ID，交给引擎自动选择合适中文人声）
            tts_engine = TTSEngineFactory.create(TTSMode.LOCAL)
            audio_data, duration = await tts_engine.generate(
                text=chunk.text,
                speaker="NARRATOR",  # LocalTTSEngine 内部会识别为逻辑标签，自动选择 zh-CN 默认语音
            )

            # Save audio file
            audio_dir = Path(settings.upload_dir) / "audio" / "projects" / project_id
            audio_dir.mkdir(parents=True, exist_ok=True)

            audio_path = audio_dir / f"{chunk.id}.mp3"
            with open(audio_path, "wb") as f:
                f.write(audio_data)

            # Update chunk with actual data
            chunk.status = "completed"
            chunk.duration = duration
            chunk.audio_path = str(audio_path)
            await db.commit()

            # Send completion notification
            await manager.send_chunk_completed(
                project_id=project_id,
                chunk_id=chunk.id,
                duration=duration
            )

            logger.info(f"Successfully generated audio for chunk {chunk_id}")

        except Exception as e:
            chunk.status = "failed"
            await db.commit()
            await manager.send_error(
                project_id=project_id,
                error_type="generation_failed",
                error_message=f"Failed to generate audio for chunk {chunk_id}: {str(e)}"
            )
            logger.error(f"Failed to generate audio for chunk {chunk_id}: {e}")


async def generate_batch_task(project_id: str, db: AsyncSession):
    """Background task for batch audio generation with WebSocket progress."""
    from app.services.tts_engine import TTSEngineFactory, TTSMode

    manager = _get_websocket_manager()

    result = await db.execute(
        select(ChunkModel).where(
            ChunkModel.project_id == project_id,
            ChunkModel.status.in_(["pending", "processing", "failed"]),
        )
    )
    chunks = result.scalars().all()
    total = len(chunks)

    if total == 0:
        await manager.send_progress_update(
            project_id=project_id,
            current=0,
            total=0,
            status="No pending chunks to generate"
        )
        return

    # Create TTS engine
    tts_engine = TTSEngineFactory.create(TTSMode.LOCAL)

    # Prepare audio directory
    audio_dir = Path(settings.upload_dir) / "audio" / "projects" / project_id
    audio_dir.mkdir(parents=True, exist_ok=True)

    completed = 0
    failed = 0

    for i, chunk in enumerate(chunks):
        try:
            # Send progress update
            percentage = (i / total) * 100
            await manager.send_progress_update(
                project_id=project_id,
                current=i,
                total=total,
                status=f"Processing chunk {i + 1}/{total}: {chunk.text[:30]}...",
                chunk_id=chunk.id
            )

            # Generate audio using TTS engine（同样使用逻辑标签，由 LocalTTSEngine 自动选择中文默认语音）
            audio_data, duration = await tts_engine.generate(
                text=chunk.text,
                speaker="NARRATOR",
            )

            # Save audio file
            audio_path = audio_dir / f"{chunk.id}.mp3"
            with open(audio_path, "wb") as f:
                f.write(audio_data)

            # Update chunk with actual data
            chunk.status = "completed"
            chunk.duration = duration
            chunk.audio_path = str(audio_path)
            await db.commit()

            completed += 1

            # Send chunk completion notification
            await manager.send_chunk_completed(
                project_id=project_id,
                chunk_id=chunk.id,
                duration=duration
            )

        except Exception as e:
            chunk.status = "failed"
            await db.commit()
            failed += 1
            logger.error(f"Failed to generate audio for chunk {chunk.id}: {e}")

    # Send final completion notification
    await manager.send_generation_complete(
        project_id=project_id,
        total_chunks=total,
        succeeded=completed,
        failed=failed
    )


@router.get("/{project_id}/chunks", response_model=ApiResponse[PaginatedResponse[Chunk]])
async def get_chunks(
    project_id: str,
    current_user: CurrentUserDep,
    db: DbDep,
    speaker: str | None = None,
    status_filter: str | None = None,
    page: int = 1,
    page_size: int = 20,
):
    """Get project audio chunks."""
    # Verify project exists
    result = await db.execute(select(ProjectModel).where(ProjectModel.id == project_id))
    if not result.scalar_one_or_none():
        raise NotFoundException("Project not found")

    # Build query
    query = select(ChunkModel).where(ChunkModel.project_id == project_id)

    if speaker:
        query = query.where(ChunkModel.speaker == speaker)
    if status_filter:
        query = query.where(ChunkModel.status == status_filter)

    # Count total
    count_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = count_result.scalar() or 0

    # Apply pagination
    query = query.order_by(ChunkModel.order_index.asc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    chunks = result.scalars().all()

    total_pages = (total + page_size - 1) // page_size

    return ApiResponse(
        data=PaginatedResponse(
            items=[ChunkModel.model_validate(c) for c in chunks],
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )
    )


@router.get("/{project_id}/chunks/progress", response_model=ApiResponse[ChunkProgressResponse])
async def get_chunks_progress(
    project_id: str,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Get audio generation progress."""
    # Verify project exists
    result = await db.execute(select(ProjectModel).where(ProjectModel.id == project_id))
    if not result.scalar_one_or_none():
        raise NotFoundException("Project not found")

    # Get counts by status
    total_result = await db.execute(
        select(func.count()).where(ChunkModel.project_id == project_id)
    )
    total = total_result.scalar() or 0

    completed_result = await db.execute(
        select(func.count()).where(
            ChunkModel.project_id == project_id,
            ChunkModel.status == "completed",
        )
    )
    completed = completed_result.scalar() or 0

    processing_result = await db.execute(
        select(func.count()).where(
            ChunkModel.project_id == project_id,
            ChunkModel.status == "processing",
        )
    )
    processing = processing_result.scalar() or 0

    pending = total - completed - processing
    percentage = (completed / total * 100) if total > 0 else 0

    return ApiResponse(
        data=ChunkProgressResponse(
            total=total,
            completed=completed,
            processing=processing,
            pending=pending,
            percentage=percentage,
        )
    )


@router.post("/{project_id}/chunks/{chunk_id}/generate", response_model=ApiResponse[dict])
async def generate_chunk(
    project_id: str,
    chunk_id: str,
    background_tasks: BackgroundTasks,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Generate audio for a single chunk."""
    result = await db.execute(
        select(ChunkModel).where(
            ChunkModel.id == chunk_id,
            ChunkModel.project_id == project_id,
        )
    )
    chunk = result.scalar_one_or_none()

    if not chunk:
        raise NotFoundException("Chunk not found")

    chunk.status = "processing"
    await db.commit()

    background_tasks.add_task(generate_chunk_task, chunk_id, project_id, db)

    return ApiResponse(
        data={
            "chunk_id": chunk_id,
            "status": "processing",
            "message": "Audio generation started",
        }
    )


@router.post("/{project_id}/chunks/generate-batch", response_model=ApiResponse[dict])
async def generate_batch(
    project_id: str,
    request_data: ChunkGenerateBatch,
    background_tasks: BackgroundTasks,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Generate audio for multiple chunks."""
    # Verify chunks exist
    result = await db.execute(
        select(ChunkModel).where(
            ChunkModel.id.in_(request_data.chunk_ids),
            ChunkModel.project_id == project_id,
        )
    )
    chunks = result.scalars().all()

    if not chunks:
        raise NotFoundException("No chunks found")

    # Update status
    for chunk in chunks:
        chunk.status = "processing"
    await db.commit()

    background_tasks.add_task(generate_batch_task, project_id, db)

    return ApiResponse(
        data={
            "task_id": project_id,
            "total_chunks": len(chunks),
            "status": "processing",
        }
    )


@router.post("/{project_id}/chunks/generate-fast", response_model=ApiResponse[dict])
async def generate_fast(
    project_id: str,
    background_tasks: BackgroundTasks,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Generate all pending chunks."""
    # Verify project exists
    result = await db.execute(select(ProjectModel).where(ProjectModel.id == project_id))
    if not result.scalar_one_or_none():
        raise NotFoundException("Project not found")

    result = await db.execute(
        select(ChunkModel).where(
            ChunkModel.project_id == project_id,
            ChunkModel.status == "pending",
        )
    )
    chunks = result.scalars().all()

    total = len(chunks)
    if total == 0:
        return ApiResponse(data={"message": "No pending chunks to generate"})

    # Update status
    for chunk in chunks:
        chunk.status = "processing"
    await db.commit()

    background_tasks.add_task(generate_batch_task, project_id, db)

    return ApiResponse(
        data={
            "task_id": project_id,
            "status": "processing",
            "total_chunks": total,
        }
    )


@router.patch("/{project_id}/chunks/{chunk_id}", response_model=ApiResponse[Chunk])
async def update_chunk(
    project_id: str,
    chunk_id: str,
    chunk_update: ChunkUpdate,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Update chunk data."""
    result = await db.execute(
        select(ChunkModel).where(
            ChunkModel.id == chunk_id,
            ChunkModel.project_id == project_id,
        )
    )
    chunk = result.scalar_one_or_none()

    if not chunk:
        raise NotFoundException("Chunk not found")

    # Update fields
    if chunk_update.text is not None:
        chunk.text = chunk_update.text
    if chunk_update.instruct is not None:
        chunk.instruct = chunk_update.instruct
    if chunk_update.speaker is not None:
        chunk.speaker = chunk_update.speaker

    # Reset status
    chunk.status = "pending"
    chunk.audio_path = None
    chunk.duration = None

    await db.commit()

    return ApiResponse(data=ChunkModel.model_validate(chunk))


@router.post("/{project_id}/audio/merge", response_model=ApiResponse[dict])
async def merge_audio(
    project_id: str,
    options: MergeAudioOptions,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Merge all chunks into final audio."""
    # Verify project exists
    result = await db.execute(select(ProjectModel).where(ProjectModel.id == project_id))
    project = result.scalar_one_or_none()

    if not project:
        raise NotFoundException("Project not found")

    # Check if all chunks are completed
    chunk_result = await db.execute(
        select(ChunkModel).where(
            ChunkModel.project_id == project_id,
            ChunkModel.status != "completed",
        )
    )
    pending = chunk_result.scalars().all()

    if pending:
        return ApiResponse(
            success=False,
            error={
                "code": "PENDING_CHUNKS",
                "message": f"{len(pending)} chunks are not completed yet",
            }
        )

    try:
        # Get all completed chunks in order
        chunk_result = await db.execute(
            select(ChunkModel).where(
                ChunkModel.project_id == project_id,
                ChunkModel.status == "completed",
            ).order_by(ChunkModel.order_index.asc())
        )
        chunks = chunk_result.scalars().all()

        if not chunks:
            return ApiResponse(
                success=False,
                error={
                    "code": "NO_CHUNKS",
                    "message": "No completed chunks to merge",
                }
            )

        # Use AudioProcessor to merge audio files
        AudioProcessor = _get_audio_processor()
        audio_processor = AudioProcessor()

        # 解析并验证所有 chunk 的 audio_path
        chunks_data = []
        for chunk in chunks:
            resolved_path = resolve_chunk_audio_path(chunk, project_id)
            if resolved_path:
                chunks_data.append({
                    "id": chunk.id,
                    "audio_path": str(resolved_path),  # 使用解析后的路径
                    "speaker": chunk.speaker,
                    "text": chunk.text,
                    "order_index": chunk.order_index,
                })
            else:
                logger.warning(
                    f"Could not resolve audio path for chunk {chunk.id} with audio_path={chunk.audio_path}"
                )

        if not chunks_data:
            return ApiResponse(
                success=False,
                error={
                    "code": "NO_VALID_AUDIO",
                    "message": "No valid audio files found for chunks",
                }
            )

        # Merge audio files
        export_dir = settings.export_dir
        output_dir = Path(settings.upload_dir) / "audio" / "projects" / project_id
        output_dir.mkdir(parents=True, exist_ok=True)

        merged_path, total_duration = await audio_processor.merge_audio_files(
            chunks_data,
            str(output_dir),
            f"{project_id}_merged",
            add_pause_ms=options.pause_ms,
            normalize=options.normalize,
            add_fades=options.add_fades,
        )

        # Update project with merged audio info
        project.status = "completed"
        project.audio_path = _audio_public_url(project_id, Path(merged_path).name)
        project.duration = total_duration
        await db.commit()

        logger.info(f"Successfully merged {len(chunks)} chunks into {merged_path}")

        return ApiResponse(
            data={
                "status": "completed",
                "audio_url": project.audio_path,
                "duration": total_duration,
                "chunks_count": len(chunks),
                "message": "Audio merge completed successfully",
            }
        )

    except Exception as e:
        logger.error(f"Audio merge failed: {e}")
        return ApiResponse(
            success=False,
            error={
                "code": "MERGE_FAILED",
                "message": f"Audio merge failed: {str(e)}",
            }
        )


@router.get("/{project_id}/audio", response_model=ApiResponse[AudioResponse])
async def get_audio(
    project_id: str,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Get merged audio file info."""
    result = await db.execute(select(ProjectModel).where(ProjectModel.id == project_id))
    project = result.scalar_one_or_none()

    if not project:
        raise NotFoundException("Project not found")

    audio_base = Path(settings.upload_dir) / "audio" / "projects" / project_id
    logger.info(
        "[get_audio] project_id=%s audio_path=%s upload_dir=%s audio_base=%s exists=%s",
        project_id, project.audio_path, settings.upload_dir, audio_base, audio_base.exists(),
    )

    # 1）优先使用数据库中已有的 audio_path（并校验文件是否真实存在）
    if project.audio_path:
        try:
            file_path = _audio_path_to_physical(project_id, project.audio_path)
            if file_path.exists():
                return ApiResponse(
                    data=AudioResponse(
                        audio_url=_audio_public_url(project_id, file_path.name),
                        duration=project.duration or 0,
                        file_size=None,
                        format="mp3",
                        bitrate="128k",
                    )
                )
            else:
                # 数据库存了路径但文件不存在，清空后走后续自动修复逻辑
                logger.warning(
                    "Project %s has audio_path %s but file not found at %s, will try to auto-recover",
                    project_id,
                    project.audio_path,
                    str(file_path),
                )
                project.audio_path = None
                await db.commit()
        except Exception as path_err:
            logger.error(
                "Failed to validate existing project.audio_path for project %s: %s",
                project_id,
                path_err,
            )

    # 2）如果数据库没有 audio_path，尝试直接从磁盘扫描该项目目录下的 mp3 文件
    #    这可以修复：数据库状态异常 / 旧数据迁移 等导致 audio_path 为空，但物理文件仍然存在的情况
    try:
        audio_dir = Path(settings.upload_dir) / "audio" / "projects" / project_id
        if audio_dir.exists():
            mp3_files = sorted(audio_dir.glob("*.mp3"))
            if mp3_files:
                # 优先选择名字中包含 "merged" 的整本有声书，其次任意一条 mp3 作为兜底
                merged_files = [p for p in mp3_files if "merged" in p.name]
                chosen = merged_files[0] if merged_files else mp3_files[0]

                public_url = _audio_public_url(project_id, chosen.name)

                # 尝试读取真实时长，如果失败则退回 0
                duration_value: float = 0.0
                try:
                    AudioProcessor = _get_audio_processor()
                    audio_processor = AudioProcessor()
                    info = await audio_processor.get_audio_info(str(chosen))
                    duration_value = float(info.get("duration", 0.0) or 0.0)
                except Exception as info_err:
                    logger.warning(
                        "Failed to read audio info for %s: %s",
                        str(chosen),
                        info_err,
                    )

                # 回填到项目，避免每次都重新扫描
                project.audio_path = public_url
                project.duration = duration_value or project.duration
                if not project.status or project.status == "draft":
                    project.status = "completed"
                await db.commit()

                return ApiResponse(
                    data=AudioResponse(
                        audio_url=public_url,
                        duration=duration_value or project.duration or 0,
                        file_size=None,
                        format="mp3",
                        bitrate="128k",
                    )
                )
    except Exception as scan_err:
        logger.error(
            "Failed to auto-discover audio files for project %s: %s",
            project_id,
            scan_err,
        )

    # 否则尝试基于已完成的 chunk 即时进行一次合并，避免用户“明明生成了音频却提示未生成”的困惑
    chunk_result = await db.execute(
        select(ChunkModel).where(
            ChunkModel.project_id == project_id,
            ChunkModel.status == "completed",
            ChunkModel.audio_path.isnot(None),
        ).order_by(ChunkModel.order_index.asc())
    )
    chunks = chunk_result.scalars().all()

    if not chunks:
        # 有些老数据或异常情况下，chunk 的 status 可能不是 "completed"，
        # 但 audio_path 已经写入磁盘，此时仍然应该允许播放。
        fallback_chunk_result = await db.execute(
            select(ChunkModel).where(
                ChunkModel.project_id == project_id,
                ChunkModel.audio_path.isnot(None),
            ).order_by(ChunkModel.order_index.asc())
        )
        chunks = fallback_chunk_result.scalars().all()

    if not chunks:
        # 既没有合并音频，也找不到任何带 audio_path 的 chunk，尝试直接扫描磁盘
        # 这可以处理 audio_path 字段为空但文件仍然存在的情况
        try:
            audio_dir = Path(settings.upload_dir) / "audio" / "projects" / project_id
            if audio_dir.exists():
                mp3_files = sorted(audio_dir.glob("*.mp3"))
                if mp3_files:
                    # 找到磁盘上的音频文件，但数据库中没有对应的 chunk 或 chunk.audio_path 为空
                    # 这种情况下仍然应该允许播放
                    chosen = mp3_files[0]
                    public_url = _audio_public_url(project_id, chosen.name)

                    # 尝试读取时长
                    duration_value: float = 0.0
                    try:
                        AudioProcessor = _get_audio_processor()
                        audio_processor = AudioProcessor()
                        info = await audio_processor.get_audio_info(str(chosen))
                        duration_value = float(info.get("duration", 0.0) or 0.0)
                    except Exception:
                        pass

                    # 更新项目状态
                    project.audio_path = public_url
                    project.duration = duration_value
                    if not project.status or project.status == "draft":
                        project.status = "completed"
                    await db.commit()

                    return ApiResponse(
                        data=AudioResponse(
                            audio_url=public_url,
                            duration=duration_value,
                            file_size=None,
                            format="mp3",
                            bitrate="128k",
                        )
                    )
        except Exception as scan_err:
            logger.error(
                "Failed to scan disk for audio files for project %s: %s",
                project_id,
                scan_err,
            )

    if not chunks:
        # 既没有合并音频，也找不到任何带音频文件的 chunk，且磁盘扫描也没发现 mp3
        mp3_count = 0
        if audio_base.exists():
            mp3_count = len(list(audio_base.glob("*.mp3")))
        logger.warning(
            "[get_audio] AUDIO_NOT_READY: project_id=%s chunks=0 audio_dir=%s mp3_on_disk=%d",
            project_id, audio_base, mp3_count,
        )
        return ApiResponse(
            success=False,
            error={
                "code": "AUDIO_NOT_READY",
                "message": "Audio not generated yet",
                "details": {
                    "audio_dir": str(audio_base),
                    "exists": audio_base.exists(),
                    "mp3_count": mp3_count,
                },
            }
        )

    # 使用与 /audio/merge 相同的逻辑即时合并一次，并缓存到 project.audio_path
    try:
        AudioProcessor = _get_audio_processor()
        audio_processor = AudioProcessor()

        # 解析并验证所有 chunk 的 audio_path
        chunks_data = []
        for chunk in chunks:
            resolved_path = resolve_chunk_audio_path(chunk, project_id)
            if resolved_path:
                chunks_data.append({
                    "id": chunk.id,
                    "audio_path": str(resolved_path),  # 使用解析后的路径
                    "speaker": chunk.speaker,
                    "text": chunk.text,
                    "order_index": chunk.order_index,
                })
            else:
                logger.warning(
                    f"Could not resolve audio path for chunk {chunk.id} with audio_path={chunk.audio_path}"
                )

        if not chunks_data:
            # 如果所有 chunk 的路径都无法解析，尝试 fallback 逻辑
            raise ValueError("No valid audio files found after path resolution")

        output_dir = Path(settings.upload_dir) / "audio" / "projects" / project_id
        output_dir.mkdir(parents=True, exist_ok=True)

        merged_path, total_duration = await audio_processor.merge_audio_files(
            chunks_data,
            str(output_dir),
            f"{project_id}_merged",
            add_pause_ms=500,
            normalize=True,
            add_fades=True,
        )

        project.audio_path = _audio_public_url(project_id, Path(merged_path).name)
        project.duration = total_duration
        project.status = "completed"
        await db.commit()

        return ApiResponse(
            data=AudioResponse(
                audio_url=project.audio_path,
                duration=total_duration,
                file_size=None,
                format="mp3",
                bitrate="128k",
            )
        )
    except Exception as e:
        # 合并失败时的兜底逻辑：
        # 很多用户在“有声书项目”页面已经看到每一段都生成了音频，但由于服务器缺少 ffmpeg /
        # pydub 依赖或个别文件损坏，整本书的合并会报错，导致这里始终返回 AUDIO_NOT_READY，
        # 在阅读页完全听不到声音。
        logger.error(f"On-demand audio merge failed for project {project_id}: {e}")

        # Fallback 1：尽量返回至少一段可播放的音频（通常是第一段），保证阅读页"有声音"
        try:
            first_chunk_file: Path | None = None
            first_chunk_duration: float | None = None

            for chunk in chunks:
                # 使用统一的路径解析函数
                resolved_path = resolve_chunk_audio_path(chunk, project_id)
                if resolved_path:
                    logger.info(f"Found audio file for chunk {chunk.id} at: {resolved_path}")
                    first_chunk_file = resolved_path
                    first_chunk_duration = chunk.duration or 0
                    break

            if first_chunk_file is not None:
                # 构造静态访问 URL（与正常合并后的路径保持一致的前缀）
                relative_name = first_chunk_file.name
                public_url = _audio_public_url(project_id, relative_name)

                # 把兜底音频也缓存到 project 上，避免下次重复计算
                project.audio_path = public_url
                project.duration = first_chunk_duration
                project.status = project.status or "completed"
                await db.commit()

                logger.info(f"Fallback: using single chunk audio for project {project_id}: {public_url}")

                return ApiResponse(
                    data=AudioResponse(
                        audio_url=public_url,
                        duration=first_chunk_duration or 0,
                        file_size=None,
                        format="mp3",
                        bitrate="128k",
                    )
                )
        except Exception as fallback_err:
            # 兜底逻辑自身失败时，不再继续尝试，回退为原有的 NOT_READY 错误
            logger.error(
                f"Fallback to first chunk audio failed for project {project_id}: {fallback_err}"
            )

        # Fallback 2：保持原有行为，提示音频尚未准备好（包含详细错误信息便于排查）
        return ApiResponse(
            success=False,
            error={
                "code": "AUDIO_NOT_READY",
                "message": "Audio not generated yet",
                "details": str(e),
            }
        )


@router.post("/{project_id}/audio/export", response_model=ApiResponse[dict])
async def export_audio(
    project_id: str,
    current_user: CurrentUserDep,
    db: DbDep,
    export_format: Literal["combined", "audacity", "voicelines"] = Query(..., description="Export format"),
    project_name: str | None = Query(None, description="Custom project name"),
    add_fades: bool = Query(True, description="Add fade in/out (for combined format)"),
    normalize: bool = Query(True, description="Normalize volume (for combined format)"),
):
    """Export audio in various professional formats.

    Formats:
    - combined: Single merged MP3 file with pauses
    - audacity: ZIP with individual files + Audacity project (.lof) + labels
    - voicelines: ZIP with individual numbered files + CSV manifest
    """
    # Verify project exists
    result = await db.execute(select(ProjectModel).where(ProjectModel.id == project_id))
    project = result.scalar_one_or_none()

    if not project:
        raise NotFoundException("Project not found")

    # Get all chunks with audio
    chunk_result = await db.execute(
        select(ChunkModel).where(
            ChunkModel.project_id == project_id,
            ChunkModel.status == "completed",
            ChunkModel.audio_path.isnot(None),
        ).order_by(ChunkModel.order_index.asc())
    )
    chunks = chunk_result.scalars().all()

    if not chunks:
        return ApiResponse(
            success=False,
            error={
                "code": "NO_AUDIO",
                "message": "No completed audio chunks found",
            }
        )

    # Prepare chunks data
    chunks_data = [
        {
            "id": chunk.id,
            "audio_path": chunk.audio_path,
            "speaker": chunk.speaker,
            "text": chunk.text,
            "order_index": chunk.order_index,
        }
        for chunk in chunks
    ]

    # Initialize audio processor
    AudioProcessor = _get_audio_processor()
    audio_processor = AudioProcessor()
    export_dir = settings.export_dir
    final_project_name = project_name or f"project_{project_id}"

    try:
        if export_format == "combined":
            audio_path, duration = await audio_processor.export_combined_audiobook(
                chunks_data,
                str(export_dir),
                final_project_name,
                add_fades=add_fades,
                normalize=normalize,
            )

            # Update project
            project.audio_path = f"/static/exports/{Path(audio_path).name}"
            project.duration = duration
            project.status = "completed"
            await db.commit()

            return ApiResponse(
                data={
                    "format": "combined",
                    "audio_url": project.audio_path,
                    "duration": duration,
                    "message": "Combined audiobook exported successfully",
                }
            )

        elif export_format == "audacity":
            zip_path = await audio_processor.export_audacity_project(
                chunks_data,
                str(export_dir),
                final_project_name,
            )

            return ApiResponse(
                data={
                    "format": "audacity",
                    "download_url": f"/static/exports/{Path(zip_path).name}",
                    "filename": Path(zip_path).name,
                    "chunks_count": len(chunks_data),
                    "message": "Audacity project exported successfully",
                }
            )

        elif export_format == "voicelines":
            zip_path = await audio_processor.export_individual_voicelines(
                chunks_data,
                str(export_dir),
                final_project_name,
            )

            return ApiResponse(
                data={
                    "format": "voicelines",
                    "download_url": f"/static/exports/{Path(zip_path).name}",
                    "filename": Path(zip_path).name,
                    "chunks_count": len(chunks_data),
                    "message": "Individual voicelines exported successfully",
                }
            )

    except Exception as e:
        return ApiResponse(
            success=False,
            error={
                "code": "EXPORT_FAILED",
                "message": f"Export failed: {str(e)}",
            }
        )


@router.get("/{project_id}/audio/download/{filename}", response_class=FileResponse)
async def download_exported_file(
    project_id: str,
    filename: str,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Download exported audio file."""
    # Verify project exists
    result = await db.execute(select(ProjectModel).where(ProjectModel.id == project_id))
    project = result.scalar_one_or_none()

    if not project:
        raise NotFoundException("Project not found")

    # Verify file exists and belongs to project
    file_path = settings.export_dir / filename

    if not file_path.exists():
        raise NotFoundException("File not found")

    # Security check: ensure file belongs to this project
    if not (f"project_{project_id}" in filename or project_id in filename):
        raise NotFoundException("File not found")

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/zip" if filename.endswith(".zip") else "audio/mpeg",
    )


@router.post("/{project_id}/chunks/retry-failed", response_model=ApiResponse[dict])
async def retry_failed_chunks(
    project_id: str,
    background_tasks: BackgroundTasks,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """
    Retry generation for all failed chunks.

    Useful for recovering from transient errors without regenerating everything.
    """
    # Verify project exists
    result = await db.execute(select(ProjectModel).where(ProjectModel.id == project_id))
    if not result.scalar_one_or_none():
        raise NotFoundException("Project not found")

    # Get all failed chunks
    result = await db.execute(
        select(ChunkModel).where(
            ChunkModel.project_id == project_id,
            ChunkModel.status == "failed",
        )
    )
    failed_chunks = result.scalars().all()

    if not failed_chunks:
        return ApiResponse(
            data={
                "message": "No failed chunks found",
                "retried_count": 0,
            }
        )

    # Reset status and queue for regeneration
    for chunk in failed_chunks:
        chunk.status = "pending"
    await db.commit()

    # Start batch generation
    background_tasks.add_task(generate_batch_task, project_id, db)

    logger.info(f"Retrying {len(failed_chunks)} failed chunks for project {project_id}")

    return ApiResponse(
        data={
            "task_id": project_id,
            "status": "processing",
            "retried_count": len(failed_chunks),
            "message": f"Retrying {len(failed_chunks)} failed chunks",
        }
    )


@router.get("/{project_id}/chunks/failed", response_model=ApiResponse[list[Chunk]])
async def get_failed_chunks(
    project_id: str,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Get all failed chunks for a project."""
    # Verify project exists
    result = await db.execute(select(ProjectModel).where(ProjectModel.id == project_id))
    if not result.scalar_one_or_none():
        raise NotFoundException("Project not found")

    # Get failed chunks
    result = await db.execute(
        select(ChunkModel)
        .where(
            ChunkModel.project_id == project_id,
            ChunkModel.status == "failed",
        )
        .order_by(ChunkModel.order_index.asc())
    )
    failed_chunks = result.scalars().all()

    return ApiResponse(
        data=[ChunkModel.model_validate(c) for c in failed_chunks]
    )


@router.post("/{project_id}/chunks/{chunk_id}/retry", response_model=ApiResponse[dict])
async def retry_single_chunk(
    project_id: str,
    chunk_id: str,
    background_tasks: BackgroundTasks,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Retry generation for a single failed chunk."""
    result = await db.execute(
        select(ChunkModel).where(
            ChunkModel.id == chunk_id,
            ChunkModel.project_id == project_id,
        )
    )
    chunk = result.scalar_one_or_none()

    if not chunk:
        raise NotFoundException("Chunk not found")

    # Reset status
    chunk.status = "pending"
    await db.commit()

    # Start generation
    background_tasks.add_task(generate_chunk_task, chunk_id, project_id, db)

    logger.info(f"Retrying chunk {chunk_id} for project {project_id}")

    return ApiResponse(
        data={
            "chunk_id": chunk_id,
            "status": "processing",
            "message": "Chunk retry started",
        }
    )

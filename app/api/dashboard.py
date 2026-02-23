"""
Dashboard API Routes
Provides overview data for the voice studio hub dashboard.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter

from app.schemas.audio_tools import DashboardData, DashboardTask
from app.schemas.common import ApiResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage for demo purposes
# In production, this would be stored in a database
_demo_tasks: list[dict[str, Any]] = [
    {
        "id": "1",
        "name": "会议录音.wav",
        "type": "语音识别",
        "status": "done",
        "time": "5分钟前",
        "created_at": datetime.now() - timedelta(minutes=5),
    },
    {
        "id": "2",
        "name": "采访音频.mp3",
        "type": "说话人分离",
        "status": "done",
        "time": "12分钟前",
        "created_at": datetime.now() - timedelta(minutes=12),
    },
    {
        "id": "3",
        "name": "播客.flac",
        "type": "音频质量检测",
        "status": "processing",
        "time": "正在处理",
        "created_at": datetime.now() - timedelta(minutes=2),
    },
    {
        "id": "4",
        "name": "歌曲片段.wav",
        "type": "RVC 转换",
        "status": "error",
        "time": "1小时前",
        "created_at": datetime.now() - timedelta(hours=1),
    },
]


def _format_time(task: dict[str, Any]) -> str:
    """Format task time as relative string."""
    if task["status"] == "processing":
        return "正在处理"

    created_at: datetime = task["created_at"]
    now = datetime.now()
    diff = now - created_at

    if diff.seconds < 60:
        return f"{diff.seconds}秒前"
    elif diff.seconds < 3600:
        return f"{diff.seconds // 60}分钟前"
    elif diff.seconds < 86400:
        return f"{diff.seconds // 3600}小时前"
    else:
        return f"{diff.days}天前"


@router.get("/tasks", response_model=ApiResponse[DashboardData])
async def get_dashboard_tasks(
    limit: int = 10,
    status: str | None = None,
):
    """
    Get recent tasks for dashboard display.

    Args:
        limit: Maximum number of tasks to return (default: 10)
        status: Filter by status (done, processing, error)

    Returns dashboard data with recent tasks.
    """
    try:
        # Filter tasks by status if specified
        filtered_tasks = _demo_tasks
        if status:
            filtered_tasks = [t for t in _demo_tasks if t["status"] == status]

        # Sort by created_at (newest first) and limit
        sorted_tasks = sorted(
            filtered_tasks,
            key=lambda x: x["created_at"],
            reverse=True,
        )[:limit]

        # Format tasks
        dashboard_tasks = [
            DashboardTask(
                id=task["id"],
                name=task["name"],
                type=task["type"],
                status=task["status"],
                time=_format_time(task),
            )
            for task in sorted_tasks
        ]

        # Calculate stats
        stats = {
            "total": len(_demo_tasks),
            "done": len([t for t in _demo_tasks if t["status"] == "done"]),
            "processing": len([t for t in _demo_tasks if t["status"] == "processing"]),
            "error": len([t for t in _demo_tasks if t["status"] == "error"]),
        }

        data = DashboardData(
            recent_tasks=dashboard_tasks,
            stats=stats,
        )

        return ApiResponse(data=data)

    except Exception as e:
        logger.error(f"Get dashboard tasks error: {e}")
        raise


@router.post("/tasks", response_model=ApiResponse[DashboardTask])
async def create_dashboard_task(
    name: str,
    task_type: str,
    status: str = "processing",
):
    """
    Create a new dashboard task entry.

    Useful for tracking when a user starts a new audio processing task.
    """
    try:
        import uuid

        new_task = {
            "id": str(uuid.uuid4()),
            "name": name,
            "type": task_type,
            "status": status,
            "time": "正在处理",
            "created_at": datetime.now(),
        }

        _demo_tasks.append(new_task)

        dashboard_task = DashboardTask(
            id=new_task["id"],
            name=new_task["name"],
            type=new_task["type"],
            status=new_task["status"],
            time=new_task["time"],
        )

        return ApiResponse(data=dashboard_task)

    except Exception as e:
        logger.error(f"Create dashboard task error: {e}")
        raise


@router.put("/tasks/{task_id}", response_model=ApiResponse[DashboardTask])
async def update_dashboard_task(
    task_id: str,
    status: str | None = None,
):
    """
    Update a task's status.

    Called when a task completes or fails.
    """
    try:
        task = next((t for t in _demo_tasks if t["id"] == task_id), None)

        if not task:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

        if status:
            task["status"] = status
            task["time"] = _format_time(task)

        dashboard_task = DashboardTask(
            id=task["id"],
            name=task["name"],
            type=task["type"],
            status=task["status"],
            time=task["time"],
        )

        return ApiResponse(data=dashboard_task)

    except Exception as e:
        logger.error(f"Update dashboard task error: {e}")
        raise


@router.delete("/tasks/{task_id}", response_model=ApiResponse[dict])
async def delete_dashboard_task(task_id: str):
    """
    Delete a task from the dashboard.
    """
    try:
        global _demo_tasks
        initial_count = len(_demo_tasks)
        _demo_tasks = [t for t in _demo_tasks if t["id"] != task_id]

        if len(_demo_tasks) == initial_count:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

        return ApiResponse(data={"message": f"Task {task_id} deleted successfully"})

    except Exception as e:
        logger.error(f"Delete dashboard task error: {e}")
        raise


@router.get("/stats", response_model=ApiResponse[dict])
async def get_dashboard_stats():
    """
    Get overall dashboard statistics.
    """
    try:
        stats = {
            "tasks": {
                "total": len(_demo_tasks),
                "done": len([t for t in _demo_tasks if t["status"] == "done"]),
                "processing": len([t for t in _demo_tasks if t["status"] == "processing"]),
                "error": len([t for t in _demo_tasks if t["status"] == "error"]),
            },
            "features": {
                "audio_quality": {"available": True, "count": 0},
                "asr": {"available": True, "count": 0},
                "diarization": {"available": True, "count": 0},
                "rvc": {"available": True, "count": 0},
                "dialect": {"available": True, "count": 0},
                "quantization": {"available": True, "count": 0},
            },
        }

        return ApiResponse(data=stats)

    except Exception as e:
        logger.error(f"Get dashboard stats error: {e}")
        raise

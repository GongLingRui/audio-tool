"""
WebSocket Manager for Real-time Progress Updates
Handles WebSocket connections for audio generation progress
"""

import asyncio
import json
import logging
from typing import Dict, Set, Optional
from datetime import datetime

from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for progress updates."""

    def __init__(self):
        # Project ID -> Set of WebSocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # WebSocket -> Project ID mapping
        self.connection_to_project: Dict[WebSocket, str] = {}

    async def connect(self, websocket: WebSocket, project_id: str):
        """Connect a WebSocket to a project."""
        await websocket.accept()

        if project_id not in self.active_connections:
            self.active_connections[project_id] = set()

        self.active_connections[project_id].add(websocket)
        self.connection_to_project[websocket] = project_id

        logger.info(f"WebSocket connected for project {project_id}")

        # Send welcome message
        await self.send_personal_message({
            "type": "connected",
            "project_id": project_id,
            "timestamp": datetime.now().isoformat(),
            "message": "已连接到实时进度更新"
        }, websocket)

    def disconnect(self, websocket: WebSocket):
        """Disconnect a WebSocket."""
        project_id = self.connection_to_project.pop(websocket, None)
        if project_id and project_id in self.active_connections:
            self.active_connections[project_id].discard(websocket)
            if not self.active_connections[project_id]:
                del self.active_connections[project_id]
        logger.info(f"WebSocket disconnected from project {project_id}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific WebSocket."""
        try:
            await websocket.send_text(json.dumps(message, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.disconnect(websocket)

    async def broadcast_to_project(self, message: dict, project_id: str):
        """Broadcast a message to all connections for a project."""
        if project_id not in self.active_connections:
            return

        disconnected = set()
        for connection in self.active_connections[project_id]:
            try:
                await connection.send_text(json.dumps(message, ensure_ascii=False))
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.add(connection)

        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

    async def send_progress_update(
        self,
        project_id: str,
        current: int,
        total: int,
        status: str,
        chunk_id: Optional[str] = None,
        error: Optional[str] = None,
        eta_seconds: Optional[float] = None,
    ):
        """Send a progress update to all connections for a project."""
        message = {
            "type": "progress",
            "project_id": project_id,
            "timestamp": datetime.now().isoformat(),
            "data": {
                "current": current,
                "total": total,
                "percentage": round((current / total * 100) if total > 0 else 0, 1),
                "status": status,
                "chunk_id": chunk_id,
                "error": error,
                "eta_seconds": eta_seconds,
            }
        }
        await self.broadcast_to_project(message, project_id)

    async def send_chunk_completed(
        self,
        project_id: str,
        chunk_id: str,
        duration: float,
        status: str = "completed",
    ):
        """Send a chunk completion notification."""
        message = {
            "type": "chunk_completed",
            "project_id": project_id,
            "timestamp": datetime.now().isoformat(),
            "data": {
                "chunk_id": chunk_id,
                "duration": duration,
                "status": status,
            }
        }
        await self.broadcast_to_project(message, project_id)

    async def send_generation_complete(
        self,
        project_id: str,
        total_chunks: int,
        succeeded: int,
        failed: int,
        total_duration: float,
    ):
        """Send generation complete notification."""
        message = {
            "type": "generation_complete",
            "project_id": project_id,
            "timestamp": datetime.now().isoformat(),
            "data": {
                "total_chunks": total_chunks,
                "succeeded": succeeded,
                "failed": failed,
                "total_duration": total_duration,
                "success_rate": round((succeeded / total_chunks * 100) if total_chunks > 0 else 0, 1),
            }
        }
        await self.broadcast_to_project(message, project_id)

    async def send_error(
        self,
        project_id: str,
        error_type: str,
        error_message: str,
        chunk_id: Optional[str] = None,
    ):
        """Send an error notification."""
        message = {
            "type": "error",
            "project_id": project_id,
            "timestamp": datetime.now().isoformat(),
            "data": {
                "error_type": error_type,
                "error_message": error_message,
                "chunk_id": chunk_id,
            }
        }
        await self.broadcast_to_project(message, project_id)

    def get_connection_count(self, project_id: str) -> int:
        """Get the number of active connections for a project."""
        return len(self.active_connections.get(project_id, set()))


# Global connection manager instance
manager = ConnectionManager()

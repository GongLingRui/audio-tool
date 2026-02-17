"""
WebSocket API Routes for Real-time Progress Updates
"""

import asyncio
import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.services.websocket_manager import manager
from app.core.deps import DbDep
from app.models.project import Project
from app.models.chunk import Chunk

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/projects/{project_id}/progress")
async def websocket_project_progress(
    websocket: WebSocket,
    project_id: str,
    token: Optional[str] = Query(None),
):
    """
    WebSocket endpoint for real-time project progress updates.

    Connects to receive live updates during audio generation:
    - Progress updates with current/total counts
    - Chunk completion notifications
    - Error notifications
    - Generation completion summary

    Query Parameters:
        token: Optional authentication token (if not using cookie auth)
    """
    await manager.connect(websocket, project_id)

    try:
        # Verify project exists
        # Note: In production, verify user owns this project
        while True:
            # Keep connection alive and handle incoming messages
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Echo back or handle client messages
                if data == "ping":
                    await websocket.send_text('{"type": "pong"}')
            except asyncio.TimeoutError:
                # Send heartbeat
                try:
                    await websocket.send_text('{"type": "heartbeat"}')
                except:
                    break

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"WebSocket disconnected from project {project_id}")
    except Exception as e:
        logger.error(f"WebSocket error for project {project_id}: {e}")
        manager.disconnect(websocket)


@router.get("/projects/{project_id}/ws-status")
async def get_websocket_status(project_id: str):
    """
    Get WebSocket connection status for a project.

    Returns the number of active WebSocket connections for real-time updates.
    """
    connection_count = manager.get_connection_count(project_id)

    return {
        "project_id": project_id,
        "websocket_enabled": True,
        "active_connections": connection_count,
        "status": "available" if connection_count > 0 else "no_connections"
    }


# =============================================================================
# Real-time Voice Chat - 实时语音对话
# =============================================================================

@router.websocket("/ws/voice-chat")
async def websocket_voice_chat(
    websocket: WebSocket,
    voice_id: str = Query("aiden", description="Voice ID for TTS"),
    language: str = Query("zh", description="Conversation language"),
):
    """
    Real-time voice chat WebSocket endpoint (实时语音对话).

    Supports full-duplex voice conversation:
    1. Client sends audio/text messages
    2. Server processes with LLM
    3. Server responds with audio stream

    Message Types:
    - "audio": Base64 encoded audio for speech-to-text
    - "text": Text message directly
    - "config": Update voice/chat settings
    - "ping": Keep-alive ping

    Response Types:
    - "text": LLM text response
    - "audio": TTS audio response (base64 or chunked)
    - "error": Error message
    - "pong": Keep-alive response

    Query Parameters:
        voice_id: Voice ID for text-to-speech
        language: Conversation language (zh, en, etc.)
    """
    await websocket.accept()

    # Conversation history
    conversation_history = []

    try:
        while True:
            # Receive message from client
            data = await asyncio.wait_for(websocket.receive_json(), timeout=60.0)

            message_type = data.get("type")
            message_content = data.get("content", {})
            message_id = data.get("id", "")

            # Handle ping/pong
            if message_type == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "id": message_id,
                })
                continue

            # Handle config updates
            if message_type == "config":
                voice_id = message_content.get("voice_id", voice_id)
                language = message_content.get("language", language)
                await websocket.send_json({
                    "type": "config_ack",
                    "id": message_id,
                    "voice_id": voice_id,
                    "language": language,
                })
                continue

            # Handle text messages
            if message_type == "text":
                user_text = message_content.get("text", "")

                if not user_text:
                    await websocket.send_json({
                        "type": "error",
                        "id": message_id,
                        "error": "Empty text message",
                    })
                    continue

                try:
                    # Add to conversation history
                    conversation_history.append({
                        "role": "user",
                        "content": user_text,
                    })

                    # Get LLM response
                    from app.utils.llm import call_llm

                    system_prompt = f"You are a helpful AI assistant having a voice conversation in {language}. Keep responses concise and natural for speech."
                    messages = [{"role": "system", "content": system_prompt}] + conversation_history[-10:]  # Keep last 10 messages

                    assistant_text = await call_llm(messages, temperature=0.7)

                    # Add to history
                    conversation_history.append({
                        "role": "assistant",
                        "content": assistant_text,
                    })

                    # Send text response first
                    await websocket.send_json({
                        "type": "text",
                        "id": message_id,
                        "text": assistant_text,
                    })

                    # Generate TTS audio
                    from app.services.tts_engine import TTSEngineFactory, TTSMode

                    tts_engine = TTSEngineFactory.create(TTSMode.LOCAL)
                    audio_data, duration = await tts_engine.generate(
                        text=assistant_text,
                        speaker=voice_id,
                    )

                    # Send audio as base64
                    import base64
                    audio_base64 = base64.b64encode(audio_data).decode("utf-8")

                    await websocket.send_json({
                        "type": "audio",
                        "id": message_id,
                        "audio": audio_base64,
                        "format": "mp3",
                        "duration": duration,
                    })

                except Exception as e:
                    logger.error(f"Voice chat error: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "id": message_id,
                        "error": str(e),
                    })
                    continue

            # Handle audio messages (speech-to-text)
            elif message_type == "audio":
                audio_base64 = message_content.get("audio", "")

                if not audio_base64:
                    await websocket.send_json({
                        "type": "error",
                        "id": message_id,
                        "error": "Empty audio message",
                    })
                    continue

                try:
                    # Decode audio
                    import base64
                    import io
                    from pydub import AudioSegment
                    import tempfile

                    audio_bytes = base64.b64decode(audio_base64)

                    # Save to temp file for STT
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        temp_file.write(audio_bytes)
                        temp_path = temp_file.name

                    # Speech-to-text using LLM whisper or similar
                    # For now, return a placeholder response
                    await websocket.send_json({
                        "type": "text",
                        "id": message_id,
                        "text": "语音识别功能正在开发中。请使用文字输入。",  # STT in development
                    })

                    import os
                    os.unlink(temp_path)

                except Exception as e:
                    logger.error(f"Audio processing error: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "id": message_id,
                        "error": str(e),
                    })

            else:
                await websocket.send_json({
                    "type": "error",
                    "id": message_id,
                    "error": f"Unknown message type: {message_type}",
                })

    except asyncio.TimeoutError:
        logger.info("Voice chat WebSocket timeout")
        await websocket.send_json({
            "type": "error",
            "error": "Connection timeout",
        })
    except WebSocketDisconnect:
        logger.info("Voice chat WebSocket disconnected")
    except Exception as e:
        logger.error(f"Voice chat WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e),
            })
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


# =============================================================================
# Streaming TTS - 流式TTS音频推送
# =============================================================================

@router.websocket("/ws/streaming-tts")
async def websocket_streaming_tts(
    websocket: WebSocket,
    voice_id: str = Query("aiden", description="Voice ID for TTS"),
    chunk_size: int = Query(50, description="Characters per chunk"),
):
    """
    Streaming TTS WebSocket endpoint (流式TTS音频推送).

    Generates audio in chunks and streams them as they're produced.
    Supports real-time playback with minimal latency.

    Message Types:
    - "generate": Start TTS generation
        {
            "type": "generate",
            "id": "msg_id",
            "text": "Text to synthesize",
            "emotion": {"neutral": 0.8, "energy": 1.0},
            "speed": 1.0
        }
    - "cancel": Cancel current generation
    - "ping": Keep-alive
    - "config": Update TTS settings

    Response Types:
    - "chunk": Audio chunk (base64 encoded)
        {
            "type": "chunk",
            "id": "msg_id",
            "chunk_index": 0,
            "total_chunks": 5,
            "audio": "base64_audio_data",
            "text": "Text for this chunk",
            "is_final": false
        }
    - "complete": Generation complete
    - "error": Error message
    - "pong": Keep-alive response

    Query Parameters:
        voice_id: Voice ID for text-to-speech
        chunk_size: Characters per audio chunk
    """
    await websocket.accept()

    # Generation state
    current_generation = None
    cancel_requested = False

    try:
        while True:
            # Receive message
            data = await asyncio.wait_for(websocket.receive_json(), timeout=60.0)

            message_type = data.get("type")
            message_content = data.get("content", {})
            message_id = data.get("id", "")

            # Handle ping
            if message_type == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "id": message_id,
                })
                continue

            # Handle cancel
            if message_type == "cancel":
                if current_generation == message_id:
                    cancel_requested = True
                await websocket.send_json({
                    "type": "cancelled",
                    "id": message_id,
                })
                continue

            # Handle config
            if message_type == "config":
                voice_id = message_content.get("voice_id", voice_id)
                chunk_size = message_content.get("chunk_size", chunk_size)
                await websocket.send_json({
                    "type": "config_ack",
                    "id": message_id,
                    "voice_id": voice_id,
                    "chunk_size": chunk_size,
                })
                continue

            # Handle generate request
            if message_type == "generate":
                text = message_content.get("text", "")
                emotion = message_content.get("emotion", {})
                speed = message_content.get("speed", 1.0)

                if not text:
                    await websocket.send_json({
                        "type": "error",
                        "id": message_id,
                        "error": "Empty text",
                    })
                    continue

                try:
                    # Import streaming TTS
                    from app.services.audio_processor import get_streaming_tts, StreamingTTSEngine
                    from app.services.tts_engine import TTSEngineFactory, TTSMode
                    import base64

                    # Setup TTS engine
                    tts_engine = TTSEngineFactory.create(TTSMode.LOCAL)
                    streaming_tts = StreamingTTSEngine(tts_engine, chunk_size=chunk_size)

                    current_generation = message_id
                    cancel_requested = False

                    # Stream generation
                    chunk_count = 0
                    total_duration = 0.0

                    async for chunk in streaming_tts.generate_stream(
                        text=text,
                        speaker=voice_id,
                        voice_config={"emotion": emotion} if emotion else None,
                    ):
                        # Check for cancellation
                        if cancel_requested:
                            await websocket.send_json({
                                "type": "cancelled",
                                "id": message_id,
                                "chunks_generated": chunk_count,
                            })
                            break

                        # Send audio chunk
                        audio_base64 = base64.b64encode(chunk["audio_data"]).decode("utf-8")

                        await websocket.send_json({
                            "type": "chunk",
                            "id": message_id,
                            "chunk_index": chunk["chunk_index"],
                            "total_chunks": chunk["total_chunks"],
                            "audio": audio_base64,
                            "format": "mp3",
                            "text": chunk["text"],
                            "duration": chunk["duration"],
                            "is_final": chunk["is_final"],
                            "metadata": chunk.get("metadata", {}),
                        })

                        chunk_count += 1
                        total_duration += chunk["duration"]

                        # Small delay to prevent overwhelming the client
                        await asyncio.sleep(0.01)

                    # Send completion message
                    if not cancel_requested:
                        await websocket.send_json({
                            "type": "complete",
                            "id": message_id,
                            "total_chunks": chunk_count,
                            "total_duration": round(total_duration, 2),
                            "voice_id": voice_id,
                        })

                except Exception as e:
                    logger.error(f"Streaming TTS error: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "id": message_id,
                        "error": str(e),
                    })
                finally:
                    current_generation = None

            else:
                await websocket.send_json({
                    "type": "error",
                    "id": message_id,
                    "error": f"Unknown message type: {message_type}",
                })

    except asyncio.TimeoutError:
        logger.info("Streaming TTS WebSocket timeout")
        await websocket.send_json({
            "type": "error",
            "error": "Connection timeout",
        })
    except WebSocketDisconnect:
        logger.info("Streaming TTS WebSocket disconnected")
    except Exception as e:
        logger.error(f"Streaming TTS WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e),
            })
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


# =============================================================================
# Real-time Audio Analysis - 实时音频分析
# =============================================================================

@router.websocket("/ws/audio-analysis")
async def websocket_audio_analysis(
    websocket: WebSocket,
    analysis_type: str = Query("emotion", description="Analysis type: emotion, vad, speaker"),
):
    """
    Real-time audio analysis WebSocket endpoint (实时音频分析).

    Analyzes incoming audio chunks for:
    - Emotion recognition (emotion)
    - Voice activity detection (vad)
    - Speaker characteristics (speaker)

    Message Types:
    - "audio_chunk": Base64 encoded audio chunk
    - "config": Update analysis settings
    - "ping": Keep-alive

    Response Types:
    - "analysis_result": Analysis results
    - "pong": Keep-alive response

    Query Parameters:
        analysis_type: Type of analysis to perform
    """
    await websocket.accept()

    # Analysis settings
    settings_analysis = {
        "type": analysis_type,
        "continuous": True,
    }

    try:
        while True:
            # Receive message
            data = await asyncio.wait_for(websocket.receive_json(), timeout=60.0)

            message_type = data.get("type")
            message_content = data.get("content", {})
            message_id = data.get("id", "")

            # Handle ping
            if message_type == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "id": message_id,
                })
                continue

            # Handle config
            if message_type == "config":
                settings_analysis.update(message_content)
                await websocket.send_json({
                    "type": "config_ack",
                    "id": message_id,
                    "settings": settings_analysis,
                })
                continue

            # Handle audio chunks
            if message_type == "audio_chunk":
                audio_base64 = message_content.get("audio", "")

                if not audio_base64:
                    continue

                try:
                    # Decode audio
                    import base64
                    import io
                    import tempfile
                    from pathlib import Path

                    audio_bytes = base64.b64decode(audio_base64)

                    # Save to temp file
                    temp_dir = Path("./static/audio/temp")
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    temp_path = temp_dir / f"analysis_{message_id}.mp3"

                    with open(temp_path, "wb") as f:
                        f.write(audio_bytes)

                    # Perform analysis based on type
                    result = {}

                    if analysis_type == "emotion":
                        # Emotion recognition
                        from app.api.voice_advanced import recognize_emotion
                        response = await recognize_emotion(
                            audio_path=str(temp_path),
                            current_user=None,
                        )
                        result = response.data if response.success else {"error": str(response)}

                    elif analysis_type == "vad":
                        # Voice activity detection
                        from app.api.voice_advanced import detect_voice_activity
                        response = await detect_voice_activity(
                            audio_path=str(temp_path),
                            current_user=None,
                        )
                        result = response.data if response.success else {"error": str(response)}

                    elif analysis_type == "speaker":
                        # Speaker characteristics
                        from app.api.voice_advanced import analyze_speaker_characteristics
                        response = await analyze_speaker_characteristics(
                            audio_path=str(temp_path),
                            current_user=None,
                        )
                        result = response.data if response.success else {"error": str(response)}

                    # Send result
                    await websocket.send_json({
                        "type": "analysis_result",
                        "id": message_id,
                        "analysis_type": analysis_type,
                        "result": result,
                    })

                    # Cleanup
                    temp_path.unlink(missing_ok=True)

                except Exception as e:
                    logger.error(f"Audio analysis error: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "id": message_id,
                        "error": str(e),
                    })

            else:
                await websocket.send_json({
                    "type": "error",
                    "id": message_id,
                    "error": f"Unknown message type: {message_type}",
                })

    except asyncio.TimeoutError:
        logger.info("Audio analysis WebSocket timeout")
    except WebSocketDisconnect:
        logger.info("Audio analysis WebSocket disconnected")
    except Exception as e:
        logger.error(f"Audio analysis WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass

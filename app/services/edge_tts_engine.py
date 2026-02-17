"""Edge TTS Engine - Microsoft's free TTS service."""
import asyncio
import io
import json
import uuid
from typing import Any

import httpx

from app.config import settings


class EdgeTTSEngine:
    """
    Microsoft Edge TTS Engine (Read Aloud API).
    This is a free TTS service provided by Microsoft.
    """

    # Edge TTS API endpoint
    VOICE_LIST_URL = "https://speech.platform.bing.com/consumer/v1/list"
    SPEECH_URL = "https://speech.platform.bing.com/consumer/v1"

    # Common voices
    DEFAULT_VOICES = {
        "zh-CN": {
            "female": "Microsoft Server Speech Text to Speech Voice (zh-CN, XiaoxiaoNeural)",
            "male": "Microsoft Server Speech Text to Speech Voice (zh-CN, YunyangNeural)",
        },
        "en-US": {
            "female": "Microsoft Server Speech Text to Speech Voice (en-US, JennyNeural)",
            "male": "Microsoft Server Speech Text to Speech Voice (en-US, GuyNeural)",
        },
    }

    def __init__(self, config: dict | None = None):
        self.timeout = config.get("timeout", 60) if config else 60
        self._voices_cache: list[dict] | None = None

    async def get_voices(self) -> list[dict]:
        """Get available voices from Edge TTS."""
        if self._voices_cache:
            return self._voices_cache

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    self.VOICE_LIST_URL + "?locale=zh-CN",
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    voices = []
                    # Parse voice list
                    if isinstance(data, dict) and "data" in data:
                        for voice in data["data"]:
                            voices.append({
                                "id": voice.get("id", voice.get("name", "")),
                                "name": voice.get("name", voice.get("friendlyName", "")),
                                "locale": voice.get("locale", ""),
                                "gender": "female" if "Female" in voice.get("friendlyName", "") else "male",
                            })
                    self._voices_cache = voices
                    return voices
        except Exception as e:
            print(f"Failed to fetch voices: {e}")

        # Return default voices
        return [
            {"id": "zh-CN-XiaoxiaoNeural", "name": "晓晓 (女)", "locale": "zh-CN", "gender": "female"},
            {"id": "zh-CN-YunyangNeural", "name": "云扬 (男)", "locale": "zh-CN", "gender": "male"},
            {"id": "en-US-JennyNeural", "name": "Jenny (Female)", "locale": "en-US", "gender": "female"},
            {"id": "en-US-GuyNeural", "name": "Guy (Male)", "locale": "en-US", "gender": "male"},
        ]

    async def generate(
        self,
        text: str,
        speaker: str = "zh-CN-XiaoxiaoNeural",
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%",
    ) -> tuple[bytes, float]:
        """
        Generate audio from text using Edge TTS.

        Args:
            text: Text to convert to speech
            speaker: Voice ID (e.g., "zh-CN-XiaoxiaoNeural")
            rate: Speaking rate (e.g., "+0%", "+10%", "-10%")
            pitch: Pitch adjustment (e.g., "+0Hz", "+10Hz")
            volume: Volume adjustment (e.g., "+0%")

        Returns:
            Tuple of (audio_data, duration_seconds)
        """
        # Generate a unique ID for this request
        request_id = str(uuid.uuid4())

        # Build the SSML request
        ssml = f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="zh-CN"><voice name="{speaker}"><prosody rate="{rate}" pitch="{pitch}" volume="{volume}">{text}</prosody></voice></speak>'

        # Prepare the request
        url = f"{self.SPEECH_URL}?trustedclienttoken=6A5AA1D4EAFF4E9FB37E23D68491D6F4"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Content-Type": "application/ssml+xml",
            "X-MS-Client-Id": request_id,
            "Origin": "chrome-extension://pdonfhimckclligkjmehcepmjlalgpaig",
        }

        params = {
            "ClientTag": request_id,
            "format": "audio-24khz-48kbitrate-mono-mp3",
            "Detection": "true",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    url,
                    headers=headers,
                    params=params,
                    content=ssml.encode('utf-8'),
                )

                if response.status_code == 200:
                    audio_data = response.content
                    duration = self._estimate_duration(audio_data)
                    return audio_data, duration
                else:
                    raise Exception(f"TTS request failed: {response.status_code}")

        except Exception as e:
            raise Exception(f"Edge TTS generation failed: {str(e)}")

    def _estimate_duration(self, audio_data: bytes) -> float:
        """Estimate audio duration from MP3 data."""
        try:
            import struct
            # Simple MP3 duration estimation
            # This is a rough estimate - for accurate duration, use audio processing library
            # Assume 24kbps mono MP3
            estimated_bitrate = 24000  # bits per second
            file_size_bits = len(audio_data) * 8
            duration = file_size_bits / estimated_bitrate
            return max(0.1, duration)
        except Exception:
            # Fallback: assume 1 character ≈ 0.2 seconds
            return 0.2

    async def generate_batch(
        self,
        items: list[dict],
        workers: int = 3,
    ) -> list[tuple[bytes, float]]:
        """Generate multiple audio files concurrently."""
        semaphore = asyncio.Semaphore(workers)

        async def generate_one(item: dict) -> tuple[bytes, float]:
            async with semaphore:
                return await self.generate(
                    text=item["text"],
                    speaker=item.get("speaker", "zh-CN-XiaoxiaoNeural"),
                    rate=item.get("rate", "+0%"),
                    pitch=item.get("pitch", "+0Hz"),
                    volume=item.get("volume", "+0%"),
                )

        tasks = [generate_one(item) for item in items]
        return await asyncio.gather(*tasks)


# Factory function
def create_edge_tts_engine(config: dict | None = None) -> EdgeTTSEngine:
    """Create Edge TTS engine instance."""
    return EdgeTTSEngine(config)

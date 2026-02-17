"""Voice management service."""
from pathlib import Path
from typing import Any

from app.config import settings
from app.services.tts_engine import TTSEngineFactory, TTSMode


class VoiceManager:
    """Service for managing voice configurations."""

    def __init__(self, tts_mode: TTSMode | str = TTSMode.EXTERNAL):
        self.tts_mode = TTSMode(tts_mode) if isinstance(tts_mode, str) else tts_mode
        self.tts_engine = TTSEngineFactory.create(self.tts_mode)

    async def get_available_voices(self) -> dict[str, list[dict]]:
        """Get all available voices by type."""
        voices = await self.tts_engine.get_voices()

        return {
            "custom": voices,
            "lora": [
                {"id": "builtin_watson", "name": "Watson", "gender": "male", "language": "en-US"},
            ],
        }

    async def preview_voice(
        self,
        text: str,
        voice_type: str,
        voice_name: str | None = None,
        instruct: str | None = None,
    ) -> dict[str, Any]:
        """Generate a voice preview."""
        voice_config = {
            "voice_type": voice_type,
            "voice_name": voice_name,
        }

        audio_data, duration = await self.tts_engine.generate(
            text=text,
            speaker=voice_name or "preview",
            instruct=instruct,
            voice_config=voice_config,
        )

        # Save preview
        import uuid
        preview_id = str(uuid.uuid4())
        preview_path = Path("./static/audio/previews")
        preview_path.mkdir(parents=True, exist_ok=True)

        wav_path = preview_path / f"{preview_id}.wav"
        with open(wav_path, "wb") as f:
            f.write(audio_data)

        # Convert to MP3
        from app.services.audio_processor import AudioProcessor
        processor = AudioProcessor()
        mp3_path = await processor.convert_to_mp3(str(wav_path))

        return {
            "audio_url": mp3_path.replace("./static", "/static"),
            "duration": duration,
        }

    async def save_reference_audio(
        self,
        user_id: str,
        audio_data: bytes,
        transcript: str,
    ) -> str:
        """Save reference audio for voice cloning."""
        import uuid

        audio_id = str(uuid.uuid4())
        user_dir = Path(settings.upload_dir) / "voices" / user_id
        user_dir.mkdir(parents=True, exist_ok=True)

        file_path = user_dir / f"{audio_id}.wav"
        with open(file_path, "wb") as f:
            f.write(audio_data)

        return str(file_path)

    def validate_reference_audio(self, audio_path: str) -> dict[str, Any]:
        """Validate reference audio for cloning."""
        from pydub import AudioSegment

        try:
            audio = AudioSegment.from_file(audio_path)
            duration = len(audio) / 1000.0

            # Reference audio should be 5-15 seconds
            if duration < 5:
                return {"valid": False, "error": "Audio too short (minimum 5 seconds)"}
            if duration > 15:
                return {"valid": False, "error": "Audio too long (maximum 15 seconds)"}

            return {
                "valid": True,
                "duration": duration,
                "sample_rate": audio.frame_rate,
                "channels": audio.channels,
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}

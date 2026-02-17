"""
Advanced Voice Features API
SSML support, quality scoring, and advanced controls
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, Dict, Any
import uuid
from pathlib import Path
from pydantic import BaseModel

from app.core.deps import CurrentUserDep, DbDep
from app.schemas.common import ApiResponse
from app.services.ssml_processor import (
    get_ssml_processor,
    SSMLProcessor,
    ProsodyControl,
    get_preset,
)
from app.services.audio_quality_scorer import (
    get_audio_quality_scorer,
    QualityMetrics,
)
from app.services.voice_cloner import get_voice_cloner

router = APIRouter()


class SSMLRequest(BaseModel):
    """Request for SSML processing."""
    text: str
    rate: Optional[float] = None
    pitch: Optional[float] = None
    volume: Optional[float] = None
    emphasis: Optional[str] = None
    voice: Optional[str] = None


class SSMLResponse(BaseModel):
    """Response for SSML processing."""
    ssml: str
    plain_text: str
    tts_params: Dict[str, Any]


class QualityScoreRequest(BaseModel):
    """Request for audio quality scoring."""
    audio_path: str
    reference_path: Optional[str] = None


class ProsodyApplyRequest(BaseModel):
    """Request to apply prosody to audio."""
    text: str
    audio_path: Optional[str] = None
    prosody_preset: Optional[str] = None
    rate: Optional[float] = None
    pitch: Optional[float] = None
    volume: Optional[float] = None
    emphasis: Optional[str] = None


@router.post("/ssml/generate", response_model=ApiResponse[SSMLResponse])
async def generate_ssml(
    request: SSMLRequest,
    current_user: CurrentUserDep,
):
    """Generate SSML from text and prosody parameters.

    Args:
        request: Text and prosody parameters

    Returns:
        Generated SSML and TTS parameters
    """
    ssml_processor = get_ssml_processor()

    # Generate SSML
    ssml = ssml_processor.generate_ssml(
        text=request.text,
        rate=request.rate,
        pitch=request.pitch,
        volume=request.volume,
        emphasis=request.emphasis,
        voice=request.voice,
    )

    # Convert to TTS parameters
    plain_text, tts_params = ssml_processor.convert_to_tts_params(ssml)

    return ApiResponse(
        data=SSMLResponse(
            ssml=ssml,
            plain_text=plain_text,
            tts_params=tts_params,
        )
    )


@router.post("/ssml/parse", response_model=ApiResponse[Dict[str, Any]])
async def parse_ssml(
    ssml: str,
    current_user: CurrentUserDep,
):
    """Parse SSML and extract segments with prosody controls.

    Args:
        ssml: SSML string to parse

    Returns:
        Parsed segments with prosody controls
    """
    ssml_processor = get_ssml_processor()

    try:
        segments = ssml_processor.parse_ssml(ssml)

        return ApiResponse(
            data={
                "segments": [
                    {
                        "text": seg.text,
                        "prosody": {
                            "rate": seg.prosody.rate,
                            "pitch": seg.prosody.pitch,
                            "volume": seg.prosody.volume,
                            "emphasis": seg.prosody.emphasis,
                        },
                        "voice": seg.voice,
                        "breaks": seg.breaks,
                    }
                    for seg in segments
                ],
                "segment_count": len(segments),
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse SSML: {str(e)}"
        )


@router.get("/prosody/presets", response_model=ApiResponse[Dict[str, Dict[str, Any]]])
async def list_prosody_presets():
    """List all available prosody presets."""
    from app.services.ssml_processor import PROSODY_PRESETS

    presets = {}
    for name, prosody in PROSODY_PRESETS.items():
        presets[name] = {
            "rate": prosody.rate,
            "pitch": prosody.pitch,
            "volume": prosody.volume,
            "emphasis": prosody.emphasis,
        }

    return ApiResponse(data=presets)


@router.post("/quality/score", response_model=ApiResponse[Dict[str, Any]])
async def score_audio_quality(
    request: QualityScoreRequest,
    current_user: CurrentUserDep,
):
    """Score audio quality automatically.

    Args:
        request: Audio file to score

    Returns:
        Quality metrics and recommendation
    """
    scorer = get_audio_quality_scorer()

    # Resolve audio path
    audio_path = Path("./static") / request.audio_path.lstrip("/")
    if not audio_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Audio file not found: {request.audio_path}"
        )

    try:
        metrics = await scorer.score_audio(
            str(audio_path),
            reference_path=request.reference_path,
        )

        report = scorer.get_quality_report(metrics)

        return ApiResponse(
            data={
                **report,
                "audio_path": request.audio_path,
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to score audio: {str(e)}"
        )


@router.post("/quality/batch-score", response_model=ApiResponse[List[Dict[str, Any]]])
async def batch_score_quality(
    audio_paths: List[str],
    current_user: CurrentUserDep,
):
    """Score multiple audio files.

    Args:
        audio_paths: List of audio file paths to score

    Returns:
        List of quality reports
    """
    scorer = get_audio_quality_scorer()

    reports = []
    for path in audio_paths:
        audio_path = Path("./static") / path.lstrip("/")
        if not audio_path.exists():
            reports.append({
                "audio_path": path,
                "error": "File not found",
                "overall_score": 0,
            })
            continue

        try:
            metrics = await scorer.score_audio(str(audio_path))
            report = scorer.get_quality_report(metrics)
            report["audio_path"] = path
            reports.append(report)
        except Exception as e:
            reports.append({
                "audio_path": path,
                "error": str(e),
                "overall_score": 0,
            })

    return ApiResponse(data=reports)


@router.post("/prosody/apply", response_model=ApiResponse[Dict[str, Any]])
async def apply_prosody_to_audio(
    request: ProsodyApplyRequest,
    current_user: CurrentUserDep,
):
    """Apply prosody controls to audio.

    Args:
        request: Text and prosody parameters

    Returns:
        Generated audio with applied prosody
    """
    from app.services.tts_engine import TTSEngineFactory, TTSMode
    from app.services.ssml_processor import PROSODY_PRESETS
    from pydub import AudioSegment
    import io
    import numpy as np

    # Determine prosody to use
    if request.prosody_preset:
        preset = get_preset(request.prosody_preset)
        if not preset:
            raise HTTPException(
                status_code=404,
                detail=f"Prosody preset not found: {request.prosody_preset}"
            )
        prosody = preset
    else:
        prosody = ProsodyControl(
            rate=request.rate or 1.0,
            pitch=request.pitch or 1.0,
            volume=request.volume or 1.0,
            emphasis=request.emphasis or "none",
        )

    # Generate audio with TTS
    tts_engine = TTSEngineFactory.create(TTSMode.LOCAL)

    try:
        audio_data, duration = await tts_engine.generate(
            text=request.text,
            speaker="aiden",  # Default voice
        )

        # Apply prosody modifications
        audio = AudioSegment.from_file(io.BytesIO(audio_data))

        # Apply rate
        if prosody.rate != 1.0:
            new_frame_rate = int(audio.frame_rate * prosody.rate)
            audio = audio._spawn(audio.raw_data, overrides={'frame_rate': new_frame_rate})
            audio = audio.set_frame_rate(22050)

        # Apply pitch
        if prosody.pitch != 1.0:
            new_sample_rate = int(audio.frame_rate * (2.0 ** ((prosody.pitch - 1) * 12 / 12)))
            audio = audio._spawn(audio.raw_data, overrides={'frame_rate': new_sample_rate})
            audio = audio.set_frame_rate(22050)

        # Apply volume
        if prosody.volume != 1.0:
            db_change = 10 * np.log10(prosody.volume)
            audio = audio + db_change

        # Export
        output = io.BytesIO()
        audio.export(output, format="mp3")
        modified_audio = output.read()

        # Save audio file
        audio_id = uuid.uuid4().hex[:8]
        audio_dir = Path("./static/audio/prosody")
        audio_dir.mkdir(parents=True, exist_ok=True)

        audio_path = audio_dir / f"prosody_{audio_id}.mp3"
        with open(audio_path, "wb") as f:
            f.write(modified_audio)

        # Score quality
        scorer = get_audio_quality_scorer()
        metrics = await scorer.score_audio(str(audio_path))

        return ApiResponse(
            data={
                "audio_url": f"/static/audio/prosody/prosody_{audio_id}.mp3",
                "duration": duration,
                "prosody_applied": {
                    "rate": prosody.rate,
                    "pitch": prosody.pitch,
                    "volume": prosody.volume,
                    "emphasis": prosody.emphasis,
                },
                "quality_score": metrics.overall_score,
                "quality_grade": scorer._score_to_grade(metrics.overall_score),
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to apply prosody: {str(e)}"
        )


@router.get("/voice-profiles", response_model=ApiResponse[List[Dict[str, Any]]])
async def list_voice_profiles(
    current_user: CurrentUserDep,
):
    """List all voice profiles for the current user.

    Returns:
        List of voice profiles
    """
    voice_cloner = get_voice_cloner()

    try:
        profiles = await voice_cloner.list_profiles(user_id=current_user.id)
        return ApiResponse(data=profiles)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list profiles: {str(e)}"
        )


@router.get("/voice-profiles/{profile_id}", response_model=ApiResponse[Dict[str, Any]])
async def get_voice_profile(
    profile_id: str,
    current_user: CurrentUserDep,
):
    """Get details of a specific voice profile.

    Args:
        profile_id: Voice profile ID

    Returns:
        Voice profile details
    """
    voice_cloner = get_voice_cloner()

    try:
        profile = await voice_cloner.load_profile(profile_id)

        # Verify ownership
        if profile.metadata.get("user_id") != current_user.id:
            raise HTTPException(
                status_code=403,
                detail="You don't have access to this profile"
            )

        return ApiResponse(
            data={
                "profile_id": profile.profile_id,
                "name": profile.name,
                "sample_count": len(profile.samples),
                "reference_audio": profile.reference_audio,
                "voice_features": profile.voice_features,
                "created_at": profile.created_at.isoformat(),
                "metadata": profile.metadata,
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load profile: {str(e)}"
        )


@router.delete("/voice-profiles/{profile_id}", response_model=ApiResponse[Dict[str, str]])
async def delete_voice_profile(
    profile_id: str,
    current_user: CurrentUserDep,
):
    """Delete a voice profile.

    Args:
        profile_id: Voice profile ID

    Returns:
        Success status
    """
    voice_cloner = get_voice_cloner()

    try:
        # Verify ownership first
        profile = await voice_cloner.load_profile(profile_id)
        if profile.metadata.get("user_id") != current_user.id:
            raise HTTPException(
                status_code=403,
                detail="You don't have access to this profile"
            )

        success = await voice_cloner.delete_profile(profile_id)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Profile not found: {profile_id}"
            )

        return ApiResponse(
            data={
                "status": "deleted",
                "profile_id": profile_id,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete profile: {str(e)}"
        )


# =============================================================================
# Voice Activity Detection (VAD) - 语音活动检测
# =============================================================================

@router.post("/vad/detect", response_model=ApiResponse[Dict[str, Any]])
async def detect_voice_activity(
    audio_path: str,
    threshold: float = 0.5,
    min_speech_duration: float = 0.3,
    current_user: CurrentUserDep = Depends,
):
    """Detect voice activity segments in audio (语音活动检测).

    Automatically identifies speech segments vs silence/background noise.

    Args:
        audio_path: Path to audio file
        threshold: Detection threshold (0-1)
        min_speech_duration: Minimum speech duration in seconds

    Returns:
        Detected voice activities with timing and confidence
    """
    from app.services.audio_processor import AudioProcessor
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent
    import numpy as np

    processor = AudioProcessor()

    # Resolve audio path
    audio_file = Path("./static") / audio_path.lstrip("/")
    if not audio_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Audio file not found: {audio_path}"
        )

    try:
        # Load audio
        audio = AudioSegment.from_file(str(audio_file))
        duration_sec = len(audio) / 1000.0

        # Detect non-silent (speech) segments
        min_silence_len = int(min_speech_duration * 1000)
        silence_thresh = audio.dBFS - (20 * (1 - threshold))

        nonspeech_ranges = detect_nonsilent(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            seek_step=10
        )

        # Convert to activities
        activities = []
        for start_ms, end_ms in nonspeech_ranges:
            segment = audio[start_ms:end_ms]
            # Calculate confidence based on energy
            energy = np.mean(np.array(segment.get_array_of_samples()) ** 2)
            confidence = min(energy / (2 ** 15) ** 2 * 1000, 1.0)

            activities.append({
                "start_time": start_ms / 1000.0,
                "end_time": end_ms / 1000.0,
                "duration": (end_ms - start_ms) / 1000.0,
                "confidence": round(confidence, 3),
                "is_speech": True,
            })

        # Calculate speech ratio
        speech_duration = sum(a["duration"] for a in activities)
        speech_ratio = speech_duration / duration_sec if duration_sec > 0 else 0

        return ApiResponse(
            data={
                "audio_path": audio_path,
                "duration": duration_sec,
                "activities": activities,
                "num_activities": len(activities),
                "speech_duration": speech_duration,
                "speech_ratio": round(speech_ratio, 3),
                "silence_duration": round(duration_sec - speech_duration, 3),
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Voice activity detection failed: {str(e)}"
        )


# =============================================================================
# Speaker Recognition - 说话人识别
# =============================================================================

@router.post("/speaker/analyze", response_model=ApiResponse[Dict[str, Any]])
async def analyze_speaker_characteristics(
    audio_path: str,
    current_user: CurrentUserDep = Depends,
):
    """Analyze speaker voice characteristics (说话人特征分析).

    Extracts detailed voice characteristics for speaker identification
    and voice cloning applications.

    Args:
        audio_path: Path to audio file

    Returns:
        Detailed voice characteristics analysis
    """
    from app.services.audio_processor import AudioProcessor
    from pydub import AudioSegment
    import numpy as np

    processor = AudioProcessor()

    # Resolve audio path
    audio_file = Path("./static") / audio_path.lstrip("/")
    if not audio_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Audio file not found: {audio_path}"
        )

    try:
        # Load audio
        audio = AudioSegment.from_file(str(audio_file))
        samples = np.array(audio.get_array_of_samples())
        sr = audio.frame_rate

        # Basic analysis
        duration = len(audio) / 1000.0

        # Energy analysis
        rms_energy = np.sqrt(np.mean(samples ** 2))
        peak_amplitude = np.max(np.abs(samples))
        dynamic_range = 20 * np.log10(peak_amplitude / (rms_energy + 1e-10))

        # Pitch estimation (zero-crossing rate approximation)
        # For each 100ms segment
        segment_samples = samples[::sr // 10]  # Downsample for speed
        zero_crossings = np.sum(np.abs(np.diff(np.sign(segment_samples))))
        zcr = zero_crossings / len(segment_samples)

        # Estimate fundamental frequency (rough approximation)
        estimated_pitch_hz = zcr * sr / 4

        # Determine characteristics
        if estimated_pitch_hz > 200:
            estimated_gender = "female"
        elif estimated_pitch_hz > 150:
            estimated_gender = "female_or_young_male"
        elif estimated_pitch_hz > 100:
            estimated_gender = "male"
        else:
            estimated_gender = "deep_male"

        # Voice type based on energy distribution
        spectral_content = np.mean(np.abs(samples))
        if spectral_content > 10000:
            voice_type = "bright"
        elif spectral_content > 5000:
            voice_type = "neutral"
        else:
            voice_type = "dark"

        # Tempo estimation (speech rate)
        # Count peaks in amplitude envelope
        envelope = np.convolve(np.abs(samples), np.ones(1000)/1000, mode='same')
        peaks = 0
        for i in range(1, len(envelope) - 1):
            if envelope[i] > envelope[i-1] and envelope[i] > envelope[i+1]:
                if envelope[i] > np.mean(envelope):
                    peaks += 1
        estimated_tempo = peaks / duration * 60  # Beats per minute equivalent

        return ApiResponse(
            data={
                "audio_path": audio_path,
                "duration": duration,
                "sample_rate": sr,
                "channels": audio.channels,
                "estimated_gender": estimated_gender,
                "estimated_pitch_hz": round(estimated_pitch_hz, 1),
                "voice_type": voice_type,
                "energy": {
                    "rms": round(rms_energy, 2),
                    "peak": round(peak_amplitude, 2),
                    "dynamic_range_db": round(dynamic_range, 2),
                },
                "tempo": {
                    "estimated_speech_rate_bpm": round(estimated_tempo, 1),
                },
                "characteristics": {
                    "zero_crossing_rate": round(zcr, 4),
                    "brightness": round(spectral_content / 10000, 2),
                },
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Speaker analysis failed: {str(e)}"
        )


# =============================================================================
# Emotion Recognition - 语音情感识别
# =============================================================================

@router.post("/emotion/recognize", response_model=ApiResponse[Dict[str, Any]])
async def recognize_emotion(
    audio_path: str,
    segment_start: Optional[float] = None,
    segment_end: Optional[float] = None,
    current_user: CurrentUserDep = Depends,
):
    """Recognize emotion from voice audio (语音情感识别).

    Analyzes acoustic features to predict emotional state including
    arousal (energy level) and valence (positive/negative).

    Args:
        audio_path: Path to audio file
        segment_start: Optional segment start time in seconds
        segment_end: Optional segment end time in seconds

    Returns:
        Emotion prediction with confidence and dimensions
    """
    from app.services.audio_processor import AudioProcessor
    from pydub import AudioSegment
    import numpy as np

    processor = AudioProcessor()

    # Resolve audio path
    audio_file = Path("./static") / audio_path.lstrip("/")
    if not audio_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Audio file not found: {audio_path}"
        )

    try:
        # Load audio
        audio = AudioSegment.from_file(str(audio_file))

        # Extract segment if specified
        if segment_start is not None:
            start_ms = int(segment_start * 1000)
            end_ms = int(segment_end * 1000) if segment_end else len(audio)
            audio = audio[start_ms:end_ms]

        samples = np.array(audio.get_array_of_samples())
        sr = audio.frame_rate
        duration = len(audio) / 1000.0

        # Extract features
        # Energy features
        rms = np.sqrt(np.mean(samples ** 2))
        energy_std = np.std([np.sqrt(np.mean(samples[i:i+sr//10]**2))
                            for i in range(0, len(samples), sr//10)])

        # Pitch estimation (zero-crossing)
        segment_samples = samples[::sr]
        zero_crossings = np.sum(np.abs(np.diff(np.sign(segment_samples))))
        zcr = zero_crossings / len(segment_samples)
        estimated_pitch = zcr * sr / 4

        # Tempo (speech rate)
        envelope = np.convolve(np.abs(samples), np.ones(sr//10)/(sr//10), mode='same')
        peaks = sum(1 for i in range(1, len(envelope)-1)
                   if envelope[i] > envelope[i-1] and envelope[i] > envelope[i+1]
                   and envelope[i] > np.mean(envelope))
        tempo = peaks / duration

        # Calculate emotion scores based on features
        scores = {
            "happy": 0.0,
            "sad": 0.0,
            "angry": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "neutral": 0.0,
        }

        # Happy: high pitch, high energy, fast tempo
        if estimated_pitch > 180 and rms > 5000 and tempo > 120:
            scores["happy"] += 0.4
        # Sad: low pitch, low energy, slow tempo
        if estimated_pitch < 150 and rms < 3000 and tempo < 100:
            scores["sad"] += 0.4
        # Angry: medium-high pitch, very high energy
        if estimated_pitch > 160 and rms > 8000:
            scores["angry"] += 0.4
        # Fear: high pitch, high energy variability, fast tempo
        if estimated_pitch > 200 and energy_std > 2000 and tempo > 140:
            scores["fear"] += 0.4
        # Surprise: very high pitch, fast tempo
        if estimated_pitch > 220 and tempo > 130:
            scores["surprise"] += 0.4
        # Neutral: medium everything
        if 120 <= estimated_pitch <= 180 and 3000 <= rms <= 7000 and 90 <= tempo <= 120:
            scores["neutral"] += 0.4

        # Add some base probability for neutral
        scores["neutral"] += 0.1

        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: round(v / total, 3) for k, v in scores.items()}

        # Get top emotion
        emotion = max(scores, key=scores.get)
        confidence = scores[emotion]

        # Calculate arousal (0-1, calm to excited)
        arousal = min((rms / 10000) + (tempo / 200) + (energy_std / 5000), 1.0)

        # Calculate valence (-1 to 1, negative to positive)
        # High pitch + low energy = negative (sad/fear)
        # High pitch + high energy = positive (happy/surprise)
        if emotion in ["happy", "surprise"]:
            valence = 0.6 + (arousal * 0.4)
        elif emotion in ["sad", "fear", "angry"]:
            valence = -0.6 - (arousal * 0.2)
        else:
            valence = 0.0

        return ApiResponse(
            data={
                "audio_path": audio_path,
                "duration": duration,
                "emotion": emotion,
                "confidence": confidence,
                "arousal": round(arousal, 3),  # 0-1 calm to excited
                "valence": round(valence, 3),  # -1 to 1 negative to positive
                "all_scores": scores,
                "features": {
                    "estimated_pitch_hz": round(estimated_pitch, 1),
                    "energy_rms": round(rms, 2),
                    "energy_variability": round(energy_std, 2),
                    "tempo_bpm": round(tempo, 1),
                },
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Emotion recognition failed: {str(e)}"
        )


# =============================================================================
# Comprehensive Voice Analysis - 综合语音分析
# =============================================================================

@router.post("/analyze/comprehensive", response_model=ApiResponse[Dict[str, Any]])
async def comprehensive_voice_analysis(
    audio_path: str,
    include_vad: bool = True,
    include_emotion: bool = True,
    include_speaker: bool = True,
    current_user: CurrentUserDep = Depends,
):
    """Perform comprehensive voice analysis (综合语音分析).

    Combines VAD, speaker analysis, and emotion recognition
    for a complete audio profile.

    Args:
        audio_path: Path to audio file
        include_vad: Include voice activity detection
        include_emotion: Include emotion recognition
        include_speaker: Include speaker characteristics

    Returns:
        Comprehensive analysis results
    """
    from app.services.audio_processor import AudioProcessor
    from pydub import AudioSegment
    from pathlib import Path as PathLib
    import numpy as np

    # Resolve audio path
    audio_file = PathLib("./static") / audio_path.lstrip("/")
    if not audio_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Audio file not found: {audio_path}"
        )

    result = {
        "audio_path": audio_path,
        "analysis_type": "comprehensive",
    }

    try:
        processor = AudioProcessor()
        audio = AudioSegment.from_file(str(audio_file))
        samples = np.array(audio.get_array_of_samples())
        sr = audio.frame_rate
        duration = len(audio) / 1000.0

        result["duration"] = duration
        result["sample_rate"] = sr
        result["channels"] = audio.channels

        # Voice Activity Detection
        if include_vad:
            from pydub.silence import detect_nonsilent

            nonspeech_ranges = detect_nonsilent(
                audio,
                min_silence_len=300,
                silence_thresh=audio.dBFS - 16,
                seek_step=10
            )

            activities = []
            speech_duration = 0
            for start_ms, end_ms in nonspeech_ranges:
                dur = (end_ms - start_ms) / 1000.0
                speech_duration += dur
                activities.append({
                    "start_time": start_ms / 1000.0,
                    "end_time": end_ms / 1000.0,
                    "duration": dur,
                })

            result["voice_activity"] = {
                "activities": activities,
                "num_activities": len(activities),
                "speech_duration": round(speech_duration, 2),
                "speech_ratio": round(speech_duration / duration, 3) if duration > 0 else 0,
                "silence_duration": round(duration - speech_duration, 2),
            }

        # Speaker Characteristics
        if include_speaker:
            rms_energy = np.sqrt(np.mean(samples ** 2))
            zero_crossings = np.sum(np.abs(np.diff(np.sign(samples[::sr//10]))))
            zcr = zero_crossings / len(samples[::sr//10])
            estimated_pitch = zcr * sr / 4

            if estimated_pitch > 200:
                gender = "female"
            elif estimated_pitch > 150:
                gender = "female_or_young_male"
            elif estimated_pitch > 100:
                gender = "male"
            else:
                gender = "deep_male"

            result["speaker_characteristics"] = {
                "estimated_gender": gender,
                "estimated_pitch_hz": round(estimated_pitch, 1),
                "energy_rms": round(rms_energy, 2),
                "voice_type": "bright" if rms_energy > 5000 else "dark",
            }

        # Emotion Recognition
        if include_emotion:
            rms = np.sqrt(np.mean(samples ** 2))
            energy_std = np.std([np.sqrt(np.mean(samples[i:i+sr//10]**2))
                                for i in range(0, len(samples), sr//10)])
            segment_samples = samples[::sr]
            zero_crossings = np.sum(np.abs(np.diff(np.sign(segment_samples))))
            zcr = zero_crossings / len(segment_samples)
            estimated_pitch = zcr * sr / 4
            envelope = np.convolve(np.abs(samples), np.ones(sr//10)/(sr//10), mode='same')
            peaks = sum(1 for i in range(1, len(envelope)-1)
                       if envelope[i] > envelope[i-1] and envelope[i] > envelope[i+1]
                       and envelope[i] > np.mean(envelope))
            tempo = peaks / duration

            # Emotion scoring
            scores = {
                "happy": 0.0, "sad": 0.0, "angry": 0.0,
                "fear": 0.0, "surprise": 0.0, "neutral": 0.0,
            }

            if estimated_pitch > 180 and rms > 5000 and tempo > 120:
                scores["happy"] += 0.4
            if estimated_pitch < 150 and rms < 3000 and tempo < 100:
                scores["sad"] += 0.4
            if estimated_pitch > 160 and rms > 8000:
                scores["angry"] += 0.4
            if estimated_pitch > 200 and energy_std > 2000 and tempo > 140:
                scores["fear"] += 0.4
            if estimated_pitch > 220 and tempo > 130:
                scores["surprise"] += 0.4
            if 120 <= estimated_pitch <= 180 and 3000 <= rms <= 7000 and 90 <= tempo <= 120:
                scores["neutral"] += 0.4

            scores["neutral"] += 0.1
            total = sum(scores.values())
            if total > 0:
                scores = {k: round(v / total, 3) for k, v in scores.items()}

            emotion = max(scores, key=scores.get)
            confidence = scores[emotion]
            arousal = min((rms / 10000) + (tempo / 200) + (energy_std / 5000), 1.0)

            if emotion in ["happy", "surprise"]:
                valence = 0.6 + (arousal * 0.4)
            elif emotion in ["sad", "fear", "angry"]:
                valence = -0.6 - (arousal * 0.2)
            else:
                valence = 0.0

            result["emotion_recognition"] = {
                "emotion": emotion,
                "confidence": confidence,
                "arousal": round(arousal, 3),
                "valence": round(valence, 3),
                "all_scores": scores,
            }

        result["success"] = True
        return ApiResponse(data=result)

    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        raise HTTPException(
            status_code=500,
            detail=f"Comprehensive analysis failed: {str(e)}"
        )


# =============================================================================
# Voice Enhancement - 语音增强
# =============================================================================

@router.post("/enhance", response_model=ApiResponse[Dict[str, Any]])
async def enhance_audio(
    audio_path: str,
    denoise: bool = True,
    normalize_volume: bool = True,
    reduce_echo: bool = False,
    current_user: CurrentUserDep = Depends,
):
    """Enhance audio quality (语音增强).

    Applies noise reduction, volume normalization, and echo reduction.

    Args:
        audio_path: Path to audio file
        denoise: Apply noise reduction
        normalize_volume: Normalize volume levels
        reduce_echo: Apply echo reduction

    Returns:
        Enhanced audio file path and processing details
    """
    from app.services.audio_processor import AudioProcessor
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent
    from pydub.effects import normalize
    from pathlib import Path as PathLib
    import uuid

    processor = AudioProcessor()

    # Resolve audio path
    audio_file = PathLib("./static") / audio_path.lstrip("/")
    if not audio_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Audio file not found: {audio_path}"
        )

    try:
        # Load audio
        audio = AudioSegment.from_file(str(audio_file))
        original_duration = len(audio) / 1000.0
        original_dbfs = audio.dBFS

        processing_steps = []
        enhanced_audio = audio

        # Noise reduction (silence gate)
        if denoise:
            # Remove very quiet sections
            nonsilent_ranges = detect_nonsilent(
                enhanced_audio,
                min_silence_len=50,
                silence_thresh=enhanced_audio.dBFS - 16,
                seek_step=10
            )

            if nonsilent_ranges:
                # Keep only non-silent parts with small gaps filled
                cleaned = AudioSegment.silent(duration=0)
                for i, (start_ms, end_ms) in enumerate(nonsilent_ranges):
                    cleaned += enhanced_audio[start_ms:end_ms]
                    if i < len(nonsilent_ranges) - 1:
                        # Add small silence between segments
                        next_start = nonsilent_ranges[i + 1][0]
                        gap = next_start - end_ms
                        if gap < 500:  # Fill gaps shorter than 500ms
                            cleaned += AudioSegment.silent(duration=gap)
                enhanced_audio = cleaned
                processing_steps.append("noise_reduction")

        # Volume normalization
        if normalize_volume:
            enhanced_audio = normalize(enhanced_audio, headroom=1.0)
            processing_steps.append("volume_normalization")

        # Echo reduction (simple high-pass filter approximation)
        if reduce_echo:
            # Cut low frequencies to reduce room resonance
            enhanced_audio = enhanced_audio.high_pass_filter(80)
            processing_steps.append("echo_reduction")

        # Export enhanced audio
        output_dir = PathLib("./static/audio/enhanced")
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_id = uuid.uuid4().hex[:8]
        output_filename = f"enhanced_{audio_id}.mp3"
        output_path = output_dir / output_filename

        enhanced_audio.export(str(output_path), format="mp3", bitrate="192k")

        # Calculate improvements
        enhanced_duration = len(enhanced_audio) / 1000.0
        enhanced_dbfs = enhanced_audio.dBFS
        db_improvement = enhanced_dbfs - original_dbfs

        return ApiResponse(
            data={
                "original_audio_path": audio_path,
                "enhanced_audio_url": f"/static/audio/enhanced/{output_filename}",
                "original_duration": round(original_duration, 2),
                "enhanced_duration": round(enhanced_duration, 2),
                "original_dbfs": round(original_dbfs, 2),
                "enhanced_dbfs": round(enhanced_dbfs, 2),
                "db_improvement": round(db_improvement, 2),
                "processing_steps": processing_steps,
                "success": True,
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Audio enhancement failed: {str(e)}"
        )


# =============================================================================
# Voice Translation - 语音翻译 (TTS + Translation)
# =============================================================================

@router.post("/translate", response_model=ApiResponse[Dict[str, Any]])
async def translate_voice(
    text: str,
    source_lang: str = "zh",
    target_lang: str = "en",
    voice_id: str = "aiden",
    emotion: str = "neutral",
    current_user: CurrentUserDep = Depends,
):
    """Translate and generate speech (语音翻译).

    Translates text from source language to target language
    and generates speech using TTS.

    Args:
        text: Source text to translate
        source_lang: Source language code (zh, en, ja, ko, etc.)
        target_lang: Target language code
        voice_id: Voice ID for TTS
        emotion: Emotion for speech

    Returns:
        Translated text and generated audio
    """
    from app.utils.llm import translate_text
    from app.services.tts_engine import TTSEngineFactory, TTSMode
    import uuid
    from pathlib import Path as PathLib

    try:
        # Step 1: Translate text
        translated_text = await translate_text(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang
        )

        # Step 2: Generate speech with translated text
        tts_engine = TTSEngineFactory.create(TTSMode.LOCAL)
        audio_data, duration = await tts_engine.generate(
            text=translated_text,
            speaker=voice_id,
        )

        # Step 3: Save audio file
        output_dir = PathLib("./static/audio/translated")
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_id = uuid.uuid4().hex[:8]
        output_filename = f"translate_{audio_id}.mp3"
        output_path = output_dir / output_filename

        with open(output_path, "wb") as f:
            f.write(audio_data)

        return ApiResponse(
            data={
                "original_text": text,
                "translated_text": translated_text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "audio_url": f"/static/audio/translated/{output_filename}",
                "duration": duration,
                "voice_id": voice_id,
                "success": True,
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Voice translation failed: {str(e)}"
        )

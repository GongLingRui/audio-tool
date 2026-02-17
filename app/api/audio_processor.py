"""Audio processor API routes."""
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import CurrentUserDep, DbDep
from app.schemas.common import ApiResponse
from app.services.audio_processor import (
    get_audio_mixer,
    get_dialogue_generator,
    get_text_segmenter,
    get_streaming_tts,
    get_rvc_converter,
    get_prosody_controller,
    get_model_manager,
    get_enhanced_assessor,
    get_ssml_processor,
    get_sound_effects_library,
)

router = APIRouter()


# =============================================================================
# Audio Mixer API - 音频混合
# =============================================================================

@router.post("/mix-audio", response_model=ApiResponse[dict])
async def mix_audio(
    speech_audio_path: str,
    background_music_path: Optional[str] = None,
    sound_effects: Optional[str] = None,
    music_volume: float = 0.2,
    ducking: bool = True,
    ducking_amount: float = 0.5,
    fade_in: float = 0.5,
    fade_out: float = 1.0,
    output_path: Optional[str] = None,
):
    """Mix speech audio with background music and sound effects."""
    mixer = get_audio_mixer()

    # Parse sound effects JSON
    effects = None
    if sound_effects:
        import json
        effects = json.loads(sound_effects)

    result = await mixer.mix_audio(
        speech_audio_path=speech_audio_path,
        background_music_path=background_music_path,
        sound_effects=effects,
        music_volume=music_volume,
        ducking=ducking,
        ducking_amount=ducking_amount,
        fade_in=fade_in,
        fade_out=fade_out,
        output_path=output_path,
    )

    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("error", "Audio mixing failed")
        )

    return ApiResponse(data=result)


# =============================================================================
# Dialogue Generation API - 对话生成
# =============================================================================

@router.post("/generate-dialogue", response_model=ApiResponse[dict])
async def generate_dialogue(
    dialogue_script: List[Dict[str, Any]],
    voice_configs: Optional[Dict[str, Any]] = None,
    add_pauses: bool = True,
    normalize_audio: bool = True,
    output_path: Optional[str] = None,
):
    """Generate multi-speaker dialogue audio."""
    generator = get_dialogue_generator()

    result = await generator.generate_dialogue(
        dialogue_script=dialogue_script,
        voice_configs=voice_configs,
        output_path=output_path,
        add_pauses=add_pauses,
        normalize_audio=normalize_audio,
    )

    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("error", "Dialogue generation failed")
        )

    return ApiResponse(data=result)


# =============================================================================
# Text Segmentation API - 文本分割
# =============================================================================

@router.post("/segment-text", response_model=ApiResponse[List[dict]])
async def segment_text(
    text: str,
    max_chars: Optional[int] = None,
    preserve_sentences: Optional[bool] = None,
    detect_dialogue: bool = True,
    add_pause_markers: bool = False,
):
    """Intelligently segment text for TTS processing."""
    segmenter = get_text_segmenter()

    segments = await segmenter.segment_text(
        text=text,
        max_chars=max_chars,
        preserve_sentences=preserve_sentences,
        detect_dialogue=detect_dialogue,
        add_pause_markers=add_pause_markers,
    )

    return ApiResponse(data=segments)


# =============================================================================
# Streaming TTS API - 流式TTS
# =============================================================================

@router.post("/streaming-tts", response_model=ApiResponse[dict])
async def streaming_tts(
    text: str,
    speaker: str = "aiden",
    voice_config: Optional[Dict[str, Any]] = None,
):
    """Generate streaming TTS audio."""
    streaming_tts_engine = get_streaming_tts()

    chunks = []
    async def on_chunk(chunk_data: bytes, index: int):
        chunks.append({"index": index, "size": len(chunk_data)})

    await streaming_tts_engine.generate_stream(
        text=text,
        speaker=speaker,
        voice_config=voice_config,
        callback=on_chunk,
    )

    return ApiResponse(data={
        "text": text,
        "speaker": speaker,
        "chunks": chunks,
        "total_chunks": len(chunks),
    })


# =============================================================================
# RVC Voice Conversion API - RVC语音转换
# =============================================================================

@router.post("/rvc-convert", response_model=ApiResponse[dict])
async def rvc_convert_voice(
    source_audio_path: str,
    target_voice_model: str,
    preserve_prosody: bool = True,
    preserve_timing: bool = True,
    pitch_shift: float = 0.0,
    output_path: Optional[str] = None,
):
    """Convert voice using RVC (Retrieval-based Voice Conversion)."""
    converter = get_rvc_converter()

    result = await converter.convert_voice(
        source_audio_path=source_audio_path,
        target_voice_model=target_voice_model,
        preserve_prosody=preserve_prosody,
        preserve_timing=preserve_timing,
        pitch_shift=pitch_shift,
        output_path=output_path,
    )

    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("error", "RVC conversion failed")
        )

    return ApiResponse(data=result)


# =============================================================================
# Prosody Control API - 韵律控制
# =============================================================================

@router.post("/apply-prosody", response_model=ApiResponse[dict])
async def apply_prosody(
    audio_path: str,
    prosody_config: Dict[str, Any],
    output_path: Optional[str] = None,
):
    """Apply advanced prosody modifications to audio."""
    controller = get_prosody_controller()

    result = await controller.apply_prosody(
        audio_path=audio_path,
        prosody_config=prosody_config,
        output_path=output_path,
    )

    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("error", "Prosody application failed")
        )

    return ApiResponse(data=result)


@router.get("/prosody/templates", response_model=ApiResponse[List[dict]])
async def get_prosody_templates():
    """Get available prosody templates."""
    controller = get_prosody_controller()

    templates = controller.get_prosody_templates()

    return ApiResponse(data=templates)


# =============================================================================
# Model Management API - 模型管理
# =============================================================================

@router.post("/models/preload", response_model=ApiResponse[dict])
async def preload_model(
    model_type: str = "tts",
    model_id: str = "aiden",
    quantized: bool = False,
):
    """Preload a model into memory for faster inference."""
    manager = get_model_manager()

    result = await manager.preload_model(
        model_type=model_type,
        model_id=model_id,
        quantized=quantized,
    )

    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("error", "Model preloading failed")
        )

    return ApiResponse(data=result)


@router.post("/models/switch", response_model=ApiResponse[dict])
async def switch_model(
    from_model: str,
    to_model: str,
    model_type: str = "tts",
):
    """Hot-switch between models without downtime."""
    manager = get_model_manager()

    result = await manager.switch_model(
        from_model=from_model,
        to_model=to_model,
        model_type=model_type,
    )

    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("error", "Model switching failed")
        )

    return ApiResponse(data=result)


@router.get("/models/status", response_model=ApiResponse[dict])
async def get_model_status():
    """Get status of loaded models."""
    manager = get_model_manager()

    status = manager.get_loaded_models()

    return ApiResponse(data=status)


@router.post("/models/cache/clear", response_model=ApiResponse[dict])
async def clear_model_cache(
    model_type: Optional[str] = None,
    model_id: Optional[str] = None,
):
    """Clear model cache."""
    manager = get_model_manager()

    result = manager.clear_cache(
        model_type=model_type,
        model_id=model_id,
    )

    return ApiResponse(data={"cleared": result})


# =============================================================================
# Quality Assessment API - 质量评估
# =============================================================================

@router.post("/assess-quality", response_model=ApiResponse[dict])
async def assess_quality(
    audio_path: str,
    reference_path: Optional[str] = None,
    detailed: bool = False,
):
    """Assess audio quality with AI-powered metrics."""
    assessor = get_enhanced_assessor()

    result = await assessor.assess_quality(
        audio_path=audio_path,
        reference_path=reference_path,
        detailed=detailed,
    )

    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("error", "Quality assessment failed")
        )

    return ApiResponse(data=result)


@router.post("/assess-quality/batch", response_model=ApiResponse[dict])
async def batch_assess_quality(
    audio_paths: List[str],
    reference_path: Optional[str] = None,
):
    """Assess quality of multiple audio files."""
    assessor = get_enhanced_assessor()

    result = await assessor.batch_assess(
        audio_paths=audio_paths,
        reference_path=reference_path,
    )

    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("error", "Batch assessment failed")
        )

    return ApiResponse(data=result)


@router.post("/enhance", response_model=ApiResponse[dict])
async def enhance_audio(
    audio_path: str,
    enhance_denoise: bool = True,
    enhance_volume: bool = True,
    add_compression: bool = True,
    target_lufs: float = -16.0,
):
    """Enhance audio quality with post-processing."""
    from app.services.audio_processor import AudioProcessor

    processor = AudioProcessor()

    # Resolve audio path
    path = Path("./static") / audio_path.lstrip("/")
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audio file not found: {audio_path}"
        )

    result = await processor.enhance_audio(
        audio_path=str(path),
        denoise=enhance_denoise,
        normalize_volume=enhance_volume,
        add_compression=add_compression,
        target_lufs=target_lufs,
    )

    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("error", "Audio enhancement failed")
        )

    return ApiResponse(data=result)


# =============================================================================
# SSML API - SSML处理器接口
# =============================================================================

@router.post("/ssml/parse", response_model=ApiResponse[dict])
async def parse_ssml(
    ssml_text: str,
):
    """Parse SSML markup and convert to TTS parameters.

    Args:
        ssml_text: Text containing SSML markup

    Returns:
        Parsed SSML with segments, breaks, and instructions
    """
    processor = get_ssml_processor()

    result = processor.parse_ssml(ssml_text)

    return ApiResponse(data=result)


@router.post("/ssml/to-segments", response_model=ApiResponse[List[dict]])
async def ssml_to_segments(
    ssml_text: str,
    default_voice: str = "aiden",
):
    """Convert SSML text to audio segments with TTS parameters.

    Args:
        ssml_text: Text with SSML markup
        default_voice: Default voice to use

    Returns:
        List of segments with text and TTS parameters
    """
    processor = get_ssml_processor()

    segments = processor.ssml_to_segments(ssml_text, default_voice)

    return ApiResponse(data=segments)


@router.post("/ssml/validate", response_model=ApiResponse[dict])
async def validate_ssml(
    ssml_text: str,
):
    """Validate SSML markup.

    Args:
        ssml_text: SSML text to validate

    Returns:
        Validation results with errors and warnings
    """
    processor = get_ssml_processor()

    result = processor.validate_ssml(ssml_text)

    return ApiResponse(data=result)


@router.post("/ssml/text-to-ssml", response_model=ApiResponse[dict])
async def text_to_ssml(
    text: str,
    pause_after_sentence: bool = True,
    pause_duration: int = 500,
    emphasize_keywords: bool = False,
):
    """Convert plain text to SSML markup.

    Args:
        text: Plain text to convert
        pause_after_sentence: Add pause after each sentence
        pause_duration: Pause duration in milliseconds
        emphasize_keywords: Emphasize important words

    Returns:
        SSML markup string
    """
    processor = get_ssml_processor()

    options = {
        "pause_after_sentence": pause_after_sentence,
        "pause_duration": pause_duration,
        "emphasize_keywords": emphasize_keywords,
    }

    ssml = processor.text_to_ssml(text, options)

    return ApiResponse(data={"ssml": ssml, "original_text": text})


@router.post("/ssml/generate", response_model=ApiResponse[dict])
async def generate_from_ssml(
    ssml_text: str,
    voice: str = "aiden",
    output_format: str = "mp3",
):
    """Generate audio from SSML markup.

    Args:
        ssml_text: SSML markup text
        voice: Default voice to use
        output_format: Output audio format

    Returns:
        Generated audio with metadata
    """
    processor = get_ssml_processor()

    # Convert SSML to segments
    segments = processor.ssml_to_segments(ssml_text, voice)

    # Generate audio for each segment
    from app.services.tts_engine import TTSEngineFactory, TTSMode
    import uuid
    from pathlib import Path
    import io
    from pydub import AudioSegment

    tts_engine = TTSEngineFactory.create(TTSMode.LOCAL)

    audio_segments = []
    total_duration = 0

    for segment in segments:
        if segment.get("type") == "break":
            # Add silence for breaks
            duration_ms = segment.get("duration_ms", 500)
            silence = AudioSegment.silent(duration=duration_ms)
            audio_segments.append(silence)
            total_duration += duration_ms / 1000
        else:
            # Generate audio for text segment
            text = segment.get("text", "")
            params = segment.get("parameters", {})

            try:
                audio_data, duration = await tts_engine.generate(
                    text=text,
                    speaker=params.get("voice", voice),
                )

                # Apply prosody modifications
                audio = AudioSegment.from_file(io.BytesIO(audio_data))

                # Apply speed
                if "speed" in params and params["speed"] != 1.0:
                    new_frame_rate = int(audio.frame_rate * params["speed"])
                    audio = audio._spawn(audio.raw_data, overrides={'frame_rate': new_frame_rate})
                    audio = audio.set_frame_rate(22050)

                # Apply volume
                if "volume" in params and params["volume"] != 1.0:
                    db_change = 10 * np.log10(params["volume"])
                    audio = audio + db_change

                # Apply pitch shift
                if "pitch_shift" in params and params["pitch_shift"] != 0:
                    new_sample_rate = int(audio.frame_rate * (2.0 ** (params["pitch_shift"] / 12.0)))
                    audio = audio._spawn(audio.raw_data, overrides={'frame_rate': new_sample_rate})
                    audio = audio.set_frame_rate(22050)

                audio_segments.append(audio)
                total_duration += duration

            except Exception as e:
                # Continue with next segment on error
                pass

    # Combine all audio segments
    if audio_segments:
        combined = sum(audio_segments)

        # Export to file
        output = io.BytesIO()
        combined.export(output, format=output_format)
        audio_data = output.read()

        # Save audio file
        audio_id = uuid.uuid4().hex[:8]
        audio_dir = Path("./static/audio/ssml")
        audio_dir.mkdir(parents=True, exist_ok=True)

        audio_path = audio_dir / f"ssml_{audio_id}.{output_format}"
        with open(audio_path, "wb") as f:
            f.write(audio_data)

        return ApiResponse(
            data={
                "audio_url": f"/static/audio/ssml/ssml_{audio_id}.{output_format}",
                "duration": total_duration,
                "segments_count": len(segments),
                "format": output_format,
                "ssml_text": ssml_text,
                "message": "SSML audio generated successfully",
            }
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate audio from SSML"
        )


# =============================================================================
# Sound Effects Library API - 音效库接口
# =============================================================================

@router.get("/sound-effects/packs", response_model=ApiResponse[dict])
async def get_sound_effect_packs():
    """Get all available sound effect preset packs."""
    library = get_sound_effects_library()

    packs = library.get_preset_packs()

    # Add effect counts
    pack_summary = {}
    for pack_id, pack in packs.items():
        pack_summary[pack_id] = {
            "id": pack_id,
            "name": pack["name"],
            "description": pack["description"],
            "effect_count": len(pack.get("effects", [])),
            "categories": list(set(e.get("category", "other") for e in pack.get("effects", []))),
        }

    return ApiResponse(data={"packs": pack_summary, "total": len(packs)})


@router.get("/sound-effects/packs/{pack_id}", response_model=ApiResponse[dict])
async def get_sound_effect_pack(pack_id: str):
    """Get details of a specific sound effect pack."""
    library = get_sound_effects_library()

    pack = library.get_preset_pack(pack_id)

    if not pack:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sound effect pack '{pack_id}' not found"
        )

    return ApiResponse(data=pack)


@router.get("/sound-effects/search", response_model=ApiResponse[List[dict]])
async def search_sound_effects(
    keyword: str = "",
    category: Optional[str] = None,
):
    """Search for sound effects by keyword and/or category."""
    library = get_sound_effects_library()

    results = library.search_effects(keyword, category)

    return ApiResponse(data=results)


@router.get("/sound-effects/categories", response_model=ApiResponse[List[str]])
async def get_sound_effect_categories():
    """Get all available sound effect categories."""
    library = get_sound_effects_library()

    categories = library.get_all_categories()

    return ApiResponse(data=categories)


@router.post("/sound-effects/custom", response_model=ApiResponse[dict])
async def add_custom_sound_effect(
    current_user: CurrentUserDep,
    effect_id: str,
    name: str,
    category: str = "custom",
    description: Optional[str] = None,
    tags: Optional[str] = None,
    audio_file: UploadFile = File(...),
):
    """Add a custom sound effect to the library.

    Args:
        effect_id: Unique effect ID
        name: Effect name
        category: Effect category
        description: Optional description
        tags: Comma-separated tags
        audio_file: Audio file to upload

    Returns:
        Created effect info
    """
    library = get_sound_effects_library()

    # Save uploaded file
    upload_dir = Path("./static/uploads/sound_effects")
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / f"{effect_id}_{audio_file.filename}"
    with open(file_path, "wb") as f:
        content = await audio_file.read()
        f.write(content)

    # Parse tags
    tag_list = tags.split(",") if tags else []

    # Add to library
    result = library.add_custom_effect(
        effect_id=effect_id,
        name=name,
        file_path=str(file_path),
        category=category,
        description=description,
        tags=tag_list,
    )

    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("error", "Failed to add sound effect")
        )

    return ApiResponse(data=result)


@router.delete("/sound-effects/custom/{effect_id}", response_model=ApiResponse[dict])
async def delete_custom_sound_effect(
    current_user: CurrentUserDep,
    effect_id: str,
):
    """Delete a custom sound effect."""
    library = get_sound_effects_library()

    success = library.delete_custom_effect(effect_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Custom sound effect '{effect_id}' not found"
        )

    return ApiResponse(data={"message": f"Sound effect '{effect_id}' deleted successfully"})


@router.post("/sound-effects/templates", response_model=ApiResponse[dict])
async def create_mix_template(
    current_user: CurrentUserDep,
    template_id: str,
    name: str,
    effects: List[Dict[str, Any]],
    description: Optional[str] = None,
):
    """Create a reusable sound effects mix template.

    Args:
        template_id: Unique template ID
        name: Template name
        effects: List of effects with timing and volume
        description: Optional description

    Returns:
        Created template
    """
    library = get_sound_effects_library()

    template = library.create_mix_template(
        template_id=template_id,
        name=name,
        effects=effects,
        description=description,
    )

    return ApiResponse(data=template)


@router.get("/sound-effects/templates", response_model=ApiResponse[List[dict]])
async def get_mix_templates():
    """Get all mix templates."""
    library = get_sound_effects_library()

    templates = library.get_all_mix_templates()

    return ApiResponse(data=templates)


@router.get("/sound-effects/templates/{template_id}", response_model=ApiResponse[dict])
async def get_mix_template(template_id: str):
    """Get a specific mix template."""
    library = get_sound_effects_library()

    template = library.get_mix_template(template_id)

    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mix template '{template_id}' not found"
        )

    return ApiResponse(data=template)


@router.post("/sound-effects/apply-template", response_model=ApiResponse[dict])
async def apply_mix_template(
    speech_audio_path: str,
    template_id: str,
):
    """Apply a sound effects mix template to speech audio.

    Args:
        speech_audio_path: Path to speech audio file
        template_id: Mix template ID to apply

    Returns:
        Mixed audio with effects applied
    """
    from app.services.audio_processor import get_audio_mixer

    library = get_sound_effects_library()
    mixer = get_audio_mixer()

    # Get template
    template = library.get_mix_template(template_id)
    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mix template '{template_id}' not found"
        )

    # Resolve effect paths
    sound_effects = []
    for effect_spec in template.get("effects", []):
        effect_id = effect_spec.get("effect_id")
        pack_id = effect_spec.get("pack", "custom")

        if pack_id == "custom":
            effect = library.custom_effects.get(effect_id)
        else:
            effect = library.get_effect_from_pack(pack_id, effect_id)

        if effect:
            effect_path = library.resolve_effect_path(effect)
            if effect_path:
                sound_effects.append({
                    "file": effect_path,
                    "time": effect_spec.get("time", 0),
                    "volume": effect_spec.get("volume", 0.3),
                    "fade": effect_spec.get("fade", 0.1),
                })

    # Mix audio with effects
    result = await mixer.mix_audio(
        speech_audio_path=speech_audio_path,
        sound_effects=sound_effects,
    )

    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("error", "Failed to apply mix template")
        )

    return ApiResponse(data={
        **result,
        "template_id": template_id,
        "template_name": template.get("name"),
    })

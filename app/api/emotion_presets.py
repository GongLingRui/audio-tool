"""
Emotion Presets API Routes for Audiobook Narration
Provides scenario-based emotion presets for different audiobook scenarios
"""

from fastapi import APIRouter
from typing import Literal
from pydantic import BaseModel

from app.schemas.common import ApiResponse

router = APIRouter()


class EmotionParameters(BaseModel):
    """Emotion parameters for TTS generation."""
    happiness: float | None = None
    sadness: float | None = None
    anger: float | None = None
    fear: float | None = None
    surprise: float | None = None
    neutral: float | None = None
    energy: float | None = None
    tempo: float | None = None
    pitch: float | None = None
    volume: float | None = None


class ScenarioPreset(BaseModel):
    """Scenario-based emotion preset for audiobook narration."""
    id: str
    name: str
    category: Literal['narration', 'dialogue', 'action', 'atmosphere']
    description: str
    emotion: EmotionParameters
    example_text: str
    usage: list[str]


# Scenario-based emotion presets for audiobook narration
SCENARIO_PRESETS: list[dict] = [
    # === Narration Presets ===
    {
        'id': 'narrator_calm',
        'name': '叙述者-平静',
        'category': 'narration',
        'description': '平稳、客观的叙述声音，适合章节开篇和背景介绍',
        'emotion': {
            'neutral': 0.9,
            'energy': 0.8,
            'tempo': 1.0,
            'pitch': 0,
        },
        'example_text': '第一章开始。这是一个关于勇气与成长的故事，发生在一个遥远的小镇。',
        'usage': ['章节开篇', '背景介绍', '过渡段落'],
    },
    {
        'id': 'narrator_tense',
        'name': '叙述者-紧张',
        'category': 'narration',
        'description': '紧凑、略带紧迫感的叙述，适合悬疑和高潮场景',
        'emotion': {
            'neutral': 0.6,
            'fear': 0.3,
            'energy': 1.3,
            'tempo': 1.2,
            'pitch': 2,
        },
        'example_text': '就在这时，一个黑影从角落里闪了出来，所有人的心都提到了嗓子眼。',
        'usage': ['悬疑场景', '紧张时刻', '高潮段落'],
    },
    {
        'id': 'narrator_mysterious',
        'name': '叙述者-神秘',
        'category': 'narration',
        'description': '低沉、略带沙哑的叙述，适合营造神秘氛围',
        'emotion': {
            'neutral': 0.7,
            'energy': 0.7,
            'tempo': 0.9,
            'pitch': -2,
            'volume': 0.9,
        },
        'example_text': '古老的诅咒被重新唤醒，而这一切，都源于一个被遗忘的约定。',
        'usage': ['神秘场景', '回忆片段', '超自然元素'],
    },
    {
        'id': 'narrator_humorous',
        'name': '叙述者-幽默',
        'category': 'narration',
        'description': '轻松、幽默的叙述风格，适合喜剧场景',
        'emotion': {
            'happiness': 0.6,
            'neutral': 0.4,
            'energy': 1.1,
            'tempo': 1.05,
            'pitch': 1,
        },
        'example_text': '当然，他很快发现，这个所谓的"绝妙计划"其实漏洞百出。',
        'usage': ['喜剧场景', '轻松时刻', '讽刺段落'],
    },

    # === Dialogue Presets ===
    {
        'id': 'dialogue_conversation',
        'name': '对话-日常',
        'category': 'dialogue',
        'description': '自然、日常的对话语气',
        'emotion': {
            'neutral': 0.8,
            'energy': 1.0,
            'tempo': 1.0,
        },
        'example_text': '你今天感觉怎么样？还可以，就是有点累。',
        'usage': ['日常交流', '闲聊', '普通对话'],
    },
    {
        'id': 'dialogue_intimate',
        'name': '对话-亲密',
        'category': 'dialogue',
        'description': '温柔、亲密的对话，适合亲密场景',
        'emotion': {
            'happiness': 0.4,
            'neutral': 0.6,
            'energy': 0.7,
            'tempo': 0.9,
            'pitch': -1,
            'volume': 0.9,
        },
        'example_text': '我会一直陪着你的，无论发生什么。',
        'usage': ['浪漫场景', '亲密对话', '情感告白'],
    },
    {
        'id': 'dialogue_argument',
        'name': '对话-争吵',
        'category': 'dialogue',
        'description': '激动、带有攻击性的对话，适合争吵场景',
        'emotion': {
            'anger': 0.7,
            'neutral': 0.3,
            'energy': 1.4,
            'tempo': 1.3,
            'pitch': 3,
            'volume': 1.2,
        },
        'example_text': '你怎么能这样说！我根本没有那么想过！',
        'usage': ['争吵场景', '激烈辩论', '愤怒爆发'],
    },
    {
        'id': 'dialogue_suspicion',
        'name': '对话-怀疑',
        'category': 'dialogue',
        'description': '怀疑、试探的对话语气',
        'emotion': {
            'neutral': 0.5,
            'fear': 0.2,
            'energy': 0.9,
            'tempo': 0.95,
            'pitch': 1,
        },
        'example_text': '你确定那天晚上一直都在图书馆吗？',
        'usage': ['质问', '试探', '怀疑'],
    },

    # === Action Presets ===
    {
        'id': 'action_chase',
        'name': '动作-追逐',
        'category': 'action',
        'description': '快速、紧张的节奏，适合动作场景',
        'emotion': {
            'neutral': 0.3,
            'fear': 0.3,
            'energy': 1.5,
            'tempo': 1.4,
            'volume': 1.1,
        },
        'example_text': '他转身就跑，身后传来了急促的脚步声！',
        'usage': ['追逐场景', '逃跑', '紧急行动'],
    },
    {
        'id': 'action_fight',
        'name': '动作-战斗',
        'category': 'action',
        'description': '激烈、充满力量的战斗场景',
        'emotion': {
            'anger': 0.6,
            'energy': 1.5,
            'tempo': 1.3,
            'volume': 1.3,
        },
        'example_text': '这一拳用尽了他全身的力气，直接命中了目标的要害！',
        'usage': ['战斗', '冲突', '武力对抗'],
    },
    {
        'id': 'action_revelation',
        'name': '动作-揭示',
        'category': 'action',
        'description': '重大真相揭示时的震惊语气',
        'emotion': {
            'surprise': 0.8,
            'neutral': 0.2,
            'energy': 1.2,
            'tempo': 0.8,
            'volume': 1.1,
        },
        'example_text': '你永远想不到，真相竟然如此！',
        'usage': ['真相揭示', '重大发现', '剧情转折'],
    },

    # === Atmosphere Presets ===
    {
        'id': 'atmosphere_sad',
        'name': '氛围-悲伤',
        'category': 'atmosphere',
        'description': '悲伤、忧郁的氛围，适合悲剧场景',
        'emotion': {
            'sadness': 0.8,
            'neutral': 0.2,
            'energy': 0.6,
            'tempo': 0.85,
            'pitch': -2,
        },
        'example_text': '他永远地离开了，只留下无尽的思念和遗憾。',
        'usage': ['悲剧场景', '离别', '死亡'],
    },
    {
        'id': 'atmosphere_hopeful',
        'name': '氛围-希望',
        'category': 'atmosphere',
        'description': '充满希望和光明的氛围',
        'emotion': {
            'happiness': 0.7,
            'neutral': 0.3,
            'energy': 1.1,
            'tempo': 1.05,
            'pitch': 1,
        },
        'example_text': '黎明终于到来，新的希望就在眼前。',
        'usage': ['转折点', '希望重现', '新生'],
    },
    {
        'id': 'atmosphere_ominous',
        'name': '氛围-不祥',
        'category': 'atmosphere',
        'description': '不祥、危险的氛围预兆',
        'emotion': {
            'neutral': 0.5,
            'fear': 0.4,
            'energy': 0.8,
            'tempo': 0.9,
            'pitch': -1,
        },
        'example_text': '天色渐渐暗了下来，风里似乎夹杂着危险的气息。',
        'usage': ['危险预兆', '不祥之兆', '暴风雨前'],
    },
    {
        'id': 'atmosphere_romantic',
        'name': '氛围-浪漫',
        'category': 'atmosphere',
        'description': '浪漫、温馨的情感氛围',
        'emotion': {
            'happiness': 0.5,
            'neutral': 0.5,
            'energy': 0.8,
            'tempo': 0.95,
            'pitch': 0,
        },
        'example_text': '月光洒在他们身上，这一刻仿佛世界都静止了。',
        'usage': ['浪漫场景', '约会', '情感升华'],
    },
]


@router.get("/emotion-presets", response_model=ApiResponse[list[ScenarioPreset]])
async def get_emotion_presets():
    """
    Get all scenario-based emotion presets for audiobook narration.

    Returns a comprehensive list of emotion presets organized by category:
    - narration: For narrative sections
    - dialogue: For character conversations
    - action: For action scenes
    - atmosphere: For setting mood/atmosphere

    Each preset includes:
    - id: Unique identifier
    - name: Display name in Chinese
    - category: Preset category
    - description: Usage description
    - emotion: Emotion parameters for TTS
    - example_text: Example text showing usage
    - usage: List of applicable scenarios
    """
    return ApiResponse(data=[ScenarioPreset(**preset) for preset in SCENARIO_PRESETS])


@router.get("/emotion-presets/category/{category}", response_model=ApiResponse[list[ScenarioPreset]])
async def get_presets_by_category(category: Literal['narration', 'dialogue', 'action', 'atmosphere']):
    """
    Get emotion presets filtered by category.

    Categories:
    - narration: Narrative voice presets
    - dialogue: Character dialogue presets
    - action: Action scene presets
    - atmosphere: Mood/atmosphere presets
    """
    filtered = [p for p in SCENARIO_PRESETS if p['category'] == category]
    return ApiResponse(data=[ScenarioPreset(**preset) for preset in filtered])


@router.get("/emotion-presets/recommend", response_model=ApiResponse[ScenarioPreset])
async def get_recommended_preset(text: str):
    """
    Get a recommended emotion preset based on text analysis.

    Analyzes the input text for keywords and suggests the most appropriate
    emotion preset for audiobook narration.

    Query Parameters:
        text: The text to analyze for emotion context
    """
    lower_text = text.lower()

    # Detect keywords and recommend preset
    if '追逐' in lower_text or '逃跑' in lower_text or '快跑' in lower_text:
        preset_id = 'action_chase'
    elif '战斗' in lower_text or '打斗' in lower_text or '攻击' in lower_text:
        preset_id = 'action_fight'
    elif '悲伤' in lower_text or '眼泪' in lower_text or '死亡' in lower_text:
        preset_id = 'atmosphere_sad'
    elif '喜欢' in lower_text or '爱' in lower_text or '月光' in lower_text:
        preset_id = 'atmosphere_romantic'
    elif '争吵' in lower_text or '愤怒' in lower_text or '吼叫' in lower_text:
        preset_id = 'dialogue_argument'
    elif '秘密' in lower_text or '隐藏' in lower_text or '奇怪' in lower_text:
        preset_id = 'narrator_mysterious'
    else:
        preset_id = 'narrator_calm'

    for preset in SCENARIO_PRESETS:
        if preset['id'] == preset_id:
            return ApiResponse(data=ScenarioPreset(**preset))

    return ApiResponse(data=ScenarioPreset(**SCENARIO_PRESETS[0]))  # Default to calm narration


@router.get("/emotion-presets/{preset_id}", response_model=ApiResponse[ScenarioPreset])
async def get_preset_by_id(preset_id: str):
    """
    Get a specific emotion preset by ID.

    Returns a single preset with all details including emotion parameters
    that can be directly used in TTS generation.
    """
    for preset in SCENARIO_PRESETS:
        if preset['id'] == preset_id:
            return ApiResponse(data=ScenarioPreset(**preset))

    return ApiResponse(
        success=False,
        error={
            'code': 'PRESET_NOT_FOUND',
            'message': f'Emotion preset "{preset_id}" not found'
        }
    )

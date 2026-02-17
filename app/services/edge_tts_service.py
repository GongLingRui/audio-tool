"""
Edge TTS Service
Microsoft Edge TTS - 轻量级高质量语音合成，无需下载模型
"""

import io
import logging
import asyncio
import tempfile
import os
from typing import Optional, Dict, Any, List
from pathlib import Path

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    edge_tts = None

logger = logging.getLogger(__name__)


class EdgeTTSService:
    """
    Edge TTS 服务
    - 无需下载模型
    - 支持多种语言和声音
    - 高质量语音输出
    - 响应快速
    """

    # 可用的中文声音（从轻量到高质量）
    CHINESE_VOICES = {
        # 标准女声
        "xiaoxiao": {
            "name": "zh-CN-XiaoxiaoNeural",
            "description": "晓晓 - 标准女声",
            "gender": "Female",
            "quality": "standard",
        },
        # 标准男声
        "yunxi": {
            "name": "zh-CN-YunxiNeural",
            "description": "云希 - 标准男声",
            "gender": "Male",
            "quality": "standard",
        },
        # 温柔女声
        "xiaoyi": {
            "name": "zh-CN-XiaoyiNeural",
            "description": "晓伊 - 温柔女声",
            "gender": "Female",
            "quality": "soft",
        },
        # 成熟男声
        "yunyang": {
            "name": "zh-CN-YunyangNeural",
            "description": "云扬 - 成熟男声",
            "gender": "Male",
            "quality": "mature",
        },
        # 活泼女声
        "xiaomeng": {
            "name": "zh-CN-XiaomengNeural",
            "description": "晓梦 - 活泼女声",
            "gender": "Female",
            "quality": "lively",
        },
    }

    def __init__(
        self,
        voice: str = "xiaoxiao",
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
    ):
        """
        初始化 Edge TTS 服务

        Args:
            voice: 声音ID (xiaoxiao, yunxi, etc.)
            rate: 语速 (如 "+20%", "-10%")
            volume: 音量 (如 "+10%", "-5%")
            pitch: 音调 (如 "+5Hz", "-10Hz")
        """
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.pitch = pitch

        voice_config = self.CHINESE_VOICES.get(voice, self.CHINESE_VOICES["xiaoxiao"])
        self.voice_name = voice_config["name"]

        logger.info(f"Edge TTS 服务初始化: {voice_config['description']}")

    async def initialize(self):
        """初始化服务（Edge TTS不需要预加载）"""
        if not EDGE_TTS_AVAILABLE:
            raise ImportError("edge-tts 未安装。请运行: pip install edge-tts")
        logger.info("Edge TTS 服务就绪")

    async def generate_speech(
        self,
        text: str,
        voice_sample: Optional[bytes] = None,
        emotion: Optional[Dict[str, float]] = None,
        speed: float = 1.0,
    ) -> Dict[str, Any]:
        """
        生成语音

        Args:
            text: 输入文本
            voice_sample: 不支持（Edge TTS不需要）
            emotion: 情感参数（可选）
            speed: 语速倍率

        Returns:
            包含音频数据的字典
        """
        await self.initialize()

        try:
            logger.info(f"生成语音: {text[:100]}...")

            # 根据情感调整参数
            rate = self.rate
            pitch = self.pitch
            volume = self.volume

            if emotion:
                # 根据情感调整语速和音调
                if emotion.get("happiness", 0) > 0.5:
                    rate = "+10%"
                    pitch = "+5Hz"
                elif emotion.get("sadness", 0) > 0.5:
                    rate = "-10%"
                    pitch = "-5Hz"
                elif emotion.get("anger", 0) > 0.5:
                    rate = "+20%"
                    volume = "+10%"
                elif emotion.get("energy", 1.0) > 1.2:
                    rate = "+15%"

            # 应用速度倍率
            if speed != 1.0:
                speed_percent = int((speed - 1.0) * 100)
                rate = f"{speed_percent:+d}%"

            # 创建communicate对象
            communicate = edge_tts.Communicate(
                text=text,
                voice=self.voice_name,
                rate=rate,
                volume=volume,
                pitch=pitch,
            )

            # 生成音频数据
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            # 计算时长
            duration = len(audio_data) / 24000 / 2  # 24kHz, 16-bit = 2 bytes per sample

            logger.info(f"✓ 语音生成成功: {duration:.2f}s, {len(audio_data)} bytes")

            return {
                "audio": audio_data,
                "sample_rate": 24000,
                "duration": duration,
                "format": "MP3",  # Edge TTS返回MP3格式
                "model": f"edge-tts-{self.voice}",
                "device": "cloud",
            }

        except Exception as e:
            logger.error(f"Edge TTS 生成失败: {e}")
            raise

    async def get_available_voices(self) -> List[Dict[str, str]]:
        """获取可用的声音列表"""
        voices = []
        for voice_id, config in self.CHINESE_VOICES.items():
            voices.append({
                "id": voice_id,
                "name": config["description"],
                "language": "zh-CN",
                "gender": config["gender"],
                "quality": config["quality"],
            })
        return voices

    async def get_supported_languages(self) -> List[Dict[str, Any]]:
        """获取支持的语言"""
        return [
            {
                "language_code": "zh-CN",
                "language_name": "Chinese (Mandarin)",
                "sample_rate": 24000,
                "model_type": "neural",
            },
            {
                "language_code": "en-US",
                "language_name": "English (US)",
                "sample_rate": 24000,
                "model_type": "neural",
            },
            {
                "language_code": "ja-JP",
                "language_name": "Japanese",
                "sample_rate": 24000,
                "model_type": "neural",
            },
        ]


# 单例实例
_edge_tts_service: Optional[EdgeTTSService] = None


def get_edge_tts_service() -> EdgeTTSService:
    """获取或创建 Edge TTS 服务单例"""
    global _edge_tts_service
    if _edge_tts_service is None:
        _edge_tts_service = EdgeTTSService()
    return _edge_tts_service

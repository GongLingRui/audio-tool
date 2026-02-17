"""
Audio Quality Checker Service
Provides quality checks and guidance for voice cloning reference audio
"""

import logging
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class QualityIssueType(Enum):
    """Types of audio quality issues."""
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    TOO_QUIET = "too_quiet"
    TOO_LOUD = "too_loud"
    LOW_DYNAMIC_RANGE = "low_dynamic_range"
    POSSIBLE_CLIPPING = "possible_clipping"
    HIGH_NOISE_FLOOR = "high_noise_floor"
    UNSUPPORTED_FORMAT = "unsupported_format"
    EMPTY_FILE = "empty_file"


@dataclass
class QualityIssue:
    """Represents a quality issue."""
    type: QualityIssueType
    severity: str  # 'error', 'warning', 'info'
    message: str
    suggestion: str
    value: Optional[float] = None


@dataclass
class AudioQualityReport:
    """Report from audio quality check."""
    is_acceptable: bool
    overall_score: float  # 0-100
    duration: float
    sample_rate: int
    channels: int
    issues: List[QualityIssue]
    recommendations: List[str]

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "is_acceptable": self.is_acceptable,
            "overall_score": self.overall_score,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "issues": [
                {
                    "type": issue.type.value,
                    "severity": issue.severity,
                    "message": issue.message,
                    "suggestion": issue.suggestion,
                    "value": issue.value,
                }
                for issue in self.issues
            ],
            "recommendations": self.recommendations,
        }


class AudioQualityChecker:
    """
    Service for checking audio quality for voice cloning.

    Provides guidance on:
    - Optimal recording length (3-5 minutes recommended)
    - Loudness targets (-16 to -3 dBFS)
    - Dynamic range
    - Noise floor
    - Format compatibility
    """

    # Quality thresholds
    MIN_DURATION = 5.0  # seconds
    OPTIMAL_MIN_DURATION = 30.0  # seconds
    OPTIMAL_MAX_DURATION = 300.0  # seconds (5 minutes)
    MAX_DURATION = 600.0  # seconds (10 minutes)

    TARGET_LOUDNESS_DBFS = -12.0  # Target loudness
    MIN_LOUDNESS_DBFS = -25.0  # Too quiet below this
    MAX_LOUDNESS_DBFS = -3.0  # Too loud above this (risk of clipping)

    MIN_DYNAMIC_RANGE_DB = 10.0  # Minimum dynamic range
    MAX_NOISE_FLOOR_DB = -50.0  # Maximum acceptable noise floor

    # Supported audio formats
    SUPPORTED_FORMATS = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg']

    @classmethod
    def get_recording_guidelines(cls) -> Dict:
        """
        Get comprehensive recording guidelines for voice cloning.

        Returns guidelines that can be displayed to users.
        """
        return {
            "overview": "高质量语音克隆需要清晰、一致的录音样本",
            "duration": {
                "minimum": cls.MIN_DURATION,
                "recommended_min": cls.OPTIMAL_MIN_DURATION,
                "recommended_max": cls.OPTIMAL_MAX_DURATION,
                "description": f"建议录制 {cls.OPTIMAL_MIN_DURATION}-{cls.OPTIMAL_MAX_DURATION/60:.0f} 分钟的音频"
            },
            "environment": {
                "noise_level": "安静环境，避免背景噪音",
                "reverb": "减少回声，避免在空旷房间录制",
                "equipment": "使用质量较好的麦克风，避免手机内置麦克风",
            },
            "technique": {
                "distance": "保持麦克风距离一致（约15-20cm）",
                "consistency": "使用相同的设备和设置录制所有样本",
                "performance": "包含多种情感和语速，但保持自然",
                "pauses": "句子间自然停顿1-1.5秒",
            },
            "technical": {
                "sample_rate": "推荐 44.1kHz 或 48kHz",
                "bit_depth": "推荐 16-bit 或 24-bit",
                "format": "WAV 格式最佳，MP3 也可接受（比特率 ≥192kbps）",
                "peak_level": f"峰值电平应在 {cls.MAX_LOUDNESS_DBFS} 到 {cls.MIN_LOUDNESS_DBFS} dBFS 之间",
            },
            "content": {
                "variety": "包含不同语速、情感和句子长度",
                "naturalness": "保持自然说话，避免过度表演",
                "clean": "去除口误、重复和过多的填充词",
                "examples": "至少3-5段不同的内容，每段5-15秒",
            }
        }

    @classmethod
    def get_consent_requirements(cls) -> Dict:
        """Get consent and ethics requirements for voice cloning."""
        return {
            "consent_required": True,
            "consent_text": "我确认我有权使用此音频样本进行语音克隆，并同意将其用于生成有声内容。",
            "agreement_points": [
                "我确认这是我的声音或我有权使用它",
                "我同意使用此样本来创建AI语音克隆",
                "我了解生成的语音将用于有声书制作",
                "我承诺不会滥用生成的语音内容",
            ],
            "usage_limitations": [
                "不得用于欺诈或误导性内容",
                "不得用于冒充他人进行非法活动",
                "遵守当地法律法规和平台政策",
            ]
        }

    @classmethod
    async def check_audio_file(
        cls,
        file_path: str,
        detailed_analysis: bool = True,
    ) -> AudioQualityReport:
        """
        Check audio file quality for voice cloning suitability.

        Args:
            file_path: Path to audio file
            detailed_analysis: Whether to perform detailed analysis (slower)

        Returns:
            AudioQualityReport with findings and recommendations
        """
        issues = []
        recommendations = []
        score = 100.0

        path = Path(file_path)

        # Check file exists
        if not path.exists():
            return AudioQualityReport(
                is_acceptable=False,
                overall_score=0,
                duration=0,
                sample_rate=0,
                channels=0,
                issues=[
                    QualityIssue(
                        type=QualityIssueType.EMPTY_FILE,
                        severity="error",
                        message="文件不存在",
                        suggestion="请上传有效的音频文件"
                    )
                ],
                recommendations=[]
            )

        # Check format
        if path.suffix.lower() not in cls.SUPPORTED_FORMATS:
            issues.append(QualityIssue(
                type=QualityIssueType.UNSUPPORTED_FORMAT,
                severity="error",
                message=f"不支持的音频格式: {path.suffix}",
                suggestion=f"请使用以下格式之一: {', '.join(cls.SUPPORTED_FORMATS)}"
            ))
            score -= 50

        # Try to analyze audio with pydub or ffmpeg
        try:
            from app.services.audio_processor import AudioProcessor
            processor = AudioProcessor()

            # Get basic audio info
            info = await processor.get_audio_info(str(path))

            duration = info.get('duration', 0)
            sample_rate = info.get('sample_rate', 0)
            channels = info.get('channels', 0)

            # Check duration
            if duration < cls.MIN_DURATION:
                issues.append(QualityIssue(
                    type=QualityIssueType.TOO_SHORT,
                    severity="error",
                    message=f"音频太短: {duration:.1f}秒",
                    suggestion=f"至少需要 {cls.MIN_DURATION:.0f}秒，建议 {cls.OPTIMAL_MIN_DURATION:.0f}秒以上",
                    value=duration
                ))
                score -= 30
            elif duration < cls.OPTIMAL_MIN_DURATION:
                issues.append(QualityIssue(
                    type=QualityIssueType.TOO_SHORT,
                    severity="warning",
                    message=f"音频偏短: {duration:.1f}秒",
                    suggestion=f"建议增加到 {cls.OPTIMAL_MIN_DURATION:.0f}秒以上以获得更好效果",
                    value=duration
                ))
                score -= 10
            elif duration > cls.MAX_DURATION:
                issues.append(QualityIssue(
                    type=QualityIssueType.TOO_LONG,
                    severity="warning",
                    message=f"音频过长: {duration:.1f}秒",
                    suggestion=f"建议分段录制，每段 {cls.OPTIMAL_MAX_DURATION/60:.0f}分钟以内",
                    value=duration
                ))
                score -= 5

            # Detailed analysis if requested
            if detailed_analysis and duration >= cls.MIN_DURATION:
                # Analyze loudness
                loudness_info = await processor.analyze_loudness(str(path))
                if 'peak_db' in loudness_info:
                    peak_db = loudness_info['peak_db']

                    if peak_db > cls.MAX_LOUDNESS_DBFS:
                        issues.append(QualityIssue(
                            type=QualityIssueType.POSSIBLE_CLIPPING,
                            severity="warning",
                            message=f"峰值电平过高: {peak_db:.1f} dBFS",
                            suggestion=f"建议降低音量，峰值应在 {cls.MAX_LOUDNESS_DBFS} dBFS 以下",
                            value=peak_db
                        ))
                        score -= 10
                    elif peak_db < cls.MIN_LOUDNESS_DBFS:
                        issues.append(QualityIssue(
                            type=QualityIssueType.TOO_QUIET,
                            severity="warning",
                            message=f"音量过低: {peak_db:.1f} dBFS",
                            suggestion=f"建议提高音量至约 {cls.TARGET_LOUDNESS_DBFS} dBFS",
                            value=peak_db
                        ))
                        score -= 10

                # Check dynamic range
                if 'dynamic_range_db' in loudness_info:
                    dynamic_range = loudness_info['dynamic_range_db']
                    if dynamic_range < cls.MIN_DYNAMIC_RANGE_DB:
                        issues.append(QualityIssue(
                            type=QualityIssueType.LOW_DYNAMIC_RANGE,
                            severity="info",
                            message=f"动态范围较小: {dynamic_range:.1f} dB",
                            suggestion="可以接受，但更大的动态范围会更好",
                            value=dynamic_range
                        ))
                        score -= 5

                # Check noise floor if available
                if 'noise_floor_db' in loudness_info:
                    noise_floor = loudness_info['noise_floor_db']
                    if noise_floor > cls.MAX_NOISE_FLOOR_DB:
                        issues.append(QualityIssue(
                            type=QualityIssueType.HIGH_NOISE_FLOOR,
                            severity="warning",
                            message=f"背景噪音较高: {noise_floor:.1f} dB",
                            suggestion="建议在更安静的环境中重新录制",
                            value=noise_floor
                        ))
                        score -= 10

        except Exception as e:
            logger.warning(f"Failed to analyze audio file: {e}")
            issues.append(QualityIssue(
                type=QualityIssueType.UNSUPPORTED_FORMAT,
                severity="error",
                message="无法分析音频文件",
                suggestion="请确保文件格式正确且未损坏"
            ))
            score -= 50

        # Generate recommendations based on issues
        if not any(i.type == QualityIssueType.TOO_SHORT for i in issues if i.severity == "error"):
            recommendations.append("音频质量检查通过，可用于语音克隆")

            if duration < cls.OPTIMAL_MIN_DURATION:
                recommendations.append(f"建议添加更多样本至 {cls.OPTIMAL_MIN_DURATION:.0f} 秒以上")

            if not any(i.type == QualityIssueType.TOO_QUIET for i in issues):
                recommendations.append("音量适中")

            if not any(i.type == QualityIssueType.HIGH_NOISE_FLOOR for i in issues):
                recommendations.append("背景噪音控制良好")

        is_acceptable = score >= 50 and not any(
            i.severity == "error" for i in issues
        )

        return AudioQualityReport(
            is_acceptable=is_acceptable,
            overall_score=max(0, min(100, score)),
            duration=duration if 'duration' in locals() else 0,
            sample_rate=sample_rate if 'sample_rate' in locals() else 0,
            channels=channels if 'channels' in locals() else 0,
            issues=issues,
            recommendations=recommendations,
        )

    @classmethod
    async def check_multiple_files(
        cls,
        file_paths: List[str],
    ) -> Tuple[AudioQualityReport, List[AudioQualityReport]]:
        """
        Check multiple audio files and provide combined report.

        Useful for batch uploads of voice cloning samples.

        Returns:
            Tuple of (combined_report, individual_reports)
        """
        individual_reports = []
        all_issues = []

        total_duration = 0
        min_sample_rate = float('inf')
        min_channels = float('inf')

        for file_path in file_paths:
            report = await cls.check_audio_file(file_path)
            individual_reports.append(report)
            all_issues.extend(report.issues)

            total_duration += report.duration
            if report.sample_rate > 0:
                min_sample_rate = min(min_sample_rate, report.sample_rate)
            if report.channels > 0:
                min_channels = min(min_channels, report.channels)

        # Check if total duration meets recommendations
        avg_score = sum(r.overall_score for r in individual_reports) / len(individual_reports)

        if total_duration < cls.OPTIMAL_MIN_DURATION:
            all_issues.append(QualityIssue(
                type=QualityIssueType.TOO_SHORT,
                severity="warning",
                message=f"总时长不足: {total_duration:.1f}秒",
                suggestion=f"建议总时长至少 {cls.OPTIMAL_MIN_DURATION:.0f}秒",
                value=total_duration
            ))

        is_acceptable = (
            avg_score >= 50 and
            total_duration >= cls.MIN_DURATION and
            not any(i.severity == "error" for i in all_issues)
        )

        combined_report = AudioQualityReport(
            is_acceptable=is_acceptable,
            overall_score=avg_score,
            duration=total_duration,
            sample_rate=int(min_sample_rate) if min_sample_rate != float('inf') else 0,
            channels=int(min_channels) if min_channels != float('inf') else 0,
            issues=all_issues,
            recommendations=[
                f"已检查 {len(file_paths)} 个文件",
                f"总时长: {total_duration:.1f}秒",
            ],
        )

        return combined_report, individual_reports


# Singleton instance
_audio_quality_checker: Optional[AudioQualityChecker] = None


def get_audio_quality_checker() -> AudioQualityChecker:
    """Get the singleton audio quality checker instance."""
    global _audio_quality_checker
    if _audio_quality_checker is None:
        _audio_quality_checker = AudioQualityChecker()
    return _audio_quality_checker

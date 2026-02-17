"""
Smart Error Handler - 智能错误处理系统
Provides user-friendly error messages with automatic fix suggestions
"""
import re
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class SmartErrorHandler:
    """
    Smart error handler that provides user-friendly error messages
    with automatic fix suggestions and error analysis.
    """

    # Error patterns and their solutions
    ERROR_PATTERNS = {
        # File/Upload errors
        "file_not_found": {
            "patterns": ["file not found", "no such file", "无法找到文件", "文件不存在"],
            "friendly_message": "找不到指定的文件",
            "causes": [
                "文件可能已被删除或移动",
                "文件路径可能输入错误",
                "可能没有上传权限"
            ],
            "solutions": [
                "请检查文件路径是否正确",
                "尝试重新上传文件",
                "联系管理员检查文件存储状态"
            ],
            "auto_fix": "无法自动修复，需要手动操作"
        },
        "file_too_large": {
            "patterns": ["file too large", "文件过大", "超出大小限制"],
            "friendly_message": "文件大小超出限制",
            "causes": [
                "文件超过了系统允许的最大大小"
            ],
            "solutions": [
                "压缩文件后重新上传",
                "分段处理大文件",
                "联系管理员增加文件大小限制"
            ],
            "auto_fix": "无法自动修复，建议压缩文件"
        },
        "invalid_file_format": {
            "patterns": ["invalid format", "unsupported format", "不支持的格式", "格式错误"],
            "friendly_message": "文件格式不支持",
            "causes": [
                "上传的文件格式不在支持列表中",
                "文件可能已损坏"
            ],
            "solutions": [
                "使用支持的格式：MP3、WAV、M4A、FLAC",
                "检查文件是否完整",
                "使用音频转换工具转换格式"
            ],
            "auto_fix": "建议转换为MP3格式"
        },

        # TTS/Audio errors
        "tts_engine_error": {
            "patterns": ["tts engine", "tts failed", "语音合成失败", "tts错误"],
            "friendly_message": "语音合成出现问题",
            "causes": [
                "TTS引擎可能正在初始化",
                "请求的文本可能包含不支持的字符",
                "系统资源可能不足"
            ],
            "solutions": [
                "稍后重试",
                "简化文本内容",
                "检查网络连接",
                "联系技术支持"
            ],
            "auto_fix": "建议稍后重试"
        },
        "voice_not_found": {
            "patterns": ["voice not found", "音色不存在", "找不到语音", "invalid voice"],
            "friendly_message": "找不到指定的音色",
            "causes": [
                "音色可能已被删除",
                "音色名称可能输入错误"
            ],
            "solutions": [
                "从音色列表中选择有效的音色",
                "检查音色名称拼写",
                "刷新页面重新加载音色列表"
            ],
            "auto_fix": "选择默认音色"
        },
        "audio_processing_failed": {
            "patterns": ["audio processing", "音频处理失败", "处理音频时出错"],
            "friendly_message": "音频处理失败",
            "causes": [
                "音频文件可能已损坏",
                "音频格式可能不支持",
                "处理所需的内存可能不足"
            ],
            "solutions": [
                "检查音频文件是否完整",
                "尝试使用其他音频格式",
                "减少音频文件大小",
                "联系技术支持"
            ],
            "auto_fix": "尝试重新处理"
        },

        # Authentication errors
        "unauthorized": {
            "patterns": ["unauthorized", "未授权", "401", "认证失败"],
            "friendly_message": "身份认证失败",
            "causes": [
                "登录会话可能已过期",
                "账号可能已被禁用"
            ],
            "solutions": [
                "重新登录",
                "检查账号状态",
                "清除浏览器缓存后重试"
            ],
            "auto_fix": "跳转到登录页面"
        },
        "permission_denied": {
            "patterns": ["permission denied", "forbidden", "403", "权限不足"],
            "friendly_message": "没有操作权限",
            "causes": [
                "您的账号可能没有执行此操作的权限",
                "资源可能属于其他用户"
            ],
            "solutions": [
                "联系管理员申请相应权限",
                "确认您有权限访问此资源"
            ],
            "auto_fix": "无法自动修复"
        },

        # Rate limit/Quota errors
        "rate_limit_exceeded": {
            "patterns": ["rate limit", "429", "请求过于频繁", "超出限制"],
            "friendly_message": "请求次数超出限制",
            "causes": [
                "短时间内请求次数过多",
                "达到了每日/每月配额上限"
            ],
            "solutions": [
                "等待一段时间后重试",
                "升级账户获得更高配额",
                "减少请求频率"
            ],
            "auto_fix": f"建议等待{60}秒后重试"
        },
        "quota_exceeded": {
            "patterns": ["quota exceeded", "配额用尽", "超出配额"],
            "friendly_message": "使用配额已用尽",
            "causes": [
                "今日/本月免费额度已用完"
            ],
            "solutions": [
                "等待配额重置（每日/每月）",
                "升级账户获得更多配额"
            ],
            "auto_fix": "无法自动修复，请升级账户"
        },

        # Network/Connection errors
        "network_error": {
            "patterns": ["network", "connection", "网络错误", "连接失败", "timeout"],
            "friendly_message": "网络连接出现问题",
            "causes": [
                "网络连接可能不稳定",
                "服务器可能暂时无法访问"
            ],
            "solutions": [
                "检查网络连接",
                "刷新页面重试",
                "稍后再试"
            ],
            "auto_fix": "尝试刷新页面"
        },
        "server_error": {
            "patterns": ["server error", "500", "内部错误", "服务器错误"],
            "friendly_message": "服务器出现错误",
            "causes": [
                "服务器可能正在维护",
                "服务可能暂时不可用"
            ],
            "solutions": [
                "稍后重试",
                "如果问题持续，联系技术支持"
            ],
            "auto_fix": "建议稍后重试"
        },

        # Validation errors
        "validation_error": {
            "patterns": ["validation", "验证失败", "格式错误", "invalid input"],
            "friendly_message": "输入数据格式不正确",
            "causes": [
                "输入的参数格式可能不正确",
                "必填字段可能缺失"
            ],
            "solutions": [
                "检查输入格式",
                "确保所有必填字段都已填写",
                "参考示例格式填写"
            ],
            "auto_fix": "检查输入格式"
        },
        "text_too_long": {
            "patterns": ["text too long", "文本过长", "超出长度限制"],
            "friendly_message": "文本内容过长",
            "causes": [
                "输入的文本超过了系统允许的最大长度"
            ],
            "solutions": [
                "分段处理文本",
                "删除不必要的字符",
                "联系管理员增加文本长度限制"
            ],
            "auto_fix": "建议分段处理"
        },

        # Voice cloning errors
        "voice_clone_failed": {
            "patterns": ["voice clone", "语音克隆", "克隆失败"],
            "friendly_message": "语音克隆失败",
            "causes": [
                "音频样本质量可能不够好",
                "样本时长可能不符合要求",
                "背景噪音可能太大"
            ],
            "solutions": [
                "使用更清晰的音频样本（3-10秒）",
                "确保环境安静，没有背景噪音",
                "使用标准的音频格式（MP3/WAV）",
                "尝试不同的音频样本"
            ],
            "auto_fix": "请使用更清晰的音频样本"
        },
        "voice_clone_too_short": {
            "patterns": ["too short", "太短", "时长不足"],
            "friendly_message": "音频样本时长不足",
            "causes": [
                "上传的音频样本太短，需要至少3秒"
            ],
            "solutions": [
                "使用至少3秒的音频样本",
                "最佳时长为5-10秒"
            ],
            "auto_fix": "请上传至少3秒的音频"
        },
    }

    # Error severity levels
    SEVERITY_LEVELS = {
        "critical": {"icon": "🔴", "color": "red", "priority": 1},
        "error": {"icon": "❌", "color": "orange", "priority": 2},
        "warning": {"icon": "⚠️", "color": "yellow", "priority": 3},
        "info": {"icon": "ℹ️", "color": "blue", "priority": 4},
    }

    def __init__(self):
        """Initialize smart error handler."""
        self.error_log: List[Dict[str, Any]] = []
        self.error_stats: Dict[str, int] = {}

    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        user_language: str = "zh"
    ) -> Dict[str, Any]:
        """
        Handle an error and return user-friendly response.

        Args:
            error: The exception that occurred
            context: Additional context about the error
            user_language: User's preferred language

        Returns:
            Dict with user-friendly error information
        """
        error_message = str(error).lower()
        error_type = type(error).__name__

        # Find matching error pattern
        error_info = self._match_error_pattern(error_message)

        # Build user-friendly response
        response = {
            "error_type": error_info["error_type"] if error_info else "unknown",
            "friendly_message": error_info.get("friendly_message", "操作失败，请稍后重试"),
            "technical_message": str(error),
            "severity": self._determine_severity(error_info),
            "causes": error_info.get("causes", ["系统出现未知错误"]),
            "solutions": error_info.get("solutions", ["请稍后重试", "如果问题持续，联系技术支持"]),
            "auto_fix": error_info.get("auto_fix", "无法自动修复"),
            "timestamp": datetime.now().isoformat(),
            "error_id": self._generate_error_id(),
            "context": context or {},
        }

        # Log error
        self._log_error(error, response)

        return response

    def _match_error_pattern(self, error_message: str) -> Optional[Dict[str, Any]]:
        """Match error message against known patterns."""
        for error_type, error_info in self.ERROR_PATTERNS.items():
            for pattern in error_info["patterns"]:
                if pattern.lower() in error_message:
                    return {
                        "error_type": error_type,
                        **error_info
                    }
        return None

    def _determine_severity(self, error_info: Optional[Dict]) -> str:
        """Determine error severity level."""
        if not error_info:
            return "error"

        error_type = error_info.get("error_type", "")

        if error_type in ["server_error", "permission_denied"]:
            return "critical"
        elif error_type in ["rate_limit_exceeded", "network_error"]:
            return "warning"
        else:
            return "error"

    def _generate_error_id(self) -> str:
        """Generate unique error ID for tracking."""
        import uuid
        return f"ERR-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"

    def _log_error(self, error: Exception, response: Dict[str, Any]):
        """Log error for analysis."""
        error_type = response["error_type"]
        self.error_stats[error_type] = self.error_stats.get(error_type, 0) + 1

        log_entry = {
            "error_id": response["error_id"],
            "error_type": error_type,
            "error_message": response["technical_message"],
            "context": response.get("context"),
            "timestamp": response["timestamp"],
        }

        self.error_log.append(log_entry)

        # Keep only last 1000 errors
        if len(self.error_log) > 1000:
            self.error_log = self.error_log[-1000:]

        # Log to file
        logger.error(
            f"Error [{response['error_id']}]: {response['technical_message']}",
            extra={"context": response.get("context")}
        )

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "total_errors": sum(self.error_stats.values()),
            "by_type": self.error_stats,
            "recent_errors": self.error_log[-100:],  # Last 100 errors
        }

    def get_common_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most common errors."""
        sorted_errors = sorted(
            self.error_stats.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]

        return [
            {
                "error_type": error_type,
                "count": count,
                "info": self.ERROR_PATTERNS.get(error_type, {})
            }
            for error_type, count in sorted_errors
        ]

    def suggest_fix(self, error_message: str) -> List[str]:
        """Get fix suggestions for an error message."""
        error_info = self._match_error_pattern(error_message.lower())
        if error_info:
            return error_info.get("solutions", [])
        return ["请稍后重试", "如果问题持续，联系技术支持"]

    def translate_technical_message(self, error_message: str, user_language: str = "zh") -> str:
        """Translate technical error message to user-friendly one."""
        error_info = self._match_error_pattern(error_message.lower())
        if error_info and user_language == "zh":
            return error_info.get("friendly_message", error_message)
        return error_message

    def create_user_friendly_response(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create API response with user-friendly error information.

        Returns:
            Dict suitable for API response
        """
        error_info = self.handle_error(error, context)

        return {
            "success": False,
            "error": {
                "code": error_info["error_type"].upper(),
                "message": error_info["friendly_message"],
                "details": error_info["technical_message"],
                "error_id": error_info["error_id"],
                "severity": error_info["severity"],
                "causes": error_info["causes"],
                "suggestions": error_info["solutions"],
                "auto_fix": error_info["auto_fix"],
            },
            "timestamp": error_info["timestamp"],
        }


# Global instance
_error_handler: Optional[SmartErrorHandler] = None


def get_error_handler() -> SmartErrorHandler:
    """Get global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = SmartErrorHandler()
    return _error_handler

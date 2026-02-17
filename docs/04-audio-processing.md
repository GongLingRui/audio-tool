# 音频处理与 TTS 集成方案

## 1. 音频处理架构

### 1.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                     音频处理管道                              │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐    │
│  │ 脚本    │──▶│ 分块    │──▶│ TTS     │──▶│ 后处理  │    │
│  │ 解析    │   │ 处理    │   │ 生成    │   │ 优化    │    │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘    │
│       │             │             │             │          │
│       ▼             ▼             ▼             ▼          │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐    │
│  │ 结构化  │   │ 音频块  │   │ WAV     │   │ 停顿    │    │
│  │ JSON    │   │ 队列    │   │ 文件    │   │ 插入    │    │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘    │
│                                           │                  │
│                                           ▼                  │
│                                    ┌─────────┐              │
│                                    │ 音频    │              │
│                                    │ 合并    │              │
│                                    └─────────┘              │
│                                           │                  │
│                                           ▼                  │
│                                    ┌─────────┐              │
│                                    │ 导出    │              │
│                                    │ MP3/WAV │              │
│                                    └─────────┘              │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 技术选型

| 组件 | 技术方案 | 说明 |
|------|---------|------|
| **TTS 引擎** | Qwen3-TTS / Edge-TTS | 本地/外部 TTS 服务 |
| **音频处理** | FFmpeg | 专业音频处理工具 |
| **音频库** | pydub, librosa | Python 音频处理库 |
| **格式转换** | FFmpeg-python | FFmpeg Python 绑定 |
| **音频播放** | WaveSurfer.js | 前端波形可视化 |

## 2. TTS 引擎集成

### 2.1 TTS 模式设计

```python
# backend/app/services/tts_engine.py
from enum import Enum
from abc import ABC, abstractmethod

class TTSMode(str, Enum):
    """TTS 模式"""
    LOCAL = "local"           # 本地 Qwen3-TTS 模型
    EXTERNAL = "external"     # 外部 TTS 服务器
    EDGE = "edge"            # Edge-TTS (备用)

class VoiceType(str, Enum):
    """语音类型"""
    CUSTOM = "custom"         # 预设语音
    CLONE = "clone"          # 语音克隆
    LORA = "lora"            # LoRA 微调
    DESIGN = "design"        # 语音设计

class TTSEngine(ABC):
    """TTS 引擎抽象基类"""

    @abstractmethod
    async def generate(self, text: str, config: dict) -> bytes:
        """生成音频"""
        pass

    @abstractmethod
    async def get_voices(self) -> list:
        """获取可用语音列表"""
        pass
```

### 2.2 本地 TTS 引擎

```python
# backend/app/services/tts_local.py
import torch
from transformers import AutoModel
import numpy as np
import io

class LocalTTSEngine(TTSEngine):
    """本地 Qwen3-TTS 引擎"""

    def __init__(self, config: dict):
        self.device = config.get("device", "auto")
        self.model_path = config.get("model_path", "Qwen/Qwen3-TTS-12B")
        self.model = None
        self._load_model()

    def _load_model(self):
        """懒加载模型"""
        if self.model is None:
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()

    async def generate(
        self,
        text: str,
        speaker: str,
        instruct: str = None,
        voice_config: dict = None
    ) -> tuple[bytes, float]:
        """
        生成音频

        Args:
            text: 要转换的文本
            speaker: 发言人
            instruct: TTS 指令（情绪、语调等）
            voice_config: 语音配置

        Returns:
            (音频数据, 时长)
        """
        voice_type = voice_config.get("voice_type", "custom") if voice_config else "custom"

        # 根据语音类型选择生成方法
        if voice_type == "custom":
            return await self._generate_custom(text, speaker, instruct)
        elif voice_type == "clone":
            return await self._generate_clone(text, speaker, voice_config)
        elif voice_type == "lora":
            return await self._generate_lora(text, speaker, voice_config)
        elif voice_type == "design":
            return await self._generate_design(text, speaker, voice_config)
        else:
            raise ValueError(f"Unsupported voice type: {voice_type}")

    async def _generate_custom(
        self,
        text: str,
        speaker: str,
        instruct: str = None
    ) -> tuple[bytes, float]:
        """使用预设语音生成"""
        # 语音名称映射
        voice_map = {
            "Aiden": "aiden",
            "Dylan": "dylan",
            "Eric": "eric",
            "Ryan": "ryan",
            "Sarah": "sarah",
            "Rachel": "rachel",
            "Emma": "emma",
            "James": "james",
            "Daniel": "daniel"
        }

        voice_name = voice_map.get(speaker, "ryan")
        instruction = instruct or "Neutral speech"

        # 调用模型生成
        with torch.no_grad():
            audio, sr = self.model.generate(
                text=text,
                voice=voice_name,
                instruction=instruction,
                stream=False
            )

        # 转换为 WAV 字节
        wav_bytes = self._to_wav(audio, sr)
        duration = len(audio) / sr

        return wav_bytes, duration

    def _to_wav(self, audio: np.ndarray, sr: int) -> bytes:
        """将音频转换为 WAV 格式"""
        import wave

        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)  # 单声道
            wf.setsampwidth(2)  # 16位
            wf.setframerate(sr)
            # 转换为 int16
            audio_int16 = (audio * 32767).astype(np.int16)
            wf.writeframes(audio_int16.tobytes())
        return buf.getvalue()

    async def get_voices(self) -> list:
        """获取可用语音列表"""
        return [
            {"id": "aiden", "name": "Aiden", "gender": "male", "language": "en-US"},
            {"id": "dylan", "name": "Dylan", "gender": "male", "language": "en-US"},
            {"id": "eric", "name": "Eric", "gender": "male", "language": "en-US"},
            {"id": "ryan", "name": "Ryan", "gender": "male", "language": "en-US"},
            {"id": "sarah", "name": "Sarah", "gender": "female", "language": "en-US"},
            {"id": "rachel", "name": "Rachel", "gender": "female", "language": "en-US"},
            {"id": "emma", "name": "Emma", "gender": "female", "language": "en-US"},
        ]
```

### 2.3 外部 TTS 引擎

```python
# backend/app/services/tts_external.py
import httpx
from typing import Optional

class ExternalTTSEngine(TTSEngine):
    """外部 TTS 服务器引擎"""

    def __init__(self, config: dict):
        self.base_url = config.get("tts_url", "http://localhost:7860")
        self.timeout = config.get("timeout", 300)
        self.client = None

    async def _get_client(self):
        """获取 HTTP 客户端"""
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=self.timeout)
        return self.client

    async def generate(
        self,
        text: str,
        speaker: str,
        instruct: str = None,
        voice_config: dict = None
    ) -> tuple[bytes, float]:
        """调用外部 TTS API"""
        client = await self._get_client()

        voice_type = voice_config.get("voice_type", "custom") if voice_config else "custom"

        # 构建请求数据
        payload = {
            "text": text,
            "speaker": speaker,
            "voice_type": voice_type
        }

        if instruct:
            payload["instruction"] = instruct

        if voice_config:
            if voice_type == "clone":
                payload["ref_audio_path"] = voice_config.get("ref_audio_path")
            elif voice_type == "lora":
                payload["lora_model_path"] = voice_config.get("lora_model_path")
            elif voice_type == "design":
                payload["description"] = voice_config.get("description")

        # 发送请求
        response = await client.post(f"{self.base_url}/api/generate", json=payload)
        response.raise_for_status()

        data = response.json()
        audio_bytes = bytes.fromhex(data.get("audio", ""))
        duration = data.get("duration", 0)

        return audio_bytes, duration

    async def generate_batch(
        self,
        items: list[dict]
    ) -> list[tuple[bytes, float]]:
        """批量生成音频"""
        client = await self._get_client()

        payload = {
            "items": items,
            "parallel_workers": 2
        }

        response = await client.post(f"{self.base_url}/api/generate-batch", json=payload)
        response.raise_for_status()

        data = response.json()
        results = []
        for item in data.get("results", []):
            audio_bytes = bytes.fromhex(item.get("audio", ""))
            duration = item.get("duration", 0)
            results.append((audio_bytes, duration))

        return results

    async def get_voices(self) -> list:
        """获取可用语音列表"""
        client = await self._get_client()
        response = await client.get(f"{self.base_url}/api/voices")
        response.raise_for_status()
        return response.json().get("voices", [])

    async def close(self):
        """关闭连接"""
        if self.client:
            await self.client.aclose()
```

### 2.4 TTS 引擎工厂

```python
# backend/app/services/tts_factory.py
from .tts_local import LocalTTSEngine
from .tts_external import ExternalTTSEngine

class TTSEngineFactory:
    """TTS 引擎工厂"""

    @staticmethod
    def create(mode: str, config: dict) -> TTSEngine:
        """根据模式创建 TTS 引擎"""
        if mode == "local":
            return LocalTTSEngine(config)
        elif mode == "external":
            return ExternalTTSEngine(config)
        else:
            raise ValueError(f"Unsupported TTS mode: {mode}")
```

## 3. 音频处理服务

### 3.1 音频处理器

```python
# backend/app/services/audio_processor.py
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import os
import asyncio
from pathlib import Path

class AudioProcessor:
    """音频处理服务"""

    def __init__(self, config: dict):
        self.sample_rate = config.get("sample_rate", 24000)
        self.channels = config.get("channels", 1)
        self.bitrate = config.get("bitrate", "128k")
        self.temp_dir = Path(config.get("temp_dir", "./temp/audio"))
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    async def save_wav(self, audio_data: bytes, output_path: str) -> str:
        """
        保存 WAV 文件

        Args:
            audio_data: 音频数据
            output_path: 输出路径

        Returns:
            文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            f.write(audio_data)

        return str(output_path)

    async def convert_to_mp3(
        self,
        wav_path: str,
        mp3_path: str = None,
        bitrate: str = None
    ) -> str:
        """
        转换 WAV 到 MP3

        Args:
            wav_path: WAV 文件路径
            mp3_path: MP3 输出路径（可选）
            bitrate: 比特率（可选）

        Returns:
            MP3 文件路径
        """
        if mp3_path is None:
            mp3_path = wav_path.replace('.wav', '.mp3')

        bitrate = bitrate or self.bitrate

        # 使用 ffmpeg 转换
        from pydub import AudioSegment

        audio = AudioSegment.from_wav(wav_path)
        audio.export(mp3_path, format='mp3', bitrate=bitrate)

        # 删除原 WAV 文件
        if os.path.exists(wav_path):
            os.remove(wav_path)

        return mp3_path

    async def get_duration(self, audio_path: str) -> float:
        """获取音频时长"""
        audio = AudioSegment.from_file(audio_path)
        return len(audio) / 1000.0  # 转换为秒

    async def add_silence(
        self,
        audio_path: str,
        duration_ms: int,
        position: str = "end"
    ) -> str:
        """
        添加静音

        Args:
            audio_path: 音频文件路径
            duration_ms: 静音时长（毫秒）
            position: 位置 (start/end)

        Returns:
            处理后的音频路径
        """
        audio = AudioSegment.from_file(audio_path)
        silence = AudioSegment.silent(duration=duration_ms)

        if position == "start":
            result = silence + audio
        else:
            result = audio + silence

        output_path = audio_path.replace('.mp3', '_temp.mp3')
        result.export(output_path, format='mp3')

        # 替换原文件
        os.replace(output_path, audio_path)

        return audio_path

    async def combine_audio_files(
        self,
        audio_paths: list[str],
        speakers: list[str],
        output_path: str,
        pause_between_speakers: int = 500,
        pause_same_speaker: int = 250
    ) -> tuple[str, float]:
        """
        合并多个音频文件

        Args:
            audio_paths: 音频文件路径列表
            speakers: 对应的发言人列表
            output_path: 输出文件路径
            pause_between_speakers: 不同发言人之间的停顿（毫秒）
            pause_same_speaker: 相同发言人之间的停顿（毫秒）

        Returns:
            (输出路径, 总时长)
        """
        if not audio_paths:
            raise ValueError("No audio files to combine")

        # 加载所有音频
        audio_segments = []
        for path in audio_paths:
            audio = AudioSegment.from_mp3(path)
            audio_segments.append(audio)

        # 合并音频，插入停顿
        combined = audio_segments[0]
        for i in range(1, len(audio_segments)):
            # 确定停顿时长
            if speakers[i] != speakers[i - 1]:
                pause = AudioSegment.silent(duration=pause_between_speakers)
            else:
                pause = AudioSegment.silent(duration=pause_same_speaker)

            combined = combined + pause + audio_segments[i]

        # 导出
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        combined.export(str(output_path), format='mp3', bitrate=self.bitrate)

        # 计算总时长
        duration = len(combined) / 1000.0

        return str(output_path), duration

    async def normalize_volume(
        self,
        audio_path: str,
        target_dBFS: float = -20.0
    ) -> str:
        """
        标准化音量

        Args:
            audio_path: 音频文件路径
            target_dBFS: 目标音量（dBFS）

        Returns:
            处理后的音频路径
        """
        audio = AudioSegment.from_file(audio_path)

        # 计算需要调整的分贝数
        change_in_dBFS = target_dBFS - audio.dBFS
        normalized = audio.apply_gain(change_in_dBFS)

        output_path = audio_path.replace('.mp3', '_temp.mp3')
        normalized.export(output_path, format='mp3')

        os.replace(output_path, audio_path)

        return audio_path

    async def fade_in_out(
        self,
        audio_path: str,
        fade_in_duration: int = 100,
        fade_out_duration: int = 100
    ) -> str:
        """
        添加淡入淡出效果

        Args:
            audio_path: 音频文件路径
            fade_in_duration: 淡入时长（毫秒）
            fade_out_duration: 淡出时长（毫秒）

        Returns:
            处理后的音频路径
        """
        audio = AudioSegment.from_file(audio_path)

        # 添加淡入淡出
        faded = audio.fade_in(fade_in_duration).fade_out(fade_out_duration)

        output_path = audio_path.replace('.mp3', '_temp.mp3')
        faded.export(output_path, format='mp3')

        os.replace(output_path, audio_path)

        return audio_path

    async def cleanup_temp_files(self):
        """清理临时文件"""
        if self.temp_dir.exists():
            for file in self.temp_dir.glob('*'):
                try:
                    file.unlink()
                except Exception as e:
                    print(f"Failed to delete {file}: {e}")
```

### 3.2 音频分块处理

```python
# backend/app/services/chunk_processor.py
from typing import List, Dict
import asyncio

class ChunkProcessor:
    """音频块处理器"""

    def __init__(self, tts_engine: TTSEngine, audio_processor: AudioProcessor):
        self.tts_engine = tts_engine
        self.audio_processor = audio_processor
        self.max_workers = 2

    async def process_chunk(
        self,
        chunk: Dict,
        voice_config: Dict
    ) -> Dict:
        """
        处理单个音频块

        Args:
            chunk: 音频块数据
            voice_config: 语音配置

        Returns:
            处理结果
        """
        speaker = chunk.get("speaker")
        text = chunk.get("text")
        instruct = chunk.get("instruct")

        # 查找对应的语音配置
        speaker_config = self._get_speaker_config(speaker, voice_config)

        # 生成音频
        audio_data, duration = await self.tts_engine.generate(
            text=text,
            speaker=speaker,
            instruct=instruct,
            voice_config=speaker_config
        )

        # 保存音频
        chunk_id = chunk.get("id")
        audio_path = f"./static/audio/chunks/{chunk_id}.wav"
        await self.audio_processor.save_wav(audio_data, audio_path)

        # 转换为 MP3
        mp3_path = await self.audio_processor.convert_to_mp3(audio_path)

        return {
            "chunk_id": chunk_id,
            "audio_path": mp3_path,
            "duration": duration,
            "status": "completed"
        }

    async def process_chunks_batch(
        self,
        chunks: List[Dict],
        voice_config: Dict,
        max_workers: int = None
    ) -> List[Dict]:
        """
        批量处理音频块

        Args:
            chunks: 音频块列表
            voice_config: 语音配置
            max_workers: 最大并发数

        Returns:
            处理结果列表
        """
        max_workers = max_workers or self.max_workers

        tasks = []
        for chunk in chunks:
            if chunk.get("status") != "completed":
                task = self.process_chunk(chunk, voice_config)
                tasks.append(task)

        # 使用信号量限制并发
        semaphore = asyncio.Semaphore(max_workers)

        async def bounded_task(task):
            async with semaphore:
                return await task

        bounded_tasks = [bounded_task(task) for task in tasks]

        results = await asyncio.gather(*bounded_tasks, return_exceptions=True)

        # 处理异常
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "chunk_id": chunks[i].get("id"),
                    "status": "failed",
                    "error": str(result)
                })
            else:
                processed_results.append(result)

        return processed_results

    def _get_speaker_config(self, speaker: str, voice_config: Dict) -> Dict:
        """获取发言人配置"""
        for config in voice_config.get("voices", []):
            if config.get("speaker") == speaker:
                return config
        # 返回默认配置
        return {"voice_type": "custom", "voice_name": "ryan"}
```

## 4. FFmpeg 集成

### 4.1 FFmpeg 封装

```python
# backend/app/utils/ffmpeg_utils.py
import subprocess
import asyncio
from pathlib import Path
from typing import Optional, List

class FFmpegWrapper:
    """FFmpeg 工具类"""

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg_path = ffmpeg_path

    async def run_command(
        self,
        args: List[str],
        input_data: Optional[bytes] = None
    ) -> tuple[bytes, bytes, int]:
        """
        运行 FFmpeg 命令

        Args:
            args: FFmpeg 参数列表
            input_data: 输入数据（可选）

        Returns:
            (stdout, stderr, return_code)
        """
        cmd = [self.ffmpeg_path] + args

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE if input_data else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate(input=input_data)

        return stdout, stderr, process.returncode

    async def convert_audio(
        self,
        input_path: str,
        output_path: str,
        format: str = "mp3",
        bitrate: str = "128k",
        sample_rate: int = 44100
    ) -> bool:
        """
        转换音频格式

        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            format: 输出格式
            bitrate: 比特率
            sample_rate: 采样率

        Returns:
            是否成功
        """
        args = [
            "-i", input_path,
            "-codec:a", "libmp3lame" if format == "mp3" else "pcm_s16le",
            "-b:a", bitrate,
            "-ar", str(sample_rate),
            "-y",  # 覆盖输出文件
            output_path
        ]

        _, _, return_code = await self.run_command(args)
        return return_code == 0

    async def merge_audio_files(
        self,
        input_paths: List[str],
        output_path: str,
        concat_file: Optional[str] = None
    ) -> bool:
        """
        合并音频文件

        Args:
            input_paths: 输入文件路径列表
            output_path: 输出文件路径
            concat_file: 临时 concat 文件路径

        Returns:
            是否成功
        """
        # 创建 concat 文件
        if concat_file is None:
            concat_file = "/tmp/ffmpeg_concat.txt"

        with open(concat_file, 'w') as f:
            for path in input_paths:
                f.write(f"file '{Path(path).absolute()}'\n")

        args = [
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            "-y",
            output_path
        ]

        _, _, return_code = await self.run_command(args)

        # 清理临时文件
        if Path(concat_file).exists():
            Path(concat_file).unlink()

        return return_code == 0

    async def add_silence(
        self,
        input_path: str,
        output_path: str,
        duration_ms: int,
        position: str = "end"
    ) -> bool:
        """
        添加静音

        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            duration_ms: 静音时长（毫秒）
            position: 位置 (start/end)

        Returns:
            是否成功
        """
        duration_sec = duration_ms / 1000.0

        if position == "start":
            filter_complex = f"[1]adelay=0s|0s[silence];[silence][0]concat=n=2:v=0:a=1"
            args = [
                "-i", input_path,
                "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo",
                "-filter_complex", filter_complex,
                "-t", str(duration_sec),
                "-y",
                output_path
            ]
        else:
            args = [
                "-i", input_path,
                "-af", f"apad=whole_dur={duration_sec}",
                "-y",
                output_path
            ]

        _, _, return_code = await self.run_command(args)
        return return_code == 0

    async def get_audio_info(self, file_path: str) -> dict:
        """
        获取音频信息

        Args:
            file_path: 文件路径

        Returns:
            音频信息字典
        """
        args = [
            "-i", file_path,
            "-f", "null",
            "-"
        ]

        _, stderr, _ = await self.run_command(args)
        stderr_text = stderr.decode('utf-8', errors='ignore')

        # 解析输出
        info = {}
        for line in stderr_text.split('\n'):
            if 'Duration:' in line:
                parts = line.split('Duration: ')[1].split(',')[0].strip()
                info['duration'] = parts
            elif 'Audio:' in line:
                audio_info = line.split('Audio: ')[1].split(',')[0].strip()
                info['codec'] = audio_info

        return info

    async def export_audacity_project(
        self,
        audio_paths: List[str],
        output_dir: str,
        project_name: str = "project"
    ) -> dict:
        """
        导出 Audacity 项目

        Args:
            audio_paths: 音频文件路径列表
            output_dir: 输出目录
            project_name: 项目名称

        Returns:
            导出结果
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 创建 data 目录
        data_dir = output_path / "data"
        data_dir.mkdir(exist_ok=True)

        # 复制音频文件到 data 目录
        for i, audio_path in enumerate(audio_paths):
            src = Path(audio_path)
            dst = data_dir / f"{project_name}_{i+1:03d}{src.suffix}"
            shutil.copy(src, dst)

        # 创建 .aup 项目文件
        aup_file = output_path / f"{project_name}.aup"
        # 这里需要生成 Audacity 项目文件格式
        # 由于格式较复杂，这里仅做简化示例

        return {
            "project_file": str(aup_file),
            "data_dir": str(data_dir),
            "files": [f.name for f in data_dir.iterdir()]
        }
```

## 5. 前端音频处理

### 5.1 音频播放器配置

```typescript
// src/hooks/useAudioPlayer.ts
import WaveSurfer from 'wavesurfer.js';

interface AudioPlayerConfig {
  container: string;
  url: string;
  waveColor?: string;
  progressColor?: string;
  cursorColor?: string;
  height?: number;
}

export function useAudioPlayer() {
  const createPlayer = (config: AudioPlayerConfig) => {
    return WaveSurfer.create({
      container: config.container,
      url: config.url,
      waveColor: config.waveColor || '#4a5568',
      progressColor: config.progressColor || '#667eea',
      cursorColor: config.cursorColor || '#ffffff',
      height: config.height || 128,
      normalize: true,
      backend: 'WebAudio',
      barWidth: 2,
      barGap: 1,
    });
  };

  return { createPlayer };
}
```

### 5.2 段落时间映射

```typescript
// src/types/audio.ts
export interface ParagraphTimeMap {
  [paragraphId: string]: {
    startTime: number;
    endTime: number;
    audioPath: string;
  };
}

export interface AudioSyncConfig {
  paragraphs: Array<{
    id: string;
    text: string;
    startTime: number;
    endTime: number;
  }>;
  onParagraphChange?: (paragraphId: string) => void;
}

// src/utils/audio.ts
export function createParagraphTimeMap(
  chunks: Chunk[],
  baseTime: number = 0
): ParagraphTimeMap {
  const map: ParagraphTimeMap = {};
  let currentTime = baseTime;

  chunks.forEach((chunk, index) => {
    map[`paragraph_${index}`] = {
      startTime: currentTime,
      endTime: currentTime + (chunk.duration || 0),
      audioPath: chunk.audio_path,
    };
    currentTime += chunk.duration || 0;
  });

  return map;
}

export function getCurrentParagraph(
  currentTime: number,
  timeMap: ParagraphTimeMap
): string | null {
  for (const [id, range] of Object.entries(timeMap)) {
    if (currentTime >= range.startTime && currentTime < range.endTime) {
      return id;
    }
  }
  return null;
}
```

## 6. 配置示例

### 6.1 TTS 配置

```yaml
# config/tts.yaml
tts:
  mode: external  # local | external | edge
  external:
    url: http://localhost:7860
    timeout: 300
    parallel_workers: 2
  local:
    model_path: Qwen/Qwen3-TTS-12B
    device: auto  # auto | cuda | cpu
    batch_size: 4
  audio:
    sample_rate: 24000
    channels: 1
    bitrate: 128k
    format: mp3
  processing:
    pause_between_speakers: 500  # ms
    pause_same_speaker: 250      # ms
    normalize_volume: true
    target_dbfs: -20.0
    fade_in: 100   # ms
    fade_out: 100  # ms
```

### 6.2 语音配置模板

```json
{
  "voices": [
    {
      "speaker": "NARRATOR",
      "voice_type": "custom",
      "voice_name": "Ryan",
      "style": "calm narrator",
      "language": "zh-CN",
      "instruct_template": "平静、客观的叙述"
    },
    {
      "speaker": "主角",
      "voice_type": "clone",
      "ref_audio_path": "/uploads/voices/user_xxx/reference.wav",
      "style": "年轻男性",
      "language": "zh-CN"
    },
    {
      "speaker": "配角A",
      "voice_type": "lora",
      "lora_model_path": "/static/lora/builtin_watson",
      "style": "成熟男性",
      "language": "zh-CN"
    }
  ]
}
```

## 7. 错误处理

### 7.1 错误类型

```python
# backend/app/core/exceptions.py
class TTSError(Exception):
    """TTS 错误基类"""
    pass

class TTSConnectionError(TTSError):
    """TTS 连接错误"""
    pass

class TTSGenerationError(TTSError):
    """TTS 生成错误"""
    pass

class AudioProcessingError(Exception):
    """音频处理错误"""
    pass

class FFmpegError(AudioProcessingError):
    """FFmpeg 错误"""
    def __init__(self, message: str, return_code: int, stderr: str):
        self.return_code = return_code
        self.stderr = stderr
        super().__init__(message)
```

### 7.2 重试策略

```python
# backend/app/services/tts_engine.py
import asyncio
from typing import Callable

class RetryPolicy:
    """重试策略"""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
        exponential_base: float = 2.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

    async def execute(self, func: Callable, *args, **kwargs):
        """执行函数并重试"""
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )
                    await asyncio.sleep(delay)

        raise last_exception
```

## 8. 性能优化

### 8.1 批量处理优化

```python
# backend/app/services/audio_processor.py
class BatchProcessor:
    """批量处理器"""

    async def process_optimized_batch(
        self,
        chunks: List[Dict],
        voice_config: Dict
    ) -> List[Dict]:
        """优化的批量处理"""
        # 按发言人分组
        grouped = self._group_by_speaker(chunks)

        # 按文本长度排序（减少填充）
        for speaker in grouped:
            grouped[speaker].sort(key=lambda x: len(x.get("text", "")))

        # 并行处理
        results = []
        for speaker, speaker_chunks in grouped.items():
            speaker_results = await self.process_chunks_batch(
                speaker_chunks,
                voice_config
            )
            results.extend(speaker_results)

        return results

    def _group_by_speaker(self, chunks: List[Dict]) -> Dict[str, List[Dict]]:
        """按发言人分组"""
        grouped = {}
        for chunk in chunks:
            speaker = chunk.get("speaker", "NARRATOR")
            if speaker not in grouped:
                grouped[speaker] = []
            grouped[speaker].append(chunk)
        return grouped
```

### 8.2 缓存策略

```python
# backend/app/services/cache.py
from functools import lru_cache
import hashlib
import pickle

class AudioCache:
    """音频缓存"""

    def __init__(self, cache_dir: str = "./cache/audio"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, text: str, speaker: str, instruct: str = None) -> str:
        """生成缓存键"""
        data = f"{text}|{speaker}|{instruct or ''}"
        return hashlib.md5(data.encode()).hexdigest()

    def get(self, text: str, speaker: str, instruct: str = None) -> bytes:
        """获取缓存"""
        key = self._get_cache_key(text, speaker, instruct)
        cache_file = self.cache_dir / f"{key}.wav"

        if cache_file.exists():
            return cache_file.read_bytes()
        return None

    def set(self, text: str, speaker: str, audio: bytes, instruct: str = None):
        """设置缓存"""
        key = self._get_cache_key(text, speaker, instruct)
        cache_file = self.cache_dir / f"{key}.wav"
        cache_file.write_bytes(audio)

    def clear(self):
        """清空缓存"""
        for file in self.cache_dir.glob("*"):
            file.unlink()
```

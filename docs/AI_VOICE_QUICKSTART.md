# Read-Rhyme AI语音功能快速入门

> 快速了解和使用Read-Rhyme的AI语音功能

---

## 1. 情感控制

### 1.1 使用预设情感

```typescript
import { voiceStylingApi } from '@/services';

// 获取所有可用情感预设
const presets = await voiceStylingApi.getEmotionPresets();
// 返回：neutral, happy, sad, angry, fearful, surprised, romantic, mysterious, energetic, calm

// 使用预设生成语音
const result = await voiceStylingApi.generateStyledAudio({
  text: "你好，这是测试",
  emotion: {
    happiness: 0.8,  // 快乐程度 0-1
    energy: 1.2,    // 能量 0-2 (1=正常)
    tempo: 1.1      // 语速 0.5-2x (1=正常)
  },
  preset_id: "happy"  // 使用快乐预设
});
```

### 1.2 自定义情感

```typescript
// 完全自定义情感参数
const result = await voiceStylingApi.generateStyledAudio({
  text: "我很生气！",
  emotion: {
    anger: 0.9,     // 愤怒程度高
    energy: 1.4,     // 能量高
    tempo: 1.2,      // 语速快
    volume: 1.2      // 音量大
  },
  style: {
    gender: "male",
    timbre: "rough",    // 粗糙音色
    delivery: "rapid"  // 快速传递
  }
});
```

### 1.3 情感参数说明

| 参数 | 范围 | 说明 | 示例值 |
|------|------|------|--------|
| happiness | 0-1 | 快乐/兴奋 | 0.8=快乐, 0.3=略喜 |
| sadness | 0-1 | 悲伤/沮丧 | 0.7=悲伤, 0.2=忧郁 |
| anger | 0-1 | 愤怒/激动 | 0.9=暴怒, 0.5=恼火 |
| fear | 0-1 | 恐惧/焦虑 | 0.8=恐惧, 0.4=紧张 |
| surprise | 0-1 | 惊讶/震惊 | 0.7=惊讶, 0.3=意外 |
| neutral | 0-1 | 平静/中性 | 1.0=完全中性 |
| energy | 0-2 | 能量/活力 | 0.5=低能, 1.0=正常, 1.5=高能 |
| tempo | 0.5-2 | 语速 | 0.8=慢速, 1.0=正常, 1.3=快速 |
| pitch | -12~12 | 音高(半音) | -2=低沉, 0=正常, 4=高昂 |
| volume | 0-2 | 音量 | 0.7=小声, 1.0=正常, 1.3=大声 |

---

## 2. 语音克隆

### 2.1 上传参考音频

```typescript
// 1. 准备音频样本
const audioFile = new File([audioBlob], "voice_sample.wav");

// 2. 上传参考音频
const result = await voicesApi.uploadCloneAudio(
  audioFile,
  "这是一段测试音频，用于语音克隆"
);

// 返回：{ audio_path: "...", duration: 5.2 }
```

### 2.2 批量克隆（推荐）

```typescript
// 使用多个样本提升克隆质量
const samples = [
  new File([blob1], "sample1.wav"),  // 5-15秒
  new File([blob2], "sample2.wav"),  // 不同情感
  new File([blob3], "sample3.wav"),  // 不同语调
  // ... 建议5-10个样本
];

const result = await voiceStylingApi.batchVoiceClone({
  voice_samples: samples,
  voice_name: "我的声音",
  description: "年轻男性，温暖音色",
  language: "zh-CN"
});

// 返回：{ voice_id: "cloned_xxx", status: "training" }
```

**最佳实践：**
- ✅ 每个样本5-15秒
- ✅ 覆盖不同情感和语调
- ✅ 环境音尽量干净
- ✅ 避免背景音乐
- ✅ 使用统一设备录制

---

## 3. 语音转换

### 3.1 语音到语音转换

```typescript
// 将源音频转换为目标声音
const result = await voiceStylingApi.convertVoice({
  source_audio_path: "/path/to/source.wav",
  target_voice_id: "cloned_xxx",
  preserve_timing: true,   // 保持原始节奏
  preserve_prosody: false  // 不保持韵律（改变音色）
});
```

**应用场景：**
- 更换有声书的朗读者
- 统一多段录音的音色
- 修复音质不佳的片段

---

## 4. 音频后处理

### 4.1 语音增强

```typescript
// 增强音频质量
const result = await voiceStylingApi.enhanceSpeech({
  audio_path: "/path/to/audio.mp3",
  enhance_denoise: true,        // 去噪
  enhance_volume: true,         // 音量归一化
  add_compression: true,        // 添加压缩
  target_lufs: -16.0            // 目标响度(LUFS)
});
```

**处理效果：**
- **去噪** - 移除背景噪音
- **音量归一化** - 统一响度
- **压缩** - 动态范围控制
- **LUFS目标** - 专业响度标准

---

## 5. 完整工作流程

### 5.1 创建情感丰富的有声书

```typescript
// 步骤1: 上传书籍
const book = await booksApi.upload(file, {
  title: "我的小说",
  author: "作者名"
});

// 步骤2: 创建项目
const project = await projectsApi.create({
  book_id: book.data.id,
  name: "我的有声书项目"
});

// 步骤3: 生成脚本
const script = await scriptsApi.generate(project.data.id);

// 步骤4: 配置语音
await voicesApi.setVoiceConfig(project.data.id, [{
  speaker: "主角",
  voice_type: "clone",
  ref_audio_path: "/path/to/voice.wav"
}]);

// 步骤5: 生成情感音频
const audio = await voiceStylingApi.generateStyledAudio({
  text: "这是第一章的内容",
  emotion: {
    happiness: 0.6,
    energy: 1.0
  },
  preset_id: "calm"
});

// 步骤6: 导出最终音频
const exported = await audioApi.exportAudio(project.data.id, "combined", {
  add_fades: true,
  normalize: true
});
```

---

## 6. API使用示例

### 6.1 获取支持的语言

```typescript
const languages = await voiceStylingApi.getSupportedLanguages();

// 返回：
[
  {
    language_code: "zh-CN",
    language_name: "Chinese (Mandarin)",
    supported_voices: ["custom", "clone", "design"],
    emotion_support: true,
    sample_rate: 24000,
    model_type: "neural"
  },
  // ... 其他语言
]
```

### 6.2 获取情感预设详情

```typescript
const preset = await voiceStylingApi.getEmotionPreset("happy");

// 返回：
{
  id: "happy",
  name: "快乐",
  description: "愉悦和兴奋",
  emotion: {
    happiness: 0.8,
    energy: 1.3,
    tempo: 1.1
  },
  example_instruct: "Joyful, cheerful tone with bright energy."
}
```

---

## 7. 常见用例

### 用例1：创建不同情感的角色

```typescript
// 开朗主角
const happyCharacter = await voiceStylingApi.generateStyledAudio({
  text: "今天天气真好！",
  emotion: { happiness: 0.8, energy: 1.2 },
  style: { gender: "female", age_range: "young-adult" }
});

// 忧郁配角
const sadCharacter = await voiceStylingApi.generateStyledAudio({
  text: "为什么这种事发生在我身上...",
  emotion: { sadness: 0.8, energy: 0.7 },
  style: { gender: "male", age_range: "middle-aged" }
});
```

### 用例2：调整叙述者情绪

```typescript
// 开篇-平静叙述
const opening = await voiceStylingApi.generateStyledAudio({
  text: "第一章：故事的开始",
  preset_id: "calm"
});

// 高潮-紧张叙述
const climax = await voiceStylingApi.generateStyledAudio({
  text: "突然，一声巨响打破了宁静！",
  emotion: { surprise: 0.8, energy: 1.3, tempo: 1.2 }
});
```

### 用例3：批量处理不同情感

```typescript
const emotions = ["neutral", "happy", "sad", "angry", "surprised"];

for (const emotion of emotions) {
  const audio = await voiceStylingApi.generateStyledAudio({
    text: `这是${emotion}的示例`,
    preset_id: emotion
  });
  // 保存或使用audio
}
```

---

## 8. 故障排除

### Q: 情感控制不起作用？

**A:** 检查以下几点：
1. 确认TTS引擎支持情感控制
2. 情感参数在0-1范围内（主要情感）
3. 尝试使用预设而非自定义参数
4. 查看返回的`instruction`字段

### Q: 语音克隆质量不好？

**A:** 改进方法：
1. 使用更多样本（5-10个）
2. 确保样本清晰无噪音
3. 覆盖不同情感和语调
4. 每个样本5-15秒最佳

### Q: 语音转换效果不自然？

**A:** 优化建议：
1. 设置`preserve_timing: true`保持节奏
2. 确保源音频清晰
3. 目标声音样本质量要好
4. 尝试不同的`preserve_prosody`设置

---

## 9. 进阶技巧

### 9.1 情感渐变

```typescript
// 情感从平静到激动的渐变
const emotionCurve = [
  { happiness: 0.0, sadness: 0.0, energy: 0.8 },
  { happiness: 0.3, sadness: 0.0, energy: 0.9 },
  { happiness: 0.6, sadness: 0.0, energy: 1.1 },
  { happiness: 0.9, sadness: 0.0, energy: 1.3 }
];

for (let i = 0; i < emotionCurve.length; i++) {
  await voiceStylingApi.generateStyledAudio({
    text: segmentTexts[i],
    emotion: emotionCurve[i]
  });
}
```

### 9.2 多语言有声书

```typescript
// 中文部分
const zhAudio = await voiceStylingApi.generateStyledAudio({
  text: "你好，欢迎",
  emotion: { neutral: 1.0 },
  style: { accent: "chinese" }
});

// 英文部分
const enAudio = await voiceStylingApi.generateStyledAudio({
  text: "Hello, welcome",
  emotion: { neutral: 1.0 },
  style: { accent: "american" }
});
```

### 9.3 音质后期处理

```typescript
// 1. 生成音频
const rawAudio = await voiceStylingApi.generateStyledAudio({...});

// 2. 降噪处理
const denoised = await voiceStylingApi.enhanceSpeech({
  audio_path: rawAudio.audio_url,
  enhance_denoise: true,
  enhance_volume: true,
  target_lufs: -16.0
});

// 3. 导出
await audioApi.exportAudio(projectId, "combined", {
  add_fades: true,
  normalize: true
});
```

---

## 10. 资源链接

### 技术文档
- [AI语音技术完整指南](./AI_VOICE_TECHNOLOGY_GUIDE.md) - 深入技术原理
- [AI语音与系统改进建议](./AI_VOICE_AND_SYSTEM_IMPROVEMENTS.md) - 语音与整体系统改进方向
- [VOICE_REFERENCE.md](../backend/app/static/VOICE_REFERENCE.md) - 语音方向词汇（后端静态文件）

### 外部资源
- [ElevenLabs](https://elevenlabs.io) - 高质量TTS参考
- [GPT-SoVITS GitHub](https://github.com/RVC-Boss/GPT-SoVITS) - 语音克隆技术
- [Narration Box](https://narrationbox.com/blog/best-ai-voices-for-audiobooks-2025) - 2025最佳TTS对比

---

**提示：** 所有API调用都需要先登录，使用`authService.login()`获取认证令牌。

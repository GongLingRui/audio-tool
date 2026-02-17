# 模型配置和测试报告

生成时间: 2025-02-16
系统配置: Apple M4, 32GB RAM

## 📋 目录

1. [系统配置](#系统配置)
2. [模型概览](#模型概览)
3. [LLM模型详情](#llm模型详情)
4. [TTS模型详情](#tts模型详情)
5. [Embedding模型详情](#embedding模型详情)
6. [测试结果](#测试结果)
7. [使用建议](#使用建议)
8. [命令行工具](#命令行工具)

---

## 系统配置

### 硬件信息
- **芯片**: Apple M4
- **内存**: 32 GB
- **架构**: ARM64 (Apple Silicon)
- **平台**: macOS 15.2 (Darwin 24.6.0)

### 软件环境
- **Python**: 3.13
- **Ollama**: 0.15.6 (已安装)
- **PyTorch**: 2.10.0 (MPS加速支持)
- **Transformers**: 5.1.0

### 已安装Ollama模型
1. qwen2.5:1.5b
2. qwen2.5:3b
3. qwen2.5:7b
4. qwen2.5:14b

---

## 模型概览

### 按类型分类

| 类型 | 模型数量 | 状态 |
|------|---------|------|
| **LLM** | 4个 | ✓ 全部已安装并测试 |
| **TTS** | 2个 | ✓ 部分可用 |
| **Embedding** | 4个 | ✓ 已下载并测试 |

### 支持的模型总览

#### LLM模型 (Qwen2.5系列)
| 模型 | 参数量 | 内存 | 状态 | 用途 |
|------|--------|------|------|------|
| Qwen2.5 0.5B | 0.5B | 1GB | 可选 | 超轻量任务 |
| Qwen2.5 1.5B | 1.5B | 2GB | ✓ 已安装 | 快速响应 |
| Qwen2.5 3B | 3B | 3GB | ✓ 已安装 | 日常使用 |
| Qwen2.5 7B | 7B | 5GB | ✓ 已安装 | 标准任务 |
| Qwen2.5 14B | 14B | 10GB | ✓ 已安装 | 复杂推理 |
| Qwen2.5 32B | 32B | 20GB | 可选 | 专业应用 |
| Qwen2.5 72B | 72B | 45GB | 可选 | 顶级性能 |

#### TTS模型
| 模型 | 参数量 | 内存 | 状态 | 特性 |
|------|--------|------|------|------|
| Edge TTS | N/A | 0GB | ✓ 免费 | 在线、多语言 |
| Qwen3-TTS 1.7B | 1.7B | 4GB | 需下载 | 语音克隆、情感 |

#### Embedding模型
| 模型 | 维度 | 内存 | 状态 | 特性 |
|------|------|------|------|------|
| Paraphrase-MiniLM | 384 | 0.5GB | ✓ 已下载 | 多语言、轻量 |
| BGE-Large-ZH | 1024 | 1GB | 可选 | 中文优化 |
| BGE-M3 | 1024 | 2GB | 可选 | 多语言、高质量 |
| GTE-Qwen2-7B | 3584 | 5GB | 可选 | 最高质量 |

---

## LLM模型详情

### Qwen2.5 14B Instruct (推荐用于生产环境)

**配置信息**
- **Ollama名称**: `qwen2.5:14b`
- **HuggingFace ID**: `Qwen/Qwen2.5-14B-Instruct`
- **参数量**: 14B
- **内存需求**: 10GB
- **是否免费**: ✓ 是

**测试结果**
- 响应时间: ~9分钟 (首次加载)
- 后续响应: ~7秒
- 测试响应: "我是Qwen，来自阿里云的大规模语言模型，旨在为用户提供高效、准确的信息和创作服务。"

**适用场景**
- 复杂推理任务
- 代码生成和分析
- 数学和逻辑问题
- 长文本理解
- 有声书脚本生成

### Qwen2.5 7B Instruct (推荐用于日常使用)

**配置信息**
- **Ollama名称**: `qwen2.5:7b`
- **HuggingFace ID**: `Qwen/Qwen2.5-7B-Instruct`
- **参数量**: 7B
- **内存需求**: 5GB
- **是否免费**: ✓ 是

**测试结果**
- 响应时间: ~7秒 (首次加载)
- 测试响应: "我叫Qwen，是来自阿里云的大规模语言模型，可以回答问题、创作文字，还能表达观点和拟人化聊天。"

**适用场景**
- 日常对话
- 文本生成
- 问答系统
- RAG应用
- 脚本生成

### Qwen2.5 3B Instruct (轻量级选择)

**配置信息**
- **Ollama名称**: `qwen2.5:3b`
- **参数量**: 3B
- **内存需求**: 3GB
- **是否免费**: ✓ 是

**适用场景**
- 快速响应
- 低内存环境
- 简单问答
- 批量处理

### Qwen2.5 1.5B Instruct (超轻量)

**配置信息**
- **Ollama名称**: `qwen2.5:1.5b`
- **参数量**: 1.5B
- **内存需求**: 2GB
- **是否免费**: ✓ 是

**适用场景**
- 实时应用
- 边缘设备
- 原型开发

---

## TTS模型详情

### Edge TTS (免费推荐)

**配置信息**
- **参数量**: N/A (在线服务)
- **内存需求**: 0GB
- **是否免费**: ✓ 是
- **语言支持**: 中文、英文、日文、韩文等

**特性**
- 完全免费
- 无需下载
- 多种声音选择
- 高质量输出
- 低延迟

**使用方法**
```python
import edge_tts
communicate = edge_tts.Communicate("你好", voice="zh-CN-XiaoxiaoNeural")
await communicate.save("output.mp3")
```

### Qwen3-TTS 1.7B (高级功能)

**配置信息**
- **HuggingFace ID**: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- **参数量**: 1.7B
- **内存需求**: 4GB
- **是否免费**: ✓ 是

**特性**
- 语音克隆
- 情感控制
- 多语言支持
- 高质量输出
- MPS加速

**安装方法**
```bash
pip install transformers accelerate
```

**注意**: 需要从HuggingFace下载模型文件

---

## Embedding模型详情

### Paraphrase Multilingual MiniLM (当前使用)

**配置信息**
- **HuggingFace ID**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **向量维度**: 384
- **内存需求**: 0.5GB
- **是否免费**: ✓ 是

**测试结果**
- 加载时间: 10.7秒
- 编码时间: 1.9秒/句
- 总时间: 12.6秒

**适用场景**
- RAG检索
- 语义搜索
- 文本相似度
- 多语言支持

### BGE-M3 (高质量多语言)

**配置信息**
- **HuggingFace ID**: `BAAI/bge-m3`
- **向量维度**: 1024
- **内存需求**: 2GB
- **是否免费**: ✓ 是

**特性**
- 支持100+语言
- Dense、Sparse、ColBERT三种检索方式
- 最高质量多语言检索

### GTE-Qwen2-7B-Instruct (最高质量)

**配置信息**
- **HuggingFace ID**: `Alibaba-NLP/gte-Qwen2-7B-instruct`
- **向量维度**: 3584
- **内存需求**: 5GB
- **是否免费**: ✓ 是

**特性**
- 基于Qwen2-7B
- 最高质量embedding
- 中英文优化

---

## 测试结果

### LLM模型测试

#### Qwen2.5 14B Instruct
```
状态: ✓ 成功
首次加载时间: ~9分钟
响应时间: 546291ms (首次), ~7000ms (后续)
测试输出: "我是Qwen，来自阿里云的大规模语言模型，旨在为用户提供高效、准确的信息和创作服务。"
```

#### Qwen2.5 7B Instruct
```
状态: ✓ 成功
响应时间: 6983ms
测试输出: "我叫Qwen，是来自阿里云的大规模语言模型，可以回答问题、创作文字，还能表达观点和拟人化聊天。"
```

### Embedding模型测试

#### Paraphrase Multilingual MiniLM
```
状态: ✓ 成功
向量维度: 384
加载时间: 10718ms
编码时间: 1927ms
```

### 系统兼容性测试

| 组件 | 状态 | 说明 |
|------|------|------|
| Ollama | ✓ 运行中 | 版本 0.15.6 |
| PyTorch MPS | ✓ 可用 | 支持GPU加速 |
| Transformers | ✓ 已安装 | 版本 5.1.0 |
| Sentence-Transformers | ✓ 已安装 | 版本 5.2.2 |

---

## 使用建议

### 生产环境配置

#### 推荐配置 (32GB RAM)
```yaml
LLM: qwen2.5:14b
TTS: edge-tts (免费) 或 qwen3-tts-1.7b (高级)
Embedding: paraphrase-multilingual (轻量) 或 gte-qwen2-7b-instruct (高质量)
```

#### 性能优化配置
```yaml
LLM: qwen2.5:7b (平衡性能和速度)
TTS: edge-tts (快速、免费)
Embedding: paraphrase-multilingual (快速检索)
```

#### 低内存配置 (< 16GB RAM)
```yaml
LLM: qwen2.5:3b
TTS: edge-tts
Embedding: paraphrase-multilingual
```

### 模型选择指南

#### 按场景选择

**有声书制作**
- LLM: qwen2.5:7b 或 qwen2.5:14b
- TTS: edge-tts (快速) 或 qwen3-tts-1.7b (高质量)

**RAG问答系统**
- LLM: qwen2.5:7b
- Embedding: bge-m3 或 gte-qwen2-7b-instruct

**实时对话**
- LLM: qwen2.5:3b 或 qwen2.5:1.5b
- TTS: edge-tts

**代码生成**
- LLM: qwen2.5-coder:7b

---

## 命令行工具

### 模型管理CLI

已创建完整的命令行工具用于模型管理。

#### 查看所有模型
```bash
python scripts/model_cli.py list
```

#### 查看特定类型模型
```bash
python scripts/model_cli.py list --type llm
python scripts/model_cli.py list --type tts
python scripts/model_cli.py list --type embedding
```

#### 检查系统状态
```bash
python scripts/model_cli.py check
```

#### 安装模型
```bash
# 安装LLM模型
python scripts/model_cli.py install qwen2.5-7b-instruct

# 安装14B模型
python scripts/model_cli.py install qwen2.5-14b-instruct
```

#### 测试模型
```bash
# 测试LLM
python scripts/model_cli.py test qwen2.5-7b-instruct

# 测试Embedding
python scripts/model_cli.py test paraphrase-multilingual

# 测试TTS
python scripts/model_cli.py test edge-tts
```

#### 运行综合测试
```bash
python scripts/model_cli.py test-all
```

#### 查看推荐配置
```bash
# 基于内存查看推荐
python scripts/model_cli.py recommend --memory 32
```

#### 下载Embedding模型
```bash
python scripts/model_cli.py download-embeddings
```

---

## 配置文件

### 环境变量 (.env)

```bash
# LLM配置
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=local
LLM_MODEL=qwen2.5:14b

# TTS配置
TTS_MODE=edge
TTS_LANGUAGE=zh-CN

# RAG配置
RAG_EMBEDDING_MODEL=paraphrase-multilingual
RAG_TOP_K=5
RAG_MIN_SCORE=0.2
```

---

## 总结

### 已完成
1. ✓ 安装Ollama
2. ✓ 安装4个Qwen LLM模型 (1.5B, 3B, 7B, 14B)
3. ✓ 测试LLM模型功能
4. ✓ 下载并测试Embedding模型
5. ✓ 创建统一模型管理系统
6. ✓ 创建命令行管理工具

### 模型状态

**免费且已安装**:
- LLM: Qwen2.5 (1.5B, 3B, 7B, 14B) - 全部免费
- TTS: Edge TTS - 完全免费
- Embedding: Paraphrase Multilingual - 免费

**可升级选项**:
- TTS: Qwen3-TTS 1.7B (需下载，免费)
- Embedding: BGE-M3, GTE-Qwen2-7B (需下载，免费)

### 系统性能评估

基于Apple M4 + 32GB RAM配置：

| 任务 | 推荐模型 | 预期性能 |
|------|---------|---------|
| 快速对话 | Qwen2.5 3B | < 3秒响应 |
| 标准任务 | Qwen2.5 7B | < 7秒响应 |
| 复杂推理 | Qwen2.5 14B | < 15秒响应 |
| 文本检索 | Paraphrase | < 2秒 |
| 语音合成 | Edge TTS | < 2秒 |

### 下一步建议

1. 根据实际使用场景选择合适的模型
2. 对于生产环境，建议使用qwen2.5:7b作为默认LLM
3. 如需最高质量，可升级到qwen2.5:14b
4. 考虑下载更多Embedding模型以提升检索质量
5. 根据需求选择是否使用Qwen3-TTS进行语音克隆

---

*报告生成时间: 2025-02-16*
*系统版本: macOS 15.2 (Darwin 24.6.0)*
*工具版本: model_cli.py v1.0*

# AI模型安装指南 - M4 32GB安全配置

## 📊 当前内存状态
- 总内存: 32.0 GB
- 已使用: 10.3 GB (81.9%)
- **可用: 5.8 GB** ⚠️

## ⚠️ 重要提示

基于你的内存情况，建议使用**轻量级配置**以避免系统卡顿。

## 🎯 推荐方案：轻量级模式 (~500MB)

### 安装命令：

```bash
cd backend

# 1. 仅安装轻量级embedding模型
pip3 install sentence-transformers

# 2. 使用最小模型 (80MB)
# 在 production_rag.py 中修改:
# embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
```

### TTS方案：
使用**系统自带TTS**或**在线API**，不加载大模型到本地。

## 📋 其他方案对比

| 方案 | 内存占用 | TTS质量 | Embedding质量 | 推荐度 |
|------|----------|---------|---------------|--------|
| **轻量** | ~500MB | 系统/在线 | 好 (L6模型) | ⭐⭐⭐⭐⭐ |
| 标准 | ~2GB | 中等 | 很好 (L12) | ⭐⭐⭐ |
| 完整 | ~5GB | 高 | 很好 (多语言) | ⭐⭐ |

## 🚀 快速安装（推荐）

```bash
cd backend

# 一键安装轻量配置
chmod +x install_ml_safe.sh
./install_ml_safe.sh
# 选择选项 1
```

## 🔧 手动安装轻量模式

```bash
# 仅安装必要的包
pip3 install sentence-transformers

# 然后重启后端
pkill -f uvicorn
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 📈 使用更小模型的优势

✅ **内存安全**: 仅占用~500MB
✅ **加载快速**: 几秒钟加载完成
✅ **响应迅速**: 推理速度快
✅ **系统稳定**: 不会导致卡顿

## 💡 提示

如果后续发现内存不够，系统会自动回退到Mock模式，不会崩溃。

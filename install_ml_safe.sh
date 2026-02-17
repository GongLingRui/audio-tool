#!/bin/bash
# 安全安装ML依赖 - 适合M4 32GB配置

echo "=== Read-Rhyme AI组件安全安装 ==="
echo ""

# 检查当前内存
echo "当前内存状态:"
python3 -c "
import psutil
mem = psutil.virtual_memory()
print(f'  总内存: {mem.total / (1024**3):.1f} GB')
print(f'  可用: {mem.available / (1024**3):.1f} GB')
print(f'  使用率: {mem.percent}%')
"

echo ""
echo "选择安装模式:"
echo "1) 轻量模式 (~500MB) - 推荐用于32GB内存"
echo "2) 标准模式 (~2GB) - 需要至少6GB可用内存"
echo "3) 完整模式 (~5GB) - 需要至少10GB可用内存"
echo ""
read -p "请选择 (1/2/3) [默认:1]: " mode
mode=${mode:-1}

case $mode in
    1)
        echo ""
        echo "安装轻量模式..."
        echo "模型: all-MiniLM-L6-v2 (80MB)"
        pip3 install --no-cache-dir sentence-transformers
        ;;
    2)
        echo ""
        echo "安装标准模式..."
        echo "模型: speecht5-tts + multilingual-MiniLM"
        pip3 install --no-cache-dir transformers sentence-transformers torch
        ;;
    3)
        echo ""
        echo "⚠️  警告: 完整模式可能占用5GB+内存"
        read -p "确认继续? (y/N): " confirm
        if [ "$confirm" = "y" ]; then
            pip3 install --no-cache-dir transformers sentence-transformers torch
        else
            echo "已取消"
            exit 0
        fi
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

echo ""
echo "✓ 安装完成!"
echo ""
echo "重启后端以应用更改:"
echo "  pkill -f uvicorn"
echo "  python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

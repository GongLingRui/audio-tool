#!/usr/bin/env python3
"""
模型管理命令行工具
用于下载、测试和管理所有模型
"""

import sys
import os
import asyncio
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.model_manager import (
    ModelManager,
    ModelType,
    get_model_manager,
    AVAILABLE_MODELS,
)


async def list_models(args):
    """List all available models."""
    manager = get_model_manager()

    if args.type:
        model_type = ModelType(args.type)
        models = manager.get_available_models(model_type=model_type)
    else:
        models = manager.get_available_models()

    print(f"\n找到 {len(models)} 个模型:\n")

    for m in models:
        print(f"  ID: {m.model_id}")
        print(f"  名称: {m.name}")
        print(f"  类型: {m.model_type.value}")
        print(f"  参数量: {m.params}")
        print(f"  内存需求: {m.memory_gb} GB")
        print(f"  描述: {m.description}")
        print(f"  特性: {', '.join(m.features)}")
        if m.ollama_name:
            print(f"  Ollama: {m.ollama_name}")
        if m.huggingface_id:
            print(f"  HuggingFace: {m.huggingface_id}")
        print()


async def check_system(args):
    """Check system requirements and Ollama status."""
    import platform
    import subprocess

    print("\n=== 系统信息 ===")
    print(f"  平台: {platform.system()} {platform.release()}")
    print(f"  架构: {platform.machine()}")

    # Check memory
    try:
        result = subprocess.run(['sysctl', 'hw.memsize'], capture_output=True, text=True)
        if result.returncode == 0:
            mem_gb = int(result.stdout.split(': ')[1]) // (1024**3)
            print(f"  内存: {mem_gb} GB")
    except:
        pass

    # Check Apple Silicon
    try:
        result = subprocess.run(['system_profiler', 'SPHardwareDataType'],
                              capture_output=True, text=True)
        if 'Apple M' in result.stdout:
            chip = [line for line in result.stdout.split('\n') if 'Chip:' in line][0]
            print(f"  芯片: {chip.strip()}")
    except:
        pass

    print("\n=== Ollama 状态 ===")
    manager = get_model_manager()
    ollama_installed = await manager.check_ollama_installed()

    if ollama_installed:
        print("  ✓ Ollama 已安装并运行")
        models = await manager.get_ollama_models()
        print(f"  已安装模型 ({len(models)}):")
        for m in models:
            print(f"    - {m}")
    else:
        print("  ✗ Ollama 未安装或未运行")
        print("\n  安装 Ollama:")
        print("    1. 访问 https://ollama.ai/download")
        print("    2. 下载 macOS 版本")
        print("    3. 安装后运行: ollama serve")


async def install_model(args):
    """Install a model via Ollama."""
    manager = get_model_manager()
    model_config = manager.get_model_info(args.model)

    if not model_config:
        print(f"错误: 未找到模型 '{args.model}'")
        return 1

    if model_config.model_type != ModelType.LLM:
        print(f"错误: 模型 '{args.model}' 不是 LLM 模型，无法通过 Ollama 安装")
        return 1

    if not model_config.ollama_name:
        print(f"错误: 模型 '{args.model}' 没有配置 Ollama 名称")
        return 1

    print(f"正在安装 {model_config.name}...")
    print(f"Ollama 模型名: {model_config.ollama_name}")
    print(f"预计需要: {model_config.memory_gb} GB 内存")

    success = await manager.pull_ollama_model(model_config.ollama_name)

    if success:
        print(f"\n✓ 成功安装 {model_config.name}")
        return 0
    else:
        print(f"\n✗ 安装失败")
        return 1


async def test_model(args):
    """Test a specific model."""
    manager = get_model_manager()
    model_config = manager.get_model_info(args.model)

    if not model_config:
        print(f"错误: 未找到模型 '{args.model}'")
        return 1

    print(f"正在测试 {model_config.name}...")
    print(f"模型 ID: {model_config.model_id}")
    print(f"类型: {model_config.model_type.value}")
    print()

    if model_config.model_type == ModelType.LLM:
        result = await manager.test_llm_model(args.model)
    elif model_config.model_type == ModelType.EMBEDDING:
        result = await manager.test_embedding_model(args.model)
    elif model_config.model_type == ModelType.TTS:
        result = await manager.test_tts_model(args.model)
    else:
        print(f"错误: 不支持的模型类型")
        return 1

    print(f"结果: {'✓ 成功' if result['success'] else '✗ 失败'}")

    if result.get('success'):
        for key, value in result.items():
            if key not in ['model_id', 'model_name', 'success']:
                if value is not None:
                    print(f"  {key}: {value}")
    else:
        print(f"  错误: {result.get('error', 'Unknown error')}")

    return 0 if result['success'] else 1


async def test_all(args):
    """Test all recommended models."""
    manager = get_model_manager()

    print("正在运行综合测试...")
    print("这可能需要几分钟时间...\n")

    report = await manager.run_comprehensive_test()

    print("\n" + "=" * 60)
    print("测试报告".center(60))
    print("=" * 60)

    # System info
    print("\n【系统信息】")
    for key, value in report['system_info'].items():
        print(f"  {key}: {value}")

    print(f"\n【Ollama 状态】")
    print(f"  已安装运行: {'是' if report['ollama_installed'] else '否'}")
    print(f"  已安装模型数: {len(report['ollama_models'])}")

    # Test results
    print(f"\n【测试结果】")

    for test_type, test_name in [('llm', 'LLM'), ('embedding', 'Embedding'), ('tts', 'TTS')]:
        if test_type in report['tests']:
            result = report['tests'][test_type]
            status = "✓ 通过" if result['success'] else "✗ 失败"
            print(f"\n  {test_name} 测试: {status}")
            print(f"    模型: {result.get('model_name', 'N/A')}")
            if result['success']:
                for key, value in result.items():
                    if key not in ['model_id', 'model_name', 'success', 'error']:
                        if value is not None:
                            print(f"    {key}: {value}")
            else:
                print(f"    错误: {result.get('error', 'Unknown')}")

    # Save report
    report_path = Path("./model_test_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n报告已保存到: {report_path.absolute()}")

    return 0


async def recommend(args):
    """Show recommended models based on system memory."""
    manager = get_model_manager()
    memory_gb = args.memory or 32

    print(f"\n基于 {memory_gb}GB 内存推荐:\n")
    recommended = manager.get_recommended_models(memory_gb)

    for model_type, model_id in recommended.items():
        model_config = manager.get_model_info(model_id)
        print(f"【{model_type.upper()}】")
        print(f"  推荐: {model_config.name}")
        print(f"  ID: {model_id}")
        print(f"  内存: {model_config.memory_gb} GB")
        print(f"  描述: {model_config.description}")
        print()


async def download_embeddings(args):
    """Download embedding models."""
    print("\n正在下载 embedding 模型...\n")

    embedding_models = [m for m in AVAILABLE_MODELS.values()
                       if m.model_type == ModelType.EMBEDDING]

    for model in embedding_models:
        print(f"准备下载: {model.name}")
        print(f"  HuggingFace ID: {model.huggingface_id}")
        print(f"  预计大小: {model.memory_gb * 500} MB")

        try:
            from sentence_transformers import SentenceTransformer
            print(f"  下载中...")
            downloaded_model = SentenceTransformer(model.huggingface_id)
            print(f"  ✓ 下载完成\n")

            # Clean up
            del downloaded_model

        except Exception as e:
            print(f"  ✗ 下载失败: {e}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="模型管理命令行工具")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # list command
    list_parser = subparsers.add_parser('list', help='列出所有可用模型')
    list_parser.add_argument('--type', choices=['llm', 'tts', 'embedding'],
                          help='过滤模型类型')

    # check command
    subparsers.add_parser('check', help='检查系统状态')

    # install command
    install_parser = subparsers.add_parser('install', help='安装模型')
    install_parser.add_argument('model', help='模型 ID (使用 list 查看)')

    # test command
    test_parser = subparsers.add_parser('test', help='测试单个模型')
    test_parser.add_argument('model', help='模型 ID')

    # test-all command
    subparsers.add_parser('test-all', help='测试所有推荐模型')

    # recommend command
    recommend_parser = subparsers.add_parser('recommend', help='显示推荐的模型')
    recommend_parser.add_argument('--memory', type=int, default=32,
                               help='系统内存 (GB)')

    # download-embeddings command
    subparsers.add_parser('download-embeddings', help='下载所有 embedding 模型')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Execute command
    if args.command == 'list':
        return asyncio.run(list_models(args))
    elif args.command == 'check':
        return asyncio.run(check_system(args))
    elif args.command == 'install':
        return asyncio.run(install_model(args))
    elif args.command == 'test':
        return asyncio.run(test_model(args))
    elif args.command == 'test-all':
        return asyncio.run(test_all(args))
    elif args.command == 'recommend':
        return asyncio.run(recommend(args))
    elif args.command == 'download-embeddings':
        return asyncio.run(download_embeddings(args))

    return 0


if __name__ == '__main__':
    sys.exit(main())

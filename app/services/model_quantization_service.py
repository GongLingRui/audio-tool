"""
Model Quantization Service for TTS Models
INT8 quantization and model optimization for faster inference
"""
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class QuantizationType(Enum):
    """Types of quantization."""
    INT8 = "int8"  # 8-bit integer quantization
    FP16 = "fp16"  # 16-bit floating point
    DYNAMIC = "dynamic"  # Dynamic quantization
    GPTQ = "gptq"  # GPTQ quantization for LLMs
    AWQ = "awq"  # Activation-aware Weight Quantization


@dataclass
class QuantizationResult:
    """Result of model quantization."""
    success: bool
    quantization_type: QuantizationType
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    model_path: str
    error: Optional[str] = None
    calibration_time: float = 0.0
    inference_speedup: float = 0.0


class ModelQuantizationService:
    """
    Service for quantizing TTS models for faster inference.

    Features:
    - INT8 static quantization
    - FP16 half-precision
    - Dynamic quantization
    - Model size reduction
    - Inference speedup
    - Model comparison

    Supports:
    - PyTorch models (.pth, .pt)
    - ONNX models (.onnx)
    - HuggingFace transformers
    """

    def __init__(
        self,
        quantized_models_dir: str = "./data/quantized_models",
        calibration_samples_dir: str = "./data/calibration_samples",
    ):
        """
        Initialize model quantization service.

        Args:
            quantized_models_dir: Directory to save quantized models
            calibration_samples_dir: Directory for calibration data
        """
        self.quantized_dir = Path(quantized_models_dir)
        self.quantized_dir.mkdir(parents=True, exist_ok=True)

        self.calibration_dir = Path(calibration_samples_dir)
        self.calibration_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(self.__class__.__name__)

    async def quantize_model(
        self,
        model_path: str,
        quantization_type: QuantizationType = QuantizationType.INT8,
        model_format: str = "pytorch",
        calibration_data: Optional[List[str]] = None,
    ) -> QuantizationResult:
        """
        Quantize a TTS model.

        Args:
            model_path: Path to the model file
            quantization_type: Type of quantization to apply
            model_format: Model format (pytorch, onnx, transformers)
            calibration_data: Sample texts for calibration (for static quantization)

        Returns:
            QuantizationResult with metrics
        """
        import time
        start_time = time.time()

        # Get original model size
        original_size = self._get_model_size_mb(model_path)

        try:
            if model_format == "pytorch":
                result = await self._quantize_pytorch(
                    model_path, quantization_type, calibration_data
                )
            elif model_format == "onnx":
                result = await self._quantize_onnx(
                    model_path, quantization_type
                )
            elif model_format == "transformers":
                result = await self._quantize_transformers(
                    model_path, quantization_type, calibration_data
                )
            else:
                raise ValueError(f"Unsupported model format: {model_format}")

            # Calculate metrics
            quantized_path = str(result["model_path"])
            quantized_size = self._get_model_size_mb(quantized_path)
            compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0

            return QuantizationResult(
                success=True,
                quantization_type=quantization_type,
                original_size_mb=original_size,
                quantized_size_mb=quantized_size,
                compression_ratio=compression_ratio,
                model_path=quantized_path,
                inference_speedup=self._estimate_speedup(quantization_type),
                calibration_time=time.time() - start_time,
            )

        except Exception as e:
            self.logger.error(f"Quantization error: {e}")
            return QuantizationResult(
                success=False,
                quantization_type=quantization_type,
                original_size_mb=original_size,
                quantized_size_mb=0.0,
                compression_ratio=1.0,
                model_path="",
                error=str(e),
            )

    async def _quantize_pytorch(
        self,
        model_path: str,
        quant_type: QuantizationType,
        calibration_data: Optional[List[str]],
    ) -> Dict[str, str]:
        """Quantize PyTorch model."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch is required for PyTorch model quantization")

        # Load model
        try:
            model = torch.load(model_path, map_location="cpu", weights_only=False)
        except TypeError:
            # Older torch versions do not support weights_only.
            model = torch.load(model_path, map_location="cpu")

        if isinstance(model, dict):
            raise ValueError(
                "Unsupported PyTorch checkpoint: got a state_dict-like object. "
                "Please upload a saved nn.Module or a TorchScript model."
            )
        model.eval()

        # Prepare calibration data
        if calibration_data and quant_type in [QuantizationType.INT8]:
            # Create calibration dataset
            calibration_dataset = self._create_calibration_dataset(
                calibration_data, model
            )
        else:
            calibration_dataset = None

        # Apply quantization
        if quant_type == QuantizationType.INT8:
            # Static INT8 quantization
            quantized_model = self._apply_int8_quantization(
                model, calibration_dataset
            )
        elif quant_type == QuantizationType.FP16:
            # Half precision
            quantized_model = model.half()
        elif quant_type == QuantizationType.DYNAMIC:
            # Dynamic quantization
            quantized_model = self._apply_dynamic_quantization(model)
        else:
            raise ValueError(f"Unsupported quantization: {quant_type}")

        # Save quantized model
        output_path = self._get_output_path(model_path, quant_type)
        torch.save(quantized_model, output_path)

        # Save metadata
        self._save_quantization_metadata(
            str(model_path), str(output_path), quant_type, quantized_model
        )

        return {"model_path": str(output_path)}

    async def _quantize_onnx(
        self,
        model_path: str,
        quant_type: QuantizationType,
    ) -> Dict[str, str]:
        """Quantize ONNX model."""
        try:
            import onnx
            from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType
        except ImportError:
            raise ImportError("onnxruntime is required for ONNX quantization")

        # Load ONNX model
        model = onnx.load(model_path)

        # Apply quantization
        if quant_type == QuantizationType.INT8:
            # Dynamic quantization for ONNX
            output_path = self._get_output_path(model_path, quant_type)

            quantize_dynamic(
                model,
                model_type='Transformers',  # Adjust based on model type
                weight_type=QuantType.QInt8,
                output_path=output_path,
            )
        elif quant_type == QuantizationType.FP16:
            # Convert to FP16
            from onnxconverter_common import float16

            output_path = self._get_output_path(model_path, quant_type)
            float16.convert_float_to_float16(model, output_path)
        else:
            raise ValueError(f"Unsupported ONNX quantization: {quant_type}")

        return {"model_path": str(output_path)}

    async def _quantize_transformers(
        self,
        model_path: str,
        quant_type: QuantizationType,
        calibration_data: Optional[List[str]],
    ) -> Dict[str, str]:
        """Quantize HuggingFace Transformers model."""
        try:
            from transformers import AutoModel
            from optimum.onnxruntime import ORTModelForTextToSpeech
            from optimum.onnxruntime.configuration import OptimizationConfig
        except ImportError:
            raise ImportError(
                "optimum is required for Transformers quantization. "
                "Install with: pip install optimum[onnxruntime]"
            )

        # Load model
        model = AutoModel.from_pretrained(model_path)

        # Create output path
        output_path = self._get_output_path(model_path, quant_type)

        if quant_type == QuantizationType.INT8:
            # ONNX Runtime INT8 quantization
            optimized_model = ORTModelForTextToSpeech.from_pretrained(
                model_path,
                export=True,
                optimization_config=OptimizationConfig(
                    optimization_level=2,  # Enable all optimizations
                    enable_cpu_mask_checker=False,
                ),
            )

            # Save optimized model
            optimized_model.save_pretrained(output_path)
        elif quant_type == QuantizationType.FP16:
            # Half precision
            model.half()
            model.save_pretrained(output_path)
        else:
            raise ValueError(f"Unsupported Transformers quantization: {quant_type}")

        return {"model_path": str(output_path)}

    def _apply_int8_quantization(
        self,
        model: Any,
        calibration_dataset: Optional[Any],
    ) -> Any:
        """Apply INT8 quantization to PyTorch model."""
        import torch
        from torch.ao.quantization import (
            quantize_dynamic,
            quantize_static,
            get_default_qconfig_spec,
            prepare,
            convert,
        )

        if calibration_dataset is not None:
            # Static quantization with calibration
            model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')

            prepare(model, inplace=True)

            # Calibration
            with torch.no_grad():
                for data in calibration_dataset:
                    model(data)

            # Convert
            quantized_model = convert(model, inplace=True)
        else:
            # Dynamic quantization (no calibration needed)
            quantized_model = quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d},
                dtype=torch.qint8,
            )

        return quantized_model

    def _apply_dynamic_quantization(self, model: Any) -> Any:
        """Apply dynamic quantization."""
        import torch
        from torch.ao.quantization import quantize_dynamic

        return quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},
            dtype=torch.qint8,
        )

    def _create_calibration_dataset(
        self,
        texts: List[str],
        model: Any,
    ) -> List[Any]:
        """Create calibration dataset from text samples."""
        # This is a simplified implementation
        # In production, you would process texts through the model's tokenizer
        # and create proper input tensors

        import torch

        calibration_data = []
        for text in texts[:100]:  # Use up to 100 samples
            # Create dummy input (adjust based on actual model input)
            # This should be replaced with actual preprocessing
            dummy_input = torch.randn(1, 80, 512)  # Example shape
            calibration_data.append(dummy_input)

        return calibration_data

    def _get_model_size_mb(self, model_path: str) -> float:
        """Get model size in MB."""
        path = Path(model_path)
        if path.is_file():
            return path.stat().st_size / (1024 * 1024)
        elif path.is_dir():
            # For model directories (transformers)
            total_size = sum(
                f.stat().st_size for f in path.rglob('*') if f.is_file()
            )
            return total_size / (1024 * 1024)
        return 0.0

    def _get_output_path(
        self,
        model_path: str,
        quant_type: QuantizationType,
    ) -> Path:
        """Get output path for quantized model."""
        model_name = Path(model_path).stem
        output_name = f"{model_name}_{quant_type.value}"

        # Preserve directory structure
        if Path(model_path).is_dir():
            return self.quantized_dir / output_name
        else:
            ext = Path(model_path).suffix
            return self.quantized_dir / f"{output_name}{ext}"

    def _save_quantization_metadata(
        self,
        original_path: str,
        quantized_path: str,
        quant_type: QuantizationType,
        model: Any,
    ):
        """Save metadata about quantization."""
        metadata = {
            "original_path": str(original_path),
            "quantized_path": str(quantized_path),
            "quantization_type": quant_type.value,
            "timestamp": __import__('time').time(),
        }

        # Try to get model-specific metadata
        try:
            if hasattr(model, 'config'):
                metadata["model_config"] = model.config.to_dict()
        except:
            pass

        # Save metadata JSON
        metadata_path = Path(quantized_path).parent / f"{Path(quantized_path).stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _estimate_speedup(self, quant_type: QuantizationType) -> float:
        """Estimate inference speedup from quantization."""
        speedups = {
            QuantizationType.INT8: 2.5,  # ~2.5x faster
            QuantizationType.FP16: 1.5,  # ~1.5x faster
            QuantizationType.DYNAMIC: 2.0,  # ~2x faster
            QuantizationType.GPTQ: 3.0,  # ~3x faster
            QuantizationType.AWQ: 2.8,  # ~2.8x faster
        }
        return speedups.get(quant_type, 1.0)

    async def compare_models(
        self,
        original_path: str,
        quantized_path: str,
        test_texts: List[str],
    ) -> Dict[str, Any]:
        """
        Compare original and quantized models.

        Args:
            original_path: Path to original model
            quantized_path: Path to quantized model
            test_texts: Sample texts for testing

        Returns:
            Comparison results
        """
        import time

        results = {
            "original_path": original_path,
            "quantized_path": quantized_path,
            "size_comparison": {},
            "latency_comparison": {},
            "quality_comparison": {},
        }

        # Size comparison
        original_size = self._get_model_size_mb(original_path)
        quantized_size = self._get_model_size_mb(quantized_path)

        results["size_comparison"] = {
            "original_mb": round(original_size, 2),
            "quantized_mb": round(quantized_size, 2),
            "reduction_mb": round(original_size - quantized_size, 2),
            "reduction_percent": round((1 - quantized_size / original_size) * 100, 2),
        }

        # Latency comparison (placeholder - would need actual inference)
        results["latency_comparison"] = {
            "note": "Actual latency comparison requires model inference",
            "estimated_speedup": self._estimate_speedup_from_paths(
                original_path, quantized_path
            ),
        }

        # Quality comparison (placeholder - would need quality metrics)
        results["quality_comparison"] = {
            "note": "Quality comparison requires running inference on test data",
            "test_samples": len(test_texts),
        }

        return results

    def _estimate_speedup_from_paths(
        self,
        original_path: str,
        quantized_path: str,
    ) -> float:
        """Estimate speedup from model paths."""
        # Check quantization type from path
        for quant_type in QuantizationType:
            if quant_type.value in quantized_path:
                return self._estimate_speedup(quant_type)
        return 1.0

    async def list_quantized_models(self) -> List[Dict[str, Any]]:
        """List all quantized models."""
        models = []

        for model_file in self.quantized_dir.rglob("*"):
            if model_file.is_file() and not model_file.name.endswith("_metadata.json"):
                # Check for metadata
                metadata_file = model_file.parent / f"{model_file.stem}_metadata.json"
                metadata = {}
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                    except:
                        pass

                models.append({
                    "path": str(model_file),
                    "size_mb": round(self._get_model_size_mb(str(model_file)), 2),
                    "type": model_file.suffix,
                    "metadata": metadata,
                })

        return models

    async def cleanup_quantized_models(self, older_than_days: int = 30):
        """Clean up old quantized models."""
        import time

        cutoff_time = time.time() - (older_than_days * 24 * 3600)

        for model_file in self.quantized_dir.rglob("*"):
            if model_file.is_file():
                # Check metadata for timestamp
                metadata_file = model_file.parent / f"{model_file.stem}_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            timestamp = metadata.get("timestamp", 0)
                            if timestamp < cutoff_time:
                                model_file.unlink()
                                metadata_file.unlink()
                                self.logger.info(f"Cleaned up old model: {model_file}")
                    except:
                        pass

    async def validate_quantized_model(
        self,
        model_path: str,
    ) -> Dict[str, Any]:
        """
        Validate a quantized model.

        Args:
            model_path: Path to quantized model

        Returns:
            Validation results
        """
        path = Path(model_path)

        results = {
            "path": model_path,
            "exists": path.exists(),
            "readable": False,
            "valid": False,
            "issues": [],
        }

        if not path.exists():
            results["issues"].append("Model file does not exist")
            return results

        # Check readability
        try:
            if path.suffix in ['.pth', '.pt']:
                import torch
                torch.load(path, map_location='cpu')
                results["readable"] = True
            elif path.suffix == '.onnx':
                import onnx
                onnx.load(path)
                results["readable"] = True
            else:
                results["readable"] = True  # Assume readable for other formats
        except Exception as e:
            results["issues"].append(f"Cannot load model: {str(e)}")
            return results

        # Check metadata
        metadata_file = path.parent / f"{path.stem}_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    results["metadata"] = metadata
            except Exception as e:
                results["issues"].append(f"Invalid metadata: {str(e)}")

        results["valid"] = len(results["issues"]) == 0
        return results


# Global instance
_quantization_service: Optional[ModelQuantizationService] = None


def get_quantization_service(
    quantized_dir: str = "./data/quantized_models",
    calibration_dir: str = "./data/calibration_samples",
) -> ModelQuantizationService:
    """Get global quantization service instance."""
    global _quantization_service
    if _quantization_service is None:
        _quantization_service = ModelQuantizationService(
            quantized_models_dir=quantized_dir,
            calibration_samples_dir=calibration_dir,
        )
    return _quantization_service

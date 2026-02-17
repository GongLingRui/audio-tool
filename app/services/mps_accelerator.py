"""
MPS (Metal Performance Shaders) Accelerator for Apple Silicon
Optimized for M4 chip audio processing acceleration
"""

import os
from typing import Optional, Tuple, List, TYPE_CHECKING
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

logger = logging.getLogger(__name__)

# Type hints for when torch is not available
if TYPE_CHECKING:
    if not TORCH_AVAILABLE:
        import torch
        Module = torch.nn.Module
        Tensor = torch.Tensor
    else:
        Module = torch.nn.Module
        Tensor = torch.Tensor
else:
    if TORCH_AVAILABLE:
        Module = torch.nn.Module
        Tensor = torch.Tensor
    else:
        Module = object  # fallback type
        Tensor = object  # fallback type


class MPSAccelerator:
    """
    Metal Performance Shaders (MPS) accelerator for Apple Silicon.
    Provides GPU acceleration for audio processing on M1/M2/M3/M4 chips.
    """

    def __init__(self):
        self.device = None
        self.is_available = self._check_mps_availability()
        self.memory_info = self._get_memory_info()

    def _check_mps_availability(self) -> bool:
        """Check if MPS (Metal Performance Shaders) is available."""
        if torch is None or not TORCH_AVAILABLE:
            logger.warning("PyTorch not installed, falling back to CPU")
            self.device = "cpu"
            return False

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info(f"✓ MPS (Metal Performance Shaders) available on {self.device}")
            logger.info(f"✓ MPS Built: {torch.backends.mps.is_built()}")
            return True
        else:
            logger.warning("MPS not available, falling back to CPU")
            self.device = torch.device("cpu")
            return False

    def _get_memory_info(self) -> dict:
        """Get GPU memory information."""
        info = {
            "device": str(self.device),
            "available": self.is_available,
        }

        if self.is_available:
            # Get system memory (shared with GPU on Apple Silicon)
            import psutil
            mem = psutil.virtual_memory()
            info["system_memory_gb"] = mem.total / (1024**3)
            info["available_memory_gb"] = mem.available / (1024**3)

        return info

    def optimize_model_for_mps(self, model: Module) -> Module:
        """
        Optimize a PyTorch model for MPS acceleration.

        Args:
            model: PyTorch model to optimize

        Returns:
            Optimized model on MPS device
        """
        if not self.is_available:
            logger.warning("MPS not available, model running on CPU")
            return model.to(self.device)

        try:
            # Move model to MPS device
            model = model.to(self.device)

            # Enable memory-efficient attention if available
            if hasattr(model, 'config'):
                # For transformer models
                try:
                    model.config.use_memory_efficient_attention = True
                except:
                    pass

            # Set to eval mode for inference
            model.eval()

            logger.info(f"✓ Model optimized for MPS: {self.device}")
            return model

        except Exception as e:
            logger.error(f"Error optimizing model for MPS: {e}")
            if torch is not None:
                return model.to(torch.device("cpu"))
            return model

    def accelerate_audio_processing(self, audio_tensor: Tensor) -> Tensor:
        """
        Accelerate audio tensor operations on MPS.

        Args:
            audio_tensor: Audio tensor to process

        Returns:
            Processed tensor on MPS device
        """
        if not self.is_available or torch is None:
            return audio_tensor

        try:
            # Ensure tensor is on MPS device
            if hasattr(audio_tensor, 'device') and str(audio_tensor.device) != str(self.device):
                audio_tensor = audio_tensor.to(self.device)

            return audio_tensor

        except Exception as e:
            logger.error(f"Error in audio acceleration: {e}")
            return audio_tensor

    def batch_process_tensors(
        self,
        tensors: List[Tensor],
        operation: str = "stack"
    ) -> Tensor:
        """
        Batch process multiple tensors on MPS.

        Args:
            tensors: List of tensors to process
            operation: Operation to perform (stack, cat, add)

        Returns:
            Processed tensor
        """
        if torch is None:
            # No torch available, return as-is
            return tensors[0] if tensors else None

        if not self.is_available:
            # CPU fallback
            if operation == "stack":
                return torch.stack(tensors)
            elif operation == "cat":
                return torch.cat(tensors)
            elif operation == "add":
                return torch.stack(tensors).sum(dim=0)

        try:
            # Move all tensors to MPS
            mps_tensors = [t.to(self.device) for t in tensors]

            if operation == "stack":
                return torch.stack(mps_tensors)
            elif operation == "cat":
                return torch.cat(mps_tensors)
            elif operation == "add":
                return torch.stack(mps_tensors).sum(dim=0)
            else:
                raise ValueError(f"Unknown operation: {operation}")

        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            # Fallback to CPU
            if operation == "stack":
                return torch.stack(tensors)
            return tensors[0]

    def get_optimal_batch_size(self, tensor_shape: Tuple[int, ...]) -> int:
        """
        Calculate optimal batch size based on available memory.

        Args:
            tensor_shape: Shape of individual tensor

        Returns:
            Optimal batch size
        """
        if not self.is_available:
            return 8  # Conservative default for CPU

        try:
            available_memory_gb = self.memory_info.get("available_memory_gb", 16)
            tensor_size_mb = np.prod(tensor_shape) * 4 / (1024**2)  # float32

            # Use 20% of available memory for batch
            usable_memory_gb = available_memory_gb * 0.2
            usable_memory_mb = usable_memory_gb * 1024

            optimal_batch = int(usable_memory_mb / tensor_size_mb)

            # Clamp between 1 and 64
            return max(1, min(64, optimal_batch))

        except:
            return 8

    def tensor_to_numpy(self, tensor: Tensor) -> "np.ndarray":
        """
        Convert MPS tensor to numpy array efficiently.

        Args:
            tensor: PyTorch tensor on MPS

        Returns:
            NumPy array
        """
        if torch is None:
            return np.array([]) if np is not None else []

        if hasattr(tensor, 'is_mps') and tensor.is_mps:
            return tensor.cpu().numpy()
        return tensor.numpy()

    def numpy_to_tensor(self, array: "np.ndarray") -> Tensor:
        """
        Convert numpy array to MPS tensor efficiently.

        Args:
            array: NumPy array

        Returns:
            PyTorch tensor on MPS
        """
        if torch is None:
            return array  # Return as-is if torch not available

        tensor = torch.from_numpy(array)
        if self.is_available and self.device != "cpu":
            tensor = tensor.to(self.device)
        return tensor

    def clear_cache(self):
        """Clear MPS cache to free memory."""
        if self.is_available and torch is not None:
            torch.mps.empty_cache()
            logger.info("MPS cache cleared")

    def get_info(self) -> dict:
        """Get accelerator information."""
        info = {
            **self.memory_info,
        }
        if torch is not None:
            info["torch_version"] = torch.__version__
            info["mps_backend_built"] = torch.backends.mps.is_built() if hasattr(torch.backends, 'mps') else False
        return info


# Singleton instance
_mps_accelerator: Optional[MPSAccelerator] = None


def get_mps_accelerator() -> MPSAccelerator:
    """Get or create MPS accelerator singleton."""
    global _mps_accelerator
    if _mps_accelerator is None:
        _mps_accelerator = MPSAccelerator()
    return _mps_accelerator


def auto_device():
    """
    Automatically select best available device (MPS > CPU).

    Returns:
        Best available torch device
    """
    accelerator = get_mps_accelerator()
    return accelerator.device

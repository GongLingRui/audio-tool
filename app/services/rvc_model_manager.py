"""
RVC Model Manager
Manages RVC (Retrieval-based Voice Conversion) models for voice conversion
"""
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import subprocess

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """RVC model status."""
    AVAILABLE = "available"
    TRAINING = "training"
    ERROR = "error"
    INCOMPLETE = "incomplete"


@dataclass
class RVCModel:
    """RVC model information."""
    model_id: str
    name: str
    description: str
    language: str = "zh-CN"
    gender: str = "female"
    sample_rate: int = 48000
    f0_method: str = "rmvpe"  # pitch extraction method: crepe, rmvpe, harvest
    f0_up_key: int = 0  # pitch shift
    filter_radius: int = 3
    resample_sr: int = 0
    rms_mix_rate: float = 0.25
    protect: float = 0.33
    version: str = "v2"
    status: ModelStatus = ModelStatus.AVAILABLE
    file_path: Optional[str] = None
    index_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    file_size_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "description": self.description,
            "language": self.language,
            "gender": self.gender,
            "sample_rate": self.sample_rate,
            "f0_method": self.f0_method,
            "f0_up_key": self.f0_up_key,
            "filter_radius": self.filter_radius,
            "resample_sr": self.resample_sr,
            "rms_mix_rate": self.rms_mix_rate,
            "protect": self.protect,
            "version": self.version,
            "status": self.status.value,
            "file_path": self.file_path,
            "index_path": self.index_path,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "file_size_mb": self.file_size_mb,
        }


class RVCModelManager:
    """
    Manager for RVC voice conversion models.

    Features:
    - Model upload and storage
    - Model training support
    - Model validation
    - Model metadata management
    - RVC inference integration
    """

    def __init__(
        self,
        models_dir: str = "./data/rvc_models",
        rvc_path: Optional[str] = None,
    ):
        """
        Initialize RVC model manager.

        Args:
            models_dir: Directory to store RVC models
            rvc_path: Path to RVC project (for inference)
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.rvc_path = Path(rvc_path) if rvc_path else None

        self.logger = logging.getLogger(self.__class__.__name__)
        self._models: Dict[str, RVCModel] = {}
        self._load_models()

    def _load_models(self):
        """Load model metadata from storage."""
        metadata_file = self.models_dir / "models_registry.json"

        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for model_data in data.get("models", []):
                    model = RVCModel(**model_data)
                    model.status = ModelStatus(model_data.get("status", "available"))
                    self._models[model.model_id] = model

                self.logger.info(f"Loaded {len(self._models)} RVC models from registry")

            except Exception as e:
                self.logger.error(f"Error loading models: {e}")

        # Also scan for .pth files
        self._scan_model_files()

    def _scan_model_files(self):
        """Scan models directory for .pth model files."""
        for pth_file in self.models_dir.glob("*.pth"):
            model_id = pth_file.stem

            # Skip if already loaded
            if model_id in self._models:
                continue

            # Create basic model entry
            self._models[model_id] = RVCModel(
                model_id=model_id,
                name=model_id.replace("_", " ").title(),
                description=f"RVC model: {model_id}",
                file_path=str(pth_file),
                file_size_mb=pth_file.stat().st_size / (1024 * 1024),
            )

            # Check for index file
            index_file = pth_file.parent / f"{model_id}.index"
            if index_file.exists():
                self._models[model_id].index_path = str(index_file)

    def _save_registry(self):
        """Save model registry to disk."""
        metadata_file = self.models_dir / "models_registry.json"

        data = {
            "models": [
                model.to_dict()
                for model in self._models.values()
            ],
            "last_updated": time.time(),
        }

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    async def register_model(
        self,
        model_id: str,
        name: str,
        description: str,
        file_path: str,
        index_path: Optional[str] = None,
        **kwargs,
    ) -> RVCModel:
        """
        Register a new RVC model.

        Args:
            model_id: Unique model identifier
            name: Model display name
            description: Model description
            file_path: Path to .pth model file
            index_path: Path to .index file (optional)
            **kwargs: Additional model parameters

        Returns:
            RVCModel object
        """
        model = RVCModel(
            model_id=model_id,
            name=name,
            description=description,
            file_path=file_path,
            index_path=index_path,
            **kwargs,
        )

        # Get file size
        if Path(file_path).exists():
            model.file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)

        self._models[model_id] = model
        self._save_registry()

        self.logger.info(f"Registered RVC model: {model_id}")
        return model

    async def upload_model(
        self,
        model_file: bytes,
        model_id: str,
        name: str,
        description: str = "",
        index_file: Optional[bytes] = None,
        **kwargs,
    ) -> RVCModel:
        """
        Upload and register an RVC model.

        Args:
            model_file: Model file bytes
            model_id: Unique model identifier
            name: Model display name
            description: Model description
            index_file: Index file bytes (optional)
            **kwargs: Additional parameters

        Returns:
            RVCModel object
        """
        # Save model file
        model_path = self.models_dir / f"{model_id}.pth"
        with open(model_path, 'wb') as f:
            f.write(model_file)

        # Save index file if provided
        index_path_str = None
        if index_file:
            index_path = self.models_dir / f"{model_id}.index"
            with open(index_path, 'wb') as f:
                f.write(index_file)
            index_path_str = str(index_path)

        # Register model
        model = await self.register_model(
            model_id=model_id,
            name=name,
            description=description,
            file_path=str(model_path),
            index_path=index_path_str,
            **kwargs,
        )

        return model

    async def delete_model(self, model_id: str) -> bool:
        """
        Delete an RVC model.

        Args:
            model_id: Model to delete

        Returns:
            True if deleted
        """
        if model_id not in self._models:
            return False

        model = self._models[model_id]

        # Delete files
        if model.file_path and Path(model.file_path).exists():
            Path(model.file_path).unlink()

        if model.index_path and Path(model.index_path).exists():
            Path(model.index_path).unlink()

        # Remove from registry
        del self._models[model_id]
        self._save_registry()

        self.logger.info(f"Deleted RVC model: {model_id}")
        return True

    def get_model(self, model_id: str) -> Optional[RVCModel]:
        """Get model by ID."""
        return self._models.get(model_id)

    def list_models(
        self,
        language: Optional[str] = None,
        gender: Optional[str] = None,
        status: Optional[ModelStatus] = None,
    ) -> List[RVCModel]:
        """
        List all models with optional filtering.

        Args:
            language: Filter by language
            gender: Filter by gender
            status: Filter by status

        Returns:
            List of RVCModel objects
        """
        models = list(self._models.values())

        if language:
            models = [m for m in models if m.language == language]

        if gender:
            models = [m for m in models if m.gender == gender]

        if status:
            models = [m for m in models if m.status == status]

        return models

    async def convert_audio(
        self,
        source_audio: bytes,
        model_id: str,
        pitch_shift: int = 0,
        f0_method: str = "rmvpe",
        filter_radius: int = 3,
        resample_sr: int = 0,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
    ) -> bytes:
        """
        Convert audio using RVC model.

        Args:
            source_audio: Source audio bytes
            model_id: RVC model to use
            pitch_shift: Pitch shift in semitones
            f0_method: F0 extraction method
            filter_radius: Filter radius
            resample_sr: Resample sample rate (0 = don't resample)
            rms_mix_rate: RMS mix rate
            protect: Consonant protection

        Returns:
            Converted audio bytes
        """
        if model_id not in self._models:
            raise ValueError(f"Model not found: {model_id}")

        model = self._models[model_id]

        if not model.file_path or not Path(model.file_path).exists():
            raise ValueError(f"Model file not found: {model.file_path}")

        # Save source audio temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_input:
            tmp_input.write(source_audio)
            input_path = tmp_input.name

        try:
            # Call RVC inference
            if self.rvc_path and Path(self.rvc_path).exists():
                # Use local RVC installation
                output_path = await self._run_rvc_inference(
                    input_path=input_path,
                    model_path=model.file_path,
                    index_path=model.index_path,
                    pitch_shift=pitch_shift,
                    f0_method=f0_method,
                    filter_radius=filter_radius,
                    resample_sr=resample_sr,
                    rms_mix_rate=rms_mix_rate,
                    protect=protect,
                )
            else:
                # Use placeholder (actual RVC integration needed)
                self.logger.warning("RVC path not configured, returning original audio")
                output_path = input_path

            # Read output
            with open(output_path, 'rb') as f:
                result = f.read()

            # Cleanup
            Path(input_path).unlink(missing_ok=True)
            if output_path != input_path:
                Path(output_path).unlink(missing_ok=True)

            return result

        except Exception as e:
            self.logger.error(f"RVC conversion error: {e}")
            raise

    async def _run_rvc_inference(
        self,
        input_path: str,
        model_path: str,
        index_path: Optional[str],
        pitch_shift: int,
        f0_method: str,
        filter_radius: int,
        resample_sr: int,
        rms_mix_rate: float,
        protect: float,
    ) -> str:
        """
        Run RVC inference subprocess.

        Returns path to output audio file.
        """
        output_path = str(Path(input_path).with_suffix("_out.wav"))

        cmd = [
            "python",
            "infer_rvc.py",
            "--input", str(input_path),
            "--model", str(model_path),
            "--output", output_path,
            "--pitch_shift", str(pitch_shift),
            "--f0_method", f0_method,
            "--filter_radius", str(filter_radius),
            "--resample_sr", str(resample_sr),
            "--rms_mix_rate", str(rms_mix_rate),
            "--protect", str(protect),
        ]

        if index_path and Path(index_path).exists():
            cmd.extend(["--index", str(index_path)])

        # Run RVC inference
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.rvc_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"RVC inference failed: {stderr.decode()}")

        return output_path

    async def train_model(
        self,
        model_id: str,
        training_data: List[bytes],
        sample_rate: int = 48000,
        epochs: int = 100,
        batch_size: int = 8,
        save_every_epoch: int = 10,
        rvc_version: str = "v2",
    ) -> Dict[str, Any]:
        """
        Train a new RVC model (framework for actual training).

        Args:
            model_id: Model ID for new model
            training_data: List of audio samples
            sample_rate: Audio sample rate
            epochs: Number of training epochs
            batch_size: Batch size
            save_every_epoch: Save checkpoint every N epochs
            rvc_version: RVC version (v1 or v2)

        Returns:
            Training result with model path
        """
        # This is a framework - actual RVC training requires
        # the RVC training pipeline

        training_dir = self.models_dir / f"{model_id}_training"
        training_dir.mkdir(exist_ok=True)

        # Save training samples
        samples_dir = training_dir / "samples"
        samples_dir.mkdir(exist_ok=True)

        for i, sample in enumerate(training_data):
            sample_path = samples_dir / f"sample_{i}.wav"
            with open(sample_path, 'wb') as f:
                f.write(sample)

        # Create training config
        config = {
            "model_id": model_id,
            "sample_rate": sample_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "save_every_epoch": save_every_epoch,
            "rvc_version": rvc_version,
            "samples_dir": str(samples_dir),
        }

        config_path = training_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        self.logger.info(f"Training data prepared for model: {model_id}")
        self.logger.info("Note: Actual RVC training requires external RVC training pipeline")

        return {
            "model_id": model_id,
            "status": "preparation_complete",
            "training_dir": str(training_dir),
            "samples_count": len(training_data),
            "config_path": str(config_path),
            "note": "Actual training requires RVC training pipeline",
        }

    async def validate_model(self, model_id: str) -> Dict[str, Any]:
        """
        Validate an RVC model.

        Args:
            model_id: Model to validate

        Returns:
            Validation results
        """
        if model_id not in self._models:
            return {
                "valid": False,
                "error": "Model not found",
            }

        model = self._models[model_id]

        results = {
            "model_id": model_id,
            "valid": True,
            "checks": {},
        }

        # Check model file exists
        if model.file_path:
            results["checks"]["file_exists"] = Path(model.file_path).exists()

        # Check index file
        if model.index_path:
            results["checks"]["index_exists"] = Path(model.index_path).exists()

        # Check file size
        if model.file_path and Path(model.file_path).exists():
            size = Path(model.file_path).stat().st_size / (1024 * 1024)
            results["checks"]["file_size_mb"] = round(size, 2)
            results["checks"]["file_size_ok"] = size > 1  # At least 1MB

        # Check parameters
        results["checks"]["has_f0_method"] = bool(model.f0_method)
        results["checks"]["has_version"] = bool(model.version)

        # Overall validity
        results["valid"] = all(
            results["checks"].get(key, True)
            for key in ["file_exists", "file_size_ok", "has_f0_method"]
        )

        return results


# Global instance
_rvc_manager: Optional[RVCModelManager] = None


def get_rvc_manager(
    models_dir: str = "./data/rvc_models",
    rvc_path: Optional[str] = None,
) -> RVCModelManager:
    """Get global RVC manager instance."""
    if rvc_path is None:
        import os
        rvc_path = os.getenv("RVC_PROJECT_PATH")

    global _rvc_manager
    if _rvc_manager is None:
        _rvc_manager = RVCModelManager(
            models_dir=models_dir,
            rvc_path=rvc_path,
        )
    return _rvc_manager

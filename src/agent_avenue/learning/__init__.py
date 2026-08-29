"""Optional NumPy/PyTorch learning components.

This package is intentionally not imported from the core package or existing agent registry.
Install the ``rl`` extra before importing it.
"""

from .checkpoint import (
    CheckpointCompatibility,
    CheckpointCompatibilityError,
    CheckpointError,
    CheckpointIntegrityError,
    LoadedCheckpoint,
    SavedCheckpoint,
    inspect_checkpoint,
    load_checkpoint,
    save_checkpoint,
    tensor_digest,
)
from .dataset import (
    DATASET_SCHEMA_VERSION,
    DatasetError,
    DatasetSplit,
    MaterializedDataset,
    TrainingSample,
    extract_game_samples,
    load_dataset,
    materialize_dataset,
    save_dataset,
)
from .model import CandidateMLP, CandidateValueModel, create_model, model_spec
from .train import TrainingConfig, TrainingData, TrainingResult, evaluate_model, train_model

__all__ = [
    "DATASET_SCHEMA_VERSION",
    "CandidateMLP",
    "CandidateValueModel",
    "CheckpointCompatibility",
    "CheckpointCompatibilityError",
    "CheckpointError",
    "CheckpointIntegrityError",
    "DatasetError",
    "DatasetSplit",
    "LoadedCheckpoint",
    "MaterializedDataset",
    "SavedCheckpoint",
    "TrainingConfig",
    "TrainingData",
    "TrainingResult",
    "TrainingSample",
    "create_model",
    "evaluate_model",
    "extract_game_samples",
    "inspect_checkpoint",
    "load_checkpoint",
    "load_dataset",
    "materialize_dataset",
    "model_spec",
    "save_checkpoint",
    "save_dataset",
    "tensor_digest",
    "train_model",
]

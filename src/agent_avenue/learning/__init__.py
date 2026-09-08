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
from .evaluation import (
    GamePredictionMetrics,
    RecordEvaluation,
    RecordEvaluationError,
    evaluate_records,
)
from .model import CandidateMLP, CandidateValueModel, create_model, model_spec
from .ranking import (
    RANKING_DATASET_VERSION,
    RANKING_LOSS_VERSION,
    RankingDataset,
    RankingError,
    RankingMetrics,
    RankingSplit,
    RankingTrainingConfig,
    RankingTrainingResult,
    load_ranking_dataset,
    materialize_ranking_dataset,
    save_ranking_dataset,
    train_ranking_model,
)
from .train import TrainingConfig, TrainingData, TrainingResult, evaluate_model, train_model

__all__ = [
    "DATASET_SCHEMA_VERSION",
    "RANKING_DATASET_VERSION",
    "RANKING_LOSS_VERSION",
    "CandidateMLP",
    "CandidateValueModel",
    "CheckpointCompatibility",
    "CheckpointCompatibilityError",
    "CheckpointError",
    "CheckpointIntegrityError",
    "DatasetError",
    "DatasetSplit",
    "GamePredictionMetrics",
    "LoadedCheckpoint",
    "MaterializedDataset",
    "RankingDataset",
    "RankingError",
    "RankingMetrics",
    "RankingSplit",
    "RankingTrainingConfig",
    "RankingTrainingResult",
    "RecordEvaluation",
    "RecordEvaluationError",
    "SavedCheckpoint",
    "TrainingConfig",
    "TrainingData",
    "TrainingResult",
    "TrainingSample",
    "create_model",
    "evaluate_model",
    "evaluate_records",
    "extract_game_samples",
    "inspect_checkpoint",
    "load_checkpoint",
    "load_dataset",
    "load_ranking_dataset",
    "materialize_dataset",
    "materialize_ranking_dataset",
    "model_spec",
    "save_checkpoint",
    "save_dataset",
    "save_ranking_dataset",
    "tensor_digest",
    "train_model",
    "train_ranking_model",
]

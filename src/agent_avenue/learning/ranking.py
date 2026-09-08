"""Deterministic all-legal-action heuristic-ranking datasets and pretraining.

The teacher consumes only the acting player's public observation. Its scores define ordinal
preferences, not counterfactual returns. Ranking-only weights are training initializers; final
inference checkpoints must still be fitted with the ordinary Monte Carlo objective.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import functional as F

from agent_avenue.agents.heuristic import GreedyHeuristicConfig, score_actions
from agent_avenue.encoding import (
    ENCODER_FINGERPRINT,
    ENCODER_VERSION,
    FEATURE_NAMES,
    FEATURE_WIDTH,
    encode_candidate,
)
from agent_avenue.engine import Phase, apply_action, new_game
from agent_avenue.observation import observe
from agent_avenue.storage import (
    GameRecord,
    game_record_fingerprint,
    rules_fingerprint,
    verify_game_record,
)

from .model import CandidateMLP, ModelError, validate_state_dict

RANKING_DATASET_SCHEMA_VERSION: Final[int] = 1
RANKING_DATASET_VERSION: Final[str] = "heuristic-all-legal-ranking-v1"
RANKING_LOSS_VERSION: Final[str] = "pairwise-logistic-equal-position-v1"
RANKING_BATCHER_VERSION: Final[str] = "seeded-position-randperm-v1"


class RankingError(ValueError):
    """Raised for malformed ranking data, configuration, or artifacts."""


@dataclass(frozen=True, slots=True)
class RankingSplit:
    candidate_features: NDArray[np.float32]
    position_candidate_offsets: NDArray[np.int64]
    preferred_candidate_index: NDArray[np.int64]
    dispreferred_candidate_index: NDArray[np.int64]
    position_pair_offsets: NDArray[np.int64]
    game_index: NDArray[np.int64]
    decision_index: NDArray[np.int64]
    phase: NDArray[np.str_]

    def __post_init__(self) -> None:
        features = np.asarray(self.candidate_features, dtype=np.float32)
        candidate_offsets = np.asarray(self.position_candidate_offsets, dtype=np.int64)
        preferred = np.asarray(self.preferred_candidate_index, dtype=np.int64)
        dispreferred = np.asarray(self.dispreferred_candidate_index, dtype=np.int64)
        pair_offsets = np.asarray(self.position_pair_offsets, dtype=np.int64)
        game_index = np.asarray(self.game_index, dtype=np.int64)
        decision_index = np.asarray(self.decision_index, dtype=np.int64)
        phase = np.asarray(self.phase, dtype=np.str_)
        if features.ndim != 2 or features.shape[1] != FEATURE_WIDTH or features.shape[0] == 0:
            raise RankingError(f"candidate_features must have shape [C, {FEATURE_WIDTH}]")
        if candidate_offsets.ndim != 1 or pair_offsets.ndim != 1 or candidate_offsets.size < 2:
            raise RankingError("position offsets must be non-empty one-dimensional arrays")
        position_count = candidate_offsets.size - 1
        if pair_offsets.size != position_count + 1:
            raise RankingError("candidate and pair offsets must describe the same positions")
        if game_index.shape != (position_count,) or decision_index.shape != (position_count,):
            raise RankingError("position metadata length mismatch")
        if phase.shape != (position_count,):
            raise RankingError("phase metadata length mismatch")
        if candidate_offsets[0] != 0 or candidate_offsets[-1] != features.shape[0]:
            raise RankingError("candidate offsets do not cover candidate features")
        if pair_offsets[0] != 0 or pair_offsets[-1] != preferred.size:
            raise RankingError("pair offsets do not cover preference pairs")
        if preferred.shape != dispreferred.shape:
            raise RankingError("preference pair arrays must have equal shape")
        if np.any(np.diff(candidate_offsets) <= 0) or np.any(np.diff(pair_offsets) < 0):
            raise RankingError("positions require candidates and monotone offsets")
        if preferred.size and (
            np.any(preferred < 0)
            or np.any(dispreferred < 0)
            or np.any(preferred >= features.shape[0])
            or np.any(dispreferred >= features.shape[0])
        ):
            raise RankingError("preference pair candidate index is out of range")
        if not np.isfinite(features).all() or np.any(game_index < 0) or np.any(decision_index < 0):
            raise RankingError("ranking split contains invalid values")
        for array in (
            features,
            candidate_offsets,
            preferred,
            dispreferred,
            pair_offsets,
            game_index,
            decision_index,
            phase,
        ):
            detached = np.ascontiguousarray(array).copy()
            detached.setflags(write=False)
            object.__setattr__(
                self,
                {
                    id(features): "candidate_features",
                    id(candidate_offsets): "position_candidate_offsets",
                    id(preferred): "preferred_candidate_index",
                    id(dispreferred): "dispreferred_candidate_index",
                    id(pair_offsets): "position_pair_offsets",
                    id(game_index): "game_index",
                    id(decision_index): "decision_index",
                    id(phase): "phase",
                }[id(array)],
                detached,
            )

    @property
    def position_count(self) -> int:
        return int(self.game_index.size)

    @property
    def candidate_count(self) -> int:
        return int(self.candidate_features.shape[0])

    @property
    def pair_count(self) -> int:
        return int(self.preferred_candidate_index.size)

    @property
    def supervised_positions(self) -> NDArray[np.int64]:
        return np.flatnonzero(np.diff(self.position_pair_offsets) > 0).astype(np.int64)


@dataclass(frozen=True, slots=True)
class RankingDataset:
    train: RankingSplit
    validation: RankingSplit
    manifest: dict[str, object]

    @property
    def fingerprint(self) -> str:
        value = self.manifest.get("dataset_fingerprint")
        if not isinstance(value, str):
            raise RankingError("ranking dataset manifest has no fingerprint")
        return value


@dataclass(frozen=True, slots=True)
class RankingMetrics:
    pair_count: int
    supervised_position_count: int
    game_count: int
    pooled_pair_loss: float
    equal_position_loss: float
    equal_game_loss: float
    pair_accuracy: float

    def to_data(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RankingEpochMetrics:
    epoch: int
    train: RankingMetrics
    validation: RankingMetrics
    improved: bool
    early_stop_counter: int


@dataclass(frozen=True, slots=True)
class RankingTrainingConfig:
    seed: int
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_positions: int = 1024
    max_epochs: int = 50
    early_stopping_patience: int = 8
    min_delta: float = 0.0
    cpu_threads: int = 1
    deterministic_algorithms: bool = True
    optimizer: str = field(default="adamw", init=False)
    loss: str = field(default=RANKING_LOSS_VERSION, init=False)
    batcher: str = field(default=RANKING_BATCHER_VERSION, init=False)

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise RankingError("ranking seed must be an integer in [0, 2**63)")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise RankingError("ranking learning rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise RankingError("ranking weight decay must be finite and non-negative")
        for name in ("batch_positions", "max_epochs", "early_stopping_patience", "cpu_threads"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise RankingError(f"{name} must be a positive integer")
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise RankingError("ranking min_delta must be finite and non-negative")
        if self.deterministic_algorithms is not True:
            raise RankingError("ranking training requires deterministic algorithms")

    def normalized(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RankingTrainingResult:
    model: CandidateMLP
    config: RankingTrainingConfig
    history: tuple[RankingEpochMetrics, ...]
    best_epoch: int
    epochs_completed: int
    stopped_early: bool
    best_validation_loss: float
    train_metrics: RankingMetrics
    validation_metrics: RankingMetrics
    positions_per_second: float = field(compare=False)
    wall_clock_seconds: float = field(compare=False)

    def metrics_dict(self) -> dict[str, object]:
        return {
            "version": RANKING_LOSS_VERSION,
            "best_epoch": self.best_epoch,
            "epochs_completed": self.epochs_completed,
            "stopped_early": self.stopped_early,
            "best_validation_loss": self.best_validation_loss,
            "train": self.train_metrics.to_data(),
            "validation": self.validation_metrics.to_data(),
            "history": [
                {
                    "epoch": item.epoch,
                    "train": item.train.to_data(),
                    "validation": item.validation.to_data(),
                    "improved": item.improved,
                    "early_stop_counter": item.early_stop_counter,
                }
                for item in self.history
            ],
            "runtime": {
                "positions_per_second": self.positions_per_second,
                "wall_clock_seconds": self.wall_clock_seconds,
            },
        }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _dataset_identity(manifest: Mapping[str, object]) -> str:
    payload = {
        key: value for key, value in manifest.items() if key not in {"dataset_fingerprint", "files"}
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _build_split(
    rows: list[tuple[int, int, Phase, tuple[tuple[float, ...], ...], tuple[tuple[int, int], ...]]],
) -> RankingSplit:
    features: list[tuple[float, ...]] = []
    candidate_offsets = [0]
    preferred: list[int] = []
    dispreferred: list[int] = []
    pair_offsets = [0]
    games: list[int] = []
    decisions: list[int] = []
    phases: list[str] = []
    for game_index, decision_index, phase, candidates, local_pairs in rows:
        base = len(features)
        features.extend(candidates)
        candidate_offsets.append(len(features))
        preferred.extend(base + left for left, _ in local_pairs)
        dispreferred.extend(base + right for _, right in local_pairs)
        pair_offsets.append(len(preferred))
        games.append(game_index)
        decisions.append(decision_index)
        phases.append(phase.value)
    if not rows:
        raise RankingError("ranking split contains no positions")
    return RankingSplit(
        np.asarray(features, dtype=np.float32),
        np.asarray(candidate_offsets, dtype=np.int64),
        np.asarray(preferred, dtype=np.int64),
        np.asarray(dispreferred, dtype=np.int64),
        np.asarray(pair_offsets, dtype=np.int64),
        np.asarray(games, dtype=np.int64),
        np.asarray(decisions, dtype=np.int64),
        np.asarray(phases, dtype=np.str_),
    )


def materialize_ranking_dataset(
    records: tuple[GameRecord, ...] | list[GameRecord],
    *,
    mc_manifest: Mapping[str, object],
    source_corpus_fingerprint: str,
    teacher_config: GreedyHeuristicConfig | None = None,
    verify_code: bool = True,
) -> RankingDataset:
    """Replay records into all-legal-action ordinal preferences using only public observations."""
    teacher = teacher_config or GreedyHeuristicConfig()
    split = mc_manifest.get("split")
    if not isinstance(split, Mapping):
        raise RankingError("MC dataset manifest has no split declaration")
    train_ids = split.get("train_game_fingerprints")
    validation_ids = split.get("validation_game_fingerprints")
    if not isinstance(train_ids, list) or not isinstance(validation_ids, list):
        raise RankingError("MC dataset split game identities are malformed")
    train_set = set(train_ids)
    validation_set = set(validation_ids)
    if train_set & validation_set:
        raise RankingError("MC dataset train and validation games overlap")

    ordered = sorted(records, key=game_record_fingerprint)
    PositionRow = tuple[
        int,
        int,
        Phase,
        tuple[tuple[float, ...], ...],
        tuple[tuple[int, int], ...],
    ]
    rows: dict[str, list[PositionRow]] = {
        "train": [],
        "validation": [],
    }
    strict_pairs = 0
    tied_pairs = 0
    tied_positions = 0
    phase_counts: dict[str, int] = {}
    candidate_histogram: dict[str, int] = {}
    seen: set[str] = set()
    for game_index, record in enumerate(ordered):
        final = verify_game_record(record, verify_code=verify_code)
        if final.outcome is None:
            raise RankingError("ranking source record is not terminal")
        fingerprint = game_record_fingerprint(record)
        seen.add(fingerprint)
        if fingerprint in train_set:
            split_name = "train"
        elif fingerprint in validation_set:
            split_name = "validation"
        else:
            raise RankingError("ranking source record is absent from the MC split")
        state = new_game(record.replay.config, record.replay.seed)
        for decision_index, chosen_action in enumerate(record.replay.actions):
            if state.phase is Phase.TERMINAL:
                raise RankingError("ranking source contains actions after terminal state")
            actor = (
                state.active_player if state.phase is Phase.PLAY else state.active_player.other()
            )
            observation = observe(state, actor)
            legal = observation.legal_actions
            scores = score_actions(observation, legal, teacher)
            candidates = tuple(encode_candidate(observation, action).vector for action in legal)
            pairs: list[tuple[int, int]] = []
            for left in range(len(legal)):
                for right in range(left + 1, len(legal)):
                    if scores[left] == scores[right]:
                        tied_pairs += 1
                    elif scores[left] > scores[right]:
                        pairs.append((left, right))
                    else:
                        pairs.append((right, left))
            if not pairs:
                tied_positions += 1
            strict_pairs += len(pairs)
            phase_counts[state.phase.value] = phase_counts.get(state.phase.value, 0) + 1
            key = str(len(legal))
            candidate_histogram[key] = candidate_histogram.get(key, 0) + 1
            rows[split_name].append(
                (game_index, decision_index, state.phase, candidates, tuple(pairs))
            )
            state = apply_action(state, chosen_action)
    expected = train_set | validation_set
    if seen != expected:
        raise RankingError("ranking corpus identities do not exactly match the MC split")

    train = _build_split(rows["train"])
    validation = _build_split(rows["validation"])
    if train.pair_count == 0 or validation.pair_count == 0:
        raise RankingError("both ranking splits require strict teacher preferences")
    manifest: dict[str, object] = {
        "schema_version": RANKING_DATASET_SCHEMA_VERSION,
        "version": RANKING_DATASET_VERSION,
        "encoder_version": ENCODER_VERSION,
        "encoder_fingerprint": ENCODER_FINGERPRINT,
        "feature_width": FEATURE_WIDTH,
        "feature_names": list(FEATURE_NAMES),
        "rules_fingerprint": rules_fingerprint(),
        "teacher": teacher.to_data(),
        "teacher_semantics": "ordinal-public-heuristic-preference-not-counterfactual-return",
        "source_corpus_fingerprint": source_corpus_fingerprint,
        "source_mc_dataset_fingerprint": mc_manifest.get("dataset_fingerprint"),
        "split": json.loads(json.dumps(split, sort_keys=True)),
        "counts": {
            "records": len(ordered),
            "positions": train.position_count + validation.position_count,
            "candidates": train.candidate_count + validation.candidate_count,
            "strict_preference_pairs": strict_pairs,
            "tied_candidate_pairs": tied_pairs,
            "tie_only_positions": tied_positions,
            "train_positions": train.position_count,
            "validation_positions": validation.position_count,
            "train_candidates": train.candidate_count,
            "validation_candidates": validation.candidate_count,
            "train_pairs": train.pair_count,
            "validation_pairs": validation.pair_count,
            "phase_positions": dict(sorted(phase_counts.items())),
            "legal_candidate_count_histogram": dict(
                sorted(candidate_histogram.items(), key=lambda item: int(item[0]))
            ),
        },
        "information_boundary": "PlayerObservation + legal semantic Action only",
    }
    manifest["dataset_fingerprint"] = _dataset_identity(manifest)
    return RankingDataset(train, validation, manifest)


def save_ranking_dataset(dataset: RankingDataset, path: Path) -> tuple[Path, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    array_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
    manifest_path = array_path.with_suffix(".json")
    if array_path.exists() != manifest_path.exists():
        array_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
    if array_path.exists() and manifest_path.exists():
        loaded = load_ranking_dataset(array_path)
        if loaded.fingerprint == dataset.fingerprint:
            return array_path, manifest_path
        raise RankingError("ranking dataset destination already contains a different artifact")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{array_path.name}.tmp-", dir=array_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            arrays: dict[str, object] = {}
            for split_name, split in (("train", dataset.train), ("validation", dataset.validation)):
                for field_name in (
                    "candidate_features",
                    "position_candidate_offsets",
                    "preferred_candidate_index",
                    "dispreferred_candidate_index",
                    "position_pair_offsets",
                    "game_index",
                    "decision_index",
                    "phase",
                ):
                    arrays[f"{split_name}_{field_name}"] = getattr(split, field_name)
            np.savez_compressed(destination, **arrays)  # type: ignore[arg-type]
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, array_path)
        dataset.manifest["files"] = {
            array_path.name: {
                "sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
                "size": array_path.stat().st_size,
            }
        }
        manifest_bytes = _canonical_json(dataset.manifest) + b"\n"
        manifest_descriptor, manifest_name = tempfile.mkstemp(
            prefix=f".{manifest_path.name}.tmp-", dir=manifest_path.parent
        )
        manifest_temporary = Path(manifest_name)
        try:
            with os.fdopen(manifest_descriptor, "wb") as destination:
                destination.write(manifest_bytes)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(manifest_temporary, manifest_path)
        except Exception:
            manifest_temporary.unlink(missing_ok=True)
            raise
        return array_path, manifest_path
    except Exception:
        temporary.unlink(missing_ok=True)
        if not manifest_path.exists():
            array_path.unlink(missing_ok=True)
        raise


def load_ranking_dataset(path: Path) -> RankingDataset:
    array_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
    manifest_path = array_path.with_suffix(".json")
    try:
        manifest = json.loads(manifest_path.read_text())
        arrays = np.load(array_path, allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RankingError("unable to load ranking dataset") from exc
    if not isinstance(manifest, dict) or manifest.get("dataset_fingerprint") != _dataset_identity(
        manifest
    ):
        raise RankingError("ranking dataset manifest fingerprint mismatch")
    file_data = manifest.get("files")
    expected = file_data.get(array_path.name) if isinstance(file_data, dict) else None
    actual_digest = hashlib.sha256(array_path.read_bytes()).hexdigest()
    if not isinstance(expected, dict) or expected.get("sha256") != actual_digest:
        raise RankingError("ranking dataset array digest mismatch")
    if (
        manifest.get("schema_version") != RANKING_DATASET_SCHEMA_VERSION
        or manifest.get("version") != RANKING_DATASET_VERSION
        or manifest.get("encoder_version") != ENCODER_VERSION
        or manifest.get("encoder_fingerprint") != ENCODER_FINGERPRINT
        or manifest.get("feature_names") != list(FEATURE_NAMES)
        or manifest.get("rules_fingerprint") != rules_fingerprint()
    ):
        raise RankingError("ranking dataset is incompatible with current rules or encoder")

    def split(name: str) -> RankingSplit:
        return RankingSplit(
            arrays[f"{name}_candidate_features"],
            arrays[f"{name}_position_candidate_offsets"],
            arrays[f"{name}_preferred_candidate_index"],
            arrays[f"{name}_dispreferred_candidate_index"],
            arrays[f"{name}_position_pair_offsets"],
            arrays[f"{name}_game_index"],
            arrays[f"{name}_decision_index"],
            arrays[f"{name}_phase"],
        )

    return RankingDataset(split("train"), split("validation"), manifest)


def _metrics(model: CandidateMLP, data: RankingSplit) -> RankingMetrics:
    features = torch.tensor(data.candidate_features, dtype=torch.float32)
    preferred = torch.tensor(data.preferred_candidate_index, dtype=torch.int64)
    dispreferred = torch.tensor(data.dispreferred_candidate_index, dtype=torch.int64)
    model.eval()
    with torch.inference_mode():
        logits = model(features)
        pair_losses = (
            F.softplus(-(logits[preferred] - logits[dispreferred]))
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
        pair_correct = (logits[preferred] > logits[dispreferred]).cpu().numpy()
    position_losses: list[float] = []
    position_games: list[int] = []
    for position in data.supervised_positions:
        start = int(data.position_pair_offsets[position])
        stop = int(data.position_pair_offsets[position + 1])
        position_losses.append(float(pair_losses[start:stop].mean()))
        position_games.append(int(data.game_index[position]))
    if not position_losses:
        raise RankingError("ranking metrics require supervised positions")
    losses_array = np.asarray(position_losses, dtype=np.float64)
    games_array = np.asarray(position_games, dtype=np.int64)
    game_losses = [
        float(losses_array[games_array == game].mean()) for game in np.unique(games_array)
    ]
    return RankingMetrics(
        pair_count=data.pair_count,
        supervised_position_count=len(position_losses),
        game_count=len(game_losses),
        pooled_pair_loss=float(pair_losses.mean()),
        equal_position_loss=float(losses_array.mean()),
        equal_game_loss=float(np.mean(game_losses)),
        pair_accuracy=float(pair_correct.mean()),
    )


def _clone_state(model: CandidateMLP) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


@contextmanager
def _strict_cpu(config: RankingTrainingConfig) -> Iterator[None]:
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(config.cpu_threads)
        torch.use_deterministic_algorithms(True, warn_only=False)
        yield
    finally:
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)
        torch.set_num_threads(previous_threads)


def train_ranking_model(
    dataset: RankingDataset,
    *,
    config: RankingTrainingConfig,
    initial_state_dict: Mapping[str, object],
) -> RankingTrainingResult:
    """Pretrain a candidate model on public heuristic preferences."""
    try:
        validate_state_dict(initial_state_dict)
    except ModelError as exc:
        raise RankingError("ranking initial state is incompatible") from exc
    model = CandidateMLP(seed=0)
    model.load_state_dict(initial_state_dict, strict=True)
    supervised = dataset.train.supervised_positions
    if supervised.size == 0:
        raise RankingError("ranking training split has no supervised positions")
    shuffle = torch.Generator(device="cpu")
    shuffle.manual_seed(config.seed)
    history: list[RankingEpochMetrics] = []
    best_state: dict[str, Tensor] | None = None
    best_loss = math.inf
    best_epoch = 0
    counter = 0
    positions_seen = 0
    started = time.perf_counter()
    all_features = torch.tensor(dataset.train.candidate_features, dtype=torch.float32)

    with _strict_cpu(config):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            foreach=False,
            fused=False,
        )
        for epoch in range(1, config.max_epochs + 1):
            model.train()
            order = supervised[torch.randperm(supervised.size, generator=shuffle).numpy()]
            for start in range(0, order.size, config.batch_positions):
                positions = order[start : start + config.batch_positions]
                pair_ranges = [
                    (
                        int(dataset.train.position_pair_offsets[position]),
                        int(dataset.train.position_pair_offsets[position + 1]),
                    )
                    for position in positions
                ]
                pair_counts = torch.tensor(
                    [stop - pair_start for pair_start, stop in pair_ranges],
                    dtype=torch.int64,
                )
                preferred = torch.tensor(
                    np.concatenate(
                        [
                            dataset.train.preferred_candidate_index[pair_start:pair_stop]
                            for pair_start, pair_stop in pair_ranges
                        ]
                    ),
                    dtype=torch.int64,
                )
                dispreferred = torch.tensor(
                    np.concatenate(
                        [
                            dataset.train.dispreferred_candidate_index[pair_start:pair_stop]
                            for pair_start, pair_stop in pair_ranges
                        ]
                    ),
                    dtype=torch.int64,
                )
                pair_positions = torch.repeat_interleave(
                    torch.arange(len(positions), dtype=torch.int64), pair_counts
                )
                optimizer.zero_grad(set_to_none=True)
                pair_losses = F.softplus(
                    -(model(all_features[preferred]) - model(all_features[dispreferred]))
                )
                position_sums = torch.zeros(len(positions), dtype=torch.float32)
                position_sums.scatter_add_(0, pair_positions, pair_losses)
                loss = (position_sums / pair_counts.to(torch.float32)).mean()
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
                positions_seen += len(positions)
            train_metrics = _metrics(model, dataset.train)
            validation_metrics = _metrics(model, dataset.validation)
            candidate_loss = validation_metrics.equal_game_loss
            improved = candidate_loss < best_loss - config.min_delta
            if improved:
                best_loss = candidate_loss
                best_epoch = epoch
                best_state = _clone_state(model)
                counter = 0
            else:
                counter += 1
            history.append(
                RankingEpochMetrics(epoch, train_metrics, validation_metrics, improved, counter)
            )
            if counter >= config.early_stopping_patience:
                break
        if best_state is None:
            raise RankingError("ranking training produced no finite checkpoint")
        model.load_state_dict(best_state, strict=True)
        model.eval()
        final_train = _metrics(model, dataset.train)
        final_validation = _metrics(model, dataset.validation)
    elapsed = time.perf_counter() - started
    return RankingTrainingResult(
        model=model,
        config=config,
        history=tuple(history),
        best_epoch=best_epoch,
        epochs_completed=len(history),
        stopped_early=len(history) < config.max_epochs,
        best_validation_loss=best_loss,
        train_metrics=final_train,
        validation_metrics=final_validation,
        positions_per_second=positions_seen / elapsed if elapsed else math.inf,
        wall_clock_seconds=elapsed,
    )

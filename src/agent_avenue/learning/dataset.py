"""Replay extraction and deterministic materialized datasets for neural training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from agent_avenue.encoding import (
    ENCODER_FINGERPRINT,
    ENCODER_VERSION,
    FEATURE_NAMES,
    FEATURE_WIDTH,
    encode_candidate,
)
from agent_avenue.engine import Phase, PlayerId, apply_action, new_game
from agent_avenue.observation import observe
from agent_avenue.storage import (
    GameRecord,
    game_record_fingerprint,
    rules_fingerprint,
    verify_game_record,
)

DATASET_SCHEMA_VERSION: Final[int] = 1
SPLIT_ALGORITHM: Final[str] = "sha256-group-threshold-v1"


class DatasetError(ValueError):
    """Raised for incompatible records or malformed dataset artifacts."""


@dataclass(frozen=True, slots=True)
class TrainingSample:
    """One safe chosen-candidate input plus separate offline audit metadata."""

    features: tuple[float, ...]
    target: float
    record_fingerprint: str
    decision_index: int
    actor: PlayerId
    winner: PlayerId
    phase: Phase
    behavior_policy_id: str


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    features: NDArray[np.float32]
    targets: NDArray[np.float32]
    game_index: NDArray[np.int64]
    phase: NDArray[np.str_]


@dataclass(frozen=True, slots=True)
class MaterializedDataset:
    train: DatasetSplit
    validation: DatasetSplit
    manifest: dict[str, object]

    @property
    def fingerprint(self) -> str:
        value = self.manifest.get("dataset_fingerprint")
        if not isinstance(value, str):
            raise DatasetError("dataset manifest has no fingerprint")
        return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _policy_id(record: GameRecord, actor: PlayerId) -> str:
    seat = record.seats[0 if actor is PlayerId.PLAYER_ONE else 1]
    config = json.dumps(dict(seat.config), sort_keys=True, separators=(",", ":"))
    return f"{seat.agent_id}:{hashlib.sha256(config.encode()).hexdigest()}"


def extract_game_samples(
    record: GameRecord, *, verify_code: bool = True
) -> tuple[TrainingSample, ...]:
    """Verify and replay a completed record into ordered safe Monte Carlo samples."""
    final_state = verify_game_record(record, verify_code=verify_code)
    if final_state.outcome is None:  # pragma: no cover - verification guarantees this
        raise DatasetError("record is not terminal")
    winner = final_state.outcome.winner
    fingerprint = game_record_fingerprint(record)
    state = new_game(record.replay.config, record.replay.seed)
    samples: list[TrainingSample] = []
    for index, action in enumerate(record.replay.actions):
        if state.phase is Phase.TERMINAL:
            raise DatasetError("record contains actions after terminal state")
        actor = state.active_player if state.phase is Phase.PLAY else state.active_player.other()
        observation = observe(state, actor)
        encoded = encode_candidate(observation, action)
        samples.append(
            TrainingSample(
                features=encoded.vector,
                target=1.0 if actor is winner else 0.0,
                record_fingerprint=fingerprint,
                decision_index=index,
                actor=actor,
                winner=winner,
                phase=state.phase,
                behavior_policy_id=_policy_id(record, actor),
            )
        )
        state = apply_action(state, action)
    if len(samples) != record.decision_count:
        raise DatasetError("extracted sample count does not match record metadata")
    return tuple(samples)


def _split_value(seed: int, group_id: str) -> int:
    payload = _canonical_json(
        {"algorithm": SPLIT_ALGORITHM, "group_id": group_id, "split_seed": seed}
    )
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def _group_id(record: GameRecord) -> str:
    return f"pair:{record.pair_id}" if record.pair_id else f"game:{game_record_fingerprint(record)}"


def _dataset_identity(manifest: dict[str, object]) -> str:
    payload = {
        key: value for key, value in manifest.items() if key not in {"dataset_fingerprint", "files"}
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def materialize_dataset(
    records: tuple[GameRecord, ...] | list[GameRecord],
    *,
    split_seed: int,
    validation_fraction: float = 0.1,
    verify_code: bool = True,
    source_corpus_fingerprint: str | None = None,
) -> MaterializedDataset:
    """Extract records and assign complete games/pairs to deterministic train/validation splits."""
    if type(split_seed) is not int:
        raise DatasetError("split_seed must be an integer")
    if not 0.0 < validation_fraction < 1.0:
        raise DatasetError("validation_fraction must be between zero and one")
    if len(records) < 2:
        raise DatasetError("at least two records are required for train/validation materialization")

    ordered = sorted(records, key=game_record_fingerprint)
    groups = sorted({_group_id(record) for record in ordered})
    if len(groups) < 2:
        raise DatasetError("at least two independent game/pair groups are required")
    modulus = 1 << 256
    threshold = int(validation_fraction * modulus)
    validation_groups = {group for group in groups if _split_value(split_seed, group) < threshold}
    # Small smoke corpora should still be usable while preserving deterministic group isolation.
    ranked = sorted(groups, key=lambda group: (_split_value(split_seed, group), group))
    if not validation_groups:
        validation_groups.add(ranked[0])
    if validation_groups == set(groups):
        validation_groups.remove(ranked[-1])

    split_samples: dict[str, list[tuple[int, TrainingSample]]] = {"train": [], "validation": []}
    split_games: dict[str, list[str]] = {"train": [], "validation": []}
    record_fingerprints: list[str] = []
    for game_index, record in enumerate(ordered):
        fingerprint = game_record_fingerprint(record)
        record_fingerprints.append(fingerprint)
        split = "validation" if _group_id(record) in validation_groups else "train"
        split_games[split].append(fingerprint)
        split_samples[split].extend(
            (game_index, sample) for sample in extract_game_samples(record, verify_code=verify_code)
        )

    def make_split(name: str) -> DatasetSplit:
        rows = split_samples[name]
        if not rows:
            raise DatasetError(f"{name} split contains no samples")
        return DatasetSplit(
            features=np.asarray([sample.features for _, sample in rows], dtype=np.float32),
            targets=np.asarray([sample.target for _, sample in rows], dtype=np.float32),
            game_index=np.asarray([index for index, _ in rows], dtype=np.int64),
            phase=np.asarray([sample.phase.value for _, sample in rows], dtype=np.str_),
        )

    train = make_split("train")
    validation = make_split("validation")
    manifest: dict[str, object] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "encoder_version": ENCODER_VERSION,
        "encoder_fingerprint": ENCODER_FINGERPRINT,
        "feature_width": FEATURE_WIDTH,
        "feature_names": list(FEATURE_NAMES),
        "rules_fingerprint": rules_fingerprint(),
        "split": {
            "algorithm": SPLIT_ALGORITHM,
            "seed": split_seed,
            "validation_fraction": validation_fraction,
            "train_game_fingerprints": split_games["train"],
            "validation_game_fingerprints": split_games["validation"],
        },
        "source_record_fingerprints": record_fingerprints,
        "source_corpus_fingerprint": source_corpus_fingerprint,
        "counts": {
            "records": len(ordered),
            "train_samples": int(train.targets.size),
            "validation_samples": int(validation.targets.size),
        },
    }
    manifest["dataset_fingerprint"] = _dataset_identity(manifest)
    return MaterializedDataset(train, validation, manifest)


def save_dataset(dataset: MaterializedDataset, path: Path) -> tuple[Path, Path]:
    """Write compressed arrays and a canonical adjacent JSON manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
    manifest_path = array_path.with_suffix(".json")
    np.savez_compressed(
        array_path,
        train_features=dataset.train.features,
        train_targets=dataset.train.targets,
        train_game_index=dataset.train.game_index,
        train_phase=dataset.train.phase,
        validation_features=dataset.validation.features,
        validation_targets=dataset.validation.targets,
        validation_game_index=dataset.validation.game_index,
        validation_phase=dataset.validation.phase,
    )
    dataset.manifest["files"] = {
        array_path.name: {
            "sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
            "size": array_path.stat().st_size,
        }
    }
    manifest_path.write_bytes(_canonical_json(dataset.manifest) + b"\n")
    return array_path, manifest_path


def load_dataset(path: Path) -> MaterializedDataset:
    """Load and validate a materialized dataset against the current encoder contract."""
    array_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
    manifest_path = array_path.with_suffix(".json")
    try:
        manifest = json.loads(manifest_path.read_text())
        arrays = np.load(array_path, allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise DatasetError("unable to load dataset artifact") from exc
    if not isinstance(manifest, dict) or manifest.get("dataset_fingerprint") != _dataset_identity(
        manifest
    ):
        raise DatasetError("dataset manifest fingerprint mismatch")
    files = manifest.get("files")
    expected_file = files.get(array_path.name) if isinstance(files, dict) else None
    if (
        not isinstance(expected_file, dict)
        or expected_file.get("sha256") != hashlib.sha256(array_path.read_bytes()).hexdigest()
        or expected_file.get("size") != array_path.stat().st_size
    ):
        raise DatasetError("dataset array digest mismatch")
    if (
        manifest.get("schema_version") != DATASET_SCHEMA_VERSION
        or manifest.get("encoder_version") != ENCODER_VERSION
        or manifest.get("encoder_fingerprint") != ENCODER_FINGERPRINT
        or manifest.get("feature_names") != list(FEATURE_NAMES)
        or manifest.get("rules_fingerprint") != rules_fingerprint()
    ):
        raise DatasetError("dataset is incompatible with the current encoder or rules")
    try:
        train = DatasetSplit(
            arrays["train_features"].astype(np.float32, copy=False),
            arrays["train_targets"].astype(np.float32, copy=False),
            arrays["train_game_index"].astype(np.int64, copy=False),
            arrays["train_phase"].astype(np.str_, copy=False),
        )
        validation = DatasetSplit(
            arrays["validation_features"].astype(np.float32, copy=False),
            arrays["validation_targets"].astype(np.float32, copy=False),
            arrays["validation_game_index"].astype(np.int64, copy=False),
            arrays["validation_phase"].astype(np.str_, copy=False),
        )
    except KeyError as exc:
        raise DatasetError("dataset array file is incomplete") from exc
    if train.features.shape[1:] != (FEATURE_WIDTH,) or validation.features.shape[1:] != (
        FEATURE_WIDTH,
    ):
        raise DatasetError("dataset feature width mismatch")
    counts = manifest.get("counts")
    if not isinstance(counts, dict) or (
        counts.get("train_samples") != int(train.targets.size)
        or counts.get("validation_samples") != int(validation.targets.size)
    ):
        raise DatasetError("dataset sample counts do not match manifest")
    return MaterializedDataset(train, validation, manifest)

"""Separate v2 dataset materialization aligned exactly to an immutable v1 dataset."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
from numpy.typing import NDArray

from agent_avenue.encoding.candidate_structured_v2 import (
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
    load_corpus,
    rules_fingerprint,
    verify_game_record,
)
from agent_avenue.storage.corpus import CorpusError

from .dataset import DATASET_SCHEMA_VERSION as V1_DATASET_SCHEMA_VERSION
from .dataset import DatasetError as V1DatasetError
from .dataset import MaterializedDataset as V1MaterializedDataset
from .dataset import load_dataset as load_v1_dataset

STRUCTURED_DATASET_SCHEMA_VERSION: Final[int] = 1
STRUCTURED_DATASET_KIND: Final[str] = "selected-action-terminal-mc-structured-v2"


class StructuredDatasetError(ValueError):
    """Raised for malformed, misaligned, or tampered structured v2 datasets."""


@dataclass(frozen=True, slots=True)
class StructuredTrainingSample:
    """A selected v2 vector plus audit provenance never used as a feature."""

    features: tuple[float, ...]
    target: float
    record_fingerprint: str
    decision_index: int
    actor: PlayerId
    winner: PlayerId
    phase: Phase
    behavior_policy_id: str


@dataclass(frozen=True, slots=True)
class StructuredDatasetSplit:
    features: NDArray[np.float32]
    targets: NDArray[np.float32]
    game_index: NDArray[np.int64]
    phase: NDArray[np.str_]
    record_fingerprint: NDArray[np.str_]
    decision_index: NDArray[np.int64]
    actor: NDArray[np.str_]
    winner: NDArray[np.str_]
    behavior_policy_id: NDArray[np.str_]


@dataclass(frozen=True, slots=True)
class StructuredMaterializedDataset:
    train: StructuredDatasetSplit
    validation: StructuredDatasetSplit
    manifest: dict[str, object]

    @property
    def fingerprint(self) -> str:
        value = self.manifest.get("dataset_fingerprint")
        if not isinstance(value, str):
            raise StructuredDatasetError("structured dataset manifest has no fingerprint")
        return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _valid_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _dataset_identity(manifest: Mapping[str, object]) -> str:
    payload = {
        key: value for key, value in manifest.items() if key not in {"dataset_fingerprint", "files"}
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _policy_id(record: GameRecord, actor: PlayerId) -> str:
    seat = record.seats[0 if actor is PlayerId.PLAYER_ONE else 1]
    config = json.dumps(dict(seat.config), sort_keys=True, separators=(",", ":"))
    return f"{seat.agent_id}:{hashlib.sha256(config.encode()).hexdigest()}"


def extract_structured_game_samples(
    record: GameRecord, *, verify_code: bool = True
) -> tuple[StructuredTrainingSample, ...]:
    """Replay one verified record into ordered selected-action terminal-MC v2 rows."""
    final_state = verify_game_record(record, verify_code=verify_code)
    if final_state.outcome is None:  # pragma: no cover - verifier promises terminal replay
        raise StructuredDatasetError("record is not terminal")
    winner = final_state.outcome.winner
    fingerprint = game_record_fingerprint(record)
    state = new_game(record.replay.config, record.replay.seed)
    samples: list[StructuredTrainingSample] = []
    for decision_index, action in enumerate(record.replay.actions):
        if state.phase is Phase.TERMINAL:
            raise StructuredDatasetError("record contains actions after terminal state")
        actor = state.active_player if state.phase is Phase.PLAY else state.active_player.other()
        observation = observe(state, actor)
        encoded = encode_candidate(observation, action)
        samples.append(
            StructuredTrainingSample(
                encoded.vector,
                1.0 if actor is winner else 0.0,
                fingerprint,
                decision_index,
                actor,
                winner,
                state.phase,
                _policy_id(record, actor),
            )
        )
        state = apply_action(state, action)
    if len(samples) != record.decision_count:
        raise StructuredDatasetError("extracted sample count does not match record metadata")
    return tuple(samples)


def load_verified_step2_corpus(
    directory: str | Path,
    *,
    expected_corpus_fingerprint: str,
    caller_verified: bool,
    verify_code: bool = False,
) -> tuple[object, tuple[GameRecord, ...]]:
    """Load historical records only after the caller has verified its frozen input identity.

    ``verify_code=False`` deliberately retains current rules and semantic replay validation while
    accepting the historical Step-2 source fingerprint.
    """
    if caller_verified is not True:
        raise StructuredDatasetError("historical Step-2 corpus requires caller_verified=True")
    if not _valid_digest(expected_corpus_fingerprint):
        raise StructuredDatasetError("expected_corpus_fingerprint must be a SHA-256 digest")
    try:
        manifest, records = load_corpus(
            Path(directory), verify_code=verify_code, verify_replays=True
        )
    except CorpusError as exc:
        raise StructuredDatasetError("historical corpus failed rules/replay verification") from exc
    actual = getattr(manifest, "corpus_fingerprint", None)
    if actual != expected_corpus_fingerprint:
        raise StructuredDatasetError("historical corpus fingerprint mismatch")
    return manifest, records


def _load_v1_source(
    value: str | Path | V1MaterializedDataset,
) -> tuple[V1MaterializedDataset, Path | None]:
    if isinstance(value, V1MaterializedDataset):
        return value, None
    path = Path(value)
    try:
        return load_v1_dataset(path), path if path.suffix == ".npz" else path.with_suffix(".npz")
    except V1DatasetError as exc:
        raise StructuredDatasetError("unable to load compatible v1 source dataset") from exc


def _as_fingerprint_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not _valid_digest(item) for item in value):
        raise StructuredDatasetError(f"v1 manifest {label} must be an ordered digest list")
    return tuple(cast(str, item) for item in value)


def _v1_alignment(
    source: V1MaterializedDataset,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    manifest = source.manifest
    if (
        manifest.get("schema_version") != V1_DATASET_SCHEMA_VERSION
        or manifest.get("encoder_version") != "candidate-public-v1"
        or manifest.get("feature_width") != 87
    ):
        raise StructuredDatasetError("source dataset is not a candidate-public-v1 artifact")
    source_records = _as_fingerprint_tuple(
        manifest.get("source_record_fingerprints"), "source records"
    )
    split = manifest.get("split")
    if not isinstance(split, Mapping):
        raise StructuredDatasetError("v1 dataset has no split declaration")
    train = _as_fingerprint_tuple(split.get("train_game_fingerprints"), "train groups")
    validation = _as_fingerprint_tuple(
        split.get("validation_game_fingerprints"), "validation groups"
    )
    if not source_records or not train or not validation or set(train).intersection(validation):
        raise StructuredDatasetError("v1 dataset split groups are malformed")
    if set(train).union(validation) != set(source_records):
        raise StructuredDatasetError("v1 split groups do not cover its source records")
    return source_records, train, validation


def _split_from_samples(
    rows: Sequence[tuple[int, StructuredTrainingSample]],
) -> StructuredDatasetSplit:
    if not rows:
        raise StructuredDatasetError("v1-aligned split has no samples")
    return StructuredDatasetSplit(
        features=np.asarray([sample.features for _, sample in rows], dtype=np.float32),
        targets=np.asarray([sample.target for _, sample in rows], dtype=np.float32),
        game_index=np.asarray([index for index, _ in rows], dtype=np.int64),
        phase=np.asarray([sample.phase.value for _, sample in rows], dtype=np.str_),
        record_fingerprint=np.asarray(
            [sample.record_fingerprint for _, sample in rows], dtype=np.str_
        ),
        decision_index=np.asarray([sample.decision_index for _, sample in rows], dtype=np.int64),
        actor=np.asarray([sample.actor.value for _, sample in rows], dtype=np.str_),
        winner=np.asarray([sample.winner.value for _, sample in rows], dtype=np.str_),
        behavior_policy_id=np.asarray(
            [sample.behavior_policy_id for _, sample in rows], dtype=np.str_
        ),
    )


def _validate_v1_row_identity(
    source: V1MaterializedDataset,
    train: StructuredDatasetSplit,
    validation: StructuredDatasetSplit,
) -> None:
    for label, structured, legacy in (
        ("train", train, source.train),
        ("validation", validation, source.validation),
    ):
        if (
            structured.targets.shape != legacy.targets.shape
            or not np.array_equal(structured.targets, legacy.targets)
            or not np.array_equal(structured.game_index, legacy.game_index)
            or not np.array_equal(structured.phase, legacy.phase)
        ):
            raise StructuredDatasetError(
                f"structured {label} rows do not match v1 labels/order/split"
            )


def materialize_structured_dataset(
    records: Sequence[GameRecord],
    *,
    v1_dataset: str | Path | V1MaterializedDataset,
    verify_code: bool = True,
    source_corpus_fingerprint: str | None = None,
    source_corpus_manifest_fingerprint: str | None = None,
) -> StructuredMaterializedDataset:
    """Re-encode exactly the v1 source row order, groups, phases, and terminal labels."""
    source, _ = _load_v1_source(v1_dataset)
    ordered_fingerprints, train_fingerprints, validation_fingerprints = _v1_alignment(source)
    by_fingerprint = {game_record_fingerprint(record): record for record in records}
    if len(by_fingerprint) != len(records) or set(by_fingerprint) != set(ordered_fingerprints):
        raise StructuredDatasetError(
            "supplied records do not exactly match v1 source record identities"
        )
    if source_corpus_fingerprint is not None and not _valid_digest(source_corpus_fingerprint):
        raise StructuredDatasetError("source_corpus_fingerprint must be a SHA-256 digest")
    if source_corpus_manifest_fingerprint is not None and not _valid_digest(
        source_corpus_manifest_fingerprint
    ):
        raise StructuredDatasetError("source_corpus_manifest_fingerprint must be a SHA-256 digest")
    train_rows: list[tuple[int, StructuredTrainingSample]] = []
    validation_rows: list[tuple[int, StructuredTrainingSample]] = []
    train_set = set(train_fingerprints)
    for game_index, fingerprint in enumerate(ordered_fingerprints):
        destination = train_rows if fingerprint in train_set else validation_rows
        destination.extend(
            (game_index, sample)
            for sample in extract_structured_game_samples(
                by_fingerprint[fingerprint], verify_code=verify_code
            )
        )
    train = _split_from_samples(train_rows)
    validation = _split_from_samples(validation_rows)
    _validate_v1_row_identity(source, train, validation)
    source_split = source.manifest.get("split")
    source_algorithm = source_split.get("algorithm") if isinstance(source_split, Mapping) else None
    manifest: dict[str, object] = {
        "schema_version": STRUCTURED_DATASET_SCHEMA_VERSION,
        "artifact_kind": STRUCTURED_DATASET_KIND,
        "encoder_version": ENCODER_VERSION,
        "encoder_fingerprint": ENCODER_FINGERPRINT,
        "feature_width": FEATURE_WIDTH,
        "feature_names": list(FEATURE_NAMES),
        "rules_fingerprint": rules_fingerprint(),
        "source_v1_dataset_fingerprint": source.fingerprint,
        "source_v1_encoder_version": "candidate-public-v1",
        "source_v1_encoder_fingerprint": source.manifest.get("encoder_fingerprint"),
        "source_corpus_fingerprint": source_corpus_fingerprint,
        "source_corpus_manifest_fingerprint": source_corpus_manifest_fingerprint,
        "split": {
            "source_algorithm": source_algorithm,
            "train_game_fingerprints": list(train_fingerprints),
            "validation_game_fingerprints": list(validation_fingerprints),
        },
        "source_record_fingerprints": list(ordered_fingerprints),
        "provenance": {
            "row_order": "v1-source-record-order+decision-order-v1",
            "behavior_policy_id": "audit-only-not-feature",
            "label": "selected-action-acting-player-terminal-win",
        },
        "counts": {
            "records": len(records),
            "train_samples": int(train.targets.size),
            "validation_samples": int(validation.targets.size),
        },
    }
    manifest["dataset_fingerprint"] = _dataset_identity(manifest)
    return StructuredMaterializedDataset(train, validation, manifest)


# Explicit long spelling helps callers distinguish the immutable-alignment API from v1's splitter.
materialize_structured_dataset_from_v1 = materialize_structured_dataset


def _array_payload(dataset: StructuredMaterializedDataset) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split_name, split in (("train", dataset.train), ("validation", dataset.validation)):
        result[f"{split_name}_features"] = split.features
        result[f"{split_name}_targets"] = split.targets
        result[f"{split_name}_game_index"] = split.game_index
        result[f"{split_name}_phase"] = split.phase
        result[f"{split_name}_record_fingerprint"] = split.record_fingerprint
        result[f"{split_name}_decision_index"] = split.decision_index
        result[f"{split_name}_actor"] = split.actor
        result[f"{split_name}_winner"] = split.winner
        result[f"{split_name}_behavior_policy_id"] = split.behavior_policy_id
    return result


def save_structured_dataset(
    dataset: StructuredMaterializedDataset, path: str | Path
) -> tuple[Path, Path]:
    """Atomically save a separate immutable v2 artifact with audit-only provenance arrays."""
    array_path = Path(path)
    array_path = array_path if array_path.suffix == ".npz" else array_path.with_suffix(".npz")
    manifest_path = array_path.with_suffix(".json")
    array_path.parent.mkdir(parents=True, exist_ok=True)
    if array_path.exists() != manifest_path.exists():
        array_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
    if array_path.exists():
        existing = load_structured_dataset(array_path)
        if existing.fingerprint == dataset.fingerprint:
            return array_path, manifest_path
        raise StructuredDatasetError("structured dataset destination has a different artifact")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{array_path.name}.tmp-", dir=array_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            np.savez_compressed(destination, **_array_payload(dataset))
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, array_path)
        dataset.manifest["files"] = {
            array_path.name: {
                "sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
                "size": array_path.stat().st_size,
            }
        }
        manifest_path.write_bytes(_canonical_json(dataset.manifest) + b"\n")
        return array_path, manifest_path
    except Exception:
        temporary.unlink(missing_ok=True)
        if not manifest_path.exists():
            array_path.unlink(missing_ok=True)
        raise


def _read_split(arrays: Mapping[str, Any], name: str) -> StructuredDatasetSplit:
    try:
        return StructuredDatasetSplit(
            np.asarray(arrays[f"{name}_features"], dtype=np.float32),
            np.asarray(arrays[f"{name}_targets"], dtype=np.float32),
            np.asarray(arrays[f"{name}_game_index"], dtype=np.int64),
            np.asarray(arrays[f"{name}_phase"], dtype=np.str_),
            np.asarray(arrays[f"{name}_record_fingerprint"], dtype=np.str_),
            np.asarray(arrays[f"{name}_decision_index"], dtype=np.int64),
            np.asarray(arrays[f"{name}_actor"], dtype=np.str_),
            np.asarray(arrays[f"{name}_winner"], dtype=np.str_),
            np.asarray(arrays[f"{name}_behavior_policy_id"], dtype=np.str_),
        )
    except KeyError as exc:
        raise StructuredDatasetError("structured dataset array file is incomplete") from exc


def load_structured_dataset(path: str | Path) -> StructuredMaterializedDataset:
    """Load and validate only a v2 dataset; v1 artifacts are rejected rather than migrated."""
    array_path = Path(path)
    array_path = array_path if array_path.suffix == ".npz" else array_path.with_suffix(".npz")
    manifest_path = array_path.with_suffix(".json")
    try:
        manifest = json.loads(manifest_path.read_text())
        arrays = np.load(array_path, allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise StructuredDatasetError("unable to load structured dataset artifact") from exc
    if not isinstance(manifest, dict) or manifest.get("dataset_fingerprint") != _dataset_identity(
        manifest
    ):
        raise StructuredDatasetError("structured dataset manifest fingerprint mismatch")
    if (
        manifest.get("schema_version") != STRUCTURED_DATASET_SCHEMA_VERSION
        or manifest.get("artifact_kind") != STRUCTURED_DATASET_KIND
        or manifest.get("encoder_version") != ENCODER_VERSION
        or manifest.get("encoder_fingerprint") != ENCODER_FINGERPRINT
        or manifest.get("feature_width") != FEATURE_WIDTH
        or manifest.get("feature_names") != list(FEATURE_NAMES)
        or manifest.get("rules_fingerprint") != rules_fingerprint()
    ):
        raise StructuredDatasetError(
            "dataset is not compatible with candidate-public-structured-v2"
        )
    files = manifest.get("files")
    expected = files.get(array_path.name) if isinstance(files, Mapping) else None
    if (
        not isinstance(expected, Mapping)
        or expected.get("sha256") != hashlib.sha256(array_path.read_bytes()).hexdigest()
        or expected.get("size") != array_path.stat().st_size
    ):
        raise StructuredDatasetError("structured dataset array digest mismatch")
    array_mapping = cast(Mapping[str, Any], arrays)
    train = _read_split(array_mapping, "train")
    validation = _read_split(array_mapping, "validation")
    for split in (train, validation):
        if (
            split.features.shape != (split.targets.size, FEATURE_WIDTH)
            or split.game_index.shape != split.targets.shape
            or split.phase.shape != split.targets.shape
            or split.record_fingerprint.shape != split.targets.shape
            or split.decision_index.shape != split.targets.shape
            or split.actor.shape != split.targets.shape
            or split.winner.shape != split.targets.shape
            or split.behavior_policy_id.shape != split.targets.shape
        ):
            raise StructuredDatasetError("structured dataset arrays have inconsistent row counts")
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping) or (
        counts.get("train_samples") != int(train.targets.size)
        or counts.get("validation_samples") != int(validation.targets.size)
    ):
        raise StructuredDatasetError("structured dataset sample counts do not match manifest")
    return StructuredMaterializedDataset(train, validation, manifest)


__all__ = [
    "STRUCTURED_DATASET_KIND",
    "STRUCTURED_DATASET_SCHEMA_VERSION",
    "StructuredDatasetError",
    "StructuredDatasetSplit",
    "StructuredMaterializedDataset",
    "StructuredTrainingSample",
    "extract_structured_game_samples",
    "load_structured_dataset",
    "load_verified_step2_corpus",
    "materialize_structured_dataset",
    "materialize_structured_dataset_from_v1",
    "save_structured_dataset",
]

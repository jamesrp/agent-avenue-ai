"""Versioned NPZ storage for safe Step-4 all-action rollout targets.

This optional NumPy module stores safe model vectors and compact rollout diagnostics only.  It does
not serialize latent card zones, source replay metadata, or production game states.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
from numpy.typing import NDArray

from agent_avenue.encoding.candidate_structured_v2 import ENCODER_FINGERPRINT, ENCODER_VERSION

from .identity import PANEL_STRATA
from .targets import (
    CONTINUATION_SLOTS,
    RNG_CONTRACT_VERSION,
    ROLLOUT_TARGET_VERSION,
    STEP4_ROOT_SEED,
    WORLD_COUNT,
    RolloutTargetSet,
)

ROLLOUT_TARGET_ARTIFACT_SCHEMA_VERSION: Final[int] = 1
ROLLOUT_TARGET_ARTIFACT_KIND: Final[str] = "counterfactual-rollout-targets-npz-v1"


class RolloutArtifactError(ValueError):
    """Raised when a target NPZ/manifest is malformed, incompatible, or tampered."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _copy_json_object(value: Mapping[str, object], *, name: str) -> dict[str, object]:
    copied = json.loads(_canonical_json(dict(value)))
    if not isinstance(copied, dict):  # pragma: no cover - Mapping guarantees object shape
        raise RolloutArtifactError(f"{name} must be a JSON object")
    return cast(dict[str, object], copied)


def _identity(manifest: Mapping[str, object]) -> str:
    payload = {
        key: value
        for key, value in manifest.items()
        if key not in {"artifact_fingerprint", "files"}
    }
    return _digest(payload)


@dataclass(frozen=True, slots=True)
class LoadedRolloutTargets:
    """Validated safe arrays and metadata loaded from a rollout target artifact."""

    features: NDArray[np.float32]
    targets: NDArray[np.float32]
    position_index: NDArray[np.int64]
    safe_identity: NDArray[np.str_]
    stratum: NDArray[np.str_]
    action_json: NDArray[np.str_]
    sample_json: NDArray[np.str_]
    manifest: dict[str, object]

    @property
    def fingerprint(self) -> str:
        value = self.manifest.get("artifact_fingerprint")
        if not isinstance(value, str):
            raise RolloutArtifactError("target manifest lacks an artifact fingerprint")
        return value


def _array_payload(targets: RolloutTargetSet) -> dict[str, NDArray[Any]]:
    position_by_id = {
        position.safe_identity: index for index, position in enumerate(targets.positions)
    }
    return {
        "features": np.asarray([row.features for row in targets.rows], dtype=np.float32),
        "targets": np.asarray([row.target for row in targets.rows], dtype=np.float32),
        "position_index": np.asarray(
            [position_by_id[row.safe_identity] for row in targets.rows], dtype=np.int64
        ),
        "safe_identity": np.asarray([row.safe_identity for row in targets.rows], dtype=np.str_),
        "stratum": np.asarray([row.stratum for row in targets.rows], dtype=np.str_),
        "action_json": np.asarray(
            [_canonical_json(row.to_data()["action"]).decode("utf-8") for row in targets.rows],
            dtype=np.str_,
        ),
        "sample_json": np.asarray(
            [_canonical_json(row.to_data()["samples"]).decode("utf-8") for row in targets.rows],
            dtype=np.str_,
        ),
    }


def _row_digest(targets: RolloutTargetSet) -> str:
    return _digest(
        [
            {
                "safe_identity": row.safe_identity,
                "stratum": row.stratum,
                "action": row.to_data()["action"],
                "target": row.target,
                "feature_digest": _digest(list(row.features)),
                "samples": row.to_data()["samples"],
            }
            for row in targets.rows
        ]
    )


def _manifest(
    targets: RolloutTargetSet,
    *,
    input_identities: Mapping[str, object],
    arrays: Mapping[str, NDArray[Any]],
) -> dict[str, object]:
    positions = [
        {
            "replicate_id": position.replicate_id,
            "stratum": position.stratum,
            "safe_identity": position.safe_identity,
            "selection_hash": position.selection_hash,
        }
        for position in targets.positions
    ]
    manifest: dict[str, object] = {
        "schema_version": ROLLOUT_TARGET_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": ROLLOUT_TARGET_ARTIFACT_KIND,
        "target_version": ROLLOUT_TARGET_VERSION,
        "encoder": {
            "version": ENCODER_VERSION,
            "fingerprint": ENCODER_FINGERPRINT,
            "width": 519,
        },
        "rng_contract": {
            "version": RNG_CONTRACT_VERSION,
            "root_seed": targets.root_seed,
            "world_count": WORLD_COUNT,
            "expanded_deck": "EXPANDED_CANONICAL_DECK",
            "sampling": "sha256-counter-rejection-v1+reverse-fisher-yates-v1",
        },
        "continuation_population": [
            {"slot": slot, "policy_id": policy_id} for slot, policy_id in CONTINUATION_SLOTS
        ],
        "panel": positions,
        "input_identities": _copy_json_object(input_identities, name="input_identities"),
        "counts": {
            "positions": len(targets.positions),
            "rows": len(targets.rows),
            "world_samples": len(targets.rows) * WORLD_COUNT,
        },
        "target_digest": targets.digest,
        "row_digest": _row_digest(targets),
        "diagnostics": {
            "terminal_fraction": targets.terminal_fraction,
            "depth_leaf_fraction": targets.depth_leaf_fraction,
            "mechanical_cap_errors": 0,
        },
        "array_layout": {key: list(value.shape) for key, value in arrays.items()},
    }
    manifest["artifact_fingerprint"] = _identity(manifest)
    return manifest


def save_rollout_targets(
    targets: RolloutTargetSet,
    path: str | Path,
    *,
    input_identities: Mapping[str, object],
) -> tuple[Path, Path]:
    """Atomically write ``targets.npz`` plus a tamper-evident JSON manifest."""
    array_path = Path(path)
    array_path = array_path if array_path.suffix == ".npz" else array_path.with_suffix(".npz")
    manifest_path = array_path.with_suffix(".json")
    array_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _array_payload(targets)
    manifest = _manifest(targets, input_identities=input_identities, arrays=arrays)
    if array_path.exists() or manifest_path.exists():
        if array_path.exists() and manifest_path.exists():
            existing = load_rollout_targets(array_path)
            if existing.fingerprint == manifest["artifact_fingerprint"]:
                return array_path, manifest_path
        raise RolloutArtifactError("target destination already contains a different artifact")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{array_path.name}.tmp-", dir=array_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            np.savez_compressed(destination, **arrays)  # type: ignore[arg-type]
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, array_path)
        manifest["files"] = {
            array_path.name: {
                "sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
                "size": array_path.stat().st_size,
            }
        }
        manifest_path.write_bytes(_canonical_json(manifest) + b"\n")
        return array_path, manifest_path
    except Exception:
        temporary.unlink(missing_ok=True)
        if not manifest_path.exists():
            array_path.unlink(missing_ok=True)
        raise


def _read_arrays(arrays: Mapping[str, Any]) -> tuple[NDArray[Any], ...]:
    expected = (
        "features",
        "targets",
        "position_index",
        "safe_identity",
        "stratum",
        "action_json",
        "sample_json",
    )
    if set(arrays) != set(expected):
        raise RolloutArtifactError("target NPZ arrays do not match the fixed schema")
    return tuple(np.asarray(arrays[name]) for name in expected)


def _validate_samples(values: NDArray[np.str_]) -> None:
    for value in values.tolist():
        try:
            samples = json.loads(str(value))
        except json.JSONDecodeError as exc:
            raise RolloutArtifactError("target sample metadata is not JSON") from exc
        if not isinstance(samples, list) or len(samples) != WORLD_COUNT:
            raise RolloutArtifactError("every target row must retain exactly ten world samples")
        for sample in samples:
            if not isinstance(sample, dict) or set(sample) != {
                "world_index",
                "world_seed",
                "latent_digest",
                "player_one_policy_id",
                "player_two_policy_id",
                "continuation_rng_seeds",
                "terminal",
                "depth_leaf",
                "actions_applied",
                "root_value",
            }:
                raise RolloutArtifactError("target sample metadata fields are malformed")
            if sample["terminal"] == sample["depth_leaf"]:
                raise RolloutArtifactError("target sample terminal/leaf flags are inconsistent")


def load_rollout_targets(path: str | Path) -> LoadedRolloutTargets:
    """Load only the exact current safe target schema and verify its digests."""
    array_path = Path(path)
    array_path = array_path if array_path.suffix == ".npz" else array_path.with_suffix(".npz")
    manifest_path = array_path.with_suffix(".json")
    try:
        raw_manifest = json.loads(manifest_path.read_text())
        archive = np.load(array_path, allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RolloutArtifactError("unable to load rollout target artifact") from exc
    if not isinstance(raw_manifest, dict):
        raise RolloutArtifactError("target manifest must be a JSON object")
    manifest = cast(dict[str, object], raw_manifest)
    if manifest.get("artifact_fingerprint") != _identity(manifest):
        raise RolloutArtifactError("target manifest fingerprint mismatch")
    if (
        manifest.get("schema_version") != ROLLOUT_TARGET_ARTIFACT_SCHEMA_VERSION
        or manifest.get("artifact_kind") != ROLLOUT_TARGET_ARTIFACT_KIND
        or manifest.get("target_version") != ROLLOUT_TARGET_VERSION
    ):
        raise RolloutArtifactError("target artifact schema is incompatible")
    encoder = manifest.get("encoder")
    if not isinstance(encoder, Mapping) or encoder != {
        "version": ENCODER_VERSION,
        "fingerprint": ENCODER_FINGERPRINT,
        "width": 519,
    }:
        raise RolloutArtifactError("target artifact does not use the frozen structured encoder")
    rng = manifest.get("rng_contract")
    if not isinstance(rng, Mapping) or rng.get("version") != RNG_CONTRACT_VERSION:
        raise RolloutArtifactError("target artifact RNG contract is incompatible")
    if rng.get("root_seed") != STEP4_ROOT_SEED:
        raise RolloutArtifactError("target artifact root seed is not the fixed Step-4 seed")
    if manifest.get("continuation_population") != [
        {"slot": slot, "policy_id": policy_id} for slot, policy_id in CONTINUATION_SLOTS
    ]:
        raise RolloutArtifactError("target artifact policy population is incompatible")
    files = manifest.get("files")
    expected_file = files.get(array_path.name) if isinstance(files, Mapping) else None
    if (
        not isinstance(expected_file, Mapping)
        or expected_file.get("sha256") != hashlib.sha256(array_path.read_bytes()).hexdigest()
        or expected_file.get("size") != array_path.stat().st_size
    ):
        raise RolloutArtifactError("target NPZ digest mismatch")
    raw_arrays = cast(Mapping[str, Any], archive)
    raw = _read_arrays(raw_arrays)
    features = np.asarray(raw[0], dtype=np.float32)
    targets = np.asarray(raw[1], dtype=np.float32)
    position_index = np.asarray(raw[2], dtype=np.int64)
    safe_identity = np.asarray(raw[3], dtype=np.str_)
    stratum = np.asarray(raw[4], dtype=np.str_)
    action_json = np.asarray(raw[5], dtype=np.str_)
    sample_json = np.asarray(raw[6], dtype=np.str_)
    count = targets.size
    if (
        features.shape != (count, 519)
        or position_index.shape != (count,)
        or safe_identity.shape != (count,)
        or stratum.shape != (count,)
        or action_json.shape != (count,)
        or sample_json.shape != (count,)
        or count == 0
    ):
        raise RolloutArtifactError("target NPZ row arrays have inconsistent dimensions")
    if (
        not np.isfinite(features).all()
        or not np.isfinite(targets).all()
        or not np.all((targets >= 0.0) & (targets <= 1.0))
    ):
        raise RolloutArtifactError("target vectors or soft labels are outside finite [0,1] bounds")
    if any(value not in PANEL_STRATA for value in stratum.tolist()):
        raise RolloutArtifactError("target rows contain an unknown fixed panel stratum")
    panel = manifest.get("panel")
    if not isinstance(panel, list) or not panel:
        raise RolloutArtifactError("target manifest panel is malformed")
    if np.any(position_index < 0) or np.any(position_index >= len(panel)):
        raise RolloutArtifactError("target position indices do not refer to the manifest panel")
    _validate_samples(sample_json)
    layout = manifest.get("array_layout")
    actual_layout = {
        name: list(value.shape)
        for name, value in zip(
            (
                "features",
                "targets",
                "position_index",
                "safe_identity",
                "stratum",
                "action_json",
                "sample_json",
            ),
            raw,
            strict=True,
        )
    }
    if layout != actual_layout:
        raise RolloutArtifactError("target manifest array layout does not match NPZ")
    return LoadedRolloutTargets(
        features,
        targets,
        position_index,
        safe_identity,
        stratum,
        action_json,
        sample_json,
        manifest,
    )


__all__ = [
    "ROLLOUT_TARGET_ARTIFACT_KIND",
    "ROLLOUT_TARGET_ARTIFACT_SCHEMA_VERSION",
    "LoadedRolloutTargets",
    "RolloutArtifactError",
    "load_rollout_targets",
    "save_rollout_targets",
]

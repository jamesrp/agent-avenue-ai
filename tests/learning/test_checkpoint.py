import hashlib
import json
from pathlib import Path

import pytest
import torch

from agent_avenue.learning.checkpoint import (
    CheckpointCompatibility,
    CheckpointCompatibilityError,
    CheckpointError,
    CheckpointIntegrityError,
    inspect_checkpoint,
    load_checkpoint,
    save_checkpoint,
    tensor_digest,
)
from agent_avenue.learning.model import CandidateMLP
from agent_avenue.storage import rules_fingerprint

FEATURE_NAMES = tuple(f"feature_{index}" for index in range(87))
Q0_FINGERPRINT = "bc6f070aa30e2155ed6953a065c9f67e10357b54f222e3e944273493ab4e6e8e"


def _set_unversioned_fingerprint(path: Path, *, legacy: bool) -> str:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("fingerprint_algorithm")
    excluded = {
        "checkpoint_fingerprint",
        "created_at",
        "environment",
        "output_path",
        "elapsed_seconds",
    }
    if not legacy:
        excluded.add("files")
    payload = {key: value for key, value in manifest.items() if key not in excluded}
    if not legacy:
        payload["metrics_summary"] = {
            key: value for key, value in payload["metrics_summary"].items() if key != "runtime"
        }
    fingerprint = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    manifest["checkpoint_fingerprint"] = fingerprint
    manifest_path.write_text(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    return fingerprint


@pytest.fixture
def archived_q0_path() -> Path:
    path = Path(__file__).resolve().parents[2] / "checkpoints" / "q0"
    required = {"manifest.json", "metrics.json", "weights.pt"}
    if not path.is_dir() or {entry.name for entry in path.iterdir()} != required:
        pytest.skip("ignored archived q0 checkpoint is not available")
    return path


def _save(path: Path) -> tuple[CandidateMLP, str]:
    model = CandidateMLP(seed=42)
    saved = save_checkpoint(
        path,
        model,
        metrics={"validation": {"equal_game_loss": 0.25}},
        encoder_version="candidate-public-v1",
        encoder_fingerprint="e" * 64,
        feature_names=FEATURE_NAMES,
        training_config={"batch_size": 8},
        training_seeds={"model": 42, "shuffle": 9},
        dataset_fingerprint="d" * 64,
        created_at="2026-08-29T00:00:00+00:00",
    )
    return model, saved.checkpoint_fingerprint


def _compatibility(**changes: object) -> CheckpointCompatibility:
    values = {
        "encoder_version": "candidate-public-v1",
        "encoder_fingerprint": "e" * 64,
        "feature_names": FEATURE_NAMES,
        "rules_fingerprint": rules_fingerprint(),
    }
    values.update(changes)
    return CheckpointCompatibility(**values)  # type: ignore[arg-type]


def test_checkpoint_round_trip_is_safe_inference_only_and_immutable(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint"
    original, identity = _save(path)
    inspected = inspect_checkpoint(path, compatibility=_compatibility())
    loaded = load_checkpoint(path, compatibility=_compatibility())

    assert inspected.checkpoint_fingerprint == loaded.checkpoint_fingerprint == identity
    assert inspected.manifest["fingerprint_algorithm"] == "sha256-canonical-json-v2"
    assert inspected.tensor_digest == tensor_digest(original.state_dict())
    assert loaded.metrics == {"validation": {"equal_game_loss": 0.25}}
    assert not loaded.model.training
    assert all(not parameter.requires_grad for parameter in loaded.model.parameters())
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            original.state_dict().values(), loaded.model.state_dict().values(), strict=True
        )
    )
    with pytest.raises(CheckpointError, match="already exists"):
        _save(path)


def test_unversioned_schema_v1_fingerprint_algorithms_remain_compatible(
    tmp_path: Path,
) -> None:
    current_path = tmp_path / "current"
    legacy_path = tmp_path / "legacy"
    _save(current_path)
    _save(legacy_path)

    current_identity = _set_unversioned_fingerprint(current_path, legacy=False)
    legacy_identity = _set_unversioned_fingerprint(legacy_path, legacy=True)

    assert (
        inspect_checkpoint(current_path, compatibility=_compatibility()).checkpoint_fingerprint
        == current_identity
    )
    assert (
        inspect_checkpoint(legacy_path, compatibility=_compatibility()).checkpoint_fingerprint
        == legacy_identity
    )
    assert current_identity != legacy_identity

    legacy_manifest_path = legacy_path / "manifest.json"
    legacy_manifest = json.loads(legacy_manifest_path.read_text())
    legacy_manifest["metadata"]["tampered"] = True
    legacy_manifest_path.write_text(json.dumps(legacy_manifest))
    with pytest.raises(CheckpointIntegrityError, match="checkpoint fingerprint"):
        inspect_checkpoint(legacy_path, compatibility=_compatibility())


def test_archived_q0_original_fingerprint_validates_unchanged(
    archived_q0_path: Path,
) -> None:
    manifest = json.loads((archived_q0_path / "manifest.json").read_text())
    encoder = manifest["encoder"]
    compatibility = manifest["compatibility"]
    inspected = inspect_checkpoint(
        archived_q0_path,
        compatibility=CheckpointCompatibility(
            encoder_version=encoder["version"],
            encoder_fingerprint=encoder["fingerprint"],
            feature_names=tuple(encoder["feature_names"]),
            rules_fingerprint=compatibility["rules_fingerprint"],
            code_fingerprint=compatibility["code_fingerprint"],
        ),
    )

    assert inspected.checkpoint_fingerprint == Q0_FINGERPRINT
    assert manifest["checkpoint_fingerprint"] == Q0_FINGERPRINT
    assert "fingerprint_algorithm" not in manifest


def test_checkpoint_identity_excludes_runtime_diagnostics(tmp_path: Path) -> None:
    model = CandidateMLP(seed=7)
    common = {
        "encoder_version": "candidate-public-v1",
        "encoder_fingerprint": "e" * 64,
        "feature_names": FEATURE_NAMES,
        "training_config": {"batch_size": 8},
        "training_seeds": {"model": 7},
        "dataset_fingerprint": "d" * 64,
    }
    first = save_checkpoint(
        tmp_path / "first",
        model,
        metrics={"validation": {"loss": 0.25}, "runtime": {"wall_clock_seconds": 1.0}},
        **common,
    )
    second = save_checkpoint(
        tmp_path / "second",
        model,
        metrics={"validation": {"loss": 0.25}, "runtime": {"wall_clock_seconds": 9.0}},
        **common,
    )
    assert first.checkpoint_fingerprint == second.checkpoint_fingerprint


def test_checkpoint_rejects_compatibility_and_digest_tampering(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint"
    _save(path)
    with pytest.raises(CheckpointCompatibilityError, match="encoder fingerprint"):
        load_checkpoint(path, compatibility=_compatibility(encoder_fingerprint="f" * 64))

    metrics_path = path / "metrics.json"
    metrics_path.write_text('{"validation":{"equal_game_loss":0.5}}\n')
    with pytest.raises(CheckpointIntegrityError, match=r"metrics\.json digest"):
        load_checkpoint(path, compatibility=_compatibility())


def test_checkpoint_lineage_requires_parent_and_positive_generation(tmp_path: Path) -> None:
    model = CandidateMLP(seed=1)
    with pytest.raises(CheckpointError, match="provided together"):
        save_checkpoint(
            tmp_path / "missing-generation", model, metrics={}, parent_checkpoint="a" * 64
        )
    with pytest.raises(CheckpointError, match="valid parent"):
        save_checkpoint(
            tmp_path / "invalid-generation",
            model,
            metrics={},
            parent_checkpoint="not-a-digest",
            generation=1,
        )


def test_checkpoint_uses_weights_only_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "checkpoint"
    _save(path)
    real_load = torch.load
    calls: list[object] = []

    def recording_load(*args: object, **kwargs: object) -> object:
        calls.append(kwargs.get("weights_only"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    load_checkpoint(path, compatibility=_compatibility())
    assert calls == [True]


def test_manifest_rejects_unknown_fingerprint_algorithm(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint"
    _save(path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["fingerprint_algorithm"] = "unknown"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(CheckpointCompatibilityError, match="fingerprint algorithm"):
        load_checkpoint(path, compatibility=_compatibility())


def test_manifest_self_hash_rejects_metadata_edits(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint"
    _save(path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model"]["version"] = "other-model"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(CheckpointIntegrityError, match="checkpoint fingerprint"):
        load_checkpoint(path, compatibility=_compatibility())

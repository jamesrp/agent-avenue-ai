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


def test_manifest_self_hash_rejects_metadata_edits(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint"
    _save(path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model"]["version"] = "other-model"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(CheckpointIntegrityError, match="checkpoint fingerprint"):
        load_checkpoint(path, compatibility=_compatibility())

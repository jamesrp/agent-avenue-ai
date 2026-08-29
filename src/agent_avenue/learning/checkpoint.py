"""Immutable, self-verifying inference checkpoint bundles.

A bundle consists of exactly ``manifest.json``, ``weights.pt``, and ``metrics.json``.  The loader
checks schema and compatibility metadata plus all file and canonical tensor digests before returning
an inference-only CPU model.  PyTorch deserialization always uses ``weights_only=True``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast

import torch
from torch import Tensor

from agent_avenue.storage.fingerprints import (
    code_fingerprint as current_code_fingerprint,
)
from agent_avenue.storage.fingerprints import (
    rules_fingerprint as current_rules_fingerprint,
)

from .model import (
    INPUT_WIDTH,
    MODEL_VERSION,
    CandidateMLP,
    ModelError,
    create_model,
    model_spec,
    validate_state_dict,
)

CHECKPOINT_SCHEMA_VERSION: Final = 1
CHECKPOINT_ARTIFACT_KIND: Final = "immutable-inference"
MANIFEST_FILENAME: Final = "manifest.json"
WEIGHTS_FILENAME: Final = "weights.pt"
METRICS_FILENAME: Final = "metrics.json"
_BUNDLE_FILES: Final = frozenset({MANIFEST_FILENAME, WEIGHTS_FILENAME, METRICS_FILENAME})


class CheckpointError(ValueError):
    """Raised when a checkpoint cannot be safely created or validated."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised when a valid checkpoint targets incompatible software or schemas."""


class CheckpointIntegrityError(CheckpointError):
    """Raised when checkpoint contents do not match their declared digests."""


@dataclass(frozen=True, slots=True)
class CheckpointCompatibility:
    """Expected inference-time compatibility values.

    ``feature_names`` is optional for callers that only retain the encoder fingerprint.  When it is
    supplied, exact name and order equality is required in addition to the fingerprint.
    """

    encoder_version: str
    encoder_fingerprint: str
    feature_names: tuple[str, ...] | None = None
    input_width: int = INPUT_WIDTH
    model_version: str = MODEL_VERSION
    rules_fingerprint: str | None = None
    code_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not self.encoder_version or not _valid_digest(self.encoder_fingerprint):
            raise CheckpointCompatibilityError(
                "encoder version and lowercase SHA-256 fingerprint are required"
            )
        if self.input_width != INPUT_WIDTH:
            raise CheckpointCompatibilityError(
                f"candidate-mlp-v1 requires encoder width {INPUT_WIDTH}"
            )
        if self.model_version != MODEL_VERSION:
            raise CheckpointCompatibilityError(f"unsupported model version {self.model_version!r}")
        if self.feature_names is not None and (
            len(self.feature_names) != self.input_width
            or len(set(self.feature_names)) != self.input_width
        ):
            raise CheckpointCompatibilityError("feature_names must be unique and match input_width")
        for label, digest in (
            ("rules", self.rules_fingerprint),
            ("code", self.code_fingerprint),
        ):
            if digest is not None and not _valid_digest(digest):
                raise CheckpointCompatibilityError(
                    f"{label} fingerprint must be a lowercase SHA-256 digest"
                )


@dataclass(frozen=True, slots=True)
class SavedCheckpoint:
    path: Path
    checkpoint_fingerprint: str
    manifest: Mapping[str, object]

    @property
    def checkpoint_id(self) -> str:
        return self.checkpoint_fingerprint


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    path: Path
    model: CandidateMLP
    checkpoint_fingerprint: str
    manifest: Mapping[str, object]
    metrics: object

    @property
    def checkpoint_id(self) -> str:
        return self.checkpoint_fingerprint


@dataclass(frozen=True, slots=True)
class CheckpointInspection:
    path: Path
    checkpoint_fingerprint: str
    manifest: Mapping[str, object]
    metrics: object
    tensor_digest: str

    @property
    def checkpoint_id(self) -> str:
        return self.checkpoint_fingerprint


def current_compatibility() -> CheckpointCompatibility:
    """Return the compatibility contract exported by the installed encoder and rules."""
    try:
        from agent_avenue.encoding import ENCODER_FINGERPRINT, ENCODER_VERSION, FEATURE_NAMES
    except ImportError as exc:  # pragma: no cover - adjacent milestone is normally installed
        raise CheckpointCompatibilityError("current encoder contract is unavailable") from exc
    return CheckpointCompatibility(
        encoder_version=ENCODER_VERSION,
        encoder_fingerprint=ENCODER_FINGERPRINT,
        feature_names=FEATURE_NAMES,
        rules_fingerprint=current_rules_fingerprint(),
    )


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise CheckpointError("checkpoint metadata must be finite canonical JSON data") from exc


def _json_copy(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    return json.loads(_canonical_json(value))


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _tensor_bytes(tensor: Tensor) -> bytes:
    if tensor.device.type != "cpu" or tensor.layout != torch.strided:
        raise CheckpointError("canonical tensor digests require dense CPU tensors")
    detached = tensor.detach().contiguous()
    return detached.view(torch.uint8).numpy().tobytes(order="C")


def tensor_digest(state_dict: Mapping[str, object]) -> str:
    """Hash tensor names, dtypes, shapes, and values independent of ``.pt`` serialization."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name]
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise CheckpointError("state dictionaries must map string names to tensors")
        metadata = {
            "name": name,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "layout": str(value.layout),
        }
        encoded = _canonical_json(metadata)
        raw = _tensor_bytes(value)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


# Descriptive alias used in reports and tests.
model_state_digest = tensor_digest


def _encoder_data(
    encoder: object | None,
    *,
    encoder_version: str | None,
    encoder_fingerprint: str | None,
    feature_names: Sequence[str] | None,
) -> dict[str, object]:
    def field_from(source: object, names: tuple[str, ...]) -> object | None:
        if isinstance(source, Mapping):
            for name in names:
                if name in source:
                    result: object = source[name]
                    return result
            return None
        for name in names:
            if hasattr(source, name):
                value = cast(object, getattr(source, name))
                return value() if callable(value) and name.endswith("fingerprint") else value
        return None

    if encoder is None and encoder_version is None and encoder_fingerprint is None:
        try:
            from agent_avenue.encoding import FEATURE_SCHEMA
        except ImportError as exc:  # pragma: no cover - adjacent milestone is normally installed
            raise CheckpointError("encoder metadata is required") from exc
        encoder = FEATURE_SCHEMA
    if encoder is not None:
        encoder_version = encoder_version or field_from(
            encoder, ("version", "encoder_version", "ENCODER_VERSION")
        )  # type: ignore[assignment]
        encoder_fingerprint = encoder_fingerprint or field_from(
            encoder, ("fingerprint", "encoder_fingerprint", "ENCODER_FINGERPRINT")
        )  # type: ignore[assignment]
        feature_names = feature_names or field_from(encoder, ("feature_names", "FEATURE_NAMES"))  # type: ignore[assignment]
        width_value = field_from(encoder, ("width", "input_width", "FEATURE_WIDTH"))
    else:
        width_value = None

    if not isinstance(encoder_version, str) or not encoder_version:
        raise CheckpointError("encoder_version is required")
    if not _valid_digest(encoder_fingerprint):
        raise CheckpointError("encoder_fingerprint must be a lowercase SHA-256 digest")
    if feature_names is None or isinstance(feature_names, (str, bytes)):
        raise CheckpointError("ordered feature_names are required")
    names = tuple(feature_names)
    if len(names) != INPUT_WIDTH or any(not isinstance(name, str) or not name for name in names):
        raise CheckpointError(f"feature_names must contain exactly {INPUT_WIDTH} non-empty strings")
    if len(set(names)) != len(names):
        raise CheckpointError("feature_names must be unique")
    if width_value is not None and (type(width_value) is not int or width_value != INPUT_WIDTH):
        raise CheckpointError(f"encoder width must be {INPUT_WIDTH}")
    return {
        "version": encoder_version,
        "fingerprint": encoder_fingerprint,
        "width": INPUT_WIDTH,
        "feature_names": list(names),
    }


def _identity_payload(manifest: Mapping[str, object]) -> dict[str, object]:
    excluded = {
        "checkpoint_fingerprint",
        "created_at",
        "environment",
        "output_path",
        "elapsed_seconds",
    }
    return {key: value for key, value in manifest.items() if key not in excluded}


def _checkpoint_fingerprint(manifest: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(_identity_payload(manifest))).hexdigest()


def save_checkpoint(
    path: str | Path,
    model: CandidateMLP,
    *,
    metrics: object,
    encoder: object | None = None,
    encoder_version: str | None = None,
    encoder_fingerprint: str | None = None,
    feature_names: Sequence[str] | None = None,
    training_config: object | None = None,
    training_seeds: Mapping[str, object] | None = None,
    dataset_fingerprint: str | None = None,
    source_corpus_fingerprints: Sequence[str] = (),
    rules_fingerprint: str | None = None,
    code_fingerprint: str | None = None,
    rules_digest: str | None = None,
    code_digest: str | None = None,
    parent_checkpoint: str | None = None,
    generation: int | None = None,
    metadata: Mapping[str, object] | None = None,
    created_at: datetime | str | None = None,
) -> SavedCheckpoint:
    """Atomically create a new immutable inference bundle.

    Existing paths are never overwritten. ``metadata`` is an optional JSON-safe extension namespace;
    core schema/model/encoder/file fields cannot be replaced through it.
    """
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise CheckpointError(f"checkpoint destination already exists: {destination}")
    if next(model.parameters()).device.type != "cpu":
        raise CheckpointError("inference checkpoint models must be on CPU")
    try:
        validate_state_dict(model.state_dict())
    except ModelError as exc:
        raise CheckpointError("model is incompatible with candidate-mlp-v1") from exc

    if rules_fingerprint is not None and rules_digest is not None:
        raise CheckpointError("provide rules_fingerprint or rules_digest, not both")
    if code_fingerprint is not None and code_digest is not None:
        raise CheckpointError("provide code_fingerprint or code_digest, not both")
    declared_rules = rules_fingerprint or rules_digest or current_rules_fingerprint()
    declared_code = code_fingerprint or code_digest or current_code_fingerprint()
    if not _valid_digest(declared_rules) or not _valid_digest(declared_code):
        raise CheckpointError("rules and code fingerprints must be lowercase SHA-256 digests")

    encoder_data = _encoder_data(
        encoder,
        encoder_version=encoder_version,
        encoder_fingerprint=encoder_fingerprint,
        feature_names=feature_names,
    )
    metrics_data = _json_copy(metrics)
    training_data = {
        "config": _json_copy(training_config if training_config is not None else {}),
        "seeds": _json_copy(training_seeds if training_seeds is not None else {}),
        "deterministic_algorithms": True,
        "device": "cpu",
    }
    sources_data = {
        "dataset_fingerprint": dataset_fingerprint,
        "corpus_fingerprints": list(source_corpus_fingerprints),
    }
    lineage_data = {"parent_checkpoint": parent_checkpoint, "generation": generation}
    extension_data = _json_copy(metadata if metadata is not None else {})

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=str(destination.parent))
    )
    try:
        state = {
            name: tensor.detach().cpu().contiguous().clone()
            for name, tensor in model.state_dict().items()
        }
        weights_path = temporary / WEIGHTS_FILENAME
        metrics_path = temporary / METRICS_FILENAME
        torch.save(state, weights_path)
        metrics_path.write_bytes(_canonical_json(metrics_data) + b"\n")

        file_data = {
            WEIGHTS_FILENAME: {
                "sha256": _sha256_file(weights_path),
                "size": weights_path.stat().st_size,
            },
            METRICS_FILENAME: {
                "sha256": _sha256_file(metrics_path),
                "size": metrics_path.stat().st_size,
            },
        }
        if isinstance(created_at, datetime):
            created = created_at.astimezone(UTC).isoformat()
        elif isinstance(created_at, str):
            created = created_at
        elif created_at is None:
            created = datetime.now(UTC).isoformat()
        else:
            raise CheckpointError("created_at must be a datetime, string, or None")

        manifest: dict[str, object] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "artifact_kind": CHECKPOINT_ARTIFACT_KIND,
            "created_at": created,
            "model": model_spec(),
            "encoder": encoder_data,
            "compatibility": {
                "rules_fingerprint": declared_rules,
                "code_fingerprint": declared_code,
            },
            "training": training_data,
            "sources": sources_data,
            "metrics_summary": metrics_data,
            "lineage": lineage_data,
            "metadata": extension_data,
            "files": file_data,
            "tensor_digest": tensor_digest(state),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "device": "cpu",
                "cpu_threads": torch.get_num_threads(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            },
        }
        manifest["checkpoint_fingerprint"] = _checkpoint_fingerprint(manifest)
        (temporary / MANIFEST_FILENAME).write_bytes(_canonical_json(manifest) + b"\n")
        os.replace(temporary, destination)
        frozen_data = _require_mapping(_json_copy(manifest), "manifest")
        frozen_manifest = cast(Mapping[str, object], _freeze_json(frozen_data))
        return SavedCheckpoint(
            destination,
            str(manifest["checkpoint_fingerprint"]),
            frozen_manifest,
        )
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _read_json(path: Path, *, label: str) -> object:
    if path.is_symlink() or not path.is_file():
        raise CheckpointIntegrityError(f"{label} must be a regular file")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointIntegrityError(f"invalid {label}") from exc


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise CheckpointIntegrityError(f"{label} must be a JSON object")
    return cast(Mapping[str, object], value)


def _manifest(path: Path) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_dir():
        raise CheckpointIntegrityError("checkpoint path must be a regular directory")
    actual_files = {entry.name for entry in path.iterdir()}
    if actual_files != _BUNDLE_FILES or any(entry.is_symlink() for entry in path.iterdir()):
        raise CheckpointIntegrityError(
            f"checkpoint bundle must contain exactly {sorted(_BUNDLE_FILES)}"
        )
    manifest = _require_mapping(_read_json(path / MANIFEST_FILENAME, label="manifest"), "manifest")
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointCompatibilityError("unsupported checkpoint schema version")
    if manifest.get("artifact_kind") != CHECKPOINT_ARTIFACT_KIND:
        raise CheckpointCompatibilityError("checkpoint is not an immutable inference artifact")
    declared_identity = manifest.get("checkpoint_fingerprint")
    if not _valid_digest(declared_identity) or declared_identity != _checkpoint_fingerprint(
        manifest
    ):
        raise CheckpointIntegrityError("checkpoint fingerprint mismatch")
    return manifest


def _validate_model_and_encoder(
    manifest: Mapping[str, object], compatibility: CheckpointCompatibility | None
) -> None:
    declared_model = _require_mapping(manifest.get("model"), "model")
    expected_model = model_spec()
    for field in (
        "version",
        "input_width",
        "hidden_width",
        "output_width",
        "parameter_count",
        "output_semantics",
        "activation",
        "forward_output",
        "initialization",
    ):
        if declared_model.get(field) != expected_model[field]:
            raise CheckpointCompatibilityError(f"model {field} mismatch")

    encoder = _require_mapping(manifest.get("encoder"), "encoder")
    names = encoder.get("feature_names")
    if (
        encoder.get("width") != INPUT_WIDTH
        or not isinstance(encoder.get("version"), str)
        or not encoder.get("version")
        or not _valid_digest(encoder.get("fingerprint"))
        or not isinstance(names, list)
        or len(set(names)) != INPUT_WIDTH
        or len(names) != INPUT_WIDTH
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise CheckpointCompatibilityError("invalid or incompatible encoder width/features")
    declared_compatibility = _require_mapping(manifest.get("compatibility"), "compatibility")
    declared_rules = declared_compatibility.get("rules_fingerprint")
    if not _valid_digest(declared_rules):
        raise CheckpointCompatibilityError("invalid checkpoint rules fingerprint")
    declared_code = declared_compatibility.get("code_fingerprint")
    if not _valid_digest(declared_code):
        raise CheckpointCompatibilityError("invalid checkpoint code fingerprint")

    if compatibility is None:
        return
    checks = {
        "encoder version": (encoder.get("version"), compatibility.encoder_version),
        "encoder fingerprint": (encoder.get("fingerprint"), compatibility.encoder_fingerprint),
        "encoder width": (encoder.get("width"), compatibility.input_width),
        "model version": (declared_model.get("version"), compatibility.model_version),
    }
    expected_rules = compatibility.rules_fingerprint
    if expected_rules is not None:
        checks["rules fingerprint"] = (declared_rules, expected_rules)
    expected_code = compatibility.code_fingerprint
    if expected_code is not None:
        checks["code fingerprint"] = (
            declared_code,
            expected_code,
        )
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise CheckpointCompatibilityError(f"{label} mismatch")
    if compatibility.feature_names is not None and tuple(names) != compatibility.feature_names:
        raise CheckpointCompatibilityError("encoder feature names/order mismatch")


def _validate_files(path: Path, manifest: Mapping[str, object]) -> object:
    files = _require_mapping(manifest.get("files"), "files")
    if set(files) != {WEIGHTS_FILENAME, METRICS_FILENAME}:
        raise CheckpointIntegrityError("manifest file list mismatch")
    for filename in (WEIGHTS_FILENAME, METRICS_FILENAME):
        record = _require_mapping(files.get(filename), f"files.{filename}")
        expected_digest = record.get("sha256")
        expected_size = record.get("size")
        artifact = path / filename
        if artifact.is_symlink() or not artifact.is_file():
            raise CheckpointIntegrityError(f"missing regular checkpoint file {filename}")
        if not _valid_digest(expected_digest) or expected_digest != _sha256_file(artifact):
            raise CheckpointIntegrityError(f"{filename} digest mismatch")
        if type(expected_size) is not int or expected_size != artifact.stat().st_size:
            raise CheckpointIntegrityError(f"{filename} size mismatch")
    metrics = _read_json(path / METRICS_FILENAME, label="metrics")
    if _json_copy(metrics) != manifest.get("metrics_summary"):
        raise CheckpointIntegrityError("metrics summary mismatch")
    return metrics


def _load_state(path: Path, manifest: Mapping[str, object]) -> Mapping[str, Tensor]:
    try:
        value = torch.load(path / WEIGHTS_FILENAME, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointIntegrityError("safe checkpoint weight loading failed") from exc
    if not isinstance(value, Mapping):
        raise CheckpointIntegrityError("weights file must contain a state dictionary")
    try:
        validate_state_dict(value)
        actual_tensor_digest = tensor_digest(value)
    except (ModelError, CheckpointError) as exc:
        raise CheckpointIntegrityError("invalid model state in weights file") from exc
    if manifest.get("tensor_digest") != actual_tensor_digest:
        raise CheckpointIntegrityError("canonical tensor digest mismatch")
    return cast(Mapping[str, Tensor], value)


def inspect_checkpoint(
    path: str | Path,
    *,
    compatibility: CheckpointCompatibility | None = None,
) -> CheckpointInspection:
    """Fully validate a bundle and return immutable metadata without creating a game session."""
    compatibility = compatibility or current_compatibility()
    checkpoint_path = Path(path)
    manifest = _manifest(checkpoint_path)
    _validate_model_and_encoder(manifest, compatibility)
    metrics = _validate_files(checkpoint_path, manifest)
    state = _load_state(checkpoint_path, manifest)
    frozen_data = _require_mapping(_json_copy(manifest), "manifest")
    frozen_manifest = cast(Mapping[str, object], _freeze_json(frozen_data))
    return CheckpointInspection(
        path=checkpoint_path,
        checkpoint_fingerprint=str(manifest["checkpoint_fingerprint"]),
        manifest=frozen_manifest,
        metrics=_freeze_json(_json_copy(metrics)),
        tensor_digest=tensor_digest(state),
    )


def load_checkpoint(
    path: str | Path,
    *,
    compatibility: CheckpointCompatibility | None = None,
    expected_encoder_version: str | None = None,
    expected_encoder_fingerprint: str | None = None,
    expected_feature_names: Sequence[str] | None = None,
    expected_rules_fingerprint: str | None = None,
    expected_code_fingerprint: str | None = None,
) -> LoadedCheckpoint:
    """Validate and safely load an inference-only CPU model.

    The explicit ``expected_*`` keywords are a convenience alternative to constructing
    :class:`CheckpointCompatibility` and make the boundary straightforward for later CLI code.
    """
    if compatibility is not None and any(
        value is not None
        for value in (
            expected_encoder_version,
            expected_encoder_fingerprint,
            expected_feature_names,
            expected_rules_fingerprint,
            expected_code_fingerprint,
        )
    ):
        raise CheckpointError("use either compatibility or expected_* arguments, not both")
    if compatibility is None and any(
        value is not None
        for value in (
            expected_encoder_version,
            expected_encoder_fingerprint,
            expected_feature_names,
            expected_rules_fingerprint,
            expected_code_fingerprint,
        )
    ):
        current = current_compatibility()
        compatibility = CheckpointCompatibility(
            encoder_version=expected_encoder_version or current.encoder_version,
            encoder_fingerprint=expected_encoder_fingerprint or current.encoder_fingerprint,
            feature_names=current.feature_names
            if expected_feature_names is None
            else tuple(expected_feature_names),
            rules_fingerprint=expected_rules_fingerprint or current.rules_fingerprint,
            code_fingerprint=expected_code_fingerprint,
        )
    if compatibility is None:
        compatibility = current_compatibility()

    checkpoint_path = Path(path)
    manifest = _manifest(checkpoint_path)
    _validate_model_and_encoder(manifest, compatibility)
    metrics = _validate_files(checkpoint_path, manifest)
    state = _load_state(checkpoint_path, manifest)
    model = create_model(seed=0)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:  # pragma: no cover - validate_state_dict already gives detail
        raise CheckpointIntegrityError("model state could not be loaded") from exc
    model.eval()
    model.requires_grad_(False)
    frozen_data = _require_mapping(_json_copy(manifest), "manifest")
    frozen_manifest = cast(Mapping[str, object], _freeze_json(frozen_data))
    return LoadedCheckpoint(
        path=checkpoint_path,
        model=model,
        checkpoint_fingerprint=str(manifest["checkpoint_fingerprint"]),
        manifest=frozen_manifest,
        metrics=_freeze_json(_json_copy(metrics)),
    )

"""Resumable compressed corpora of verified completed-game records."""

from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, TextIO

from agent_avenue.engine.setup import RULES_VERSION, SHUFFLE_VERSION

from .fingerprints import code_fingerprint, rules_fingerprint
from .game_record import (
    GAME_RECORD_SCHEMA_VERSION,
    GameRecord,
    GameRecordError,
    game_record_fingerprint,
    game_record_from_data,
    game_record_to_data,
    verify_game_record,
)

CORPUS_SCHEMA_VERSION: Final[int] = 2
CORPUS_PROGRESS_SCHEMA_VERSION: Final[int] = 1
_RECORDS_NAME: Final[str] = "games.jsonl.gz"
_PROGRESS_NAME: Final[str] = "progress.json"
_STAGING_NAME: Final[str] = "staging"
_LOCK_NAME: Final[str] = ".lock"


class CorpusError(ValueError):
    """Raised when a corpus is malformed, tampered with, or incompatible."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _identity(data: Mapping[str, object], *excluded: str) -> str:
    payload = {key: value for key, value in data.items() if key not in excluded}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _json_copy(value: object) -> object:
    return json.loads(_canonical_json(value))


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(data)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _valid_game_id(value: str) -> bool:
    return bool(value) and value not in {".", ".."} and "/" not in value and "\\" not in value


@dataclass(frozen=True, slots=True)
class CorpusDeclaration:
    """Immutable identity of a scheduled corpus before any games are generated."""

    run_id: str
    behavior_policy: str
    expected_game_ids: tuple[str, ...]
    root_seed: int | None = None
    generation: int | None = None
    configuration: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id or not self.behavior_policy:
            raise CorpusError("run_id and behavior_policy must be non-empty")
        if not self.expected_game_ids:
            raise CorpusError("corpus declaration must contain at least one game")
        if len(set(self.expected_game_ids)) != len(self.expected_game_ids) or any(
            not _valid_game_id(game_id) for game_id in self.expected_game_ids
        ):
            raise CorpusError("expected game ids must be unique safe names")
        copied = _json_copy(dict(self.configuration))
        if not isinstance(copied, dict):  # pragma: no cover - mapping guarantees this
            raise CorpusError("corpus configuration must be a JSON object")
        object.__setattr__(self, "configuration", copied)

    def to_data(self) -> dict[str, object]:
        return {
            "schema_version": CORPUS_PROGRESS_SCHEMA_VERSION,
            "run_id": self.run_id,
            "behavior_policy": self.behavior_policy,
            "expected_game_ids": list(self.expected_game_ids),
            "root_seed": self.root_seed,
            "generation": self.generation,
            "configuration": _json_copy(self.configuration),
            "rules_fingerprint": rules_fingerprint(),
            "code_fingerprint": code_fingerprint(),
        }

    @property
    def fingerprint(self) -> str:
        return _identity(self.to_data())


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    schema_version: int
    corpus_fingerprint: str
    declaration_fingerprint: str | None
    run_id: str
    created_at: str
    records_file: str
    record_count: int
    decision_count: int
    record_fingerprints: tuple[str, ...]
    rules_version: str
    shuffle_version: str
    game_record_schema_version: int
    rules_fingerprint: str
    code_fingerprint: str
    root_seed: int | None
    generation: int | None
    behavior_policy: str
    configuration: Mapping[str, object]

    def to_data(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "corpus_fingerprint": self.corpus_fingerprint,
            "declaration_fingerprint": self.declaration_fingerprint,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "records_file": self.records_file,
            "record_count": self.record_count,
            "decision_count": self.decision_count,
            "record_fingerprints": list(self.record_fingerprints),
            "rules_version": self.rules_version,
            "shuffle_version": self.shuffle_version,
            "game_record_schema_version": self.game_record_schema_version,
            "rules_fingerprint": self.rules_fingerprint,
            "code_fingerprint": self.code_fingerprint,
            "root_seed": self.root_seed,
            "generation": self.generation,
            "behavior_policy": self.behavior_policy,
            "configuration": _json_copy(self.configuration),
        }


class ResumableCorpus:
    """Locked append-by-game corpus that can safely resume after interruption."""

    def __init__(self, directory: Path, declaration: CorpusDeclaration, lock: TextIO) -> None:
        self.directory = directory
        self.declaration = declaration
        self._lock = lock
        self._closed = False

    def __enter__(self) -> ResumableCorpus:
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()
            self._closed = True

    def _shard_path(self, index: int) -> Path:
        return self.directory / _STAGING_NAME / f"game-{index:06d}.json.gz"

    def _load_shard(self, index: int, *, verify_code: bool = True) -> GameRecord:
        path = self._shard_path(index)
        if path.is_symlink() or not path.is_file():
            raise CorpusError(f"corpus shard {index} must be a regular file")
        try:
            with gzip.open(path, "rt", encoding="utf-8") as source:
                lines = tuple(line for line in source if line.strip())
            if len(lines) != 1:
                raise CorpusError(f"corpus shard {index} must contain exactly one record")
            record = game_record_from_data(json.loads(lines[0]))
            verify_game_record(record, verify_code=verify_code)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, GameRecordError) as exc:
            raise CorpusError(f"unable to verify corpus shard {index}") from exc
        expected_id = self.declaration.expected_game_ids[index]
        if record.run_id != self.declaration.run_id or record.game_id != expected_id:
            raise CorpusError(f"corpus shard {index} does not match its scheduled game")
        return record

    def completed_records(self) -> tuple[GameRecord, ...]:
        """Return every verified staged record in declaration order."""
        staging = self.directory / _STAGING_NAME
        expected_names = {
            self._shard_path(index).name for index in range(len(self.declaration.expected_game_ids))
        }
        actual_names = {entry.name for entry in staging.iterdir()}
        if not actual_names <= expected_names:
            raise CorpusError("corpus staging directory contains unexpected shards")
        return tuple(
            self._load_shard(index)
            for index in range(len(self.declaration.expected_game_ids))
            if self._shard_path(index).exists() or self._shard_path(index).is_symlink()
        )

    def completed_game_ids(self) -> tuple[str, ...]:
        return tuple(record.game_id for record in self.completed_records())

    def missing_game_ids(self) -> tuple[str, ...]:
        completed = set(self.completed_game_ids())
        return tuple(
            game_id for game_id in self.declaration.expected_game_ids if game_id not in completed
        )

    def append(self, record: GameRecord) -> None:
        if self._closed:
            raise CorpusError("cannot append to a closed corpus")
        try:
            index = self.declaration.expected_game_ids.index(record.game_id)
        except ValueError as exc:
            raise CorpusError(f"unexpected corpus game id: {record.game_id}") from exc
        if record.run_id != self.declaration.run_id:
            raise CorpusError("record run id does not match corpus declaration")
        path = self._shard_path(index)
        if path.exists() or path.is_symlink():
            existing = self._load_shard(index)
            if game_record_fingerprint(existing) != game_record_fingerprint(record):
                raise CorpusError("existing corpus shard conflicts with generated record")
            return
        verify_game_record(record)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".shard-{index:06d}.tmp-", dir=self.directory
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                    compressed.write(_canonical_json(game_record_to_data(record)) + b"\n")
                raw.flush()
                os.fsync(raw.fileno())
            os.replace(temporary, path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def finalize(self) -> CorpusManifest:
        records = tuple(
            self._load_shard(index) for index in range(len(self.declaration.expected_game_ids))
        )
        temporary = self.directory / f".{_RECORDS_NAME}.tmp"
        try:
            with temporary.open("wb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                    for record in records:
                        compressed.write(_canonical_json(game_record_to_data(record)) + b"\n")
                raw.flush()
                os.fsync(raw.fileno())
            os.replace(temporary, self.directory / _RECORDS_NAME)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        data: dict[str, object] = {
            "schema_version": CORPUS_SCHEMA_VERSION,
            "corpus_fingerprint": "",
            "declaration_fingerprint": self.declaration.fingerprint,
            "run_id": self.declaration.run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "records_file": _RECORDS_NAME,
            "record_count": len(records),
            "decision_count": sum(record.decision_count for record in records),
            "record_fingerprints": [game_record_fingerprint(record) for record in records],
            "rules_version": RULES_VERSION,
            "shuffle_version": SHUFFLE_VERSION,
            "game_record_schema_version": GAME_RECORD_SCHEMA_VERSION,
            "rules_fingerprint": rules_fingerprint(),
            "code_fingerprint": code_fingerprint(),
            "root_seed": self.declaration.root_seed,
            "generation": self.declaration.generation,
            "behavior_policy": self.declaration.behavior_policy,
            "configuration": _json_copy(self.declaration.configuration),
        }
        data["corpus_fingerprint"] = _identity(data, "corpus_fingerprint", "created_at")
        _atomic_write(self.directory / "manifest.json", _canonical_json(data) + b"\n")
        shutil.rmtree(self.directory / _STAGING_NAME)
        (self.directory / _PROGRESS_NAME).unlink(missing_ok=True)
        return _manifest_from_data(data)


def open_resumable_corpus(directory: Path, declaration: CorpusDeclaration) -> ResumableCorpus:
    """Open or create a declared corpus and exclusively lock it until closed."""
    if directory.is_symlink():
        raise CorpusError("corpus destination cannot be a symlink")
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / _LOCK_NAME).open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise CorpusError("corpus is already open by another writer") from exc
    try:
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            manifest, _ = load_corpus(directory)
            if manifest.declaration_fingerprint != declaration.fingerprint:
                raise CorpusError("completed corpus declaration does not match requested schedule")
            raise CorpusError("completed corpus does not need a resumable writer")
        progress_path = directory / _PROGRESS_NAME
        declared_data = declaration.to_data()
        declared_data["declaration_fingerprint"] = declaration.fingerprint
        if progress_path.exists() or progress_path.is_symlink():
            if progress_path.is_symlink() or not progress_path.is_file():
                raise CorpusError("corpus progress must be a regular file")
            try:
                actual = json.loads(progress_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise CorpusError("unable to load corpus progress") from exc
            if actual != declared_data:
                raise CorpusError("corpus progress declaration mismatch")
        else:
            _atomic_write(progress_path, _canonical_json(declared_data) + b"\n")
        staging = directory / _STAGING_NAME
        if staging.is_symlink():
            raise CorpusError("corpus staging directory cannot be a symlink")
        staging.mkdir(exist_ok=True)
        for entry in staging.iterdir():
            if entry.name.startswith(".game-") and ".json.gz.tmp-" in entry.name:
                if entry.is_symlink() or not entry.is_file():
                    raise CorpusError("stale corpus temporary shard is not a regular file")
                entry.unlink()
        return ResumableCorpus(directory, declaration, lock)
    except Exception:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
        raise


def write_corpus(
    directory: Path,
    records: Iterable[GameRecord],
    *,
    run_id: str,
    behavior_policy: str,
    root_seed: int | None = None,
    generation: int | None = None,
    verify_code: bool = True,
) -> CorpusManifest:
    """Write a finite record iterable through the resumable corpus format."""
    if not verify_code:
        raise CorpusError("resumable corpus writing requires current-code record verification")
    buffered = tuple(records)
    if not buffered:
        raise CorpusError("cannot create an empty corpus")

    declaration = CorpusDeclaration(
        run_id,
        behavior_policy,
        tuple(record.game_id for record in buffered),
        root_seed,
        generation,
    )
    with open_resumable_corpus(directory, declaration) as corpus:
        for record in buffered:
            verify_game_record(record)
            corpus.append(record)
        return corpus.finalize()


def _manifest_from_data(data: object) -> CorpusManifest:
    if not isinstance(data, dict):
        raise CorpusError("corpus manifest must be an object")
    version = data.get("schema_version")
    common = {
        "schema_version",
        "corpus_fingerprint",
        "run_id",
        "created_at",
        "records_file",
        "record_count",
        "decision_count",
        "record_fingerprints",
        "rules_version",
        "shuffle_version",
        "game_record_schema_version",
        "rules_fingerprint",
        "code_fingerprint",
        "root_seed",
        "generation",
        "behavior_policy",
    }
    expected = common if version == 1 else common | {"declaration_fingerprint", "configuration"}
    if set(data) != expected or version not in {1, CORPUS_SCHEMA_VERSION}:
        raise CorpusError("corpus manifest fields are invalid")
    try:
        configuration = data.get("configuration")
        if version == CORPUS_SCHEMA_VERSION and not isinstance(configuration, dict):
            raise CorpusError("corpus configuration must be an object")
        manifest = CorpusManifest(
            schema_version=int(data["schema_version"]),
            corpus_fingerprint=str(data["corpus_fingerprint"]),
            declaration_fingerprint=(
                str(data["declaration_fingerprint"]) if version == CORPUS_SCHEMA_VERSION else None
            ),
            run_id=str(data["run_id"]),
            created_at=str(data["created_at"]),
            records_file=str(data["records_file"]),
            record_count=int(data["record_count"]),
            decision_count=int(data["decision_count"]),
            record_fingerprints=tuple(data["record_fingerprints"]),
            rules_version=str(data["rules_version"]),
            shuffle_version=str(data["shuffle_version"]),
            game_record_schema_version=int(data["game_record_schema_version"]),
            rules_fingerprint=str(data["rules_fingerprint"]),
            code_fingerprint=str(data["code_fingerprint"]),
            root_seed=data["root_seed"],
            generation=data["generation"],
            behavior_policy=str(data["behavior_policy"]),
            configuration=configuration if isinstance(configuration, dict) else {},
        )
    except (TypeError, ValueError) as exc:
        raise CorpusError("malformed corpus manifest") from exc
    excluded = ("corpus_fingerprint", "created_at")
    if manifest.records_file != _RECORDS_NAME or manifest.corpus_fingerprint != _identity(
        data, *excluded
    ):
        raise CorpusError("corpus manifest fingerprint mismatch")
    if manifest.rules_fingerprint != rules_fingerprint():
        raise CorpusError("corpus rules fingerprint mismatch")
    return manifest


def load_corpus(
    directory: Path, *, verify_code: bool = True, verify_replays: bool = True
) -> tuple[CorpusManifest, tuple[GameRecord, ...]]:
    """Load a corpus, verifying replay semantics by default.

    ``verify_replays=False`` is for repeated derived passes only after the same caller has already
    completed a full verified load. Record and corpus fingerprints, ordering, counts, and code/rules
    compatibility are still checked.
    """
    try:
        data = json.loads((directory / "manifest.json").read_text())
        manifest = _manifest_from_data(data)
        if verify_code and manifest.code_fingerprint != code_fingerprint():
            raise CorpusError("corpus code fingerprint mismatch")
        records: list[GameRecord] = []
        records_path = directory / manifest.records_file
        if records_path.is_symlink() or not records_path.is_file():
            raise CorpusError("corpus records must be a regular file")
        with gzip.open(records_path, "rt", encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    record = game_record_from_data(json.loads(line))
                    if verify_replays:
                        verify_game_record(record, verify_code=verify_code)
                    records.append(record)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, GameRecordError) as exc:
        if isinstance(exc, CorpusError):
            raise
        raise CorpusError("unable to load corpus") from exc
    fingerprints = tuple(game_record_fingerprint(record) for record in records)
    if fingerprints != manifest.record_fingerprints:
        raise CorpusError("corpus record order or fingerprint mismatch")
    if (
        len(records) != manifest.record_count
        or sum(record.decision_count for record in records) != manifest.decision_count
    ):
        raise CorpusError("corpus counts do not match manifest")
    return manifest, tuple(records)

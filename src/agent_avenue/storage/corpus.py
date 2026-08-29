"""Append-only compressed corpora of verified completed-game records."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

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

CORPUS_SCHEMA_VERSION: Final[int] = 1


class CorpusError(ValueError):
    """Raised when a corpus is malformed, tampered with, or incompatible."""


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    schema_version: int
    corpus_fingerprint: str
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

    def to_data(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "corpus_fingerprint": self.corpus_fingerprint,
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
        }


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _identity(data: dict[str, object]) -> str:
    payload = dict(data)
    payload.pop("corpus_fingerprint", None)
    payload.pop("created_at", None)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


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
    """Stream verified records to a new gzip JSONL corpus and write its manifest."""
    if directory.exists():
        raise CorpusError(f"corpus destination already exists: {directory}")
    if not run_id or not behavior_policy:
        raise CorpusError("run_id and behavior_policy must be non-empty")
    directory.mkdir(parents=True)
    records_name = "games.jsonl.gz"
    record_fingerprints: list[str] = []
    decisions = 0
    try:
        # mtime=0 keeps compression metadata deterministic; semantic identity does not depend on it.
        with (
            (directory / records_name).open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        ):
            for record in records:
                verify_game_record(record, verify_code=verify_code)
                fingerprint = game_record_fingerprint(record)
                compressed.write(_canonical_json(game_record_to_data(record)) + b"\n")
                record_fingerprints.append(fingerprint)
                decisions += record.decision_count
        if not record_fingerprints:
            raise CorpusError("cannot create an empty corpus")
        data: dict[str, object] = {
            "schema_version": CORPUS_SCHEMA_VERSION,
            "corpus_fingerprint": "",
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "records_file": records_name,
            "record_count": len(record_fingerprints),
            "decision_count": decisions,
            "record_fingerprints": record_fingerprints,
            "rules_version": RULES_VERSION,
            "shuffle_version": SHUFFLE_VERSION,
            "game_record_schema_version": GAME_RECORD_SCHEMA_VERSION,
            "rules_fingerprint": rules_fingerprint(),
            "code_fingerprint": code_fingerprint(),
            "root_seed": root_seed,
            "generation": generation,
            "behavior_policy": behavior_policy,
        }
        data["corpus_fingerprint"] = _identity(data)
        (directory / "manifest.json").write_bytes(_canonical_json(data) + b"\n")
        return _manifest_from_data(data)
    except Exception:
        for path in directory.glob("*"):
            path.unlink()
        directory.rmdir()
        raise


def _manifest_from_data(data: object) -> CorpusManifest:
    if not isinstance(data, dict):
        raise CorpusError("corpus manifest must be an object")
    expected = {
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
    if set(data) != expected:
        raise CorpusError("corpus manifest fields are invalid")
    try:
        manifest = CorpusManifest(
            schema_version=int(data["schema_version"]),
            corpus_fingerprint=str(data["corpus_fingerprint"]),
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
        )
    except (TypeError, ValueError) as exc:
        raise CorpusError("malformed corpus manifest") from exc
    if (
        manifest.schema_version != CORPUS_SCHEMA_VERSION
        or manifest.records_file != "games.jsonl.gz"
    ):
        raise CorpusError("unsupported corpus schema")
    if manifest.corpus_fingerprint != _identity(data):
        raise CorpusError("corpus manifest fingerprint mismatch")
    if manifest.rules_fingerprint != rules_fingerprint():
        raise CorpusError("corpus rules fingerprint mismatch")
    return manifest


def load_corpus(
    directory: Path, *, verify_code: bool = True
) -> tuple[CorpusManifest, tuple[GameRecord, ...]]:
    """Load and fully verify a corpus and its ordered record identities."""
    try:
        data = json.loads((directory / "manifest.json").read_text())
        manifest = _manifest_from_data(data)
        if verify_code and manifest.code_fingerprint != code_fingerprint():
            raise CorpusError("corpus code fingerprint mismatch")
        records: list[GameRecord] = []
        with gzip.open(directory / manifest.records_file, "rt", encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    record = game_record_from_data(json.loads(line))
                    verify_game_record(record, verify_code=verify_code)
                    records.append(record)
    except (OSError, json.JSONDecodeError, GameRecordError) as exc:
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

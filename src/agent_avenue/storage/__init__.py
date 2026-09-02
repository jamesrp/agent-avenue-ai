"""Versioned storage formats and compatibility verification."""

from .corpus import (
    CORPUS_SCHEMA_VERSION,
    CorpusDeclaration,
    CorpusError,
    CorpusManifest,
    ResumableCorpus,
    load_corpus,
    open_resumable_corpus,
    write_corpus,
)
from .fingerprints import code_fingerprint, rules_fingerprint
from .game_record import (
    GAME_RECORD_SCHEMA_VERSION,
    AgentSeatRecord,
    GameRecord,
    GameRecordError,
    create_game_record,
    game_record_fingerprint,
    game_record_from_data,
    game_record_to_data,
    load_game_record,
    save_game_record,
    verify_game_record,
)
from .provenance import (
    SourceIdentity,
    SourceIdentityError,
    inspect_source_identity,
    repository_root,
)

__all__ = [
    "CORPUS_SCHEMA_VERSION",
    "GAME_RECORD_SCHEMA_VERSION",
    "AgentSeatRecord",
    "CorpusDeclaration",
    "CorpusError",
    "CorpusManifest",
    "GameRecord",
    "GameRecordError",
    "ResumableCorpus",
    "SourceIdentity",
    "SourceIdentityError",
    "code_fingerprint",
    "create_game_record",
    "game_record_fingerprint",
    "game_record_from_data",
    "game_record_to_data",
    "inspect_source_identity",
    "load_corpus",
    "load_game_record",
    "open_resumable_corpus",
    "repository_root",
    "rules_fingerprint",
    "save_game_record",
    "verify_game_record",
    "write_corpus",
]

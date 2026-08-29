"""Stable fingerprints for reproducible run artifacts."""

import hashlib
import json
from pathlib import Path

from agent_avenue.engine import CANONICAL_DECK, CARD_DEFINITIONS
from agent_avenue.engine.setup import RULES_VERSION, SHUFFLE_VERSION


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def rules_fingerprint() -> str:
    """Fingerprint the complete public rules/card/shuffle definition."""
    payload = {
        "rules_version": RULES_VERSION,
        "shuffle_version": SHUFFLE_VERSION,
        "deck": [card.value for card in CANONICAL_DECK],
        "cards": {
            card.value: {
                "copies": definition.copies,
                "effects": [
                    {"kind": effect.kind, "points": effect.points} for effect in definition.effects
                ],
            }
            for card, definition in CARD_DEFINITIONS.items()
        },
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def code_fingerprint(package_root: Path | None = None) -> str:
    """Fingerprint checked-in Python source, independent of timestamps and bytecode."""
    root = package_root or Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()

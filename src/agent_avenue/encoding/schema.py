"""Stable schema and fingerprint contracts for encoder inputs."""

import hashlib
import json
from dataclasses import dataclass, fields
from typing import Final

from agent_avenue.engine.cards import CardName

SCHEMA_VERSION: Final[str] = "candidate-public-v1"
CARD_ORDER: Final[tuple[CardName, ...]] = tuple(CardName)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


@dataclass(frozen=True, slots=True)
class CandidateEncoderConfig:
    """All constants that affect candidate-public-v1 normalization."""

    version: str = SCHEMA_VERSION
    card_order: tuple[CardName, ...] = CARD_ORDER
    turn_scale: int = 19
    deck_scale: int = 30
    score_limit: int = 14
    gap_limit: int = 7
    hand_scale: int = 4

    def __post_init__(self) -> None:
        if self.version != SCHEMA_VERSION:
            raise ValueError(f"unsupported encoder version: {self.version!r}")
        if self.card_order != CARD_ORDER:
            raise ValueError("candidate-public-v1 requires the canonical card order")
        for item in fields(self):
            if item.name in {"version", "card_order"}:
                continue
            value = getattr(self, item.name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{item.name} must be a positive integer")

    def to_data(self) -> dict[str, object]:
        """Return the normalized, JSON-compatible schema configuration."""
        return {
            "card_order": [card.value for card in self.card_order],
            "deck_scale": self.deck_scale,
            "gap_limit": self.gap_limit,
            "hand_scale": self.hand_scale,
            "score_limit": self.score_limit,
            "turn_scale": self.turn_scale,
            "version": self.version,
        }

    @classmethod
    def from_data(cls, data: object) -> "CandidateEncoderConfig":
        if not isinstance(data, dict):
            raise ValueError("encoder configuration must be an object")
        expected = {item.name for item in fields(cls)}
        if set(data) != expected:
            raise ValueError("encoder configuration fields do not match its version")
        card_order = data["card_order"]
        if not isinstance(card_order, list) or any(
            not isinstance(item, str) for item in card_order
        ):
            raise ValueError("card_order must be a list of card names")
        try:
            cards = tuple(CardName(item) for item in card_order)
        except ValueError as exc:
            raise ValueError("card_order contains an unknown card") from exc
        values = {key: data[key] for key in expected if key != "card_order"}
        return cls(card_order=cards, **values)


@dataclass(frozen=True, slots=True)
class CandidateFeatureSchema:
    """Named feature layout and normalization constants for one encoder version."""

    config: CandidateEncoderConfig
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.feature_names) is not tuple or not self.feature_names:
            raise ValueError("feature_names must be a non-empty tuple")
        if any(type(name) is not str or not name for name in self.feature_names):
            raise ValueError("feature names must be non-empty strings")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature names must be unique")

    @property
    def version(self) -> str:
        return self.config.version

    @property
    def width(self) -> int:
        return len(self.feature_names)

    def to_data(self) -> dict[str, object]:
        """Return the complete schema payload used for fingerprinting."""
        return {
            "config": self.config.to_data(),
            "feature_names": list(self.feature_names),
            "version": self.version,
            "width": self.width,
        }

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_data())).hexdigest()


def schema_fingerprint(schema: CandidateFeatureSchema) -> str:
    """Return the SHA-256 fingerprint of a complete feature schema."""
    return schema.fingerprint

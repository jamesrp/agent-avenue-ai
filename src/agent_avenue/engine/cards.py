"""Card definitions for the two-player Agent Avenue base game."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Literal


class CardName(StrEnum):
    """Stable names used by actions, replay files, and observations."""

    DOUBLE_AGENT = "double_agent"
    ENFORCER = "enforcer"
    CODEBREAKER = "codebreaker"
    DAREDEVIL = "daredevil"
    SABOTEUR = "saboteur"
    SENTINEL = "sentinel"
    SIDEKICK = "sidekick"
    MOLE = "mole"


EffectKind = Literal["score", "win", "lose"]


@dataclass(frozen=True, slots=True)
class CardEffect:
    """One recruit effect."""

    kind: EffectKind
    points: int = 0


@dataclass(frozen=True, slots=True)
class CardDefinition:
    """Definition and deck count for one card name."""

    name: CardName
    copies: int
    effects: tuple[CardEffect, ...]

    def effect_for_count(self, count: int) -> CardEffect:
        """Return the effect for the resulting recruited copy count."""
        if count < 1:
            raise ValueError("copy count must be positive")
        return self.effects[min(count, len(self.effects)) - 1]


def _score(points: int) -> CardEffect:
    return CardEffect("score", points)


CARD_DEFINITIONS: Final[Mapping[CardName, CardDefinition]] = MappingProxyType(
    {
        CardName.DOUBLE_AGENT: CardDefinition(
            CardName.DOUBLE_AGENT, 6, (_score(-1), _score(6), _score(-1))
        ),
        CardName.ENFORCER: CardDefinition(CardName.ENFORCER, 6, (_score(1), _score(2), _score(3))),
        CardName.CODEBREAKER: CardDefinition(
            CardName.CODEBREAKER,
            6,
            (_score(0), _score(0), CardEffect("win")),
        ),
        CardName.DAREDEVIL: CardDefinition(
            CardName.DAREDEVIL,
            6,
            (_score(2), _score(3), CardEffect("lose")),
        ),
        CardName.SABOTEUR: CardDefinition(
            CardName.SABOTEUR, 6, (_score(-1), _score(-1), _score(-2))
        ),
        CardName.SENTINEL: CardDefinition(CardName.SENTINEL, 6, (_score(0), _score(2), _score(6))),
        CardName.SIDEKICK: CardDefinition(CardName.SIDEKICK, 1, (_score(4),)),
        CardName.MOLE: CardDefinition(CardName.MOLE, 1, (_score(-3),)),
    }
)

CANONICAL_DECK: Final[tuple[CardName, ...]] = tuple(
    card for card in CardName for _ in range(CARD_DEFINITIONS[card].copies)
)

if len(CANONICAL_DECK) != 38:  # pragma: no cover - import-time invariant
    raise RuntimeError("canonical deck must contain exactly 38 cards")


def recruit_effect(card: CardName, resulting_count: int) -> CardEffect:
    """Return the effect triggered by recruiting ``card`` at this count."""
    return CARD_DEFINITIONS[card].effect_for_count(resulting_count)

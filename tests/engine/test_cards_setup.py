from collections import Counter

import pytest

from agent_avenue.engine import (
    CANONICAL_DECK,
    CARD_DEFINITIONS,
    CardName,
    EngineError,
    ErrorCode,
    GameConfig,
    PlayerId,
    new_game,
    recruit_effect,
)


def test_canonical_deck_composition() -> None:
    assert len(CANONICAL_DECK) == 38
    counts = Counter(CANONICAL_DECK)
    assert counts[CardName.SIDEKICK] == counts[CardName.MOLE] == 1
    for card in CardName:
        assert counts[card] == CARD_DEFINITIONS[card].copies


@pytest.mark.parametrize(
    ("card", "effects"),
    [
        (CardName.DOUBLE_AGENT, (-1, 6, -1, -1)),
        (CardName.ENFORCER, (1, 2, 3, 3)),
        (CardName.SABOTEUR, (-1, -1, -2, -2)),
        (CardName.SENTINEL, (0, 2, 6, 6)),
    ],
)
def test_repeated_score_effects(card: CardName, effects: tuple[int, ...]) -> None:
    assert tuple(recruit_effect(card, count).points for count in range(1, 5)) == effects


def test_special_and_singleton_effects() -> None:
    assert tuple(recruit_effect(CardName.CODEBREAKER, count).points for count in (1, 2)) == (0, 0)
    assert tuple(recruit_effect(CardName.DAREDEVIL, count).points for count in (1, 2)) == (2, 3)
    assert recruit_effect(CardName.CODEBREAKER, 3).kind == "win"
    assert recruit_effect(CardName.DAREDEVIL, 3).kind == "lose"
    assert recruit_effect(CardName.SIDEKICK, 1).points == 4
    assert recruit_effect(CardName.MOLE, 1).points == -3


def test_seeded_setup_is_reproducible_and_respects_starting_player() -> None:
    config = GameConfig(starting_player=PlayerId.PLAYER_TWO)
    first = new_game(config, seed=42)
    second = new_game(config, seed=42)
    other = new_game(config, seed=43)
    assert first == second
    assert first != other
    assert tuple(map(len, first.hands)) == (4, 4)
    assert len(first.deck) == 30
    assert first.active_player is PlayerId.PLAYER_TWO


@pytest.mark.parametrize("seed", [-1, 2**64, True])
def test_invalid_seed_fails_clearly(seed: int) -> None:
    with pytest.raises(EngineError) as error:
        new_game(seed=seed)
    assert error.value.code is ErrorCode.INVALID_CONFIG


def test_noncanonical_configuration_is_rejected() -> None:
    with pytest.raises(EngineError) as error:
        new_game(GameConfig(deck=CANONICAL_DECK[:-1]))
    assert error.value.code is ErrorCode.INVALID_CONFIG


def test_configuration_rejects_mutable_or_enum_like_decks() -> None:
    mutable = GameConfig(deck=list(CANONICAL_DECK))  # type: ignore[arg-type]
    string_cards = GameConfig(deck=tuple(card.value for card in CANONICAL_DECK))  # type: ignore[arg-type]
    for config in (mutable, string_cards):
        with pytest.raises(EngineError) as error:
            new_game(config)
        assert error.value.code is ErrorCode.INVALID_CONFIG

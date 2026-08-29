"""Validated deterministic game setup."""

from collections import Counter

from .cards import CANONICAL_DECK, CardName
from .errors import EngineError, ErrorCode
from .model import GameConfig, GameState, Phase, PlayerId

RULES_VERSION = "base-1"
SHUFFLE_VERSION = "splitmix64-fisher-yates-1"
_MIN_SEED = 0
_MAX_SEED = (1 << 64) - 1


def validate_config(config: GameConfig) -> None:
    """Validate a Milestone 1 base-game configuration."""
    if not isinstance(config.starting_player, PlayerId):
        raise EngineError(ErrorCode.INVALID_CONFIG, "starting_player must be a PlayerId")
    if type(config.deck) is not tuple:
        raise EngineError(ErrorCode.INVALID_CONFIG, "configuration deck must be an immutable tuple")
    if any(not isinstance(card, CardName) for card in config.deck):
        raise EngineError(ErrorCode.INVALID_CONFIG, "configuration cards must be CardName values")
    if Counter(config.deck) != Counter(CANONICAL_DECK) or len(config.deck) != 38:
        raise EngineError(
            ErrorCode.INVALID_CONFIG,
            "configuration deck must be the canonical 38-card base deck",
            card_count=len(config.deck),
        )


def _splitmix64(value: int) -> tuple[int, int]:
    state = (value + 0x9E3779B97F4A7C15) & _MAX_SEED
    mixed = state
    mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & _MAX_SEED
    mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & _MAX_SEED
    return state, mixed ^ (mixed >> 31)


def shuffled_deck(deck: tuple[CardName, ...], seed: int) -> tuple[CardName, ...]:
    """Shuffle with a pinned, language-independent 64-bit Fisher-Yates algorithm."""
    if not isinstance(seed, int) or isinstance(seed, bool) or not _MIN_SEED <= seed <= _MAX_SEED:
        raise EngineError(
            ErrorCode.INVALID_CONFIG,
            "seed must be an unsigned 64-bit integer",
            seed=seed,
        )
    cards = list(deck)
    state = seed
    for index in range(len(cards) - 1, 0, -1):
        state, random_value = _splitmix64(state)
        swap_index = random_value % (index + 1)
        cards[index], cards[swap_index] = cards[swap_index], cards[index]
    return tuple(cards)


def normalize_config(config: GameConfig) -> dict[str, object]:
    """Return the canonical replay representation of a configuration."""
    validate_config(config)
    return {
        "starting_player": config.starting_player.value,
        "deck": [card.value for card in config.deck],
    }


def config_from_normalized(value: object) -> GameConfig:
    """Parse and validate a normalized replay configuration."""
    if not isinstance(value, dict):
        raise EngineError(ErrorCode.INVALID_CONFIG, "configuration must be an object")
    try:
        starting = value["starting_player"]
        deck = value["deck"]
        if not isinstance(starting, str) or not isinstance(deck, list):
            raise TypeError
        config = GameConfig(
            starting_player=PlayerId(starting),
            deck=tuple(CardName(item) for item in deck if isinstance(item, str)),
        )
        if len(config.deck) != len(deck):
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise EngineError(ErrorCode.INVALID_CONFIG, "malformed normalized configuration") from exc
    validate_config(config)
    return config


def new_game(config: GameConfig | None = None, seed: int = 0) -> GameState:
    """Create a reproducibly shuffled game with four cards dealt to each player.

    The shuffled tuple's first element is the deck top. Player one receives the first four cards,
    then player two receives the next four cards.
    """
    actual_config = config or GameConfig()
    validate_config(actual_config)
    shuffled = shuffled_deck(actual_config.deck, seed)
    return GameState(
        config=actual_config,
        seed=seed,
        deck=shuffled[8:],
        hands=(shuffled[:4], shuffled[4:8]),
        recruited=((), ()),
        scores=(0, 0),
        active_player=actual_config.starting_player,
        turn=1,
        phase=Phase.PLAY,
        revision=0,
    )

"""Pull-based execution of independent game specifications."""

from collections.abc import Iterable, Iterator

from agent_avenue.storage import GameRecord

from .game import GameSpec, run_game


def iter_games(specs: Iterable[GameSpec]) -> Iterator[GameRecord]:
    """Yield completed records one at a time without retaining the run."""
    for spec in specs:
        yield run_game(spec)

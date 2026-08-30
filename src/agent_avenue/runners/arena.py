"""Balanced, reproducible baseline arena scheduling and statistics."""

import json
import math
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from agent_avenue.agents import (
    RNG_ALGORITHM,
    SEED_DERIVATION,
    DeterministicRandom,
    derive_seed,
)
from agent_avenue.engine import GameConfig, PlayerId
from agent_avenue.storage import GameRecord, code_fingerprint, rules_fingerprint

from .batch import iter_games
from .game import AgentSpec, GameSpec

PAIRED_BOOTSTRAP_METHOD = "paired-percentile-bootstrap-v1"
PAIRED_BOOTSTRAP_RESAMPLES = 20_000
PAIRED_BOOTSTRAP_DOMAIN = "arena:paired-bootstrap:agent-a-win-rate:v1"
PAIRED_BOOTSTRAP_LOWER_INDEX = 499
PAIRED_BOOTSTRAP_UPPER_INDEX = 19_499
ARENA_REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class PairedSeedOutcome:
    pair_id: str
    agent_a_wins: int

    def __post_init__(self) -> None:
        if (
            not self.pair_id
            or type(self.agent_a_wins) is not int
            or not 0 <= self.agent_a_wins <= 2
        ):
            raise ValueError("paired outcome must identify a pair with zero, one, or two wins")


@dataclass(frozen=True, slots=True)
class PairedBootstrapInterval:
    interval: tuple[float, float]
    point_estimate: float
    pair_count: int
    bootstrap_seed: int

    def to_data(self) -> dict[str, object]:
        return {
            "method": PAIRED_BOOTSTRAP_METHOD,
            "statistic": "agent_a_win_rate",
            "unit": "two-game paired seed block",
            "confidence_level": 0.95,
            "point_estimate": self.point_estimate,
            "pair_count": self.pair_count,
            "resample_count": PAIRED_BOOTSTRAP_RESAMPLES,
            "order_statistic_indices": {
                "lower": PAIRED_BOOTSTRAP_LOWER_INDEX,
                "upper": PAIRED_BOOTSTRAP_UPPER_INDEX,
            },
            "interval": list(self.interval),
            "bootstrap_seed": self.bootstrap_seed,
            "bootstrap_rng_domain": PAIRED_BOOTSTRAP_DOMAIN,
            "rng_algorithm": RNG_ALGORITHM,
            "seed_derivation": SEED_DERIVATION,
        }


def paired_bootstrap_interval(
    pair_wins: tuple[int, ...], master_seed: int
) -> PairedBootstrapInterval:
    """Bootstrap paired two-game seed blocks with the project deterministic RNG."""
    if not pair_wins:
        raise ValueError("paired bootstrap requires at least one pair")
    if any(type(wins) is not int or not 0 <= wins <= 2 for wins in pair_wins):
        raise ValueError("each paired outcome must contain zero, one, or two wins")
    bootstrap_seed = derive_seed(master_seed, PAIRED_BOOTSTRAP_DOMAIN)
    rng = DeterministicRandom(bootstrap_seed, PAIRED_BOOTSTRAP_DOMAIN)
    pair_count = len(pair_wins)
    totals = [
        sum(pair_wins[rng.randbelow(pair_count)] for _ in range(pair_count))
        for _ in range(PAIRED_BOOTSTRAP_RESAMPLES)
    ]
    totals.sort()
    denominator = 2 * pair_count
    return PairedBootstrapInterval(
        interval=(
            totals[PAIRED_BOOTSTRAP_LOWER_INDEX] / denominator,
            totals[PAIRED_BOOTSTRAP_UPPER_INDEX] / denominator,
        ),
        point_estimate=sum(pair_wins) / denominator,
        pair_count=pair_count,
        bootstrap_seed=bootstrap_seed,
    )


@dataclass(frozen=True, slots=True)
class ArenaConfig:
    run_id: str
    agent_a: AgentSpec
    agent_b: AgentSpec
    pair_count: int
    master_seed: int
    game_config: GameConfig = field(default_factory=GameConfig)

    def __post_init__(self) -> None:
        if self.pair_count < 1:
            raise ValueError("pair_count must be positive")
        if self.agent_a.agent_id == self.agent_b.agent_id:
            raise ValueError("arena agent identifiers must be distinct")


@dataclass(frozen=True, slots=True)
class SeatStats:
    games: int
    wins: int
    win_rate: float


@dataclass(frozen=True, slots=True)
class ArenaReport:
    run_id: str
    agent_a_id: str
    agent_b_id: str
    total_games: int
    paired_seed_count: int
    agent_a_wins: int
    agent_b_wins: int
    agent_a_win_rate: float
    confidence_interval_95: tuple[float, float]
    paired_seed_outcomes: tuple[PairedSeedOutcome, ...]
    paired_bootstrap_confidence_interval_95: PairedBootstrapInterval
    agent_a_by_seat: dict[str, SeatStats]
    terminal_reasons: dict[str, int]
    average_score_margin: float
    average_turns: float
    average_decisions: float
    elapsed_seconds: float
    games_per_second: float
    master_seed: int
    seed_derivation: str
    rng_algorithm: str
    game_config: dict[str, object]
    agent_a_config: dict[str, object]
    agent_b_config: dict[str, object]
    rules_fingerprint: str
    code_fingerprint: str

    def to_data(self) -> dict[str, object]:
        return {
            "arena_report_schema_version": ARENA_REPORT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "agents": {
                "a": {
                    "id": self.agent_a_id,
                    "config": json.loads(json.dumps(self.agent_a_config, sort_keys=True)),
                },
                "b": {
                    "id": self.agent_b_id,
                    "config": json.loads(json.dumps(self.agent_b_config, sort_keys=True)),
                },
            },
            "total_games": self.total_games,
            "paired_seed_count": self.paired_seed_count,
            "wins": {"a": self.agent_a_wins, "b": self.agent_b_wins},
            "agent_a_win_rate": self.agent_a_win_rate,
            "confidence_interval_95": list(self.confidence_interval_95),
            "wilson_confidence_interval_95": list(self.confidence_interval_95),
            "paired_seed_outcomes": [
                {"pair_id": outcome.pair_id, "agent_a_wins": outcome.agent_a_wins}
                for outcome in self.paired_seed_outcomes
            ],
            "paired_bootstrap_confidence_interval_95": (
                self.paired_bootstrap_confidence_interval_95.to_data()
            ),
            "agent_a_by_seat": {
                seat: {"games": stats.games, "wins": stats.wins, "win_rate": stats.win_rate}
                for seat, stats in self.agent_a_by_seat.items()
            },
            "terminal_reasons": dict(self.terminal_reasons),
            "average_score_margin": self.average_score_margin,
            "average_turns": self.average_turns,
            "average_decisions": self.average_decisions,
            "elapsed_seconds": self.elapsed_seconds,
            "games_per_second": self.games_per_second,
            "master_seed": self.master_seed,
            "seed_derivation": self.seed_derivation,
            "rng_algorithm": self.rng_algorithm,
            "game_config": json.loads(json.dumps(self.game_config, sort_keys=True)),
            "rules_fingerprint": self.rules_fingerprint,
            "code_fingerprint": self.code_fingerprint,
        }


def wilson_interval(wins: int, games: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return the two-sided Wilson score interval for a binomial proportion."""
    if games <= 0 or not 0 <= wins <= games:
        raise ValueError("wins and games must describe a non-empty binomial sample")
    proportion = wins / games
    z2 = z * z
    denominator = 1 + z2 / games
    center = (proportion + z2 / (2 * games)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1 - proportion) / games + z2 / (4 * games * games))
        / denominator
    )
    return (max(0.0, center - radius), min(1.0, center + radius))


def schedule_arena(config: ArenaConfig) -> Iterator[GameSpec]:
    """Yield paired setup seeds with logical-agent RNG streams preserved across seats."""
    for pair_index in range(config.pair_count):
        pair_domain = f"arena:pair:{pair_index}"
        setup_seed = derive_seed(config.master_seed, f"{pair_domain}:setup") & ((1 << 64) - 1)
        seed_a = derive_seed(config.master_seed, f"{pair_domain}:agent:{config.agent_a.agent_id}")
        seed_b = derive_seed(config.master_seed, f"{pair_domain}:agent:{config.agent_b.agent_id}")
        pair_id = f"pair-{pair_index:06d}"
        yield GameSpec(
            config.run_id,
            f"{pair_id}-a-first",
            pair_id,
            config.game_config,
            setup_seed,
            (config.agent_a, config.agent_b),
            (seed_a, seed_b),
            (SEED_DERIVATION, SEED_DERIVATION),
        )
        yield GameSpec(
            config.run_id,
            f"{pair_id}-b-first",
            pair_id,
            config.game_config,
            setup_seed,
            (config.agent_b, config.agent_a),
            (seed_b, seed_a),
            (SEED_DERIVATION, SEED_DERIVATION),
        )


def run_arena(
    config: ArenaConfig,
    record_sink: Callable[[GameRecord], None] | None = None,
) -> ArenaReport:
    """Execute a balanced arena and aggregate statistics without retaining records."""
    wins = Counter[str]()
    terminal_reasons = Counter[str]()
    seat_games = Counter[PlayerId]()
    seat_wins = Counter[PlayerId]()
    pair_games = Counter[str]()
    pair_wins = Counter[str]()
    margin_total = 0
    turns_total = 0
    decisions_total = 0
    started = time.perf_counter()
    total = 0
    for record in iter_games(schedule_arena(config)):
        total += 1
        if record_sink is not None:
            record_sink(record)
        winner_id = record.seats[0 if record.winner is PlayerId.PLAYER_ONE else 1].agent_id
        wins[winner_id] += 1
        if record.pair_id is None:
            raise ValueError("paired arena record is missing pair_id")
        pair_games[record.pair_id] += 1
        if winner_id == config.agent_a.agent_id:
            pair_wins[record.pair_id] += 1
        terminal_reasons[record.terminal_reason] += 1
        a_seat = (
            PlayerId.PLAYER_ONE
            if record.seats[0].agent_id == config.agent_a.agent_id
            else PlayerId.PLAYER_TWO
        )
        seat_games[a_seat] += 1
        if winner_id == config.agent_a.agent_id:
            seat_wins[a_seat] += 1
        a_index = 0 if a_seat is PlayerId.PLAYER_ONE else 1
        margin_total += record.final_scores[a_index] - record.final_scores[1 - a_index]
        turns_total += record.turn_count
        decisions_total += record.decision_count
    elapsed = time.perf_counter() - started
    pair_ids = tuple(sorted(pair_games))
    if len(pair_ids) != config.pair_count or any(pair_games[pair_id] != 2 for pair_id in pair_ids):
        raise ValueError("arena did not produce exactly two games for every paired seed")
    paired_outcomes = tuple(PairedSeedOutcome(pair_id, pair_wins[pair_id]) for pair_id in pair_ids)
    paired_interval = paired_bootstrap_interval(
        tuple(outcome.agent_a_wins for outcome in paired_outcomes), config.master_seed
    )
    a_wins = wins[config.agent_a.agent_id]
    by_seat = {
        seat.value: SeatStats(
            seat_games[seat],
            seat_wins[seat],
            seat_wins[seat] / seat_games[seat],
        )
        for seat in PlayerId
    }
    from agent_avenue.engine.setup import normalize_config

    return ArenaReport(
        run_id=config.run_id,
        agent_a_id=config.agent_a.agent_id,
        agent_b_id=config.agent_b.agent_id,
        total_games=total,
        paired_seed_count=config.pair_count,
        agent_a_wins=a_wins,
        agent_b_wins=wins[config.agent_b.agent_id],
        agent_a_win_rate=a_wins / total,
        confidence_interval_95=wilson_interval(a_wins, total),
        paired_seed_outcomes=paired_outcomes,
        paired_bootstrap_confidence_interval_95=paired_interval,
        agent_a_by_seat=by_seat,
        terminal_reasons=dict(sorted(terminal_reasons.items())),
        average_score_margin=margin_total / total,
        average_turns=turns_total / total,
        average_decisions=decisions_total / total,
        elapsed_seconds=elapsed,
        games_per_second=total / elapsed if elapsed else math.inf,
        master_seed=config.master_seed,
        seed_derivation=SEED_DERIVATION,
        rng_algorithm=RNG_ALGORITHM,
        game_config=normalize_config(config.game_config),
        agent_a_config=json.loads(json.dumps(dict(config.agent_a.config), sort_keys=True)),
        agent_b_config=json.loads(json.dumps(dict(config.agent_b.config), sort_keys=True)),
        rules_fingerprint=rules_fingerprint(),
        code_fingerprint=code_fingerprint(),
    )

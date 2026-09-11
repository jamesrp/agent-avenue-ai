"""Frozen Step-2 population-corpus scheduling and behavior-policy composition.

This module deliberately stops at schedule declaration.  It does not collect games, train models,
or make experiment claims; callers can freeze its ``GameSpec`` values in the existing resumable
corpus machinery.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal

from agent_avenue.agents import (
    RNG_ALGORITHM,
    SEED_DERIVATION,
    Agent,
    DeterministicRandom,
    EpsilonConfig,
    EpsilonGreedyAgent,
    TerminalOffenseAgent,
    TerminalSafetyAgent,
    derive_seed,
)
from agent_avenue.agents.terminal_offense import (
    SELECTION_VERSION,
    TERMINAL_OFFENSE_VERSION,
    WIN_DEFINITION_VERSION,
)
from agent_avenue.agents.terminal_safety import (
    FALLBACK_VERSION,
    RESOLUTION_SCOPE,
    TERMINAL_SAFETY_VERSION,
    UNCERTAINTY_VERSION,
)
from agent_avenue.engine import GameConfig
from agent_avenue.engine.setup import normalize_config
from agent_avenue.engine.terminal import TERMINAL_EVALUATOR_VERSION
from agent_avenue.storage import GameRecord

from .game import AgentFactory, AgentSpec, GameSpec

POPULATION_REPLAY_CYCLE_ID: Final = "m7-population-replay-v1"
POPULATION_REPLAY_ROOT_SEED: Final = 2026091102
POPULATION_REPLAY_REPLICATE_IDS: Final[tuple[str, str, str]] = (
    "replicate-1",
    "replicate-2",
    "replicate-3",
)
POPULATION_CORPUS_CONFIG_VERSION: Final = "population-replay-corpus-config-v1"
POPULATION_ASSIGNMENT_VERSION: Final = "population-replay-assignment-v1"
POPULATION_PLAN_VERSION: Final = "population-replay-corpus-plan-v1"
POPULATION_AUDIT_VERSION: Final = "population-replay-schedule-audit-v1"
POPULATION_SETUP_HOLDOUT_VERSION: Final = "population-replay-training-setup-holdout-v1"
POPULATION_COMPOSITION_VERSION: Final = "terminal-offense-safety-epsilon-composition-v1"
POPULATION_PAIR_COUNT: Final = 2_000
POPULATION_GAMES_PER_ARM: Final = 4_000
POPULATION_LOGICAL_SLOTS_PER_ARM: Final = 4_000
POPULATION_EPSILON: Final = EpsilonConfig(1, 5)
POPULATION_POLICY_IDS: Final[tuple[str, str, str, str, str, str, str]] = (
    "q0",
    "q1",
    "q2",
    "q3",
    "q4",
    "greedy-public-v1",
    "random",
)

CORPUS_SEED_DOMAIN: Final = "corpus"
POPULATION_ASSIGNMENT_SEED_DOMAIN: Final = "population-assignment"
SETUP_SEED_DOMAIN: Final = "setup"
AGENT_LANE_SEED_DOMAIN: Final = "agent-lane"
SPLIT_SEED_DOMAIN: Final = "split"
INITIALIZATION_SEED_DOMAIN: Final = "initialization"
SHUFFLE_SEED_DOMAIN: Final = "shuffle"
ARENA_SEED_DOMAIN: Final = "arena"
BOOTSTRAP_SEED_DOMAIN: Final = "bootstrap"

PopulationArm = Literal["control", "treatment"]
LogicalLane = Literal["lane-a", "lane-b"]
LOGICAL_LANES: Final[tuple[LogicalLane, LogicalLane]] = ("lane-a", "lane-b")
ScheduledPopulationEntry = GameSpec | GameRecord


class PopulationScheduleError(ValueError):
    """Raised when the frozen population schedule or its evidence is malformed."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _copy_json_object(value: Mapping[str, object], *, label: str) -> dict[str, object]:
    copied = json.loads(_canonical_json(dict(value)))
    if not isinstance(copied, dict):  # pragma: no cover - protected by Mapping annotation
        raise PopulationScheduleError(f"{label} must be a JSON object")
    return copied


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_replicate_id(replicate_id: str) -> None:
    if replicate_id not in POPULATION_REPLAY_REPLICATE_IDS:
        raise PopulationScheduleError(f"unknown population replay replicate: {replicate_id!r}")


def _require_arm(arm: str) -> None:
    if arm not in {"control", "treatment"}:
        raise PopulationScheduleError(f"unknown population replay arm: {arm!r}")


def _require_lane(lane: str) -> None:
    if lane not in {"lane-a", "lane-b"}:
        raise PopulationScheduleError(f"unknown logical lane: {lane!r}")


@dataclass(frozen=True, slots=True)
class PolicySlotCount:
    """The number of logical policy slots allocated to one population member."""

    policy_id: str
    logical_slots: int

    def __post_init__(self) -> None:
        if not self.policy_id or type(self.logical_slots) is not int or self.logical_slots < 1:
            raise PopulationScheduleError(
                "policy slot counts require a policy id and positive count"
            )

    def to_data(self) -> dict[str, object]:
        return {"policy_id": self.policy_id, "logical_slots": self.logical_slots}


TREATMENT_LOGICAL_SLOT_COUNTS: Final[tuple[PolicySlotCount, ...]] = (
    PolicySlotCount("q0", 1_600),
    PolicySlotCount("q1", 400),
    PolicySlotCount("q2", 400),
    PolicySlotCount("q3", 400),
    PolicySlotCount("q4", 400),
    PolicySlotCount("greedy-public-v1", 400),
    PolicySlotCount("random", 400),
)
CONTROL_LOGICAL_SLOT_COUNTS: Final[tuple[PolicySlotCount, ...]] = (
    PolicySlotCount("q0", POPULATION_LOGICAL_SLOTS_PER_ARM),
)


@dataclass(frozen=True, slots=True)
class PopulationMember:
    """One named logical population member and its actual unwrapped base policy."""

    policy_id: str
    base_agent: AgentSpec

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise PopulationScheduleError("population member policy_id must be non-empty")

    def to_data(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "base_agent_id": self.base_agent.agent_id,
            "base_agent_config": _copy_json_object(
                self.base_agent.config, label="base agent configuration"
            ),
        }


def _unwrapped_base_config(base_config: Mapping[str, object]) -> dict[str, object]:
    """Copy a raw base-policy config and reject accidental historical wrapper nesting."""
    normalized = _copy_json_object(base_config, label="base policy configuration")
    if (
        normalized.get("type") in {"terminal_offense", "terminal_safety"}
        or normalized.get("version") == "epsilon-wrapper-v1"
    ):
        raise PopulationScheduleError(
            "population base policies must be raw; terminal offense, safety, and epsilon are "
            "composed exactly once"
        )
    return normalized


def compose_terminal_population_config(
    base_config: Mapping[str, object], *, epsilon: EpsilonConfig = POPULATION_EPSILON
) -> dict[str, object]:
    """Normalize ``TerminalOffense(TerminalSafety(EpsilonGreedy(base, 1/5)))`` metadata.

    The helper only manipulates already-recorded JSON configuration.  In particular, it never
    imports a learned-agent module or PyTorch, so learned factories can remain lazy.
    """
    if epsilon != POPULATION_EPSILON:
        raise PopulationScheduleError("population replay requires epsilon exactly 1/5")
    raw_base = _unwrapped_base_config(base_config)
    epsilon_config = {
        "version": "epsilon-wrapper-v1",
        "base": raw_base,
        "epsilon": epsilon.to_data(),
    }
    safety_config = {
        "type": "terminal_safety",
        "version": TERMINAL_SAFETY_VERSION,
        "base": epsilon_config,
        "fallback": FALLBACK_VERSION,
        "public_uncertainty": UNCERTAINTY_VERSION,
        "resolution_scope": RESOLUTION_SCOPE,
        "terminal_evaluator": TERMINAL_EVALUATOR_VERSION,
    }
    return {
        "type": "terminal_offense",
        "version": TERMINAL_OFFENSE_VERSION,
        "base": safety_config,
        "win_definition": WIN_DEFINITION_VERSION,
        "selection": SELECTION_VERSION,
    }


def terminal_population_agent_factory(
    base_factory: AgentFactory, *, epsilon: EpsilonConfig = POPULATION_EPSILON
) -> AgentFactory:
    """Return a lazy factory for the fixed population behavior-policy envelope."""
    if epsilon != POPULATION_EPSILON:
        raise PopulationScheduleError("population replay requires epsilon exactly 1/5")

    def factory() -> Agent:
        return TerminalOffenseAgent(
            TerminalSafetyAgent(EpsilonGreedyAgent(base_factory(), epsilon))
        )

    return factory


def compose_terminal_population_agent(
    base_agent: AgentSpec,
    *,
    rng_identity: str,
    epsilon: EpsilonConfig = POPULATION_EPSILON,
) -> AgentSpec:
    """Compose the fixed envelope around any raw learned, heuristic, or random base agent spec."""
    return AgentSpec(
        base_agent.agent_id,
        compose_terminal_population_config(base_agent.config, epsilon=epsilon),
        terminal_population_agent_factory(base_agent.factory, epsilon=epsilon),
        rng_identity=rng_identity,
    )


@dataclass(frozen=True, slots=True)
class PopulationCorpusConfig:
    """The immutable Step-2 corpus design, including real base-policy provenance."""

    members: tuple[PopulationMember, ...]
    run_id_prefix: str = POPULATION_REPLAY_CYCLE_ID
    game_config: GameConfig = field(default_factory=GameConfig)
    root_seed: int = POPULATION_REPLAY_ROOT_SEED
    cycle_id: str = POPULATION_REPLAY_CYCLE_ID
    replicate_ids: tuple[str, str, str] = POPULATION_REPLAY_REPLICATE_IDS
    pair_count: int = POPULATION_PAIR_COUNT
    epsilon: EpsilonConfig = POPULATION_EPSILON
    treatment_slot_counts: tuple[PolicySlotCount, ...] = TREATMENT_LOGICAL_SLOT_COUNTS

    def __post_init__(self) -> None:
        if self.cycle_id != POPULATION_REPLAY_CYCLE_ID:
            raise PopulationScheduleError("population replay cycle id is frozen")
        if type(self.root_seed) is not int or self.root_seed != POPULATION_REPLAY_ROOT_SEED:
            raise PopulationScheduleError("population replay root seed is frozen at 2026091102")
        if self.replicate_ids != POPULATION_REPLAY_REPLICATE_IDS:
            raise PopulationScheduleError("population replay replicate ids are frozen")
        if self.pair_count != POPULATION_PAIR_COUNT:
            raise PopulationScheduleError("population replay requires exactly 2,000 paired blocks")
        if self.epsilon != POPULATION_EPSILON:
            raise PopulationScheduleError("population replay epsilon is frozen at 1/5")
        if not self.run_id_prefix:
            raise PopulationScheduleError("population replay run_id_prefix must be non-empty")
        if not isinstance(self.game_config, GameConfig):
            raise PopulationScheduleError("population replay requires a GameConfig")
        member_ids = tuple(member.policy_id for member in self.members)
        if member_ids != POPULATION_POLICY_IDS:
            raise PopulationScheduleError(
                "population members must be q0, q1, q2, q3, q4, greedy-public-v1, random in order"
            )
        actual_agent_ids = tuple(member.base_agent.agent_id for member in self.members)
        if len(set(actual_agent_ids)) != len(actual_agent_ids):
            raise PopulationScheduleError(
                "population base agent ids must uniquely identify members"
            )
        if self.treatment_slot_counts != TREATMENT_LOGICAL_SLOT_COUNTS:
            raise PopulationScheduleError("population replay treatment weights/counts are frozen")
        if sum(item.logical_slots for item in self.treatment_slot_counts) != 2 * self.pair_count:
            raise PopulationScheduleError("population treatment logical slot total is incorrect")
        for member in self.members:
            compose_terminal_population_config(member.base_agent.config, epsilon=self.epsilon)

    @property
    def member_by_id(self) -> dict[str, PopulationMember]:
        return {member.policy_id: member for member in self.members}

    @property
    def member_by_agent_id(self) -> dict[str, PopulationMember]:
        return {member.base_agent.agent_id: member for member in self.members}

    def run_id(self, replicate_id: str, arm: PopulationArm) -> str:
        _require_replicate_id(replicate_id)
        _require_arm(arm)
        return f"{self.run_id_prefix}-{replicate_id}-{arm}"

    def to_data(self) -> dict[str, object]:
        return {
            "version": POPULATION_CORPUS_CONFIG_VERSION,
            "cycle_id": self.cycle_id,
            "root_seed": self.root_seed,
            "seed_derivation": SEED_DERIVATION,
            "rng_algorithm": RNG_ALGORITHM,
            "replicate_ids": list(self.replicate_ids),
            "run_id_prefix": self.run_id_prefix,
            "pair_count_per_arm": self.pair_count,
            "games_per_arm": POPULATION_GAMES_PER_ARM,
            "epsilon": self.epsilon.to_data(),
            "behavior_policy_composition": {
                "version": POPULATION_COMPOSITION_VERSION,
                "expression": "TerminalOffense(TerminalSafety(EpsilonGreedy(base, 1/5)))",
            },
            "game_config": normalize_config(self.game_config),
            "members": [member.to_data() for member in self.members],
            "behavior_policy_configs": {
                member.policy_id: {
                    "agent_id": member.base_agent.agent_id,
                    "config": compose_terminal_population_config(
                        member.base_agent.config, epsilon=self.epsilon
                    ),
                }
                for member in self.members
            },
            "treatment_logical_slot_counts": [
                item.to_data() for item in self.treatment_slot_counts
            ],
            "control_logical_slot_counts": [item.to_data() for item in CONTROL_LOGICAL_SLOT_COUNTS],
        }


@dataclass(frozen=True, slots=True)
class PopulationPairSeeds:
    """The arm- and policy-independent random material for one physical paired block."""

    pair_index: int
    pair_id: str
    setup_seed: int
    lane_a_seed: int
    lane_b_seed: int

    def __post_init__(self) -> None:
        if self.pair_index < 0 or not self.pair_id:
            raise PopulationScheduleError("pair seeds require a non-negative index and pair id")
        if any(
            type(seed) is not int for seed in (self.setup_seed, self.lane_a_seed, self.lane_b_seed)
        ):
            raise PopulationScheduleError("population pair seeds must be integers")
        if self.lane_a_seed == self.lane_b_seed:
            raise PopulationScheduleError("logical lanes must have independent RNG seeds")

    def to_data(self, replicate_id: str) -> dict[str, object]:
        return {
            "pair_index": self.pair_index,
            "pair_id": self.pair_id,
            "setup_seed": self.setup_seed,
            "setup_seed_domain": population_seed_domain(
                replicate_id, SETUP_SEED_DOMAIN, pair_index=self.pair_index
            ),
            "lane_a_seed": self.lane_a_seed,
            "lane_a_seed_domain": population_seed_domain(
                replicate_id,
                AGENT_LANE_SEED_DOMAIN,
                pair_index=self.pair_index,
                lane="lane-a",
            ),
            "lane_b_seed": self.lane_b_seed,
            "lane_b_seed_domain": population_seed_domain(
                replicate_id,
                AGENT_LANE_SEED_DOMAIN,
                pair_index=self.pair_index,
                lane="lane-b",
            ),
        }


def population_seed_domain(
    replicate_id: str,
    purpose: str,
    *,
    pair_index: int | None = None,
    lane: LogicalLane | None = None,
) -> str:
    """Return one named ``sha256-domain-v1`` domain from the frozen Step-2 namespace."""
    _require_replicate_id(replicate_id)
    scalar_purposes = {
        CORPUS_SEED_DOMAIN,
        POPULATION_ASSIGNMENT_SEED_DOMAIN,
        SPLIT_SEED_DOMAIN,
        INITIALIZATION_SEED_DOMAIN,
        SHUFFLE_SEED_DOMAIN,
        ARENA_SEED_DOMAIN,
        BOOTSTRAP_SEED_DOMAIN,
    }
    if purpose in scalar_purposes:
        if pair_index is not None or lane is not None:
            raise PopulationScheduleError(
                f"{purpose} seed domain does not take pair/lane arguments"
            )
        return f"{POPULATION_REPLAY_CYCLE_ID}:replicate:{replicate_id}:{purpose}:v1"
    if purpose == SETUP_SEED_DOMAIN:
        if type(pair_index) is not int or pair_index < 0 or lane is not None:
            raise PopulationScheduleError("setup seed domain requires one non-negative pair index")
        return (
            f"{POPULATION_REPLAY_CYCLE_ID}:replicate:{replicate_id}:setup:pair:{pair_index:06d}:v1"
        )
    if purpose == AGENT_LANE_SEED_DOMAIN:
        if type(pair_index) is not int or pair_index < 0 or lane is None:
            raise PopulationScheduleError("agent-lane seed domain requires pair index and lane")
        _require_lane(lane)
        return (
            f"{POPULATION_REPLAY_CYCLE_ID}:replicate:{replicate_id}:agent-lane:"
            f"pair:{pair_index:06d}:{lane}:v1"
        )
    raise PopulationScheduleError(f"unknown population replay seed domain purpose: {purpose!r}")


def population_lane_rng_identity(replicate_id: str, lane: LogicalLane) -> str:
    """Return the arm-independent runtime RNG identity for one logical policy lane."""
    _require_replicate_id(replicate_id)
    _require_lane(lane)
    return f"{POPULATION_REPLAY_CYCLE_ID}:replicate:{replicate_id}:logical-{lane}"


@dataclass(frozen=True, slots=True)
class PopulationReplicateSeedPlan:
    """All named seeds needed by a Step-2 replicate, including future runner-owned stages."""

    replicate_id: str
    corpus_seed: int
    assignment_seed: int
    split_seed: int
    initialization_seed: int
    shuffle_seed: int
    arena_seed: int
    bootstrap_seed: int
    pair_seeds: tuple[PopulationPairSeeds, ...]

    def __post_init__(self) -> None:
        _require_replicate_id(self.replicate_id)
        scalars = (
            self.corpus_seed,
            self.assignment_seed,
            self.split_seed,
            self.initialization_seed,
            self.shuffle_seed,
            self.arena_seed,
            self.bootstrap_seed,
        )
        if any(type(seed) is not int for seed in scalars):
            raise PopulationScheduleError(
                "population replicate seed plan contains a non-integer seed"
            )
        if len(self.pair_seeds) != POPULATION_PAIR_COUNT:
            raise PopulationScheduleError(
                "population replicate seed plan must materialize 2,000 pairs"
            )
        if tuple(pair.pair_index for pair in self.pair_seeds) != tuple(
            range(POPULATION_PAIR_COUNT)
        ):
            raise PopulationScheduleError("population replicate pair seed indexes are not complete")
        if len({pair.pair_id for pair in self.pair_seeds}) != POPULATION_PAIR_COUNT:
            raise PopulationScheduleError("population replicate pair ids are not unique")
        if len({pair.setup_seed for pair in self.pair_seeds}) != POPULATION_PAIR_COUNT:
            raise PopulationScheduleError("population replicate setup seeds are not unique")

    @property
    def lane_rng_identities(self) -> tuple[str, str]:
        return (
            population_lane_rng_identity(self.replicate_id, "lane-a"),
            population_lane_rng_identity(self.replicate_id, "lane-b"),
        )

    def to_data(self) -> dict[str, object]:
        scalar_domains = (
            ("corpus_seed", CORPUS_SEED_DOMAIN),
            ("assignment_seed", POPULATION_ASSIGNMENT_SEED_DOMAIN),
            ("split_seed", SPLIT_SEED_DOMAIN),
            ("initialization_seed", INITIALIZATION_SEED_DOMAIN),
            ("shuffle_seed", SHUFFLE_SEED_DOMAIN),
            ("arena_seed", ARENA_SEED_DOMAIN),
            ("bootstrap_seed", BOOTSTRAP_SEED_DOMAIN),
        )
        return {
            "replicate_id": self.replicate_id,
            "scalar_seeds": {
                name: {
                    "seed": getattr(self, name),
                    "domain": population_seed_domain(self.replicate_id, purpose),
                }
                for name, purpose in scalar_domains
            },
            "lane_rng_identities": {
                "lane_a": self.lane_rng_identities[0],
                "lane_b": self.lane_rng_identities[1],
            },
            "pair_seeds": [pair.to_data(self.replicate_id) for pair in self.pair_seeds],
        }


def materialize_population_replicate_seeds(
    config: PopulationCorpusConfig, replicate_id: str
) -> PopulationReplicateSeedPlan:
    """Derive and retain every named seed for one frozen population replay replicate."""
    _require_replicate_id(replicate_id)

    def scalar(purpose: str) -> int:
        return derive_seed(config.root_seed, population_seed_domain(replicate_id, purpose))

    pair_seeds = tuple(
        PopulationPairSeeds(
            pair_index=index,
            pair_id=f"{replicate_id}-pair-{index:06d}",
            setup_seed=derive_seed(
                config.root_seed,
                population_seed_domain(replicate_id, SETUP_SEED_DOMAIN, pair_index=index),
            )
            & ((1 << 64) - 1),
            lane_a_seed=derive_seed(
                config.root_seed,
                population_seed_domain(
                    replicate_id, AGENT_LANE_SEED_DOMAIN, pair_index=index, lane="lane-a"
                ),
            ),
            lane_b_seed=derive_seed(
                config.root_seed,
                population_seed_domain(
                    replicate_id, AGENT_LANE_SEED_DOMAIN, pair_index=index, lane="lane-b"
                ),
            ),
        )
        for index in range(config.pair_count)
    )
    return PopulationReplicateSeedPlan(
        replicate_id=replicate_id,
        corpus_seed=scalar(CORPUS_SEED_DOMAIN),
        assignment_seed=scalar(POPULATION_ASSIGNMENT_SEED_DOMAIN),
        split_seed=scalar(SPLIT_SEED_DOMAIN),
        initialization_seed=scalar(INITIALIZATION_SEED_DOMAIN),
        shuffle_seed=scalar(SHUFFLE_SEED_DOMAIN),
        arena_seed=scalar(ARENA_SEED_DOMAIN),
        bootstrap_seed=scalar(BOOTSTRAP_SEED_DOMAIN),
        pair_seeds=pair_seeds,
    )


@dataclass(frozen=True, slots=True)
class LogicalMatchup:
    """The two logical population slots paired in one paired setup block."""

    pair_index: int
    pair_id: str
    lane_a_policy_id: str
    lane_b_policy_id: str

    def __post_init__(self) -> None:
        if self.pair_index < 0 or not self.pair_id:
            raise PopulationScheduleError("logical matchup requires pair index and pair id")
        if self.lane_a_policy_id not in POPULATION_POLICY_IDS:
            raise PopulationScheduleError("logical matchup contains an unknown lane-a policy")
        if self.lane_b_policy_id not in POPULATION_POLICY_IDS:
            raise PopulationScheduleError("logical matchup contains an unknown lane-b policy")

    def to_data(self) -> dict[str, object]:
        return {
            "pair_index": self.pair_index,
            "pair_id": self.pair_id,
            "lane_a_policy_id": self.lane_a_policy_id,
            "lane_b_policy_id": self.lane_b_policy_id,
        }


@dataclass(frozen=True, slots=True)
class PopulationAssignment:
    """A complete ordered logical-slot assignment for one arm and replicate."""

    arm: PopulationArm
    replicate_id: str
    matchups: tuple[LogicalMatchup, ...]
    slot_counts: tuple[PolicySlotCount, ...]
    assignment_seed: int | None
    assignment_rng_domain: str | None
    version: str = POPULATION_ASSIGNMENT_VERSION

    def __post_init__(self) -> None:
        _require_arm(self.arm)
        _require_replicate_id(self.replicate_id)
        if self.version != POPULATION_ASSIGNMENT_VERSION:
            raise PopulationScheduleError("unsupported population assignment version")
        if len(self.matchups) != POPULATION_PAIR_COUNT:
            raise PopulationScheduleError(
                "population assignment must contain exactly 2,000 matchups"
            )
        if tuple(item.pair_index for item in self.matchups) != tuple(range(POPULATION_PAIR_COUNT)):
            raise PopulationScheduleError("population assignment matchup indexes are incomplete")
        if len({item.pair_id for item in self.matchups}) != POPULATION_PAIR_COUNT:
            raise PopulationScheduleError("population assignment pair ids are not unique")
        expected_counts = (
            CONTROL_LOGICAL_SLOT_COUNTS if self.arm == "control" else TREATMENT_LOGICAL_SLOT_COUNTS
        )
        if self.slot_counts != expected_counts:
            raise PopulationScheduleError(
                "population assignment slot counts do not match the frozen arm"
            )
        actual = Counter(
            policy_id
            for matchup in self.matchups
            for policy_id in (matchup.lane_a_policy_id, matchup.lane_b_policy_id)
        )
        expected = {item.policy_id: item.logical_slots for item in self.slot_counts}
        if dict(actual) != expected:
            raise PopulationScheduleError(
                "population assignment logical slots do not match its counts"
            )
        if self.arm == "control":
            if self.assignment_seed is not None or self.assignment_rng_domain is not None:
                raise PopulationScheduleError(
                    "q0-only control assignment must not consume assignment RNG"
                )
            if any(
                matchup.lane_a_policy_id != "q0" or matchup.lane_b_policy_id != "q0"
                for matchup in self.matchups
            ):
                raise PopulationScheduleError(
                    "q0-only control assignment must use q0 in both lanes"
                )
        elif type(
            self.assignment_seed
        ) is not int or self.assignment_rng_domain != population_seed_domain(
            self.replicate_id, POPULATION_ASSIGNMENT_SEED_DOMAIN
        ):
            raise PopulationScheduleError(
                "treatment assignment requires its named deterministic assignment seed"
            )

    @property
    def logical_slots(self) -> tuple[str, ...]:
        return tuple(
            policy_id
            for matchup in self.matchups
            for policy_id in (matchup.lane_a_policy_id, matchup.lane_b_policy_id)
        )

    @property
    def fingerprint(self) -> str:
        return _sha256(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "version": self.version,
            "arm": self.arm,
            "replicate_id": self.replicate_id,
            "assignment_seed": self.assignment_seed,
            "assignment_rng_domain": self.assignment_rng_domain,
            "rng_algorithm": RNG_ALGORITHM if self.arm == "treatment" else None,
            "seed_derivation": SEED_DERIVATION if self.arm == "treatment" else None,
            "logical_slot_counts": [item.to_data() for item in self.slot_counts],
            "matchups": [matchup.to_data() for matchup in self.matchups],
        }


def _control_assignment(seed_plan: PopulationReplicateSeedPlan) -> PopulationAssignment:
    return PopulationAssignment(
        arm="control",
        replicate_id=seed_plan.replicate_id,
        matchups=tuple(
            LogicalMatchup(pair.pair_index, pair.pair_id, "q0", "q0")
            for pair in seed_plan.pair_seeds
        ),
        slot_counts=CONTROL_LOGICAL_SLOT_COUNTS,
        assignment_seed=None,
        assignment_rng_domain=None,
    )


def _treatment_assignment(seed_plan: PopulationReplicateSeedPlan) -> PopulationAssignment:
    slots = [
        slot_count.policy_id
        for slot_count in TREATMENT_LOGICAL_SLOT_COUNTS
        for _ in range(slot_count.logical_slots)
    ]
    domain = population_seed_domain(seed_plan.replicate_id, POPULATION_ASSIGNMENT_SEED_DOMAIN)
    rng = DeterministicRandom(seed_plan.assignment_seed, domain)
    for index in range(len(slots) - 1, 0, -1):
        other = rng.randbelow(index + 1)
        slots[index], slots[other] = slots[other], slots[index]
    matchups = tuple(
        LogicalMatchup(
            pair.pair_index,
            pair.pair_id,
            slots[pair.pair_index * 2],
            slots[pair.pair_index * 2 + 1],
        )
        for pair in seed_plan.pair_seeds
    )
    return PopulationAssignment(
        arm="treatment",
        replicate_id=seed_plan.replicate_id,
        matchups=matchups,
        slot_counts=TREATMENT_LOGICAL_SLOT_COUNTS,
        assignment_seed=seed_plan.assignment_seed,
        assignment_rng_domain=domain,
    )


@dataclass(frozen=True, slots=True)
class PopulationLaneAgent:
    """One actual wrapped member specification assigned to a stable logical RNG lane."""

    policy_id: str
    lane: LogicalLane
    agent: AgentSpec

    def __post_init__(self) -> None:
        if self.policy_id not in POPULATION_POLICY_IDS:
            raise PopulationScheduleError("lane agent has an unknown population policy")
        _require_lane(self.lane)


@dataclass(frozen=True, slots=True)
class PopulationReplicatePlan:
    """Complete two-arm paired schedule material for one fixed replicate."""

    seed_plan: PopulationReplicateSeedPlan
    control_assignment: PopulationAssignment
    treatment_assignment: PopulationAssignment
    lane_agents: tuple[PopulationLaneAgent, ...]
    game_config: GameConfig
    control_run_id: str
    treatment_run_id: str

    def __post_init__(self) -> None:
        replicate_id = self.seed_plan.replicate_id
        if self.control_assignment.arm != "control":
            raise PopulationScheduleError(
                "replicate plan control assignment must be the control arm"
            )
        if self.treatment_assignment.arm != "treatment":
            raise PopulationScheduleError(
                "replicate plan treatment assignment must be the treatment arm"
            )
        if self.control_assignment.replicate_id != replicate_id:
            raise PopulationScheduleError(
                "control assignment does not belong to its seed replicate"
            )
        if self.treatment_assignment.replicate_id != replicate_id:
            raise PopulationScheduleError(
                "treatment assignment does not belong to its seed replicate"
            )
        if tuple(item.pair_id for item in self.control_assignment.matchups) != tuple(
            pair.pair_id for pair in self.seed_plan.pair_seeds
        ):
            raise PopulationScheduleError("control assignment pair ids do not match the seed plan")
        if tuple(item.pair_id for item in self.treatment_assignment.matchups) != tuple(
            pair.pair_id for pair in self.seed_plan.pair_seeds
        ):
            raise PopulationScheduleError(
                "treatment assignment pair ids do not match the seed plan"
            )
        expected = {
            (policy_id, lane)
            for policy_id in POPULATION_POLICY_IDS
            for lane in ("lane-a", "lane-b")
        }
        actual = {(item.policy_id, item.lane) for item in self.lane_agents}
        if actual != expected or len(self.lane_agents) != len(expected):
            raise PopulationScheduleError("replicate plan requires one agent spec per policy/lane")
        expected_identities = {
            population_lane_rng_identity(replicate_id, "lane-a"),
            population_lane_rng_identity(replicate_id, "lane-b"),
        }
        if {item.agent.rng_identity for item in self.lane_agents} != expected_identities:
            raise PopulationScheduleError(
                "replicate lane agents do not use the declared RNG identities"
            )
        if not self.control_run_id or not self.treatment_run_id:
            raise PopulationScheduleError(
                "replicate plan requires stable control/treatment run ids"
            )

    @property
    def replicate_id(self) -> str:
        return self.seed_plan.replicate_id

    def assignment(self, arm: PopulationArm) -> PopulationAssignment:
        _require_arm(arm)
        return self.control_assignment if arm == "control" else self.treatment_assignment

    def run_id(self, arm: PopulationArm) -> str:
        _require_arm(arm)
        return self.control_run_id if arm == "control" else self.treatment_run_id

    def agent_for(self, policy_id: str, lane: LogicalLane) -> AgentSpec:
        matches = tuple(
            item.agent
            for item in self.lane_agents
            if item.policy_id == policy_id and item.lane == lane
        )
        if len(matches) != 1:  # pragma: no cover - protected by __post_init__
            raise PopulationScheduleError("replicate plan has no unique logical lane agent")
        return matches[0]

    def game_specs(self, arm: PopulationArm) -> tuple[GameSpec, ...]:
        """Materialize both physical seat-swapped games for every logical paired matchup."""
        assignment = self.assignment(arm)
        games: list[GameSpec] = []
        for pair_seed, matchup in zip(self.seed_plan.pair_seeds, assignment.matchups, strict=True):
            lane_a = self.agent_for(matchup.lane_a_policy_id, "lane-a")
            lane_b = self.agent_for(matchup.lane_b_policy_id, "lane-b")
            games.append(
                GameSpec(
                    self.run_id(arm),
                    f"{pair_seed.pair_id}-lane-a-first",
                    pair_seed.pair_id,
                    self.game_config,
                    pair_seed.setup_seed,
                    (lane_a, lane_b),
                    (pair_seed.lane_a_seed, pair_seed.lane_b_seed),
                    (SEED_DERIVATION, SEED_DERIVATION),
                )
            )
            games.append(
                GameSpec(
                    self.run_id(arm),
                    f"{pair_seed.pair_id}-lane-b-first",
                    pair_seed.pair_id,
                    self.game_config,
                    pair_seed.setup_seed,
                    (lane_b, lane_a),
                    (pair_seed.lane_b_seed, pair_seed.lane_a_seed),
                    (SEED_DERIVATION, SEED_DERIVATION),
                )
            )
        return tuple(games)

    def to_data(self) -> dict[str, object]:
        return {
            "replicate_id": self.replicate_id,
            "run_ids": {"control": self.control_run_id, "treatment": self.treatment_run_id},
            "seeds": self.seed_plan.to_data(),
            "assignments": {
                "control": self.control_assignment.to_data(),
                "treatment": self.treatment_assignment.to_data(),
            },
            "game_id_pattern": "{replicate_id}-pair-{pair_index:06d}-lane-{a|b}-first",
            "seat_swap": "logical-lane-a-first-and-logical-lane-b-first-v1",
        }


@dataclass(frozen=True, slots=True)
class PopulationCorpusPlan:
    """All three frozen corpus replicates, with complete assignment and seed material."""

    config: PopulationCorpusConfig
    replicates: tuple[PopulationReplicatePlan, ...]
    version: str = POPULATION_PLAN_VERSION

    def __post_init__(self) -> None:
        if self.version != POPULATION_PLAN_VERSION:
            raise PopulationScheduleError("unsupported population corpus plan version")
        if tuple(item.replicate_id for item in self.replicates) != self.config.replicate_ids:
            raise PopulationScheduleError(
                "population corpus plan does not contain frozen replicates"
            )
        all_setup_seeds = [
            pair.setup_seed
            for replicate in self.replicates
            for pair in replicate.seed_plan.pair_seeds
        ]
        if len(set(all_setup_seeds)) != len(all_setup_seeds):
            raise PopulationScheduleError("population replay replicate setup families overlap")

    @property
    def fingerprint(self) -> str:
        return _sha256(self.to_data())

    @property
    def assignment_fingerprint(self) -> str:
        return _sha256(
            [
                {
                    "replicate_id": replicate.replicate_id,
                    "control": replicate.control_assignment.fingerprint,
                    "treatment": replicate.treatment_assignment.fingerprint,
                }
                for replicate in self.replicates
            ]
        )

    def replicate(self, replicate_id: str) -> PopulationReplicatePlan:
        _require_replicate_id(replicate_id)
        matches = tuple(item for item in self.replicates if item.replicate_id == replicate_id)
        if len(matches) != 1:  # pragma: no cover - protected by __post_init__
            raise PopulationScheduleError("population corpus plan has no unique replicate")
        return matches[0]

    def game_specs(self, replicate_id: str, arm: PopulationArm) -> tuple[GameSpec, ...]:
        return self.replicate(replicate_id).game_specs(arm)

    def all_game_specs(self, arm: PopulationArm) -> tuple[GameSpec, ...]:
        _require_arm(arm)
        return tuple(spec for replicate in self.replicates for spec in replicate.game_specs(arm))

    def to_data(self) -> dict[str, object]:
        return {
            "version": self.version,
            "configuration": self.config.to_data(),
            "replicates": [replicate.to_data() for replicate in self.replicates],
            "assignment_fingerprint": self.assignment_fingerprint,
        }


def build_population_corpus_plan(config: PopulationCorpusConfig) -> PopulationCorpusPlan:
    """Build the exact three-replicate Step-2 corpus declaration without running any games."""
    replicates: list[PopulationReplicatePlan] = []
    for replicate_id in config.replicate_ids:
        seeds = materialize_population_replicate_seeds(config, replicate_id)
        lane_agents = tuple(
            PopulationLaneAgent(
                member.policy_id,
                lane,
                compose_terminal_population_agent(
                    member.base_agent,
                    rng_identity=population_lane_rng_identity(replicate_id, lane),
                    epsilon=config.epsilon,
                ),
            )
            for member in config.members
            for lane in LOGICAL_LANES
        )
        replicates.append(
            PopulationReplicatePlan(
                seed_plan=seeds,
                control_assignment=_control_assignment(seeds),
                treatment_assignment=_treatment_assignment(seeds),
                lane_agents=lane_agents,
                game_config=config.game_config,
                control_run_id=config.run_id(replicate_id, "control"),
                treatment_run_id=config.run_id(replicate_id, "treatment"),
            )
        )
    return PopulationCorpusPlan(config, tuple(replicates))


def schedule_population_corpus(
    plan: PopulationCorpusPlan, replicate_id: str, arm: PopulationArm
) -> tuple[GameSpec, ...]:
    """Return one complete arm/replicate schedule suitable for ``corpus_declaration``."""
    return plan.game_specs(replicate_id, arm)


@dataclass(frozen=True, slots=True)
class PopulationSetupBlock:
    """One reconstructed paired setup block from a scheduled arm or completed records."""

    replicate_id: str
    pair_index: int
    pair_id: str
    setup_seed: int

    def to_data(self) -> dict[str, object]:
        return {
            "replicate_id": self.replicate_id,
            "pair_index": self.pair_index,
            "pair_id": self.pair_id,
            "setup_seed": self.setup_seed,
        }


@dataclass(frozen=True, slots=True)
class SetupHoldoutCandidate:
    """A unique training setup identity that a later dispatcher must exclude from prior corpora."""

    replicate_id: str
    pair_index: int
    pair_id: str
    setup_seed: int
    game_config: Mapping[str, object]
    setup_identity: str
    version: str = POPULATION_SETUP_HOLDOUT_VERSION

    def __post_init__(self) -> None:
        _require_replicate_id(self.replicate_id)
        if self.pair_index < 0 or not self.pair_id or type(self.setup_seed) is not int:
            raise PopulationScheduleError(
                "setup holdout candidate has invalid pair or setup metadata"
            )
        normalized_config = _copy_json_object(self.game_config, label="setup holdout game config")
        immutable_config: Mapping[str, object] = MappingProxyType(normalized_config)
        expected_identity = population_setup_identity(immutable_config, self.setup_seed)
        if self.setup_identity != expected_identity:
            raise PopulationScheduleError(
                "setup holdout candidate identity does not match its setup"
            )
        object.__setattr__(self, "game_config", immutable_config)

    def to_data(self) -> dict[str, object]:
        return {
            "version": self.version,
            "replicate_id": self.replicate_id,
            "pair_index": self.pair_index,
            "pair_id": self.pair_id,
            "setup_seed": self.setup_seed,
            "game_config": _copy_json_object(self.game_config, label="setup holdout game config"),
            "setup_identity": self.setup_identity,
            "corpus_arms": ["control", "treatment"],
        }


def population_setup_identity(game_config: Mapping[str, object], setup_seed: int) -> str:
    """Fingerprint normalized game configuration plus setup seed, independent of behavior policy."""
    if type(setup_seed) is not int:
        raise PopulationScheduleError("setup identity requires an integer setup seed")
    return _sha256(
        {
            "game_config": _copy_json_object(game_config, label="setup identity game config"),
            "setup_seed": setup_seed,
        }
    )


def enumerate_population_training_setup_blocks(
    plan: PopulationCorpusPlan,
) -> tuple[SetupHoldoutCandidate, ...]:
    """Enumerate all 6,000 unique Step-2 training setup blocks before any dispatch occurs."""
    normalized_game_config = normalize_config(plan.config.game_config)
    candidates = tuple(
        SetupHoldoutCandidate(
            replicate_id=replicate.replicate_id,
            pair_index=pair.pair_index,
            pair_id=pair.pair_id,
            setup_seed=pair.setup_seed,
            game_config=normalized_game_config,
            setup_identity=population_setup_identity(normalized_game_config, pair.setup_seed),
        )
        for replicate in plan.replicates
        for pair in replicate.seed_plan.pair_seeds
    )
    if len(candidates) != len({candidate.setup_identity for candidate in candidates}):
        raise PopulationScheduleError("population training setup holdout candidates overlap")
    return candidates


@dataclass(frozen=True, slots=True)
class PopulationMatchupCount:
    """Count of unordered logical policy matchups in one reconstructed assignment."""

    policy_a: str
    policy_b: str
    paired_blocks: int

    def to_data(self) -> dict[str, object]:
        return {
            "policy_a": self.policy_a,
            "policy_b": self.policy_b,
            "paired_blocks": self.paired_blocks,
        }


@dataclass(frozen=True, slots=True)
class PopulationArmAudit:
    """Reconstructed schedule facts from one arm's ``GameSpec`` values or completed records."""

    replicate_id: str
    arm: PopulationArm
    assignment: PopulationAssignment
    setup_blocks: tuple[PopulationSetupBlock, ...]
    player_one_marginals: tuple[PolicySlotCount, ...]
    player_two_marginals: tuple[PolicySlotCount, ...]
    matchup_counts: tuple[PopulationMatchupCount, ...]
    version: str = POPULATION_AUDIT_VERSION

    def to_data(self) -> dict[str, object]:
        return {
            "version": self.version,
            "replicate_id": self.replicate_id,
            "arm": self.arm,
            "assignment": self.assignment.to_data(),
            "setup_blocks": [block.to_data() for block in self.setup_blocks],
            "per_seat_marginals": {
                "player_one": [item.to_data() for item in self.player_one_marginals],
                "player_two": [item.to_data() for item in self.player_two_marginals],
            },
            "matchup_counts": [item.to_data() for item in self.matchup_counts],
        }


@dataclass(frozen=True, slots=True)
class PopulationAlignmentAudit:
    """Arm-independent setup/lane alignment reconstructed from control and treatment evidence."""

    paired_block_count: int
    game_count_per_arm: int
    shared_setup_blocks: tuple[PopulationSetupBlock, ...]
    version: str = POPULATION_AUDIT_VERSION

    def to_data(self) -> dict[str, object]:
        return {
            "version": self.version,
            "paired_block_count": self.paired_block_count,
            "game_count_per_arm": self.game_count_per_arm,
            "shared_setup_blocks": [block.to_data() for block in self.shared_setup_blocks],
            "status": "passed",
        }


@dataclass(frozen=True, slots=True)
class _ObservedSeat:
    agent_id: str
    config: Mapping[str, object]
    seed: int
    rng_identity: str


@dataclass(frozen=True, slots=True)
class _ObservedGame:
    run_id: str
    game_id: str
    pair_id: str | None
    game_config: Mapping[str, object]
    setup_seed: int
    seats: tuple[_ObservedSeat, _ObservedSeat]


def _observed_game(entry: ScheduledPopulationEntry) -> _ObservedGame:
    if isinstance(entry, GameSpec):
        return _ObservedGame(
            entry.run_id,
            entry.game_id,
            entry.pair_id,
            normalize_config(entry.config),
            entry.setup_seed,
            (
                _ObservedSeat(
                    entry.seats[0].agent_id,
                    entry.seats[0].config,
                    entry.agent_seeds[0],
                    entry.seats[0].rng_identity,
                ),
                _ObservedSeat(
                    entry.seats[1].agent_id,
                    entry.seats[1].config,
                    entry.agent_seeds[1],
                    entry.seats[1].rng_identity,
                ),
            ),
        )
    return _ObservedGame(
        entry.run_id,
        entry.game_id,
        entry.pair_id,
        normalize_config(entry.replay.config),
        entry.replay.seed,
        (
            _ObservedSeat(
                entry.seats[0].agent_id,
                entry.seats[0].config,
                entry.seats[0].seed,
                entry.seats[0].rng_identity,
            ),
            _ObservedSeat(
                entry.seats[1].agent_id,
                entry.seats[1].config,
                entry.seats[1].seed,
                entry.seats[1].rng_identity,
            ),
        ),
    )


def _same_observed_metadata(expected: GameSpec, observed: _ObservedGame) -> bool:
    return (
        expected.run_id == observed.run_id
        and expected.game_id == observed.game_id
        and expected.pair_id == observed.pair_id
        and normalize_config(expected.config) == dict(observed.game_config)
        and expected.setup_seed == observed.setup_seed
        and all(
            expected.seats[index].agent_id == observed.seats[index].agent_id
            and dict(expected.seats[index].config) == dict(observed.seats[index].config)
            and expected.agent_seeds[index] == observed.seats[index].seed
            and expected.seats[index].rng_identity == observed.seats[index].rng_identity
            for index in range(2)
        )
    )


def _policy_id(config: PopulationCorpusConfig, seat: _ObservedSeat) -> str:
    member = config.member_by_agent_id.get(seat.agent_id)
    if member is None:
        raise PopulationScheduleError(
            "scheduled evidence contains a base agent outside the population"
        )
    return member.policy_id


def _marginal_slot_counts(counts: Counter[str]) -> tuple[PolicySlotCount, ...]:
    return tuple(
        PolicySlotCount(policy_id, counts[policy_id])
        for policy_id in POPULATION_POLICY_IDS
        if counts[policy_id]
    )


def _matchup_counts(matchups: Iterable[LogicalMatchup]) -> tuple[PopulationMatchupCount, ...]:
    ordering = {policy_id: index for index, policy_id in enumerate(POPULATION_POLICY_IDS)}
    counts: Counter[tuple[str, str]] = Counter()
    for matchup in matchups:
        first, second = sorted(
            (matchup.lane_a_policy_id, matchup.lane_b_policy_id), key=lambda item: ordering[item]
        )
        counts[(first, second)] += 1
    return tuple(
        PopulationMatchupCount(first, second, counts[(first, second)])
        for first, second in sorted(counts, key=lambda item: (ordering[item[0]], ordering[item[1]]))
    )


def audit_population_arm(
    plan: PopulationCorpusPlan,
    replicate_id: str,
    arm: PopulationArm,
    entries: Iterable[ScheduledPopulationEntry],
) -> PopulationArmAudit:
    """Reconstruct and validate one arm's assignment, setup blocks, marginals, and matchups."""
    _require_arm(arm)
    replicate = plan.replicate(replicate_id)
    expected_specs = replicate.game_specs(arm)
    observed = tuple(_observed_game(entry) for entry in entries)
    expected_ids = tuple(spec.game_id for spec in expected_specs)
    by_game_id = {item.game_id: item for item in observed}
    if len(observed) != len(expected_specs) or len(by_game_id) != len(observed):
        raise PopulationScheduleError(
            "population arm evidence has an incomplete or duplicate game set"
        )
    if set(by_game_id) != set(expected_ids):
        raise PopulationScheduleError("population arm evidence game ids do not match its schedule")
    for expected in expected_specs:
        if not _same_observed_metadata(expected, by_game_id[expected.game_id]):
            raise PopulationScheduleError(
                "population arm evidence does not match its declared schedule"
            )

    lane_a_identity, lane_b_identity = replicate.seed_plan.lane_rng_identities
    reconstructed: list[LogicalMatchup] = []
    setup_blocks: list[PopulationSetupBlock] = []
    player_one_counts: Counter[str] = Counter()
    player_two_counts: Counter[str] = Counter()
    for pair_seed in replicate.seed_plan.pair_seeds:
        first = by_game_id[f"{pair_seed.pair_id}-lane-a-first"]
        second = by_game_id[f"{pair_seed.pair_id}-lane-b-first"]
        if first.pair_id != pair_seed.pair_id or second.pair_id != pair_seed.pair_id:
            raise PopulationScheduleError("population arm evidence pair ids do not match")
        if first.setup_seed != second.setup_seed:
            raise PopulationScheduleError("seat-swapped games do not share a setup seed")
        first_lanes = {seat.rng_identity: seat for seat in first.seats}
        second_lanes = {seat.rng_identity: seat for seat in second.seats}
        if set(first_lanes) != {lane_a_identity, lane_b_identity} or set(second_lanes) != {
            lane_a_identity,
            lane_b_identity,
        }:
            raise PopulationScheduleError(
                "population arm evidence does not preserve logical lane identities"
            )
        lane_a_policy = _policy_id(plan.config, first_lanes[lane_a_identity])
        lane_b_policy = _policy_id(plan.config, first_lanes[lane_b_identity])
        if (
            _policy_id(plan.config, second_lanes[lane_a_identity]) != lane_a_policy
            or _policy_id(plan.config, second_lanes[lane_b_identity]) != lane_b_policy
        ):
            raise PopulationScheduleError("seat swap changed a logical policy assignment")
        reconstructed.append(
            LogicalMatchup(pair_seed.pair_index, pair_seed.pair_id, lane_a_policy, lane_b_policy)
        )
        setup_blocks.append(
            PopulationSetupBlock(
                replicate_id,
                pair_seed.pair_index,
                pair_seed.pair_id,
                first.setup_seed,
            )
        )
        for game in (first, second):
            player_one_counts[_policy_id(plan.config, game.seats[0])] += 1
            player_two_counts[_policy_id(plan.config, game.seats[1])] += 1

    expected_assignment = replicate.assignment(arm)
    rebuilt_assignment = PopulationAssignment(
        arm=arm,
        replicate_id=replicate_id,
        matchups=tuple(reconstructed),
        slot_counts=expected_assignment.slot_counts,
        assignment_seed=expected_assignment.assignment_seed,
        assignment_rng_domain=expected_assignment.assignment_rng_domain,
    )
    if rebuilt_assignment != expected_assignment:
        raise PopulationScheduleError("population arm evidence reconstructs a different assignment")
    expected_marginals = {
        item.policy_id: item.logical_slots for item in expected_assignment.slot_counts
    }
    if (
        dict(player_one_counts) != expected_marginals
        or dict(player_two_counts) != expected_marginals
    ):
        raise PopulationScheduleError("seat-swapped population marginals are not exact")
    return PopulationArmAudit(
        replicate_id=replicate_id,
        arm=arm,
        assignment=rebuilt_assignment,
        setup_blocks=tuple(setup_blocks),
        player_one_marginals=_marginal_slot_counts(player_one_counts),
        player_two_marginals=_marginal_slot_counts(player_two_counts),
        matchup_counts=_matchup_counts(rebuilt_assignment.matchups),
    )


def reconstruct_population_assignment(
    plan: PopulationCorpusPlan,
    replicate_id: str,
    arm: PopulationArm,
    entries: Iterable[ScheduledPopulationEntry],
) -> PopulationAssignment:
    """Return the complete assignment reconstructed from schedule entries or completed records."""
    return audit_population_arm(plan, replicate_id, arm, entries).assignment


def reconstruct_population_setup_blocks(
    plan: PopulationCorpusPlan,
    replicate_id: str,
    arm: PopulationArm,
    entries: Iterable[ScheduledPopulationEntry],
) -> tuple[PopulationSetupBlock, ...]:
    """Return paired setup blocks reconstructed from schedule entries or completed records."""
    return audit_population_arm(plan, replicate_id, arm, entries).setup_blocks


def reconstruct_population_per_seat_marginals(
    plan: PopulationCorpusPlan,
    replicate_id: str,
    arm: PopulationArm,
    entries: Iterable[ScheduledPopulationEntry],
) -> tuple[tuple[PolicySlotCount, ...], tuple[PolicySlotCount, ...]]:
    """Return player-one/player-two policy marginals reconstructed from evidence."""
    audit = audit_population_arm(plan, replicate_id, arm, entries)
    return audit.player_one_marginals, audit.player_two_marginals


def reconstruct_population_matchup_counts(
    plan: PopulationCorpusPlan,
    replicate_id: str,
    arm: PopulationArm,
    entries: Iterable[ScheduledPopulationEntry],
) -> tuple[PopulationMatchupCount, ...]:
    """Return unordered logical matchup counts reconstructed from evidence."""
    return audit_population_arm(plan, replicate_id, arm, entries).matchup_counts


def audit_population_alignment(
    control_entries: Iterable[ScheduledPopulationEntry],
    treatment_entries: Iterable[ScheduledPopulationEntry],
) -> PopulationAlignmentAudit:
    """Validate common setup and arm-independent lane streams without comparing policy labels."""
    controls = tuple(_observed_game(entry) for entry in control_entries)
    treatments = tuple(_observed_game(entry) for entry in treatment_entries)
    control_by_id = {game.game_id: game for game in controls}
    treatment_by_id = {game.game_id: game for game in treatments}
    if (
        not controls
        or len(control_by_id) != len(controls)
        or len(treatment_by_id) != len(treatments)
        or set(control_by_id) != set(treatment_by_id)
    ):
        raise PopulationScheduleError(
            "control/treatment alignment requires matching unique non-empty games"
        )
    blocks: dict[str, PopulationSetupBlock] = {}
    pair_games: dict[str, list[_ObservedGame]] = {}
    for game_id in sorted(control_by_id):
        control = control_by_id[game_id]
        treatment = treatment_by_id[game_id]
        if (
            control.pair_id is None
            or control.pair_id != treatment.pair_id
            or control.setup_seed != treatment.setup_seed
            or dict(control.game_config) != dict(treatment.game_config)
            or tuple(seat.rng_identity for seat in control.seats)
            != tuple(seat.rng_identity for seat in treatment.seats)
            or tuple(seat.seed for seat in control.seats)
            != tuple(seat.seed for seat in treatment.seats)
        ):
            raise PopulationScheduleError(
                "control/treatment setup or logical lane RNG streams differ"
            )
        if control.seats[0].rng_identity == control.seats[1].rng_identity:
            raise PopulationScheduleError("logical lanes are not independent within a game")
        if control.seats[0].seed == control.seats[1].seed:
            raise PopulationScheduleError("logical lanes share an RNG seed within a game")
        pair_games.setdefault(control.pair_id, []).append(control)
    for pair_id, games in pair_games.items():
        if len(games) != 2:
            raise PopulationScheduleError(
                "alignment evidence must contain both seat swaps per pair"
            )
        first, second = sorted(games, key=lambda game: game.game_id)
        if (
            first.setup_seed != second.setup_seed
            or tuple(seat.rng_identity for seat in first.seats)
            != tuple(reversed(tuple(seat.rng_identity for seat in second.seats)))
            or tuple(seat.seed for seat in first.seats)
            != tuple(reversed(tuple(seat.seed for seat in second.seats)))
        ):
            raise PopulationScheduleError("seat swap does not preserve logical lane streams")
        pair_index_text = pair_id.rsplit("-pair-", 1)[-1]
        if not pair_index_text.isdigit():
            raise PopulationScheduleError("population pair id has no stable pair index")
        replicate_id = pair_id.rsplit("-pair-", 1)[0]
        blocks[pair_id] = PopulationSetupBlock(
            replicate_id,
            int(pair_index_text),
            pair_id,
            first.setup_seed,
        )
    return PopulationAlignmentAudit(
        paired_block_count=len(blocks),
        game_count_per_arm=len(controls),
        shared_setup_blocks=tuple(sorted(blocks.values(), key=lambda block: block.pair_id)),
    )

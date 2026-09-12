"""Fixed-root, common-random-number rollout target construction.

The rollout layer accepts only safe observations, synthetic latent states, and policy
interfaces.  It never imports a checkpoint implementation or passes a latent state to an encoder or
policy.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol, cast, runtime_checkable

from agent_avenue.agents import (
    Agent,
    DeterministicRandom,
    TerminalOffenseAgent,
    TerminalSafetyAgent,
)
from agent_avenue.agents.random_source import derive_seed
from agent_avenue.agents.terminal_offense import filter_immediate_win_actions
from agent_avenue.agents.terminal_safety import filter_terminal_actions
from agent_avenue.encoding.candidate_structured_v2 import FEATURE_WIDTH, encode_candidate
from agent_avenue.engine.model import Action, PlayerId, TerminalOutcome
from agent_avenue.observation.model import ObservationDecision, PlayerObservation

from .identity import PanelPosition, canonical_legal_actions, canonical_panel_order
from .latent import (
    LatentRolloutError,
    LatentRolloutState,
    latent_decision,
    rollout_transition_v1,
    safe_latent_observation,
    sample_latent_world,
)

STEP4_ROOT_SEED = 2026091204
ROLLOUT_TARGET_VERSION = "counterfactual-rollout-targets-v1"
RNG_CONTRACT_VERSION = "counterfactual-rng-contract-v1"
WORLD_COUNT = 10
ROLLOUT_DEPTH = 9
CONTINUATION_SLOTS: tuple[tuple[str, str], ...] = (
    ("q0-1", "q0"),
    ("q0-2", "q0"),
    ("q0-3", "q0"),
    ("q0-4", "q0"),
    ("q1", "q1"),
    ("q2", "q2"),
    ("q3", "q3"),
    ("q4", "q4"),
    ("greedy-public-v1", "heuristic"),
    ("random", "random"),
)
REQUIRED_POLICY_IDS = frozenset(policy_id for _, policy_id in CONTINUATION_SLOTS)


class RolloutTargetError(ValueError):
    """Raised when a fixed Step-4 target cannot be constructed safely or mechanically."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _action_data(action: Action) -> dict[str, object]:
    if hasattr(action, "slot"):
        return {
            "type": "recruit",
            "revision": action.revision,
            "actor": action.actor.value,
            "slot": action.slot.value,
        }
    return {
        "type": "play_offer",
        "revision": action.revision,
        "actor": action.actor.value,
        "face_up": action.face_up.value,
        "face_down": action.face_down.value,
    }


def _state_digest(state: LatentRolloutState) -> str:
    """Retain a hidden-world audit digest without retaining any hidden zone in target rows."""
    offer = None
    if state.offer is not None:
        offer = {
            "offered_by": state.offer.offered_by.value,
            "face_up": state.offer.face_up.value,
            "face_down": state.offer.face_down.value,
        }
    return _digest(
        {
            "deck": [card.value for card in state.deck],
            "hands": [[card.value for card in hand] for hand in state.hands],
            "recruited": [[card.value for card in cards] for cards in state.recruited],
            "scores": list(state.scores),
            "active_player": state.active_player.value,
            "turn": state.turn,
            "phase": state.phase.value,
            "revision": state.revision,
            "offer": offer,
        }
    )


def _permutation(items: Sequence[str], rng: DeterministicRandom) -> tuple[str, ...]:
    values = list(items)
    for upper in range(len(values) - 1, 0, -1):
        lower = rng.randbelow(upper + 1)
        values[upper], values[lower] = values[lower], values[upper]
    return tuple(values)


def world_seed(*, root_seed: int, replicate_id: str, safe_identity: str, world_index: int) -> int:
    if not 0 <= world_index < WORLD_COUNT:
        raise RolloutTargetError("world index is outside the fixed ten-world design")
    return derive_seed(root_seed, f"step4:world:{replicate_id}:{safe_identity}:{world_index}")


def policy_permutation_seed(
    *, root_seed: int, replicate_id: str, safe_identity: str, seat: PlayerId
) -> int:
    return derive_seed(
        root_seed, f"step4:policy-permutation:{replicate_id}:{safe_identity}:{seat.value}"
    )


def rollout_rng_seed(
    *,
    root_seed: int,
    replicate_id: str,
    safe_identity: str,
    world_index: int,
    seat: PlayerId,
    decision_ordinal: int,
) -> int:
    if world_index < 0 or decision_ordinal < 0:
        raise RolloutTargetError("world and decision ordinals must be non-negative")
    return derive_seed(
        root_seed,
        "step4:rollout-rng:"
        f"{replicate_id}:{safe_identity}:{world_index}:{seat.value}:{decision_ordinal}",
    )


@dataclass(frozen=True, slots=True)
class WorldPolicyAssignment:
    """Physical-seat continuation identities for one world."""

    world_index: int
    player_one_slot: str
    player_two_slot: str

    @property
    def player_one_policy_id(self) -> str:
        return dict(CONTINUATION_SLOTS)[self.player_one_slot]

    @property
    def player_two_policy_id(self) -> str:
        return dict(CONTINUATION_SLOTS)[self.player_two_slot]

    def policy_for(self, player: PlayerId) -> str:
        return (
            self.player_one_policy_id
            if player is PlayerId.PLAYER_ONE
            else self.player_two_policy_id
        )


def continuation_policy_assignments(
    *, root_seed: int, replicate_id: str, safe_identity: str
) -> tuple[WorldPolicyAssignment, ...]:
    """Create independent ten-slot permutations with exact per-seat marginals."""
    slots = tuple(slot for slot, _ in CONTINUATION_SLOTS)
    p1_seed = policy_permutation_seed(
        root_seed=root_seed,
        replicate_id=replicate_id,
        safe_identity=safe_identity,
        seat=PlayerId.PLAYER_ONE,
    )
    p2_seed = policy_permutation_seed(
        root_seed=root_seed,
        replicate_id=replicate_id,
        safe_identity=safe_identity,
        seat=PlayerId.PLAYER_TWO,
    )
    one = _permutation(slots, DeterministicRandom(p1_seed, "step4:policy-permutation/p1"))
    two = _permutation(slots, DeterministicRandom(p2_seed, "step4:policy-permutation/p2"))
    return tuple(
        WorldPolicyAssignment(index, one[index], two[index]) for index in range(WORLD_COUNT)
    )


@dataclass(frozen=True, slots=True)
class SampledWorld:
    """One sampled hidden allocation identified only by an audit digest outside transcripts."""

    world_index: int
    seed: int
    state: LatentRolloutState
    latent_digest: str
    policies: WorldPolicyAssignment


def sample_position_worlds(
    position: PanelPosition, *, root_seed: int = STEP4_ROOT_SEED
) -> tuple[SampledWorld, ...]:
    """Sample exactly ten candidate-independent worlds for one public panel identity."""
    assignments = continuation_policy_assignments(
        root_seed=root_seed,
        replicate_id=position.replicate_id,
        safe_identity=position.safe_identity,
    )
    worlds: list[SampledWorld] = []
    for world_index in range(WORLD_COUNT):
        seed = world_seed(
            root_seed=root_seed,
            replicate_id=position.replicate_id,
            safe_identity=position.safe_identity,
            world_index=world_index,
        )
        state = sample_latent_world(
            position.observation,
            DeterministicRandom(seed, f"step4:world/{position.safe_identity}/{world_index}"),
        )
        worlds.append(
            SampledWorld(world_index, seed, state, _state_digest(state), assignments[world_index])
        )
    return tuple(worlds)


def wrap_continuation_policy(policy: Agent) -> Agent:
    """Apply the frozen greedy terminal-offense then terminal-safety deployment envelope."""
    return TerminalOffenseAgent(TerminalSafetyAgent(policy))


def enveloped_legal_actions(observation: PlayerObservation) -> tuple[Action, ...]:
    """Apply the same envelope without invoking a policy (used only for q0 leaf scoring)."""
    initial = canonical_legal_actions(observation.legal_actions)
    offense = filter_immediate_win_actions(observation, initial)
    offense_observation = replace(observation, legal_actions=offense.allowed_actions)
    safety = filter_terminal_actions(offense_observation, offense.allowed_actions)
    return canonical_legal_actions(safety.allowed_actions)


@runtime_checkable
class LeafLogitScorer(Protocol):
    """Minimal optional learned-policy adapter, deliberately defined without Torch imports."""

    def score_candidates(
        self,
        observation: PlayerObservation,
        decision: ObservationDecision,
        legal_actions: tuple[Action, ...],
    ) -> object:
        """Return semantic-order action logits."""


LeafScorer = LeafLogitScorer | Callable[[PlayerObservation, tuple[Action, ...]], object]


def _coerce_logits(value: object, actions: tuple[Action, ...]) -> tuple[float, ...]:
    if isinstance(value, Mapping):
        try:
            values = tuple(
                float(cast(float, cast(Mapping[Action, object], value)[action]))
                for action in actions
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RolloutTargetError("leaf mapping does not cover the allowed legal set") from exc
    elif hasattr(value, "actions") and hasattr(value, "logits"):
        scored_actions = value.actions
        logits = value.logits
        if scored_actions != actions:
            raise RolloutTargetError("leaf scorer did not preserve canonical allowed action order")
        try:
            values = tuple(float(item) for item in logits)
        except (TypeError, ValueError) as exc:
            raise RolloutTargetError("leaf scorer logits are malformed") from exc
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        try:
            values = tuple(float(item) for item in value)
        except (TypeError, ValueError) as exc:
            raise RolloutTargetError("leaf scorer logits are malformed") from exc
    else:
        raise RolloutTargetError(
            "leaf scorer must return logits by action or in allowed action order"
        )
    if len(values) != len(actions) or any(not math.isfinite(item) for item in values):
        raise RolloutTargetError("leaf logits must be finite and aligned to all allowed actions")
    return values


def _leaf_logits(
    scorer: LeafScorer, observation: PlayerObservation, actions: tuple[Action, ...]
) -> tuple[float, ...]:
    if isinstance(scorer, LeafLogitScorer):
        value = scorer.score_candidates(observation, observation.decision, actions)
    else:
        value = scorer(observation, actions)
    return _coerce_logits(value, actions)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


@dataclass(frozen=True, slots=True)
class RolloutSample:
    """One no-hidden-transcript candidate/world result retained in a target row."""

    world_index: int
    world_seed: int
    latent_digest: str
    player_one_policy_id: str
    player_two_policy_id: str
    continuation_rng_seeds: tuple[int, ...]
    terminal: bool
    depth_leaf: bool
    actions_applied: int
    root_value: float

    def __post_init__(self) -> None:
        if not (self.terminal ^ self.depth_leaf):
            raise RolloutTargetError("a rollout sample must be exactly terminal or a depth leaf")
        if not 1 <= self.actions_applied <= ROLLOUT_DEPTH:
            raise RolloutTargetError("rollout action count is outside the inclusive depth contract")
        if not 0.0 <= self.root_value <= 1.0 or not math.isfinite(self.root_value):
            raise RolloutTargetError("rollout root value must be finite in [0, 1]")

    def to_data(self) -> dict[str, object]:
        return {
            "world_index": self.world_index,
            "world_seed": self.world_seed,
            "latent_digest": self.latent_digest,
            "player_one_policy_id": self.player_one_policy_id,
            "player_two_policy_id": self.player_two_policy_id,
            "continuation_rng_seeds": list(self.continuation_rng_seeds),
            "terminal": self.terminal,
            "depth_leaf": self.depth_leaf,
            "actions_applied": self.actions_applied,
            "root_value": self.root_value,
        }


def _select_continuation_action(
    policy: Agent,
    state: LatentRolloutState,
    *,
    root_seed: int,
    replicate_id: str,
    safe_identity: str,
    world_index: int,
) -> tuple[Action, int]:
    decision = latent_decision(state)
    if isinstance(decision, TerminalOutcome):
        raise RolloutTargetError("continuation policy was requested after terminal resolution")
    actor = decision.actor
    observation = safe_latent_observation(state, actor)
    seed = rollout_rng_seed(
        root_seed=root_seed,
        replicate_id=replicate_id,
        safe_identity=safe_identity,
        world_index=world_index,
        seat=actor,
        decision_ordinal=decision.revision,
    )
    rng = DeterministicRandom(
        seed, f"step4:rollout-rng/{world_index}/{actor.value}/{decision.revision}"
    )
    action = policy.choose_action(observation, observation.decision, observation.legal_actions, rng)
    if action not in observation.legal_actions:
        raise RolloutTargetError(
            "continuation policy returned an action outside its safe legal set"
        )
    return action, seed


def rollout_candidate_world(
    position: PanelPosition,
    candidate: Action,
    world: SampledWorld,
    *,
    continuation_policies: Mapping[str, Agent],
    leaf_scorer: LeafScorer,
    root_seed: int = STEP4_ROOT_SEED,
) -> RolloutSample:
    """Apply one forced root action then at most eight safe continuation policy decisions."""
    required = REQUIRED_POLICY_IDS
    if set(continuation_policies) != required:
        raise RolloutTargetError(
            "continuation policies must be exactly q0-q4, heuristic, and random"
        )
    root_actions = canonical_legal_actions(position.observation.legal_actions)
    if candidate not in root_actions:
        raise RolloutTargetError("forced root candidate is not a legal public action")
    root_actor = position.observation.viewer
    try:
        state = rollout_transition_v1(world.state, candidate)
    except LatentRolloutError as exc:
        raise RolloutTargetError(
            "forced root action could not be applied to a sampled world"
        ) from exc
    rng_seeds: list[int] = []
    actions_applied = 1
    while state.phase.value != "terminal" and actions_applied < ROLLOUT_DEPTH:
        policy_id = world.policies.policy_for(
            state.active_player if state.phase.value == "play" else state.active_player.other()
        )
        policy = wrap_continuation_policy(continuation_policies[policy_id])
        action, seed = _select_continuation_action(
            policy,
            state,
            root_seed=root_seed,
            replicate_id=position.replicate_id,
            safe_identity=position.safe_identity,
            world_index=world.world_index,
        )
        rng_seeds.append(seed)
        try:
            state = rollout_transition_v1(state, action)
        except LatentRolloutError as exc:
            raise RolloutTargetError(
                "continuation action could not be applied to its latent world"
            ) from exc
        actions_applied += 1
    if state.phase.value == "terminal":
        assert state.outcome is not None
        value = 1.0 if state.outcome.winner is root_actor else 0.0
        return RolloutSample(
            world.world_index,
            world.seed,
            world.latent_digest,
            world.policies.player_one_policy_id,
            world.policies.player_two_policy_id,
            tuple(rng_seeds),
            True,
            False,
            actions_applied,
            value,
        )
    if state.turn > 19:
        raise RolloutTargetError("mechanical turn cap reached before leaf evaluation")
    decision = latent_decision(state)
    if isinstance(decision, TerminalOutcome):  # pragma: no cover - phase check above is exhaustive
        raise RolloutTargetError("terminal outcome escaped terminal phase")
    leaf_actor = decision.actor
    observation = safe_latent_observation(state, leaf_actor)
    allowed = enveloped_legal_actions(observation)
    leaf_observation = replace(observation, legal_actions=allowed)
    logits = _leaf_logits(leaf_scorer, leaf_observation, allowed)
    value = _sigmoid(max(logits))
    root_value = value if leaf_actor is root_actor else 1.0 - value
    return RolloutSample(
        world.world_index,
        world.seed,
        world.latent_digest,
        world.policies.player_one_policy_id,
        world.policies.player_two_policy_id,
        tuple(rng_seeds),
        False,
        True,
        actions_applied,
        root_value,
    )


Encoder = Callable[[PlayerObservation, Action], Sequence[float]]


def _structured_vector(observation: PlayerObservation, action: Action) -> Sequence[float]:
    return encode_candidate(observation, action).vector


@dataclass(frozen=True, slots=True)
class RolloutTargetRow:
    """All-action, position-balanced auxiliary supervision for one public candidate."""

    replicate_id: str
    stratum: str
    safe_identity: str
    action: Action
    features: tuple[float, ...]
    target: float
    samples: tuple[RolloutSample, ...]

    def __post_init__(self) -> None:
        if len(self.features) != FEATURE_WIDTH or any(
            not math.isfinite(value) for value in self.features
        ):
            raise RolloutTargetError(
                "rollout rows must retain one finite 519-value structured vector"
            )
        if len(self.samples) != WORLD_COUNT or not 0.0 <= self.target <= 1.0:
            raise RolloutTargetError("rollout rows require ten finite [0,1] world samples")
        mean = sum(sample.root_value for sample in self.samples) / WORLD_COUNT
        if not math.isclose(self.target, mean, rel_tol=0.0, abs_tol=1e-12):
            raise RolloutTargetError("rollout row target is not the fixed ten-world mean")

    def to_data(self) -> dict[str, object]:
        return {
            "replicate_id": self.replicate_id,
            "stratum": self.stratum,
            "safe_identity": self.safe_identity,
            "action": _action_data(self.action),
            "feature_digest": _digest(list(self.features)),
            "target": self.target,
            "samples": [sample.to_data() for sample in self.samples],
        }


@dataclass(frozen=True, slots=True)
class RolloutTargetSet:
    """A canonical all-action target panel; no audit record fingerprints are included."""

    positions: tuple[PanelPosition, ...]
    rows: tuple[RolloutTargetRow, ...]
    root_seed: int = STEP4_ROOT_SEED

    def __post_init__(self) -> None:
        ordered = canonical_panel_order(self.positions)
        if self.positions != ordered:
            raise RolloutTargetError("target positions must use canonical panel order")
        valid = {
            (position.safe_identity, action)
            for position in self.positions
            for action in canonical_legal_actions(position.observation.legal_actions)
        }
        keys = {(row.safe_identity, row.action) for row in self.rows}
        if keys != valid or len(keys) != len(self.rows):
            raise RolloutTargetError("target rows must cover every legal action exactly once")

    @property
    def depth_leaf_fraction(self) -> float:
        samples = [sample for row in self.rows for sample in row.samples]
        return sum(sample.depth_leaf for sample in samples) / len(samples) if samples else 0.0

    @property
    def terminal_fraction(self) -> float:
        samples = [sample for row in self.rows for sample in row.samples]
        return sum(sample.terminal for sample in samples) / len(samples) if samples else 0.0

    @property
    def digest(self) -> str:
        return _digest(
            {
                "version": ROLLOUT_TARGET_VERSION,
                "root_seed": self.root_seed,
                "positions": [
                    {
                        "replicate_id": position.replicate_id,
                        "stratum": position.stratum,
                        "safe_identity": position.safe_identity,
                        "selection_hash": position.selection_hash,
                    }
                    for position in self.positions
                ],
                "rows": [row.to_data() for row in self.rows],
            }
        )


def generate_rollout_targets(
    positions: Sequence[PanelPosition],
    *,
    continuation_policies: Mapping[str, Agent],
    leaf_scorer: LeafScorer,
    root_seed: int = STEP4_ROOT_SEED,
    encoder: Encoder = _structured_vector,
    enforce_depth_leaf_gate: bool = True,
) -> RolloutTargetSet:
    """Generate deterministic ten-world targets for every legal action in canonical panel order."""
    ordered = canonical_panel_order(positions)
    rows: list[RolloutTargetRow] = []
    for position in ordered:
        worlds = sample_position_worlds(position, root_seed=root_seed)
        actions = canonical_legal_actions(position.observation.legal_actions)
        for action in actions:
            samples = tuple(
                rollout_candidate_world(
                    position,
                    action,
                    world,
                    continuation_policies=continuation_policies,
                    leaf_scorer=leaf_scorer,
                    root_seed=root_seed,
                )
                for world in worlds
            )
            vector = tuple(float(value) for value in encoder(position.observation, action))
            if len(vector) != FEATURE_WIDTH:
                raise RolloutTargetError(
                    "encoder must return the frozen 519-value structured vector"
                )
            rows.append(
                RolloutTargetRow(
                    position.replicate_id,
                    position.stratum,
                    position.safe_identity,
                    action,
                    vector,
                    sum(sample.root_value for sample in samples) / WORLD_COUNT,
                    samples,
                )
            )
    targets = RolloutTargetSet(ordered, tuple(rows), root_seed)
    if enforce_depth_leaf_gate and targets.depth_leaf_fraction > 0.5:
        raise RolloutTargetError("depth-leaf fraction exceeds the fixed 50% eligibility gate")
    return targets


__all__ = [
    "CONTINUATION_SLOTS",
    "REQUIRED_POLICY_IDS",
    "RNG_CONTRACT_VERSION",
    "ROLLOUT_DEPTH",
    "ROLLOUT_TARGET_VERSION",
    "STEP4_ROOT_SEED",
    "WORLD_COUNT",
    "LeafLogitScorer",
    "LeafScorer",
    "RolloutSample",
    "RolloutTargetError",
    "RolloutTargetRow",
    "RolloutTargetSet",
    "SampledWorld",
    "WorldPolicyAssignment",
    "continuation_policy_assignments",
    "enveloped_legal_actions",
    "generate_rollout_targets",
    "policy_permutation_seed",
    "rollout_candidate_world",
    "rollout_rng_seed",
    "sample_position_worlds",
    "world_seed",
    "wrap_continuation_policy",
]

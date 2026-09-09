"""Replay-derived strength and tactical-leak diagnostics."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from typing import Final

from agent_avenue.agents import (
    DeterministicRandom,
    derive_seed,
    filter_immediate_win_actions,
    filter_terminal_actions,
    terminal_outcomes_for_action,
)
from agent_avenue.engine import (
    Action,
    OfferSlot,
    OutcomeReason,
    OutcomeResolution,
    Phase,
    PlayerId,
    PlayOfferAction,
    RecruitAction,
    TerminalOutcome,
    apply_action,
    new_game,
)
from agent_avenue.engine.model import player_index
from agent_avenue.observation import observe
from agent_avenue.observation.model import PlayerObservation, RecruitContext
from agent_avenue.storage import GameRecord, game_record_fingerprint, verify_game_record

from .arena import (
    PAIRED_BOOTSTRAP_LOWER_INDEX,
    PAIRED_BOOTSTRAP_RESAMPLES,
    PAIRED_BOOTSTRAP_UPPER_INDEX,
)

FORCED_WIN_AUDIT_VERSION: Final = "public-forced-win-audit-v1"
RECRUIT_PATTERN_AUDIT_VERSION: Final = "recruit-pattern-audit-v1"
PAIRED_DIFFERENCE_VERSION: Final = "paired-policy-win-rate-difference-bootstrap-v1"
CandidateScorer = Callable[[PlayerObservation, tuple[Action, ...]], tuple[float, ...]]


class StrengthAuditError(ValueError):
    """Raised when retained records cannot support a valid strength audit."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _artifact_fingerprint(data: Mapping[str, object]) -> str:
    payload = {key: value for key, value in data.items() if key != "artifact_fingerprint"}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _action_data(action: Action) -> dict[str, object]:
    if isinstance(action, PlayOfferAction):
        return {
            "type": "play_offer",
            "revision": action.revision,
            "actor": action.actor.value,
            "face_up": action.face_up.value,
            "face_down": action.face_down.value,
        }
    return {
        "type": "recruit",
        "revision": action.revision,
        "actor": action.actor.value,
        "slot": action.slot.value,
    }


def _public_player_data(observation: PlayerObservation) -> list[dict[str, object]]:
    return [
        {
            "player": player.player.value,
            "score": player.score,
            "recruited": [card.value for card in player.recruited],
            "hand_size": player.hand_size,
        }
        for player in observation.players
    ]


def _mechanisms(outcome: TerminalOutcome, actor: PlayerId) -> tuple[str, ...]:
    values: list[str] = []
    if outcome.reason is OutcomeReason.DECK_EXHAUSTION:
        values.append("deck_exhaustion")
    if actor in outcome.facts.score_gap_winners:
        values.append("score_gap")
    if actor in outcome.facts.instant_winners:
        values.append("codebreaker")
    if actor.other() in outcome.facts.instant_losers:
        values.append("opponent_daredevil")
    if outcome.resolution in {
        OutcomeResolution.ACTIVE_CONDITION_TIE,
        OutcomeResolution.ACTIVE_SCORE_TIE,
    }:
        values.append("active_tiebreak")
    return tuple(values or ("other_terminal",))


def _forced_win_mechanisms(
    observation: PlayerObservation, actions: tuple[Action, ...]
) -> tuple[str, ...]:
    mechanisms: set[str] = set()
    for action in actions:
        for outcome in terminal_outcomes_for_action(observation, action):
            if outcome is None or outcome.winner is not observation.viewer:
                raise StrengthAuditError("forced-win action did not resolve to a viewer win")
            mechanisms.update(_mechanisms(outcome, observation.viewer))
    return tuple(sorted(mechanisms))


def _empty_tactical_counts() -> Counter[str]:
    return Counter(
        {
            "decisions": 0,
            "play_decisions": 0,
            "recruit_decisions": 0,
            "decisions_with_forced_win": 0,
            "forced_win_actions": 0,
            "executed_forced_win_actions": 0,
            "missed_forced_wins": 0,
            "missed_non_immediate_wins": 0,
            "missed_eventual_losses": 0,
            "chosen_immediate_wins": 0,
            "decisions_with_provable_loss": 0,
            "executed_provable_losses": 0,
            "executed_avoidable_provable_losses": 0,
            "all_actions_provably_losing": 0,
        }
    )


def _counts_data(counts: Mapping[str, int]) -> dict[str, int]:
    return {key: counts[key] for key in _empty_tactical_counts()}


def _resolved_chosen_state(
    state: object, action: Action, following_action: Action | None
) -> object:
    # The concrete GameState annotation is avoided here only to keep the helper's call site compact.
    from agent_avenue.engine import GameState

    if not isinstance(state, GameState):
        raise TypeError("expected GameState")
    after = apply_action(state, action)
    if isinstance(action, PlayOfferAction):
        if not isinstance(following_action, RecruitAction):
            raise StrengthAuditError("play action is not followed by a recruit action")
        after = apply_action(after, following_action)
    return after


def _example(
    record: GameRecord,
    *,
    action_index: int,
    observation: PlayerObservation,
    action: Action,
    opponent_id: str,
    forced_actions: tuple[Action, ...],
    mechanisms: tuple[str, ...],
    chosen_immediate_win: bool,
    scorer: CandidateScorer | None,
) -> dict[str, object]:
    decision = observation.decision
    decision_data: dict[str, object] = {"kind": getattr(decision, "kind", "unknown")}
    if isinstance(decision, RecruitContext):
        decision_data.update(
            {
                "face_up": decision.face_up.value,
                "known_face_down": (
                    None if decision.known_face_down is None else decision.known_face_down.value
                ),
            }
        )
    scored: list[dict[str, object]] = []
    if scorer is not None:
        scoring_actions = filter_terminal_actions(
            observation, observation.legal_actions
        ).allowed_actions
        values = scorer(observation, scoring_actions)
        if len(values) != len(scoring_actions):
            raise StrengthAuditError("candidate scorer returned the wrong number of values")
        scored = [
            {"action": _action_data(candidate), "logit": value}
            for candidate, value in sorted(
                zip(scoring_actions, values, strict=True),
                key=lambda item: item[1],
                reverse=True,
            )
        ]
    return {
        "game_id": record.game_id,
        "run_id": record.run_id,
        "pair_id": record.pair_id,
        "action_index": action_index,
        "turn": observation.turn,
        "phase": observation.phase.value,
        "actor": observation.viewer.value,
        "opponent_id": opponent_id,
        "players": _public_player_data(observation),
        "own_hand": [card.value for card in observation.own_hand],
        "remaining_deck_count": observation.remaining_deck_count,
        "decision": decision_data,
        "chosen_action": _action_data(action),
        "forced_win_actions": [_action_data(candidate) for candidate in forced_actions],
        "mechanisms": list(mechanisms),
        "chosen_line_won_immediately": chosen_immediate_win,
        "actor_eventually_won": record.winner is observation.viewer,
        "safe_candidate_logits": scored,
        "record_fingerprint": game_record_fingerprint(record),
    }


def audit_public_forced_wins(
    records: Iterable[GameRecord],
    *,
    source_label: str,
    candidate_scorers: Mapping[str, CandidateScorer] | None = None,
    example_agent_ids: frozenset[str] = frozenset(),
    max_examples_per_agent: int = 24,
    verify_records: bool = True,
) -> dict[str, object]:
    """Replay records and measure exact public current-turn win conversion."""
    if not source_label:
        raise StrengthAuditError("forced-win audit requires a source label")
    if max_examples_per_agent < 0:
        raise StrengthAuditError("max_examples_per_agent must be non-negative")
    scorers = candidate_scorers or {}
    overall = _empty_tactical_counts()
    by_agent: dict[str, Counter[str]] = {}
    by_phase: dict[str, dict[str, Counter[str]]] = {}
    by_opponent: dict[str, dict[str, Counter[str]]] = {}
    mechanisms_by_agent: dict[str, Counter[str]] = {}
    examples: dict[str, list[dict[str, object]]] = {}
    record_count = 0
    decision_count = 0

    for record in records:
        if verify_records:
            verify_game_record(record)
        record_count += 1
        state = new_game(record.replay.config, record.replay.seed)
        actions = record.replay.actions
        for action_index, action in enumerate(actions):
            if state.phase is Phase.TERMINAL:
                raise StrengthAuditError("record contains actions after terminal state")
            actor = (
                state.active_player if state.phase is Phase.PLAY else state.active_player.other()
            )
            actor_index = player_index(actor)
            agent_id = record.seats[actor_index].agent_id
            opponent_id = record.seats[1 - actor_index].agent_id
            observation = observe(state, actor)
            forced = filter_immediate_win_actions(observation, observation.legal_actions)
            safety = filter_terminal_actions(observation, observation.legal_actions)
            following = actions[action_index + 1] if action_index + 1 < len(actions) else None
            resolved = _resolved_chosen_state(state, action, following)
            from agent_avenue.engine import GameState

            if not isinstance(resolved, GameState):  # pragma: no cover - helper guarantees this
                raise TypeError("expected resolved GameState")
            chosen_immediate_win = (
                resolved.phase is Phase.TERMINAL
                and resolved.outcome is not None
                and resolved.outcome.winner is actor
            )
            phase_name = state.phase.value
            counts = by_agent.setdefault(agent_id, _empty_tactical_counts())
            phase_counts = by_phase.setdefault(agent_id, {}).setdefault(
                phase_name, _empty_tactical_counts()
            )
            opponent_counts = by_opponent.setdefault(agent_id, {}).setdefault(
                opponent_id, _empty_tactical_counts()
            )
            values = {
                "decisions": 1,
                "play_decisions": int(state.phase is Phase.PLAY),
                "recruit_decisions": int(state.phase is Phase.RECRUIT),
                "decisions_with_forced_win": int(bool(forced.forced_win_actions)),
                "forced_win_actions": len(forced.forced_win_actions),
                "executed_forced_win_actions": int(action in forced.forced_win_actions),
                "missed_forced_wins": int(
                    bool(forced.forced_win_actions) and action not in forced.forced_win_actions
                ),
                "missed_non_immediate_wins": int(
                    bool(forced.forced_win_actions)
                    and action not in forced.forced_win_actions
                    and not chosen_immediate_win
                ),
                "missed_eventual_losses": int(
                    bool(forced.forced_win_actions)
                    and action not in forced.forced_win_actions
                    and record.winner is not actor
                ),
                "chosen_immediate_wins": int(chosen_immediate_win),
                "decisions_with_provable_loss": int(bool(safety.provable_loss_actions)),
                "executed_provable_losses": int(action in safety.provable_loss_actions),
                "executed_avoidable_provable_losses": int(
                    action in safety.provable_loss_actions and not safety.forced_loss_fallback
                ),
                "all_actions_provably_losing": int(safety.forced_loss_fallback),
            }
            for key, value in values.items():
                overall[key] += value
                counts[key] += value
                phase_counts[key] += value
                opponent_counts[key] += value
            if forced.forced_win_actions:
                mechanisms = _forced_win_mechanisms(observation, forced.forced_win_actions)
                mechanism_counts = mechanisms_by_agent.setdefault(agent_id, Counter())
                signature = "+".join(mechanisms)
                mechanism_counts[f"opportunity_signature:{signature}"] += 1
                for mechanism in mechanisms:
                    mechanism_counts[f"opportunity:{mechanism}"] += 1
                missed = action not in forced.forced_win_actions
                if missed:
                    mechanism_counts[f"miss_signature:{signature}"] += 1
                    for mechanism in mechanisms:
                        mechanism_counts[f"miss:{mechanism}"] += 1
                    agent_examples = examples.setdefault(agent_id, [])
                    if (
                        agent_id in example_agent_ids
                        and len(agent_examples) < max_examples_per_agent
                    ):
                        agent_examples.append(
                            _example(
                                record,
                                action_index=action_index,
                                observation=observation,
                                action=action,
                                opponent_id=opponent_id,
                                forced_actions=forced.forced_win_actions,
                                mechanisms=mechanisms,
                                chosen_immediate_win=chosen_immediate_win,
                                scorer=scorers.get(agent_id),
                            )
                        )
            decision_count += 1
            state = apply_action(state, action)

    if record_count == 0:
        raise StrengthAuditError("forced-win audit requires at least one record")
    data: dict[str, object] = {
        "version": FORCED_WIN_AUDIT_VERSION,
        "source_label": source_label,
        "record_count": record_count,
        "decision_count": decision_count,
        "counts": _counts_data(overall),
        "by_agent": {
            agent_id: {
                "counts": _counts_data(by_agent[agent_id]),
                "by_phase": {
                    phase: _counts_data(counts)
                    for phase, counts in sorted(by_phase.get(agent_id, {}).items())
                },
                "by_opponent": {
                    opponent: _counts_data(counts)
                    for opponent, counts in sorted(by_opponent.get(agent_id, {}).items())
                },
                "mechanisms": dict(sorted(mechanisms_by_agent.get(agent_id, {}).items())),
                "examples": examples.get(agent_id, []),
            }
            for agent_id in sorted(by_agent)
        },
        "artifact_fingerprint": "",
    }
    data["artifact_fingerprint"] = _artifact_fingerprint(data)
    return data


def _empty_pattern_counts() -> Counter[str]:
    return Counter(
        {
            "decisions": 0,
            "face_up_choices": 0,
            "target_game_wins": 0,
            "target_immediate_wins": 0,
            "chosen_score_swing_sum": 0,
            "alternative_score_swing_sum": 0,
            "alternative_score_swing_better": 0,
            "hindsight_alternative_immediate_win": 0,
        }
    )


def _score_swing(state: object, actor: PlayerId, action: RecruitAction) -> tuple[int, bool]:
    from agent_avenue.engine import GameState

    if not isinstance(state, GameState):
        raise TypeError("expected GameState")
    resolved = apply_action(state, action)
    completed = resolved.history[-1]
    index = player_index(actor)
    swing = completed.score_changes[index] - completed.score_changes[1 - index]
    immediate_win = (
        resolved.phase is Phase.TERMINAL
        and resolved.outcome is not None
        and resolved.outcome.winner is actor
    )
    return swing, immediate_win


def _pattern_data(counts: Mapping[str, int]) -> dict[str, object]:
    decisions = counts["decisions"]
    if decisions <= 0:
        raise StrengthAuditError("recruit pattern row must contain decisions")
    return {
        **{key: counts[key] for key in _empty_pattern_counts()},
        "face_up_choice_rate": counts["face_up_choices"] / decisions,
        "target_game_win_rate": counts["target_game_wins"] / decisions,
        "target_immediate_win_rate": counts["target_immediate_wins"] / decisions,
        "average_chosen_score_swing": counts["chosen_score_swing_sum"] / decisions,
        "average_alternative_score_swing": counts["alternative_score_swing_sum"] / decisions,
    }


def audit_recruit_patterns(
    records: Iterable[GameRecord],
    *,
    target_agent_id: str,
    source_label: str,
    verify_records: bool = True,
) -> dict[str, object]:
    """Describe a target policy's recruit response patterns, including offline true offers."""
    if not target_agent_id or not source_label:
        raise StrengthAuditError("recruit pattern audit requires target and source labels")
    overall = _empty_pattern_counts()
    by_face_up: dict[str, Counter[str]] = {}
    by_face_up_copy_count: dict[str, Counter[str]] = {}
    by_opponent: dict[str, Counter[str]] = {}
    by_ordered_offer: dict[str, Counter[str]] = {}
    by_opponent_offer: dict[str, Counter[str]] = {}
    record_count = 0

    for record in records:
        if verify_records:
            verify_game_record(record)
        record_count += 1
        state = new_game(record.replay.config, record.replay.seed)
        for action in record.replay.actions:
            if state.phase is Phase.RECRUIT:
                if not isinstance(action, RecruitAction):
                    raise StrengthAuditError("recruit phase contains a non-recruit action")
                actor = state.active_player.other()
                actor_index = player_index(actor)
                agent_id = record.seats[actor_index].agent_id
                if agent_id == target_agent_id:
                    if state.offer is None:  # pragma: no cover - valid state guarantees this
                        raise StrengthAuditError("recruit state has no offer")
                    opponent_id = record.seats[1 - actor_index].agent_id
                    observation = observe(state, actor)
                    face_up = state.offer.face_up.value
                    face_down = state.offer.face_down.value
                    viewer = next(
                        player
                        for player in observation.players
                        if player.player is observation.viewer
                    )
                    visible_copies = min(viewer.recruited.count(state.offer.face_up), 2)
                    alternative = RecruitAction(
                        action.revision,
                        action.actor,
                        OfferSlot.FACE_DOWN
                        if action.slot is OfferSlot.FACE_UP
                        else OfferSlot.FACE_UP,
                    )
                    chosen_swing, chosen_immediate = _score_swing(state, actor, action)
                    alternative_swing, alternative_immediate = _score_swing(
                        state, actor, alternative
                    )
                    values = {
                        "decisions": 1,
                        "face_up_choices": int(action.slot is OfferSlot.FACE_UP),
                        "target_game_wins": int(record.winner is actor),
                        "target_immediate_wins": int(chosen_immediate),
                        "chosen_score_swing_sum": chosen_swing,
                        "alternative_score_swing_sum": alternative_swing,
                        "alternative_score_swing_better": int(alternative_swing > chosen_swing),
                        "hindsight_alternative_immediate_win": int(
                            alternative_immediate and not chosen_immediate
                        ),
                    }
                    rows = (
                        overall,
                        by_face_up.setdefault(face_up, _empty_pattern_counts()),
                        by_face_up_copy_count.setdefault(
                            f"{face_up}:{visible_copies}", _empty_pattern_counts()
                        ),
                        by_opponent.setdefault(opponent_id, _empty_pattern_counts()),
                        by_ordered_offer.setdefault(
                            f"{face_up}|{face_down}", _empty_pattern_counts()
                        ),
                        by_opponent_offer.setdefault(
                            f"{opponent_id}|{face_up}|{face_down}", _empty_pattern_counts()
                        ),
                    )
                    for counts in rows:
                        counts.update(values)
            state = apply_action(state, action)

    if overall["decisions"] == 0:
        raise StrengthAuditError("target agent has no recruit decisions in the supplied records")
    data: dict[str, object] = {
        "version": RECRUIT_PATTERN_AUDIT_VERSION,
        "source_label": source_label,
        "target_agent_id": target_agent_id,
        "record_count": record_count,
        "overall": _pattern_data(overall),
        "by_face_up": {key: _pattern_data(value) for key, value in sorted(by_face_up.items())},
        "by_face_up_copy_count": {
            key: _pattern_data(value) for key, value in sorted(by_face_up_copy_count.items())
        },
        "by_opponent": {key: _pattern_data(value) for key, value in sorted(by_opponent.items())},
        "by_ordered_offer": {
            key: _pattern_data(value) for key, value in sorted(by_ordered_offer.items())
        },
        "by_opponent_ordered_offer": {
            key: _pattern_data(value) for key, value in sorted(by_opponent_offer.items())
        },
        "artifact_fingerprint": "",
    }
    data["artifact_fingerprint"] = _artifact_fingerprint(data)
    return data


def paired_policy_difference_interval(
    treatment_pair_wins: tuple[int, ...],
    control_pair_wins: tuple[int, ...],
    *,
    master_seed: int,
    domain: str,
) -> dict[str, object]:
    """Bootstrap a matched difference in policy win rate over shared two-game setup blocks."""
    if not treatment_pair_wins or len(treatment_pair_wins) != len(control_pair_wins):
        raise StrengthAuditError("paired policy comparison requires aligned non-empty blocks")
    if any(
        type(value) is not int or not 0 <= value <= 2
        for value in (*treatment_pair_wins, *control_pair_wins)
    ):
        raise StrengthAuditError("paired policy wins must be integers from zero through two")
    if not domain:
        raise StrengthAuditError("paired policy comparison requires an RNG domain")
    differences = tuple(
        (treatment - control) / 2
        for treatment, control in zip(treatment_pair_wins, control_pair_wins, strict=True)
    )
    seed = derive_seed(master_seed, domain)
    rng = DeterministicRandom(seed, domain)
    block_count = len(differences)
    resampled = [
        sum(differences[rng.randbelow(block_count)] for _ in range(block_count)) / block_count
        for _ in range(PAIRED_BOOTSTRAP_RESAMPLES)
    ]
    resampled.sort()
    return {
        "version": PAIRED_DIFFERENCE_VERSION,
        "statistic": "treatment_minus_control_win_rate",
        "unit": "shared two-game paired setup block",
        "block_count": block_count,
        "point_estimate": sum(differences) / block_count,
        "confidence_level": 0.95,
        "interval": [
            resampled[PAIRED_BOOTSTRAP_LOWER_INDEX],
            resampled[PAIRED_BOOTSTRAP_UPPER_INDEX],
        ],
        "resample_count": PAIRED_BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": seed,
        "bootstrap_rng_domain": domain,
    }

from __future__ import annotations

import ast
import hashlib
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from agent_avenue.agents import DeterministicRandom, RandomAgent
from agent_avenue.engine import GameConfig, Phase, PlayerId, apply_action, legal_actions, new_game
from agent_avenue.engine.model import GameState
from agent_avenue.observation import observe
from agent_avenue.rollout import (
    EXPANDED_CANONICAL_DECK,
    REQUIRED_POLICY_IDS,
    LatentRolloutState,
    PanelPosition,
    RolloutTargetError,
    canonical_observation,
    continuation_policy_assignments,
    latent_legal_actions,
    rollout_candidate_world,
    rollout_transition_v1,
    safe_latent_observation,
    safe_position_identity,
    sample_latent_world,
    sample_position_worlds,
    stratum_for_observation,
    validate_latent_state,
)


def _latent_from_engine(state: GameState) -> LatentRolloutState:
    return LatentRolloutState(
        state.deck,
        state.hands,
        state.recruited,
        state.scores,
        state.active_player,
        state.turn,
        state.phase,
        state.revision,
        state.offer,
        state.history,
        state.outcome,
    )


def _position(seed: int = 5) -> PanelPosition:
    state = new_game(seed=seed)
    observation = observe(state, state.active_player)
    stratum = stratum_for_observation(observation)
    assert stratum is not None
    safe_id = safe_position_identity(observation, replicate_id="replicate-test", stratum=stratum)
    selection_hash = hashlib.sha256(
        f"step4:panel:replicate-test:{stratum}:{safe_id}".encode()
    ).hexdigest()
    return PanelPosition(
        "replicate-test",
        stratum,
        safe_id,
        selection_hash,
        canonical_observation(observation),
        "0" * 64,
    )


def test_expanded_deck_and_safe_recruit_observer_preserve_copy_and_hidden_boundaries() -> None:
    state = new_game(seed=9)
    assert Counter(EXPANDED_CANONICAL_DECK) == Counter(state.config.deck)
    played = apply_action(state, legal_actions(state)[0])
    recruiter = played.active_player.other()
    observation = observe(played, recruiter)
    latent = sample_latent_world(observation, DeterministicRandom(17, "sampler"))

    validate_latent_state(latent)
    safe = safe_latent_observation(latent, recruiter)
    assert safe.own_hand == tuple(sorted(safe.own_hand, key=tuple(type(safe.own_hand[0])).index))
    assert getattr(safe.decision, "known_face_down", None) is None
    cards = (
        latent.deck + latent.hands[0] + latent.hands[1] + latent.recruited[0] + latent.recruited[1]
    )
    assert latent.offer is not None
    assert Counter((*cards, latent.offer.face_up, latent.offer.face_down)) == Counter(
        EXPANDED_CANONICAL_DECK
    )


@pytest.mark.parametrize(
    ("seed", "starting_player"),
    (
        (3, PlayerId.PLAYER_ONE),
        (5, PlayerId.PLAYER_TWO),
        (17, PlayerId.PLAYER_ONE),
        (33, PlayerId.PLAYER_TWO),
    ),
)
def test_latent_transitions_are_engine_equivalent_for_both_decision_types(
    seed: int, starting_player: PlayerId
) -> None:
    state = new_game(GameConfig(starting_player=starting_player), seed=seed)
    for _ in range(18):
        action = legal_actions(state)[-1]
        latent = _latent_from_engine(state)
        assert latent_legal_actions(latent) == legal_actions(state)
        next_latent = rollout_transition_v1(latent, action)
        next_engine = apply_action(state, action)
        assert next_latent == _latent_from_engine(next_engine)
        state = next_engine
        if state.phase is Phase.TERMINAL:
            break
    assert state.phase is Phase.TERMINAL


def test_identity_and_policy_design_ignore_source_audit_and_candidate_order() -> None:
    position = _position()
    changed_audit = replace(position, audit_record_fingerprint="f" * 64)
    assert position.safe_identity == changed_audit.safe_identity
    first = continuation_policy_assignments(
        root_seed=2026091204,
        replicate_id=position.replicate_id,
        safe_identity=position.safe_identity,
    )
    second = continuation_policy_assignments(
        root_seed=2026091204,
        replicate_id=position.replicate_id,
        safe_identity=changed_audit.safe_identity,
    )
    assert first == second
    for seat in (PlayerId.PLAYER_ONE, PlayerId.PLAYER_TWO):
        policies = [assignment.policy_for(seat) for assignment in first]
        assert policies.count("q0") == 4
        assert {"q1", "q2", "q3", "q4", "heuristic", "random"} <= set(policies)


def test_static_latent_boundary_does_not_import_authoritative_observer_or_game_state() -> None:
    tree = ast.parse(Path("src/agent_avenue/rollout/latent.py").read_text())
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "agent_avenue.observation.build" not in imports
    assert "GameState" not in names


class _RecordingRandom(RandomAgent):
    calls: int = 0

    def choose_action(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        type(self).calls += 1
        return super().choose_action(*args, **kwargs)


def test_forced_root_is_not_policy_selected_and_leaf_uses_root_viewpoint() -> None:
    position = _position()
    worlds = sample_position_worlds(position)
    _RecordingRandom.calls = 0
    policies = {policy_id: _RecordingRandom() for policy_id in REQUIRED_POLICY_IDS}
    sample = rollout_candidate_world(
        position,
        next(reversed(position.observation.legal_actions)),
        worlds[0],
        continuation_policies=policies,
        leaf_scorer=lambda _observation, actions: [2.0] * len(actions),
    )
    assert _RecordingRandom.calls == len(sample.continuation_rng_seeds)
    assert sample.actions_applied >= 1
    if sample.depth_leaf:
        # The initial player acts at the root; at depth nine the recruit decision belongs to P2.
        assert sample.root_value < 0.5
    else:
        assert sample.root_value in (0.0, 1.0)


def test_depth_leaf_gate_rejects_all_leaf_target_design() -> None:
    position = _position()
    worlds = sample_position_worlds(position)
    policies = {policy_id: RandomAgent() for policy_id in REQUIRED_POLICY_IDS}
    samples = [
        rollout_candidate_world(
            position,
            action,
            worlds[0],
            continuation_policies=policies,
            leaf_scorer=lambda _observation, actions: [0.0] * len(actions),
        )
        for action in position.observation.legal_actions
    ]
    assert all(sample.depth_leaf or sample.terminal for sample in samples)
    assert not any(sample.actions_applied > 9 for sample in samples)
    with pytest.raises(RolloutTargetError):
        rollout_candidate_world(
            position,
            position.observation.legal_actions[0],
            worlds[0],
            continuation_policies={"q0": RandomAgent()},
            leaf_scorer=lambda _observation, actions: [0.0] * len(actions),
        )

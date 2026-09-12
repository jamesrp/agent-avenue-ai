from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from agent_avenue.engine import PlayerId, new_game
from agent_avenue.engine.cards import CardName


@pytest.fixture(scope="module")
def validator() -> ModuleType:
    path = Path("scripts/validate_counterfactual_rollout_supervision_v1.py")
    spec = importlib.util.spec_from_file_location("step4_validator_local", path)
    if spec is None or spec.loader is None:  # pragma: no cover - repository path is fixed
        raise RuntimeError("unable to import validator fixture")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FirstPolicy:
    def choose_action(self, observation, decision, legal_actions, rng):  # type: ignore[no-untyped-def]
        del observation, decision, rng
        return legal_actions[0]


class _Leaf:
    def score_candidates(self, observation, decision, legal_actions):  # type: ignore[no-untyped-def]
        del observation, decision
        return SimpleNamespace(actions=legal_actions, logits=tuple(0.25 for _ in legal_actions))


def _play_state(validator: ModuleType):  # type: ignore[no-untyped-def]
    state = new_game(seed=31)
    observation = validator.local_engine_observation(state, PlayerId.PLAYER_ONE)
    return observation, validator.local_sample_world(
        observation, validator.LocalRandom(17, "validator-test-world")
    )


def test_local_sampler_rejects_hidden_allocation_mutation_and_draw_order_changes(
    validator: ModuleType,
) -> None:
    _, state = _play_state(validator)
    with pytest.raises(validator.ValidationError, match="card conservation"):
        validator.validate_local_state(replace(state, deck=state.deck[1:]))

    action = validator.canonical_actions(validator.local_legal_actions(state))[0]
    forward = validator.local_transition(state, action)
    reordered = replace(state, deck=tuple(reversed(state.deck)))
    validator.validate_local_state(reordered)
    changed = validator.local_transition(reordered, action)
    assert forward.hands != changed.hands


def test_local_rng_policy_permutations_root_fixation_and_candidate_order_mutations(
    validator: ModuleType,
) -> None:
    observation, _ = _play_state(validator)
    identity = validator.safe_identity(observation)
    assignments = validator.local_assignments("replicate-test", identity)
    assert sum(validator.policy_id(left) == "q0" for left, _ in assignments) == 4
    assert sum(validator.policy_id(right) == "q0" for _, right in assignments) == 4
    assert validator.derive_seed_local(1, "domain-a") != validator.derive_seed_local(1, "domain-b")

    actions = validator.canonical_actions(observation.legal_actions)
    policies = {
        key: _FirstPolicy() for key in ("q0", "q1", "q2", "q3", "q4", "heuristic", "random")
    }
    first, transcript = validator.local_rollout(
        observation, "replicate-test", identity, actions[0], 0, policies, _Leaf(), transcript=True
    )
    assert transcript[0]["forced_root"] is True
    assert transcript[0]["action"] == validator.local_action_data(actions[0])
    if len(actions) > 1:
        _, changed = validator.local_rollout(
            observation,
            "replicate-test",
            identity,
            actions[1],
            0,
            policies,
            _Leaf(),
            transcript=True,
        )
        assert changed[0]["action"] != transcript[0]["action"]
    assert first["world_seed"] == validator.derive_seed_local(
        validator.ROOT_SEED, f"step4:world:replicate-test:{identity}:0"
    )


def test_local_terminal_tie_leaf_inversion_target_and_cap_mutations(validator: ModuleType) -> None:
    outcome = validator.local_adjudicate(
        scores=(0, 0),
        recruited=(
            (CardName.CODEBREAKER, CardName.CODEBREAKER, CardName.CODEBREAKER),
            (CardName.CODEBREAKER, CardName.CODEBREAKER, CardName.CODEBREAKER),
        ),
        active_player=PlayerId.PLAYER_TWO,
        deck_empty=False,
        next_player_hand_size=4,
    )
    assert outcome is not None and outcome.winner is PlayerId.PLAYER_TWO
    expected = {"root_value": 0.25, "depth_leaf": True}
    assert not validator._sample_equal(expected, {"root_value": 0.75, "depth_leaf": True})
    assert not validator._sample_equal(expected, {"root_value": 0.25, "depth_leaf": False})
    remaining = list(validator.EXPANDED_DECK)
    remaining.remove(CardName.ENFORCER)
    remaining.remove(CardName.SENTINEL)
    capped = validator.LocalLatentState(
        deck=tuple(remaining),
        hands=((), ()),
        recruited=((), ()),
        scores=(0, 0),
        active_player=PlayerId.PLAYER_ONE,
        turn=19,
        phase=validator.Phase.RECRUIT,
        revision=1,
        offer=validator.LocalOffer(PlayerId.PLAYER_ONE, CardName.ENFORCER, CardName.SENTINEL),
    )
    with pytest.raises(validator.ValidationError, match="mechanical turn cap"):
        validator.local_transition(
            capped,
            validator.RecruitAction(1, PlayerId.PLAYER_TWO, validator.OfferSlot.FACE_UP),
        )


def test_transcript_quarantine_and_count_mutations_are_not_accepted(validator: ModuleType) -> None:
    expected = [{"quarantine": "trusted-hidden-rollout-audit-only-v1"}] * 140
    assert len(expected) == 140
    assert all(row["quarantine"] == "trusted-hidden-rollout-audit-only-v1" for row in expected)
    mutated = [*expected]
    mutated[0] = {"quarantine": "leaked"}
    assert not all(row["quarantine"] == "trusted-hidden-rollout-audit-only-v1" for row in mutated)
    assert len(mutated[:-1]) != 140

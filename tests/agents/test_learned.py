from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from agent_avenue.agents.learned import LearnedValueAgent, LearnedValueConfig  # noqa: E402
from agent_avenue.agents.ordering import semantic_action_key  # noqa: E402
from agent_avenue.engine import PlayerId, new_game  # noqa: E402
from agent_avenue.learning import create_model, load_checkpoint, save_checkpoint  # noqa: E402
from agent_avenue.observation import observe  # noqa: E402


class FixedRandom:
    def __init__(self, value: int) -> None:
        self.value = value
        self.calls = 0

    def randbelow(self, upper_bound: int) -> int:
        self.calls += 1
        return self.value % upper_bound


def _checkpoint(path: Path):  # type: ignore[no-untyped-def]
    model = create_model(seed=7)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    save_checkpoint(path, model, metrics={"fixture": True})
    return load_checkpoint(path)


def test_learned_config_round_trip_excludes_checkpoint_path(tmp_path: Path) -> None:
    loaded = _checkpoint(tmp_path / "checkpoint")
    config = LearnedValueConfig.from_checkpoint(loaded)
    data = config.to_data()
    assert LearnedValueConfig.from_data(data) == config
    assert str(tmp_path) not in repr(data)

    malformed = dict(data)
    malformed["unexpected"] = True
    with pytest.raises(ValueError, match="fields"):
        LearnedValueConfig.from_data(malformed)


def test_learned_agent_batches_candidates_and_ties_by_semantic_order(tmp_path: Path) -> None:
    loaded = _checkpoint(tmp_path / "checkpoint")
    agent = LearnedValueAgent.from_checkpoint(loaded)
    state = new_game(seed=17)
    observation = observe(state, PlayerId.PLAYER_ONE)
    reversed_actions = tuple(reversed(observation.legal_actions))
    reversed_observation = replace(observation, legal_actions=reversed_actions)

    first_rng = FixedRandom(0)
    reversed_rng = FixedRandom(0)
    first = agent.choose_action(
        observation, observation.decision, observation.legal_actions, first_rng
    )
    second = agent.choose_action(
        reversed_observation,
        reversed_observation.decision,
        reversed_observation.legal_actions,
        reversed_rng,
    )
    assert first == second
    assert first_rng.calls == reversed_rng.calls == 1


def test_learned_scoring_diagnostic_exposes_exact_ties_in_semantic_order(
    tmp_path: Path,
) -> None:
    agent = LearnedValueAgent.from_checkpoint(_checkpoint(tmp_path / "checkpoint"))
    observation = observe(new_game(seed=17), PlayerId.PLAYER_ONE)
    reversed_actions = tuple(reversed(observation.legal_actions))
    reversed_observation = replace(observation, legal_actions=reversed_actions)

    scores = agent.score_candidates(
        reversed_observation,
        reversed_observation.decision,
        reversed_observation.legal_actions,
    )

    assert set(scores.actions) == set(observation.legal_actions)
    assert scores.actions == tuple(sorted(reversed_actions, key=semantic_action_key))
    assert scores.logits == tuple(0.0 for _ in scores.actions)
    assert scores.maximum_indices == tuple(range(len(scores.actions)))
    assert scores.tie_count == len(scores.actions)
    assert scores.maximum_actions == scores.actions


def test_learned_choose_action_uses_the_scoring_diagnostic_without_behavior_change(
    tmp_path: Path,
) -> None:
    agent = LearnedValueAgent.from_checkpoint(_checkpoint(tmp_path / "checkpoint"))
    observation = observe(new_game(seed=17), PlayerId.PLAYER_ONE)
    scores = agent.score_candidates(observation, observation.decision, observation.legal_actions)
    rng = FixedRandom(2)

    chosen = agent.choose_action(observation, observation.decision, observation.legal_actions, rng)

    assert chosen == scores.actions[scores.maximum_indices[2]]
    assert rng.calls == 1


def test_learned_agent_rejects_inconsistent_turn(tmp_path: Path) -> None:
    agent = LearnedValueAgent.from_checkpoint(_checkpoint(tmp_path / "checkpoint"))
    observation = observe(new_game(seed=5), PlayerId.PLAYER_ONE)
    with pytest.raises(ValueError, match="consistent"):
        agent.choose_action(
            observation,
            observation.decision,
            tuple(reversed(observation.legal_actions)),
            FixedRandom(0),
        )

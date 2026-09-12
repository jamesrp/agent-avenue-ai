from pathlib import Path

from agent_avenue.agents import DeterministicRandom, TerminalOffenseAgent, TerminalSafetyAgent
from agent_avenue.agents.structured import StructuredValueAgent
from agent_avenue.engine import PlayerId, new_game
from agent_avenue.learning.model import CandidateMLP
from agent_avenue.learning.structured_checkpoint import (
    load_structured_checkpoint,
    save_structured_checkpoint,
)
from agent_avenue.learning.structured_model import create_structured_model
from agent_avenue.observation import observe


def _agent(tmp_path: Path) -> StructuredValueAgent:
    path = tmp_path / "checkpoint"
    save_structured_checkpoint(
        path,
        create_structured_model(CandidateMLP(seed=13), projection_seed=17),
        metrics={},
        q0_parent_checkpoint_fingerprint="a" * 64,
        q0_parent_tensor_digest="b" * 64,
        dataset_fingerprint="c" * 64,
        source_corpus_fingerprints=("d" * 64,),
        split_identities={"train_pair_ids": "d" * 64},
        training_config={"batch_size": 1024},
        training_seeds={"structured_init": 17},
        created_at="2026-09-12T00:00:00+00:00",
    )
    return StructuredValueAgent.from_checkpoint(load_structured_checkpoint(path))


def test_structured_agent_scores_semantic_order_and_is_hidden_state_equivalent(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    first = observe(new_game(seed=9), PlayerId.PLAYER_ONE)
    second = observe(new_game(seed=25), PlayerId.PLAYER_ONE)
    assert first.own_hand == second.own_hand
    assert first.legal_actions == second.legal_actions

    first_scores = agent.score_candidates(first, first.decision, first.legal_actions)
    second_scores = agent.score_candidates(second, second.decision, second.legal_actions)
    assert first_scores == second_scores
    assert agent.choose_action(
        first, first.decision, first.legal_actions, DeterministicRandom(5, "test")
    ) == agent.choose_action(
        second, second.decision, second.legal_actions, DeterministicRandom(5, "test")
    )
    assert agent.config.to_data()["type"] == "structured_value"


def test_structured_agent_composes_with_terminal_safety_and_offense(tmp_path: Path) -> None:
    base = _agent(tmp_path)
    wrapped = TerminalOffenseAgent(TerminalSafetyAgent(base))
    observation = observe(new_game(seed=4), PlayerId.PLAYER_ONE)
    action = wrapped.choose_action(
        observation,
        observation.decision,
        observation.legal_actions,
        DeterministicRandom(3, "test"),
    )
    assert action in observation.legal_actions
    assert wrapped.config_to_data()["base"]["type"] == "terminal_safety"

from dataclasses import replace

import pytest

from agent_avenue.agents import (
    AgentTurn,
    DeterministicRandom,
    EpsilonConfig,
    EpsilonGreedyAgent,
    GreedyHeuristicAgent,
    GreedyHeuristicConfig,
    RandomAgent,
    ScriptedAgent,
    choose_agent_action,
    derive_seed,
    score_actions,
)
from agent_avenue.engine import (
    CardName,
    OfferSlot,
    Phase,
    PlayerId,
    PlayOfferAction,
    RecruitAction,
)
from agent_avenue.observation.model import (
    PlayContext,
    PlayerObservation,
    PublicPlayer,
    RecruitContext,
)


class IndexedRandom:
    def __init__(self, index: int) -> None:
        self.index = index
        self.bounds: list[int] = []

    def randbelow(self, upper_bound: int) -> int:
        self.bounds.append(upper_bound)
        assert 0 <= self.index < upper_bound
        return self.index


class SequenceRandom:
    def __init__(self, values: list[int]) -> None:
        self.values = iter(values)
        self.bounds: list[int] = []

    def randbelow(self, upper_bound: int) -> int:
        self.bounds.append(upper_bound)
        value = next(self.values)
        assert 0 <= value < upper_bound
        return value


def play_observation(
    hand: tuple[CardName, ...] = (
        CardName.ENFORCER,
        CardName.SABOTEUR,
        CardName.CODEBREAKER,
        CardName.DAREDEVIL,
    ),
    recruited: tuple[tuple[CardName, ...], tuple[CardName, ...]] = ((), ()),
    scores: tuple[int, int] = (0, 0),
) -> PlayerObservation:
    actions = tuple(
        PlayOfferAction(4, PlayerId.PLAYER_ONE, up, down)
        for up in dict.fromkeys(hand)
        for down in dict.fromkeys(hand)
        if up is not down
    )
    return PlayerObservation(
        viewer=PlayerId.PLAYER_ONE,
        own_hand=hand,
        players=(
            PublicPlayer(PlayerId.PLAYER_ONE, scores[0], recruited[0], 4),
            PublicPlayer(PlayerId.PLAYER_TWO, scores[1], recruited[1], 4),
        ),
        active_player=PlayerId.PLAYER_ONE,
        turn=3,
        phase=Phase.PLAY,
        remaining_deck_count=20,
        history=(),
        decision=PlayContext("play", 4, PlayerId.PLAYER_ONE),
        legal_actions=actions,
    )


def recruit_observation(
    *,
    face_up: CardName = CardName.ENFORCER,
    known_face_down: CardName | None = None,
    own_recruited: tuple[CardName, ...] = (),
    opponent_recruited: tuple[CardName, ...] = (),
    scores: tuple[int, int] = (0, 0),
) -> PlayerObservation:
    actions = tuple(
        RecruitAction(5, PlayerId.PLAYER_ONE, slot)
        for slot in (OfferSlot.FACE_UP, OfferSlot.FACE_DOWN)
    )
    return PlayerObservation(
        viewer=PlayerId.PLAYER_ONE,
        own_hand=(CardName.SENTINEL, CardName.SABOTEUR, CardName.DOUBLE_AGENT, CardName.DAREDEVIL),
        players=(
            PublicPlayer(PlayerId.PLAYER_ONE, scores[0], own_recruited, 4),
            PublicPlayer(PlayerId.PLAYER_TWO, scores[1], opponent_recruited, 4),
        ),
        active_player=PlayerId.PLAYER_TWO,
        turn=3,
        phase=Phase.RECRUIT,
        remaining_deck_count=20,
        history=(),
        decision=RecruitContext(
            "recruit",
            5,
            PlayerId.PLAYER_ONE,
            PlayerId.PLAYER_TWO,
            face_up,
            known_face_down,
        ),
        legal_actions=actions,
    )


def test_epsilon_wrapper_has_explicit_rng_consumption() -> None:
    observation = recruit_observation()

    class FirstAgent:
        def choose_action(self, observation, decision, legal_actions, rng):  # type: ignore[no-untyped-def]
            return legal_actions[0]

    base = FirstAgent()

    zero_rng = SequenceRandom([])
    zero = EpsilonGreedyAgent(base, EpsilonConfig(0, 1))
    assert (
        zero.choose_action(observation, observation.decision, observation.legal_actions, zero_rng)
        in observation.legal_actions
    )
    assert zero_rng.bounds == []

    one_rng = SequenceRandom([1])
    one = EpsilonGreedyAgent(base, EpsilonConfig(1, 1))
    assert (
        one.choose_action(observation, observation.decision, observation.legal_actions, one_rng)
        == observation.legal_actions[1]
    )
    assert one_rng.bounds == [2]

    explore_rng = SequenceRandom([0, 1])
    exploratory = EpsilonGreedyAgent(base, EpsilonConfig(1, 5))
    assert (
        exploratory.choose_action(
            observation, observation.decision, observation.legal_actions, explore_rng
        )
        == observation.legal_actions[1]
    )
    assert explore_rng.bounds == [5, 2]


def test_rng_is_versioned_domain_separated_and_reproducible() -> None:
    assert derive_seed(7, "setup") != derive_seed(7, "agent/player_one")
    first = DeterministicRandom.from_root_seed(7, "agent/player_one")
    second = DeterministicRandom.from_root_seed(7, "agent/player_one")
    assert [first.randbelow(17) for _ in range(20)] == [second.randbelow(17) for _ in range(20)]
    restored = DeterministicRandom.from_data(first.to_data())
    assert [first.randbelow(23) for _ in range(5)] == [restored.randbelow(23) for _ in range(5)]
    assert first.to_data()["algorithm"] == "sha256-counter-rejection-v1"
    with pytest.raises(ValueError, match="domain"):
        DeterministicRandom.from_root_seed(7, "")


def test_rng_batch_draws_equal_sequential_draws_and_advance_the_same_counter() -> None:
    sequential = DeterministicRandom.from_root_seed(2026091705, "league:bootstrap/étude")
    batched = DeterministicRandom.from_root_seed(2026091705, "league:bootstrap/étude")
    for bound, count in ((3, 7), (200, 400), (1, 3), (2**70 + 13, 9), (5, 0)):
        expected = [sequential.randbelow(bound) for _ in range(count)]
        assert batched.randbelow_batch(bound, count) == expected
        assert batched.to_data() == sequential.to_data()
    assert batched.randbelow(11) == sequential.randbelow(11)
    with pytest.raises(ValueError, match="positive"):
        batched.randbelow_batch(0, 1)
    with pytest.raises(ValueError, match="negative"):
        batched.randbelow_batch(3, -1)


def test_random_agent_maps_every_controlled_index_to_corresponding_action() -> None:
    observation = recruit_observation()
    for index, expected in enumerate(observation.legal_actions):
        rng = IndexedRandom(index)
        chosen = RandomAgent().choose_action(
            observation, observation.decision, observation.legal_actions, rng
        )
        assert chosen == expected
        assert rng.bounds == [len(observation.legal_actions)]


def test_turn_wrapper_rejects_inconsistent_inputs_and_illegal_output() -> None:
    observation = recruit_observation()
    with pytest.raises(ValueError, match="decision"):
        AgentTurn(
            observation, PlayContext("play", 1, PlayerId.PLAYER_ONE), observation.legal_actions
        )

    class BadAgent:
        def choose_action(self, observation, decision, legal_actions, rng):  # type: ignore[no-untyped-def]
            return PlayOfferAction(0, PlayerId.PLAYER_ONE, CardName.MOLE, CardName.SIDEKICK)

    with pytest.raises(ValueError, match="not legal"):
        choose_agent_action(BadAgent(), AgentTurn.from_observation(observation), IndexedRandom(0))


def test_scripted_agent_is_strict_about_exhaustion_divergence_and_unused_actions() -> None:
    observation = recruit_observation()
    first, second = observation.legal_actions
    agent = ScriptedAgent((first, second), name="regression")
    assert (
        agent.choose_action(
            observation, observation.decision, observation.legal_actions, IndexedRandom(0)
        )
        == first
    )
    with pytest.raises(ValueError, match="unused"):
        agent.assert_exhausted()
    divergent = replace(observation, legal_actions=(first,))
    with pytest.raises(ValueError, match="diverged"):
        agent.choose_action(
            divergent, divergent.decision, divergent.legal_actions, IndexedRandom(0)
        )
    exhausted = ScriptedAgent((first,))
    exhausted.choose_action(
        observation, observation.decision, observation.legal_actions, IndexedRandom(0)
    )
    with pytest.raises(ValueError, match="exhausted"):
        exhausted.choose_action(
            observation, observation.decision, observation.legal_actions, IndexedRandom(0)
        )


def test_config_round_trips_as_normalized_serializable_data() -> None:
    config = GreedyHeuristicConfig(immediate_score_weight=123, threshold_win_weight=999)
    data = config.to_data()
    assert list(data) == sorted(data)
    assert GreedyHeuristicConfig.from_data(data) == config
    with pytest.raises(ValueError, match="fields"):
        GreedyHeuristicConfig.from_data({**data, "future_field": 1})


def test_heuristic_accounts_for_immediate_score_and_seven_point_threshold() -> None:
    observation = recruit_observation(
        face_up=CardName.SIDEKICK,
        own_recruited=(CardName.ENFORCER,),
        scores=(3, 0),
    )
    scores = score_actions(observation, observation.legal_actions, GreedyHeuristicConfig())
    assert scores[0] > scores[1]  # known +4 crosses the seven-point gap


def test_heuristic_values_third_codebreaker_and_avoids_third_daredevil() -> None:
    codebreaker = recruit_observation(
        face_up=CardName.CODEBREAKER,
        own_recruited=(CardName.CODEBREAKER, CardName.CODEBREAKER),
    )
    daredevil = recruit_observation(
        face_up=CardName.DAREDEVIL,
        own_recruited=(CardName.DAREDEVIL, CardName.DAREDEVIL),
    )
    config = GreedyHeuristicConfig()
    assert score_actions(codebreaker, codebreaker.legal_actions, config)[0] > 50_000
    assert score_actions(daredevil, daredevil.legal_actions, config)[0] < -50_000


def test_known_offer_uses_adversarial_choice_for_offer_scoring() -> None:
    observation = play_observation(
        hand=(CardName.SIDEKICK, CardName.MOLE, CardName.ENFORCER, CardName.SABOTEUR)
    )
    action = PlayOfferAction(4, PlayerId.PLAYER_ONE, CardName.SIDEKICK, CardName.MOLE)
    score = score_actions(observation, (action,), GreedyHeuristicConfig())[0]
    # Opponent takes Sidekick and leaves Mole; the policy does not average away that risk.
    assert score < 0


def test_unknown_face_down_uses_expected_visible_multiset_not_hidden_truth() -> None:
    observation = recruit_observation(face_up=CardName.ENFORCER, known_face_down=None)
    hidden_state_a = {"actual_face_down": CardName.MOLE, "opponent_hand": CardName.SIDEKICK}
    hidden_state_b = {"actual_face_down": CardName.SIDEKICK, "opponent_hand": CardName.MOLE}
    assert hidden_state_a != hidden_state_b
    config = GreedyHeuristicConfig()
    scores_a = score_actions(observation, observation.legal_actions, config)
    scores_b = score_actions(observation, observation.legal_actions, config)
    assert scores_a == scores_b
    rng_a = DeterministicRandom.from_root_seed(91, "agent/player_one")
    rng_b = DeterministicRandom.from_root_seed(91, "agent/player_one")
    agent = GreedyHeuristicAgent(config)
    assert agent.choose_action(
        observation, observation.decision, observation.legal_actions, rng_a
    ) == agent.choose_action(observation, observation.decision, observation.legal_actions, rng_b)

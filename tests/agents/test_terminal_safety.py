import subprocess
import sys
from dataclasses import replace

from agent_avenue.agents import (
    EpsilonConfig,
    EpsilonGreedyAgent,
    RandomAgent,
    TerminalSafetyAgent,
    filter_terminal_actions,
    information_consistent_face_down_cards,
)
from agent_avenue.engine import (
    CardName,
    OfferSlot,
    Phase,
    PlayerId,
    PlayOfferAction,
    RecruitAction,
    apply_action,
    legal_actions,
    new_game,
)
from agent_avenue.observation import observe
from agent_avenue.observation.model import (
    PlayContext,
    PlayerObservation,
    PublicPlayer,
    RecruitContext,
)
from agent_avenue.runners.game import AgentSpec, GameSpec, run_game
from agent_avenue.storage import game_record_fingerprint


class RecordingRandom:
    def __init__(self, values: tuple[int, ...]) -> None:
        self.values = iter(values)
        self.bounds: list[int] = []

    def randbelow(self, upper_bound: int) -> int:
        self.bounds.append(upper_bound)
        value = next(self.values)
        assert 0 <= value < upper_bound
        return value


def recruit_observation(
    *,
    face_up: CardName,
    known_face_down: CardName | None = None,
    own_recruited: tuple[CardName, ...] = (),
    opponent_recruited: tuple[CardName, ...] = (),
    own_hand: tuple[CardName, ...] = (
        CardName.ENFORCER,
        CardName.SABOTEUR,
        CardName.SENTINEL,
        CardName.DOUBLE_AGENT,
    ),
    scores: tuple[int, int] = (0, 0),
    remaining_deck_count: int = 20,
    opponent_hand_size: int = 4,
) -> PlayerObservation:
    actions = tuple(RecruitAction(5, PlayerId.PLAYER_ONE, slot) for slot in OfferSlot)
    return PlayerObservation(
        viewer=PlayerId.PLAYER_ONE,
        own_hand=own_hand,
        players=(
            PublicPlayer(PlayerId.PLAYER_ONE, scores[0], own_recruited, len(own_hand)),
            PublicPlayer(PlayerId.PLAYER_TWO, scores[1], opponent_recruited, opponent_hand_size),
        ),
        active_player=PlayerId.PLAYER_TWO,
        turn=3,
        phase=Phase.RECRUIT,
        remaining_deck_count=remaining_deck_count,
        history=(),
        decision=RecruitContext(
            "recruit", 5, PlayerId.PLAYER_ONE, PlayerId.PLAYER_TWO, face_up, known_face_down
        ),
        legal_actions=actions,
    )


def play_observation(
    *,
    hand: tuple[CardName, ...],
    recruited: tuple[tuple[CardName, ...], tuple[CardName, ...]] = ((), ()),
    actions: tuple[PlayOfferAction, ...],
) -> PlayerObservation:
    return PlayerObservation(
        viewer=PlayerId.PLAYER_ONE,
        own_hand=hand,
        players=(
            PublicPlayer(PlayerId.PLAYER_ONE, 0, recruited[0], len(hand)),
            PublicPlayer(PlayerId.PLAYER_TWO, 0, recruited[1], 4),
        ),
        active_player=PlayerId.PLAYER_ONE,
        turn=3,
        phase=Phase.PLAY,
        remaining_deck_count=20,
        history=(),
        decision=PlayContext("play", 4, PlayerId.PLAYER_ONE),
        legal_actions=actions,
    )


def test_visible_third_daredevil_is_vetoed_when_other_recruit_is_not_provably_losing() -> None:
    observation = recruit_observation(
        face_up=CardName.DAREDEVIL,
        own_recruited=(CardName.DAREDEVIL, CardName.DAREDEVIL),
    )
    result = filter_terminal_actions(observation, observation.legal_actions)
    face_up, face_down = observation.legal_actions
    assert result.allowed_actions == (face_down,)
    assert result.provable_loss_actions == (face_up,)
    assert result.vetoed_actions == (face_up,)
    assert not result.forced_loss_fallback


def test_offer_that_can_leave_offeror_a_third_daredevil_is_vetoed() -> None:
    fatal = PlayOfferAction(4, PlayerId.PLAYER_ONE, CardName.ENFORCER, CardName.DAREDEVIL)
    safe = PlayOfferAction(4, PlayerId.PLAYER_ONE, CardName.ENFORCER, CardName.SIDEKICK)
    observation = play_observation(
        hand=(CardName.ENFORCER, CardName.DAREDEVIL, CardName.SIDEKICK, CardName.SABOTEUR),
        recruited=((CardName.DAREDEVIL, CardName.DAREDEVIL), ()),
        actions=(fatal, safe),
    )
    result = filter_terminal_actions(observation, observation.legal_actions)
    assert result.allowed_actions == (safe,)
    assert result.provable_loss_actions == (fatal,)
    assert result.vetoed_actions == (fatal,)


def test_active_player_condition_ties_use_engine_adjudication() -> None:
    for card in (CardName.DAREDEVIL, CardName.CODEBREAKER):
        action = PlayOfferAction(4, PlayerId.PLAYER_ONE, card, card)
        observation = play_observation(
            hand=(card, card, CardName.ENFORCER, CardName.SABOTEUR),
            recruited=((card, card), (card, card)),
            actions=(action,),
        )
        result = filter_terminal_actions(observation, observation.legal_actions)
        assert result.allowed_actions == (action,)
        assert result.provable_loss_actions == ()
        assert result.vetoed_actions == ()
        assert not result.forced_loss_fallback


def test_simultaneous_win_and_loss_conditions_reinforce_the_same_winner() -> None:
    observation = recruit_observation(
        face_up=CardName.DAREDEVIL,
        known_face_down=CardName.CODEBREAKER,
        own_recruited=(
            CardName.DAREDEVIL,
            CardName.DAREDEVIL,
            CardName.CODEBREAKER,
            CardName.CODEBREAKER,
        ),
        opponent_recruited=(
            CardName.DAREDEVIL,
            CardName.DAREDEVIL,
            CardName.CODEBREAKER,
            CardName.CODEBREAKER,
        ),
    )
    result = filter_terminal_actions(observation, observation.legal_actions)
    # Face-up makes the recruiter lose while the offerer wins; face-down does the reverse.
    assert result.allowed_actions == (observation.legal_actions[1],)
    assert result.provable_loss_actions == (observation.legal_actions[0],)


def test_hidden_authoritative_identity_cannot_change_filtered_recruit_actions() -> None:
    state = new_game(seed=0)
    matching = [
        action
        for action in legal_actions(state)
        if isinstance(action, PlayOfferAction) and action.face_up is CardName.SABOTEUR
    ]
    assert len(matching) >= 2
    first = apply_action(state, matching[0])
    second = apply_action(state, matching[1])
    assert first.offer is not None and second.offer is not None
    assert first.offer.face_down is not second.offer.face_down
    first_observation = observe(first, PlayerId.PLAYER_TWO)
    second_observation = observe(second, PlayerId.PLAYER_TWO)
    assert first_observation == second_observation
    assert filter_terminal_actions(
        first_observation, first_observation.legal_actions
    ) == filter_terminal_actions(second_observation, second_observation.legal_actions)


def test_information_set_excludes_cards_with_exhausted_visible_counts() -> None:
    observation = recruit_observation(
        face_up=CardName.ENFORCER,
        own_recruited=(CardName.DAREDEVIL, CardName.DAREDEVIL),
        opponent_recruited=(CardName.DAREDEVIL, CardName.DAREDEVIL, CardName.DAREDEVIL),
        own_hand=(
            CardName.DAREDEVIL,
            CardName.SABOTEUR,
            CardName.SENTINEL,
            CardName.DOUBLE_AGENT,
        ),
    )
    candidates = information_consistent_face_down_cards(observation, CardName.ENFORCER)
    assert CardName.DAREDEVIL not in candidates
    assert CardName.CODEBREAKER in candidates


def test_all_losing_actions_preserve_original_tuple_as_forced_loss_fallback() -> None:
    observation = recruit_observation(
        face_up=CardName.DAREDEVIL,
        known_face_down=CardName.DAREDEVIL,
        own_recruited=(CardName.DAREDEVIL, CardName.DAREDEVIL),
        opponent_recruited=(CardName.DAREDEVIL, CardName.DAREDEVIL),
    )
    result = filter_terminal_actions(observation, observation.legal_actions)
    assert result.allowed_actions is observation.legal_actions
    assert result.provable_loss_actions == observation.legal_actions
    assert result.vetoed_actions == ()
    assert result.forced_loss_fallback


def test_deck_exhaustion_uses_next_player_hand_and_active_score_tie_rules() -> None:
    observation = recruit_observation(
        face_up=CardName.SENTINEL,
        known_face_down=CardName.CODEBREAKER,
        own_hand=(CardName.ENFORCER,),
        scores=(0, 1),
        remaining_deck_count=0,
    )
    result = filter_terminal_actions(observation, observation.legal_actions)
    assert result.forced_loss_fallback
    assert result.provable_loss_actions == observation.legal_actions

    tied = replace(
        observation,
        players=(replace(observation.players[0], score=1), observation.players[1]),
    )
    tied_result = filter_terminal_actions(tied, tied.legal_actions)
    assert tied_result.forced_loss_fallback
    assert tied_result.provable_loss_actions == tied.legal_actions


def test_epsilon_exploration_sees_only_the_filtered_action_set() -> None:
    observation = recruit_observation(
        face_up=CardName.DAREDEVIL,
        own_recruited=(CardName.DAREDEVIL, CardName.DAREDEVIL),
    )
    wrapped = TerminalSafetyAgent(EpsilonGreedyAgent(RandomAgent(), EpsilonConfig(1, 1)))
    rng = RecordingRandom((0,))
    chosen = wrapped.choose_action(
        observation, observation.decision, observation.legal_actions, rng
    )
    assert chosen is observation.legal_actions[1]
    assert rng.bounds == [1]
    config = wrapped.config_to_data()
    assert config["version"] == "terminal-safety-v1"
    assert config["base"]["version"] == "epsilon-wrapper-v1"  # type: ignore[index]
    assert config["terminal_evaluator"] == "engine-terminal-v1"


def test_wrapped_policy_cannot_return_a_vetoed_action() -> None:
    observation = recruit_observation(
        face_up=CardName.DAREDEVIL,
        own_recruited=(CardName.DAREDEVIL, CardName.DAREDEVIL),
    )

    class BypassingAgent:
        def choose_action(self, observation, decision, legal_actions, rng):  # type: ignore[no-untyped-def]
            del observation, decision, legal_actions, rng
            return RecruitAction(5, PlayerId.PLAYER_ONE, OfferSlot.FACE_UP)

    wrapped = TerminalSafetyAgent(BypassingAgent())
    try:
        wrapped.choose_action(
            observation, observation.decision, observation.legal_actions, RecordingRandom(())
        )
    except ValueError as exc:
        assert "vetoed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("wrapped policy bypassed terminal safety")


def test_seeded_shielded_game_records_reproduce_exactly() -> None:
    def spec() -> GameSpec:
        wrapped = TerminalSafetyAgent(RandomAgent())
        config = wrapped.config_to_data()
        agent = AgentSpec("safe-random", config, lambda: TerminalSafetyAgent(RandomAgent()))
        return GameSpec(
            "terminal-safety-test",
            "game-000001",
            None,
            new_game(seed=1).config,
            71,
            (agent, agent),
            (101, 202),
        )

    first = run_game(spec())
    second = run_game(spec())
    assert first.replay.actions == second.replay.actions
    assert game_record_fingerprint(first) == game_record_fingerprint(second)


def test_engine_and_observation_imports_do_not_load_agents_or_torch() -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import agent_avenue.engine; import agent_avenue.observation; "
                "assert not any(name.startswith('agent_avenue.agents') for name in sys.modules); "
                "assert 'torch' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr


def test_terminal_safety_passes_a_consistent_filtered_observation_to_base_policy() -> None:
    observation = recruit_observation(
        face_up=CardName.DAREDEVIL,
        own_recruited=(CardName.DAREDEVIL, CardName.DAREDEVIL),
    )
    wrapped = TerminalSafetyAgent(RandomAgent())
    chosen = wrapped.choose_action(
        observation, observation.decision, observation.legal_actions, RecordingRandom((0,))
    )
    assert chosen == observation.legal_actions[1]
    # The original immutable observation remains the complete public engine view.
    assert len(observation.legal_actions) == 2
    assert replace(observation, legal_actions=observation.legal_actions) == observation

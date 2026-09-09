from dataclasses import replace

from agent_avenue.agents import (
    RandomAgent,
    TerminalOffenseAgent,
    filter_immediate_win_actions,
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


class RecordingRandom:
    def __init__(self, values: tuple[int, ...]) -> None:
        self.values = iter(values)
        self.bounds: list[int] = []

    def randbelow(self, upper_bound: int) -> int:
        self.bounds.append(upper_bound)
        value = next(self.values)
        assert 0 <= value < upper_bound
        return value


def play_observation(
    *,
    hand: tuple[CardName, ...],
    actions: tuple[PlayOfferAction, ...],
    own_recruited: tuple[CardName, ...] = (),
    opponent_recruited: tuple[CardName, ...] = (),
    scores: tuple[int, int] = (0, 0),
) -> PlayerObservation:
    return PlayerObservation(
        viewer=PlayerId.PLAYER_ONE,
        own_hand=hand,
        players=(
            PublicPlayer(PlayerId.PLAYER_ONE, scores[0], own_recruited, len(hand)),
            PublicPlayer(PlayerId.PLAYER_TWO, scores[1], opponent_recruited, 4),
        ),
        active_player=PlayerId.PLAYER_ONE,
        turn=5,
        phase=Phase.PLAY,
        remaining_deck_count=12,
        history=(),
        decision=PlayContext("play", 8, PlayerId.PLAYER_ONE),
        legal_actions=actions,
    )


def recruit_observation(
    *,
    face_up: CardName,
    own_recruited: tuple[CardName, ...] = (),
    opponent_recruited: tuple[CardName, ...] = (),
    scores: tuple[int, int] = (0, 0),
) -> PlayerObservation:
    actions = tuple(RecruitAction(9, PlayerId.PLAYER_ONE, slot) for slot in OfferSlot)
    hand = (
        CardName.ENFORCER,
        CardName.SABOTEUR,
        CardName.SENTINEL,
        CardName.DOUBLE_AGENT,
    )
    return PlayerObservation(
        viewer=PlayerId.PLAYER_ONE,
        own_hand=hand,
        players=(
            PublicPlayer(PlayerId.PLAYER_ONE, scores[0], own_recruited, len(hand)),
            PublicPlayer(PlayerId.PLAYER_TWO, scores[1], opponent_recruited, 4),
        ),
        active_player=PlayerId.PLAYER_TWO,
        turn=5,
        phase=Phase.RECRUIT,
        remaining_deck_count=12,
        history=(),
        decision=RecruitContext(
            "recruit", 9, PlayerId.PLAYER_ONE, PlayerId.PLAYER_TWO, face_up, None
        ),
        legal_actions=actions,
    )


def test_sentinel_double_agent_offer_is_recognized_as_forced_lethal() -> None:
    lethal = PlayOfferAction(8, PlayerId.PLAYER_ONE, CardName.SENTINEL, CardName.DOUBLE_AGENT)
    quiet = PlayOfferAction(8, PlayerId.PLAYER_ONE, CardName.ENFORCER, CardName.SABOTEUR)
    observation = play_observation(
        hand=(
            CardName.SENTINEL,
            CardName.DOUBLE_AGENT,
            CardName.ENFORCER,
            CardName.SABOTEUR,
        ),
        actions=(quiet, lethal),
        own_recruited=(CardName.SENTINEL, CardName.SENTINEL, CardName.DOUBLE_AGENT),
        scores=(2, 0),
    )

    result = filter_immediate_win_actions(observation, observation.legal_actions)

    assert result.forced_win_actions == (lethal,)
    assert result.allowed_actions == (lethal,)


def test_recruiting_visible_third_codebreaker_is_robust_to_hidden_card() -> None:
    observation = recruit_observation(
        face_up=CardName.CODEBREAKER,
        own_recruited=(CardName.CODEBREAKER, CardName.CODEBREAKER),
    )

    result = filter_immediate_win_actions(observation, observation.legal_actions)

    assert result.forced_win_actions == (observation.legal_actions[0],)
    assert result.allowed_actions == (observation.legal_actions[0],)


def test_active_player_wins_simultaneous_daredevil_condition_tie() -> None:
    action = PlayOfferAction(8, PlayerId.PLAYER_ONE, CardName.DAREDEVIL, CardName.DAREDEVIL)
    observation = play_observation(
        hand=(CardName.DAREDEVIL,) * 4,
        actions=(action,),
        opponent_recruited=(CardName.DAREDEVIL, CardName.DAREDEVIL),
    )

    result = filter_immediate_win_actions(observation, observation.legal_actions)

    assert result.forced_win_actions == (action,)


def test_no_forced_win_preserves_the_complete_legal_set() -> None:
    actions = (
        PlayOfferAction(8, PlayerId.PLAYER_ONE, CardName.ENFORCER, CardName.SABOTEUR),
        PlayOfferAction(8, PlayerId.PLAYER_ONE, CardName.SABOTEUR, CardName.ENFORCER),
    )
    observation = play_observation(
        hand=(
            CardName.ENFORCER,
            CardName.SABOTEUR,
            CardName.SENTINEL,
            CardName.DOUBLE_AGENT,
        ),
        actions=actions,
    )

    result = filter_immediate_win_actions(observation, observation.legal_actions)

    assert result.forced_win_actions == ()
    assert result.allowed_actions is observation.legal_actions


def test_wrapper_passes_only_forced_wins_to_the_base_policy() -> None:
    lethal = PlayOfferAction(8, PlayerId.PLAYER_ONE, CardName.SENTINEL, CardName.DOUBLE_AGENT)
    quiet = PlayOfferAction(8, PlayerId.PLAYER_ONE, CardName.ENFORCER, CardName.SABOTEUR)
    observation = play_observation(
        hand=(
            CardName.SENTINEL,
            CardName.DOUBLE_AGENT,
            CardName.ENFORCER,
            CardName.SABOTEUR,
        ),
        actions=(quiet, lethal),
        own_recruited=(CardName.SENTINEL, CardName.SENTINEL, CardName.DOUBLE_AGENT),
        scores=(2, 0),
    )
    wrapper = TerminalOffenseAgent(RandomAgent())
    rng = RecordingRandom((0,))

    chosen = wrapper.choose_action(
        observation, observation.decision, observation.legal_actions, rng
    )

    assert chosen == lethal
    assert rng.bounds == [1]
    assert wrapper.config_to_data()["version"] == "terminal-offense-v1"
    assert replace(observation, legal_actions=observation.legal_actions) == observation

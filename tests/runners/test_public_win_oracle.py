from __future__ import annotations

from dataclasses import replace

import pytest

from agent_avenue.agents import RandomAgent, RandomAgentConfig, filter_immediate_win_actions
from agent_avenue.engine import (
    CardName,
    OfferSlot,
    Phase,
    PlayerId,
    PlayOfferAction,
    RecruitAction,
    apply_action,
    new_game,
)
from agent_avenue.observation import observe
from agent_avenue.observation.model import (
    PlayContext,
    PlayerObservation,
    PublicPlayer,
    RecruitContext,
)
from agent_avenue.runners import (
    AgentSpec,
    ArenaConfig,
    independent_public_forced_win_oracle,
    run_arena,
)
from agent_avenue.storage import GameRecord


def play_observation(
    *,
    hand: tuple[CardName, ...],
    actions: tuple[PlayOfferAction, ...],
    own_recruited: tuple[CardName, ...] = (),
    opponent_recruited: tuple[CardName, ...] = (),
    scores: tuple[int, int] = (0, 0),
    remaining_deck_count: int = 12,
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
        remaining_deck_count=remaining_deck_count,
        history=(),
        decision=PlayContext("play", 8, PlayerId.PLAYER_ONE),
        legal_actions=actions,
    )


def recruit_observation(
    *,
    face_up: CardName,
    hand: tuple[CardName, ...] = (
        CardName.ENFORCER,
        CardName.SABOTEUR,
        CardName.SENTINEL,
        CardName.DOUBLE_AGENT,
    ),
    own_recruited: tuple[CardName, ...] = (),
    opponent_recruited: tuple[CardName, ...] = (),
    scores: tuple[int, int] = (0, 0),
    remaining_deck_count: int = 12,
    known_face_down: CardName | None = None,
    own_hand_size: int | None = None,
) -> PlayerObservation:
    actions = tuple(RecruitAction(9, PlayerId.PLAYER_ONE, slot) for slot in OfferSlot)
    return PlayerObservation(
        viewer=PlayerId.PLAYER_ONE,
        own_hand=hand,
        players=(
            PublicPlayer(
                PlayerId.PLAYER_ONE,
                scores[0],
                own_recruited,
                len(hand) if own_hand_size is None else own_hand_size,
            ),
            PublicPlayer(PlayerId.PLAYER_TWO, scores[1], opponent_recruited, 4),
        ),
        active_player=PlayerId.PLAYER_TWO,
        turn=5,
        phase=Phase.RECRUIT,
        remaining_deck_count=remaining_deck_count,
        history=(),
        decision=RecruitContext(
            "recruit",
            9,
            PlayerId.PLAYER_ONE,
            PlayerId.PLAYER_TWO,
            face_up,
            known_face_down,
        ),
        legal_actions=actions,
    )


def test_play_oracle_cross_checks_both_assignments_with_exact_engine_transitions() -> None:
    state = new_game(seed=17)
    observation = observe(state, PlayerId.PLAYER_ONE)

    oracle = independent_public_forced_win_oracle(
        observation, observation.legal_actions, authoritative_state=state
    )

    assert oracle.exact_transition_cross_checks == 2 * len(observation.legal_actions)
    assert (
        oracle.forced_win_actions
        == filter_immediate_win_actions(observation, observation.legal_actions).forced_win_actions
    )


def test_independent_oracle_does_not_call_production_classifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action = PlayOfferAction(8, PlayerId.PLAYER_ONE, CardName.SENTINEL, CardName.DOUBLE_AGENT)
    observation = play_observation(
        hand=(
            CardName.SENTINEL,
            CardName.DOUBLE_AGENT,
            CardName.ENFORCER,
            CardName.SABOTEUR,
        ),
        actions=(action,),
        own_recruited=(CardName.SENTINEL, CardName.SENTINEL, CardName.DOUBLE_AGENT),
        scores=(2, 0),
    )

    def fail(*args: object, **kwargs: object) -> object:
        raise AssertionError("production classifier was called")

    monkeypatch.setattr("agent_avenue.agents.terminal_offense.filter_immediate_win_actions", fail)
    oracle = independent_public_forced_win_oracle(observation, observation.legal_actions)
    assert oracle.forced_win_actions == (action,)


def test_oracle_handles_score_codebreaker_daredevil_and_active_ties() -> None:
    score_action = PlayOfferAction(8, PlayerId.PLAYER_ONE, CardName.SENTINEL, CardName.DOUBLE_AGENT)
    score = play_observation(
        hand=(
            CardName.SENTINEL,
            CardName.DOUBLE_AGENT,
            CardName.ENFORCER,
            CardName.SABOTEUR,
        ),
        actions=(score_action,),
        own_recruited=(CardName.SENTINEL, CardName.SENTINEL, CardName.DOUBLE_AGENT),
        scores=(2, 0),
    )
    assert independent_public_forced_win_oracle(score, score.legal_actions).forced_win_actions == (
        score_action,
    )

    codebreaker_action = PlayOfferAction(
        8, PlayerId.PLAYER_ONE, CardName.CODEBREAKER, CardName.CODEBREAKER
    )
    codebreaker_tie = play_observation(
        hand=(CardName.CODEBREAKER, CardName.CODEBREAKER),
        actions=(codebreaker_action,),
        own_recruited=(CardName.CODEBREAKER, CardName.CODEBREAKER),
        opponent_recruited=(CardName.CODEBREAKER, CardName.CODEBREAKER),
    )
    codebreaker_oracle = independent_public_forced_win_oracle(
        codebreaker_tie, codebreaker_tie.legal_actions
    )
    assert codebreaker_oracle.forced_win_actions == (codebreaker_action,)
    assert all(
        outcome is not None and outcome.resolution.value == "active_condition_tie"
        for outcome in codebreaker_oracle.action_outcomes[0].outcomes
    )

    daredevil_action = PlayOfferAction(
        8, PlayerId.PLAYER_ONE, CardName.DAREDEVIL, CardName.DAREDEVIL
    )
    daredevil = play_observation(
        hand=(CardName.DAREDEVIL,) * 4,
        actions=(daredevil_action,),
        opponent_recruited=(CardName.DAREDEVIL, CardName.DAREDEVIL),
    )
    assert independent_public_forced_win_oracle(
        daredevil, daredevil.legal_actions
    ).forced_win_actions == (daredevil_action,)


def test_recruit_oracle_enumerates_hidden_identities_and_is_hidden_equivalent() -> None:
    observation = recruit_observation(
        face_up=CardName.CODEBREAKER,
        own_recruited=(CardName.CODEBREAKER, CardName.CODEBREAKER),
    )
    permuted = replace(observation, own_hand=tuple(reversed(observation.own_hand)))

    first = independent_public_forced_win_oracle(observation, observation.legal_actions)
    second = independent_public_forced_win_oracle(permuted, permuted.legal_actions)

    assert first.hidden_identities == second.hidden_identities
    assert first.action_outcomes == second.action_outcomes
    assert first.forced_win_actions == (observation.legal_actions[0],)
    assert len(first.action_outcomes[0].outcomes) == len(first.hidden_identities)


def test_recruit_oracle_is_identical_across_authoritative_hidden_face_down_worlds() -> None:
    state = new_game(seed=0)
    first_state = apply_action(
        state,
        PlayOfferAction(0, PlayerId.PLAYER_ONE, CardName.DOUBLE_AGENT, CardName.SABOTEUR),
    )
    second_state = apply_action(
        state,
        PlayOfferAction(0, PlayerId.PLAYER_ONE, CardName.DOUBLE_AGENT, CardName.CODEBREAKER),
    )
    assert first_state.offer is not None and second_state.offer is not None
    assert first_state.offer.face_down is not second_state.offer.face_down
    first_observation = observe(first_state, PlayerId.PLAYER_TWO)
    second_observation = observe(second_state, PlayerId.PLAYER_TWO)
    assert first_observation == second_observation

    first = independent_public_forced_win_oracle(first_observation, first_observation.legal_actions)
    second = independent_public_forced_win_oracle(
        second_observation, second_observation.legal_actions
    )

    assert first == second


def test_recruit_oracle_covers_daredevil_and_deck_exhaustion() -> None:
    daredevil = recruit_observation(
        face_up=CardName.DAREDEVIL,
        hand=(
            CardName.DAREDEVIL,
            CardName.DAREDEVIL,
            CardName.DAREDEVIL,
            CardName.ENFORCER,
        ),
        opponent_recruited=(CardName.DAREDEVIL, CardName.DAREDEVIL),
    )
    daredevil_oracle = independent_public_forced_win_oracle(daredevil, daredevil.legal_actions)
    assert daredevil_oracle.forced_win_actions == (daredevil.legal_actions[1],)

    exhausted = recruit_observation(
        face_up=CardName.CODEBREAKER,
        hand=(CardName.ENFORCER,),
        scores=(1, 0),
        remaining_deck_count=0,
        known_face_down=CardName.SENTINEL,
        own_hand_size=1,
    )
    exhausted_oracle = independent_public_forced_win_oracle(exhausted, exhausted.legal_actions)
    assert exhausted_oracle.forced_win_actions == exhausted.legal_actions
    assert all(
        outcome is not None and outcome.reason.value == "deck_exhaustion"
        for row in exhausted_oracle.action_outcomes
        for outcome in row.outcomes
    )


def _random_spec(agent_id: str) -> AgentSpec:
    config = RandomAgentConfig()
    return AgentSpec(agent_id, config.to_data(), RandomAgent)


def test_independent_and_production_oracles_agree_on_production_records() -> None:
    records: list[GameRecord] = []
    run_arena(
        ArenaConfig("oracle-agreement", _random_spec("a"), _random_spec("b"), 2, 91),
        records.append,
    )
    for record in records:
        state = new_game(record.replay.config, record.replay.seed)
        for action in record.replay.actions:
            actor = (
                state.active_player if state.phase is Phase.PLAY else state.active_player.other()
            )
            observation = observe(state, actor)
            independent = independent_public_forced_win_oracle(
                observation, observation.legal_actions, authoritative_state=state
            )
            production = filter_immediate_win_actions(observation, observation.legal_actions)
            assert independent.forced_win_actions == production.forced_win_actions
            state = apply_action(state, action)

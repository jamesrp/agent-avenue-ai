from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from agent_avenue.encoding.candidate_structured_v2 import (
    ENCODER_FINGERPRINT,
    ENCODER_VERSION,
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    FEATURE_SOURCES,
    FEATURE_WIDTH,
    HISTORY_WIDTH,
    PLAY_CONSEQUENCE_WIDTH,
    RECRUIT_CONSEQUENCE_WIDTH,
    V1_PREFIX_WIDTH,
    encode_candidate,
)
from agent_avenue.encoding.candidate_v1 import encode_candidate as encode_v1_candidate
from agent_avenue.engine import (
    CardName,
    OfferSlot,
    Phase,
    PlayerId,
    PlayOfferAction,
    apply_action,
    new_game,
)
from agent_avenue.engine.model import CompletedTurn
from agent_avenue.observation import observe
from agent_avenue.observation.model import PlayContext, PlayerObservation, PublicPlayer


def _play_observation(card: CardName) -> tuple[PlayerObservation, PlayOfferAction]:
    hand = (card, *tuple(other for other in CardName if other is not card)[:3])
    action = PlayOfferAction(0, PlayerId.PLAYER_ONE, card, hand[1])
    observation = PlayerObservation(
        viewer=PlayerId.PLAYER_ONE,
        own_hand=hand,
        players=(
            PublicPlayer(PlayerId.PLAYER_ONE, 0, (), 4),
            PublicPlayer(PlayerId.PLAYER_TWO, 0, (), 4),
        ),
        active_player=PlayerId.PLAYER_ONE,
        turn=1,
        phase=Phase.PLAY,
        remaining_deck_count=30,
        history=(),
        decision=PlayContext("play", 0, PlayerId.PLAYER_ONE),
        legal_actions=(action,),
    )
    return observation, action


def _hidden_equivalent_recruit_states() -> tuple[object, object, PlayerObservation]:
    seen: dict[PlayerObservation, object] = {}
    for seed in range(500):
        state = new_game(seed=seed)
        offered = apply_action(state, observe(state, PlayerId.PLAYER_ONE).legal_actions[0])
        observation = observe(offered, PlayerId.PLAYER_TWO)
        prior = seen.get(observation)
        if (
            prior is not None
            and prior.offer is not None
            and offered.offer is not None
            and observation.decision.face_up is not CardName.CODEBREAKER
            and prior.offer.face_down is not offered.offer.face_down
        ):
            return prior, offered, observation
        seen[observation] = offered


def _recruit_observation() -> tuple[object, PlayerObservation]:
    _, offered, observation = _hidden_equivalent_recruit_states()
    return offered, observation

    observation = observe(new_game(seed=7), PlayerId.PLAYER_ONE)
    action = observation.legal_actions[0]
    encoded = encode_candidate(observation, action)

    assert ENCODER_VERSION == "candidate-public-structured-v2"
    assert FEATURE_WIDTH == 519
    assert len(FEATURE_NAMES) == FEATURE_WIDTH
    assert len(set(FEATURE_NAMES)) == FEATURE_WIDTH
    assert FEATURE_SCHEMA.fingerprint == ENCODER_FINGERPRINT
    assert tuple(FEATURE_SOURCES) == FEATURE_NAMES
    assert encoded.vector[:V1_PREFIX_WIDTH] == encode_v1_candidate(observation, action).vector
    assert (
        encoded.vector[V1_PREFIX_WIDTH + HISTORY_WIDTH + PLAY_CONSEQUENCE_WIDTH :]
        == (0.0,) * RECRUIT_CONSEQUENCE_WIDTH
    )

    _, recruit = _recruit_observation()
    recruit_encoded = encode_candidate(recruit, recruit.legal_actions[0])
    start = V1_PREFIX_WIDTH + HISTORY_WIDTH
    assert (
        recruit_encoded.vector[start : start + PLAY_CONSEQUENCE_WIDTH]
        == (0.0,) * PLAY_CONSEQUENCE_WIDTH
    )


def test_history_is_left_padded_truncated_oldest_to_newest_and_viewpoint_relative() -> None:
    observation = observe(new_game(seed=7), PlayerId.PLAYER_ONE)
    cards = tuple(CardName)
    history = tuple(
        CompletedTurn(
            turn=index + 1,
            active_player=PlayerId.PLAYER_ONE if index % 2 == 0 else PlayerId.PLAYER_TWO,
            face_up=cards[index % len(cards)],
            face_down=cards[(index + 1) % len(cards)],
            chosen_slot=OfferSlot.FACE_UP if index % 2 == 0 else OfferSlot.FACE_DOWN,
            opponent_recruited=cards[index % len(cards)]
            if index % 2 == 0
            else cards[(index + 1) % len(cards)],
            active_recruited=cards[(index + 1) % len(cards)]
            if index % 2 == 0
            else cards[index % len(cards)],
            score_changes=(index - 4, 6 - index),
        )
        for index in range(9)
    )
    encoded = encode_candidate(replace(observation, history=history), observation.legal_actions[0])
    first = history[1]
    assert encoded.vector[FEATURE_NAMES.index("history_0_valid")] == 1.0
    assert encoded.vector[FEATURE_NAMES.index(f"history_0_face_up_{first.face_up.value}")] == 1.0
    assert encoded.vector[FEATURE_NAMES.index("history_0_active_player_opponent")] == 1.0
    assert encoded.vector[FEATURE_NAMES.index("history_length_fraction")] == pytest.approx(9 / 19)
    assert encoded.vector[FEATURE_NAMES.index("history_truncated_fraction")] == pytest.approx(
        1 / 19
    )

    offered, recruit = _recruit_observation()
    selected = CompletedTurn(
        1,
        PlayerId.PLAYER_ONE,
        CardName.ENFORCER,
        CardName.SABOTEUR,
        OfferSlot.FACE_UP,
        CardName.ENFORCER,
        CardName.SABOTEUR,
        (2, -1),
    )
    encoded_recruit = encode_candidate(
        replace(recruit, history=(selected,)), recruit.legal_actions[0]
    )
    assert encoded_recruit.vector[FEATURE_NAMES.index("history_7_active_player_opponent")] == 1.0
    assert encoded_recruit.vector[
        FEATURE_NAMES.index("history_7_self_score_change")
    ] == pytest.approx(-1 / 6)
    assert encoded_recruit.vector[
        FEATURE_NAMES.index("history_7_opponent_score_change")
    ] == pytest.approx(2 / 6)
    assert offered.offer is not None


def test_every_card_effect_is_in_play_consequences() -> None:
    for card in CardName:
        observation, action = _play_observation(card)
        encoded = encode_candidate(observation, action)
        assert (
            encoded.vector[FEATURE_NAMES.index(f"play_recruit_face_down_self_card_{card.value}")]
            == 1.0
        )
        expected_kind = "score"
        if card is CardName.CODEBREAKER or card is CardName.DAREDEVIL:
            # First copies of these cards score; the encoder must use the actual card definition,
            # rather than a card-name shortcut for their eventual threshold effects.
            expected_kind = "score"
        assert (
            encoded.vector[
                FEATURE_NAMES.index(f"play_recruit_face_down_self_effect_{expected_kind}")
            ]
            == 1.0
        )


def test_recruit_uses_safe_unseen_support_and_never_current_hidden_identity() -> None:
    first_state, second_state, observation = _hidden_equivalent_recruit_states()
    assert first_state.offer is not None and second_state.offer is not None
    assert first_state.offer.face_down is not second_state.offer.face_down
    hidden_equivalent = observe(second_state, PlayerId.PLAYER_TWO)
    assert observation == hidden_equivalent
    assert all(
        encode_candidate(observation, action).vector
        == encode_candidate(hidden_equivalent, action).vector
        for action in observation.legal_actions
    )

    self_player = next(
        player for player in observation.players if player.player is observation.viewer
    )
    players = tuple(
        replace(player, recruited=(CardName.CODEBREAKER, CardName.CODEBREAKER))
        if player.player is observation.viewer
        else player
        for player in observation.players
    )
    support_observation = replace(
        observation,
        players=players,
        remaining_deck_count=observation.remaining_deck_count - 2,
    )
    action = next(
        candidate
        for candidate in support_observation.legal_actions
        if candidate.slot is OfferSlot.FACE_DOWN
    )
    baseline = encode_candidate(support_observation, action).vector
    modified_hand = (CardName.CODEBREAKER,) * 4
    altered = replace(support_observation, own_hand=modified_hand)
    altered_vector = encode_candidate(altered, action).vector
    support_names = tuple(
        f"recruit_hidden_effect_support_{recipient}_{kind}"
        for recipient in ("self", "opponent")
        for kind in ("score", "win", "lose")
    )
    assert self_player.player is observation.viewer
    assert any(
        baseline[FEATURE_NAMES.index(name)] != altered_vector[FEATURE_NAMES.index(name)]
        for name in support_names
    )
    assert all(0.0 <= altered_vector[FEATURE_NAMES.index(name)] <= 1.0 for name in support_names)


def test_encoder_source_has_no_authoritative_successor_imports() -> None:
    source = Path(__file__).parents[2] / "src/agent_avenue/encoding/candidate_structured_v2.py"
    tree = ast.parse(source.read_text())
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert all(node.module != "agent_avenue.engine.transitions" for node in imports)
    assert all(
        not (
            node.module == "agent_avenue.engine.model"
            and any(name.name == "GameState" for name in node.names)
        )
        for node in imports
    )
    assert all(
        not any(
            name.name in {"apply_action", "current_decision", "legal_actions"}
            for name in node.names
        )
        for node in imports
    )


def _threshold_play_observation(
    card: CardName, *, opponent_has_two: bool = False, duplicate_offer: bool = False
) -> tuple[PlayerObservation, PlayOfferAction]:
    if duplicate_offer:
        hand = (card, card)
        face_down = card
        self_hand_size = 2
    else:
        hand = (card, CardName.ENFORCER, CardName.SABOTEUR, CardName.SENTINEL)
        face_down = CardName.ENFORCER
        self_hand_size = 4
    action = PlayOfferAction(0, PlayerId.PLAYER_ONE, card, face_down)
    self_recruited = (card, card)
    opponent_recruited = (card, card) if opponent_has_two else ()
    visible = len(hand) + len(self_recruited) + len(opponent_recruited)
    observation = PlayerObservation(
        viewer=PlayerId.PLAYER_ONE,
        own_hand=hand,
        players=(
            PublicPlayer(PlayerId.PLAYER_ONE, 0, self_recruited, self_hand_size),
            PublicPlayer(PlayerId.PLAYER_TWO, 0, opponent_recruited, 4),
        ),
        active_player=PlayerId.PLAYER_ONE,
        turn=3,
        phase=Phase.PLAY,
        remaining_deck_count=38 - visible - 4,
        history=(),
        decision=PlayContext("play", 0, PlayerId.PLAYER_ONE),
        legal_actions=(action,),
    )
    return observation, action


def test_play_consequences_use_exact_terminal_condition_and_active_tie_resolution() -> None:
    code_observation, code_action = _threshold_play_observation(CardName.CODEBREAKER)
    code = encode_candidate(code_observation, code_action).vector
    assert code[FEATURE_NAMES.index("play_recruit_face_down_self_terminal_win")] == 1.0
    assert code[FEATURE_NAMES.index("play_recruit_face_down_self_terminal_loss")] == 0.0

    dare_observation, dare_action = _threshold_play_observation(CardName.DAREDEVIL)
    dare = encode_candidate(dare_observation, dare_action).vector
    assert dare[FEATURE_NAMES.index("play_recruit_face_down_self_terminal_loss")] == 1.0

    tie_observation, tie_action = _threshold_play_observation(
        CardName.CODEBREAKER, opponent_has_two=True, duplicate_offer=True
    )
    tie = encode_candidate(tie_observation, tie_action).vector
    # Both players reach three Codebreakers; exact evaluator rules award an active-condition tie.
    assert tie[FEATURE_NAMES.index("play_recruit_face_up_self_terminal_win")] == 1.0
    assert tie[FEATURE_NAMES.index("play_recruit_face_up_opponent_terminal_loss")] == 1.0

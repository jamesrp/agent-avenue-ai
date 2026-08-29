from dataclasses import replace

import pytest

from agent_avenue.encoding import (
    CARD_ORDER,
    ENCODER_CONFIG,
    ENCODER_FINGERPRINT,
    ENCODER_VERSION,
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    FEATURE_WIDTH,
    CandidateEncoderConfig,
    encode_candidate,
)
from agent_avenue.engine import (
    CardName,
    Phase,
    PlayerId,
    PlayOfferAction,
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


def _boundary_play_observation() -> PlayerObservation:
    repeated = (
        CardName.DOUBLE_AGENT,
        CardName.ENFORCER,
        CardName.SABOTEUR,
        CardName.SENTINEL,
    )
    self_recruited = (
        *(card for card in repeated for _ in range(3)),
        CardName.CODEBREAKER,
        CardName.CODEBREAKER,
        CardName.DAREDEVIL,
        CardName.DAREDEVIL,
        CardName.SIDEKICK,
    )
    opponent_recruited = (
        *(card for card in repeated for _ in range(3)),
        CardName.CODEBREAKER,
        CardName.CODEBREAKER,
        CardName.DAREDEVIL,
        CardName.DAREDEVIL,
        CardName.MOLE,
    )
    action = (PlayOfferAction(0, PlayerId.PLAYER_ONE, CardName.CODEBREAKER, CardName.CODEBREAKER),)
    return PlayerObservation(
        viewer=PlayerId.PLAYER_ONE,
        own_hand=(CardName.CODEBREAKER, CardName.CODEBREAKER),
        players=(
            PublicPlayer(PlayerId.PLAYER_ONE, -99, self_recruited, 2),
            PublicPlayer(PlayerId.PLAYER_TWO, 99, opponent_recruited, 2),
        ),
        active_player=PlayerId.PLAYER_ONE,
        turn=19,
        phase=Phase.PLAY,
        remaining_deck_count=0,
        history=(),
        decision=PlayContext("play", 0, PlayerId.PLAYER_ONE),
        legal_actions=action,
    )


def test_candidate_public_v1_schema_is_fixed_and_fingerprinted() -> None:
    assert ENCODER_VERSION == "candidate-public-v1"
    assert tuple(CardName) == CARD_ORDER
    assert FEATURE_SCHEMA.config == ENCODER_CONFIG
    assert FEATURE_WIDTH == 87
    assert len(FEATURE_NAMES) == FEATURE_WIDTH
    assert len(set(FEATURE_NAMES)) == FEATURE_WIDTH
    assert len(ENCODER_FINGERPRINT) == 64
    assert FEATURE_SCHEMA.fingerprint == ENCODER_FINGERPRINT
    assert (
        CandidateEncoderConfig.from_data(FEATURE_SCHEMA.config.to_data()) == FEATURE_SCHEMA.config
    )


def test_play_encoding_has_expected_action_suffix_and_named_vector() -> None:
    observation = observe(new_game(seed=4), PlayerId.PLAYER_ONE)
    action = observation.legal_actions[0]
    encoded = encode_candidate(observation, action)

    assert encoded.width == FEATURE_WIDTH
    assert encoded.feature_names == FEATURE_NAMES
    assert encoded.version == ENCODER_VERSION
    assert encoded.values == encoded.vector == encoded.features
    assert encoded.vector[0:2] == (1.0, 0.0)
    assert encoded.vector[2] == pytest.approx(1 / 19)
    assert encoded.vector[3] == 1.0
    assert encoded.vector[61:69] == (0.0,) * 8
    up = FEATURE_NAMES.index(f"candidate_play_face_up_{action.face_up.value}")
    down = FEATURE_NAMES.index(f"candidate_play_face_down_{action.face_down.value}")
    assert encoded.vector[up] == 1.0
    assert encoded.vector[down] == 1.0


def test_recruit_encoding_has_unknown_face_down_safe_suffix() -> None:
    state = new_game(seed=31)
    offer = apply_action(state, observe(state, PlayerId.PLAYER_ONE).legal_actions[0])
    observation = observe(offer, PlayerId.PLAYER_TWO)
    assert isinstance(observation.decision, RecruitContext)
    assert observation.decision.known_face_down is None
    for action in observation.legal_actions:
        encoded = encode_candidate(observation, action)
        assert encoded.vector[0:2] == (0.0, 1.0)
        assert encoded.vector[-2:] == (
            1.0 if action.slot.value == "face_up" else 0.0,
            1.0 if action.slot.value == "face_down" else 0.0,
        )
        assert sum(encoded.vector[61:69]) == 1.0
        assert sum(encoded.vector[69:85]) == 0.0


def test_hidden_equivalent_states_encode_identically() -> None:
    first = observe(new_game(seed=9), PlayerId.PLAYER_ONE)
    second = observe(new_game(seed=25), PlayerId.PLAYER_ONE)
    assert first.own_hand == second.own_hand
    assert first.legal_actions == second.legal_actions
    assert all(
        encode_candidate(first, action).vector == encode_candidate(second, action).vector
        for action in first.legal_actions
    )


def test_hand_order_does_not_change_count_or_candidate_encoding() -> None:
    observation = observe(new_game(seed=7), PlayerId.PLAYER_ONE)
    action = observation.legal_actions[0]
    reordered = replace(observation, own_hand=tuple(reversed(observation.own_hand)))
    assert (
        encode_candidate(observation, action).vector == encode_candidate(reordered, action).vector
    )


def test_duplicate_hands_and_every_card_type_have_distinguishable_play_one_hots() -> None:
    duplicate_observation = observe(new_game(seed=2), PlayerId.PLAYER_ONE)
    assert duplicate_observation.own_hand.count(CardName.CODEBREAKER) == 2

    for card in CardName:
        hand = (card, *tuple(candidate for candidate in CardName if candidate is not card)[:3])
        other = hand[1]
        action = PlayOfferAction(0, PlayerId.PLAYER_ONE, card, other)
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
        encoded = encode_candidate(observation, action)
        index = FEATURE_NAMES.index(f"candidate_play_face_up_{card.value}")
        assert encoded.vector[index] == 1.0


def test_boundary_scaling_and_threshold_features_are_encoded() -> None:
    encoded = encode_candidate(
        _boundary_play_observation(), _boundary_play_observation().legal_actions[0]
    )
    assert encoded.vector[2:4] == (1.0, 0.0)
    assert encoded.vector[4:7] == (-1.0, 1.0, -1.0)
    assert encoded.vector[7:9] == (0.5, 0.5)
    for card in (
        CardName.DOUBLE_AGENT,
        CardName.ENFORCER,
        CardName.SABOTEUR,
        CardName.SENTINEL,
    ):
        for threshold in (1, 2, 3):
            index = FEATURE_NAMES.index(f"self_recruited_{card.value}_at_least_{threshold}")
            assert encoded.vector[index] == 1.0
    assert encoded.vector[FEATURE_NAMES.index("self_recruited_sidekick_at_least_1")] == 1.0
    assert encoded.vector[FEATURE_NAMES.index("opponent_recruited_mole_at_least_1")] == 1.0


def test_encoder_rejects_terminal_nonactor_and_hidden_face_down_payloads() -> None:
    observation = observe(new_game(seed=2), PlayerId.PLAYER_ONE)
    with pytest.raises(ValueError, match="member"):
        encode_candidate(
            observation,
            PlayOfferAction(
                observation.legal_actions[0].revision,
                PlayerId.PLAYER_ONE,
                CardName.MOLE,
                CardName.SIDEKICK,
            ),
        )

    nonactor_action = replace(observation.legal_actions[0], actor=PlayerId.PLAYER_TWO)
    nonactor = replace(
        observation,
        decision=replace(observation.decision, actor=PlayerId.PLAYER_TWO),
        legal_actions=(nonactor_action,),
    )
    with pytest.raises(ValueError, match="owned by"):
        encode_candidate(nonactor, nonactor_action)

    state = new_game(seed=3)
    offered = apply_action(state, observe(state, PlayerId.PLAYER_ONE).legal_actions[0])
    recruit = observe(offered, PlayerId.PLAYER_TWO)
    assert isinstance(recruit.decision, RecruitContext)
    assert offered.offer is not None
    forged = replace(
        recruit,
        decision=replace(recruit.decision, known_face_down=offered.offer.face_down),
    )
    with pytest.raises(ValueError, match="face-down"):
        encode_candidate(forged, recruit.legal_actions[0])


def test_schema_configuration_rejects_reordering_and_unknown_fields() -> None:
    data = FEATURE_SCHEMA.config.to_data()
    bad_order = dict(data)
    bad_order["card_order"] = list(reversed(data["card_order"]))
    with pytest.raises(ValueError, match="canonical card order"):
        CandidateEncoderConfig.from_data(bad_order)
    bad_fields = dict(data)
    bad_fields["extra"] = True
    with pytest.raises(ValueError, match="fields"):
        CandidateEncoderConfig.from_data(bad_fields)

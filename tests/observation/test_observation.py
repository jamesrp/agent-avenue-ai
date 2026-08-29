import json

from agent_avenue.engine import (
    PlayerId,
    PlayOfferAction,
    apply_action,
    legal_actions,
    new_game,
)
from agent_avenue.observation import observation_to_data, observe


def test_observation_contains_only_own_hand_and_public_counts() -> None:
    state = new_game(seed=18)
    data = observation_to_data(observe(state, PlayerId.PLAYER_ONE))
    encoded = json.dumps(data, sort_keys=True)
    assert data["own_hand"] == [card.value for card in state.hands[0]]
    assert data["players"][1]["hand_size"] == 4
    assert "deck" not in data
    assert "seed" not in data
    assert "fingerprint" not in data
    assert "player_two" in encoded
    for card in state.hands[1]:
        # Opponent cards might coincidentally occur in the viewer's own hand, so field-level
        # assertions above are the security guarantee rather than substring absence.
        assert card.value not in data["players"][1]


def test_opponent_hand_and_deck_order_changes_do_not_change_observation() -> None:
    first = new_game(seed=9)
    second = new_game(seed=25)
    assert first.hands[0] == second.hands[0]
    assert first.hands[1] != second.hands[1]
    assert observation_to_data(observe(first, PlayerId.PLAYER_ONE)) == observation_to_data(
        observe(second, PlayerId.PLAYER_ONE)
    )


def test_face_down_offer_is_known_only_to_offering_player() -> None:
    state = new_game(seed=31)
    action = legal_actions(state)[0]
    assert isinstance(action, PlayOfferAction)
    offered = apply_action(state, action)
    active_data = observation_to_data(observe(offered, PlayerId.PLAYER_ONE))
    opponent_data = observation_to_data(observe(offered, PlayerId.PLAYER_TWO))
    assert active_data["decision"]["known_face_down"] == action.face_down.value
    assert opponent_data["decision"]["known_face_down"] is None
    assert active_data["legal_actions"] == []
    assert {item["slot"] for item in opponent_data["legal_actions"]} == {"face_up", "face_down"}
    assert all("face_down" not in item for item in opponent_data["legal_actions"])


def test_unknown_face_down_value_cannot_affect_recruiter_serialization() -> None:
    state = new_game(seed=0)
    matching_actions = [
        action
        for action in legal_actions(state)
        if isinstance(action, PlayOfferAction) and action.face_up.value == "saboteur"
    ]
    assert len(matching_actions) >= 2
    first = apply_action(state, matching_actions[0])
    second = apply_action(state, matching_actions[1])
    assert first.offer is not None and second.offer is not None
    assert first.offer.face_down is not second.offer.face_down
    assert observation_to_data(observe(first, PlayerId.PLAYER_TWO)) == observation_to_data(
        observe(second, PlayerId.PLAYER_TWO)
    )


def test_serialized_observation_recursively_excludes_authoritative_fields() -> None:
    data = observation_to_data(observe(new_game(seed=99), PlayerId.PLAYER_ONE))
    forbidden = {"deck", "seed", "fingerprint", "opponent_hand", "config", "actions"}

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value)) if value else set()
        return set()

    assert keys(data).isdisjoint(forbidden)

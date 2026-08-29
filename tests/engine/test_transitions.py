from collections import Counter
from dataclasses import replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agent_avenue.engine import (
    CANONICAL_DECK,
    CardName,
    EngineError,
    ErrorCode,
    OfferSlot,
    Phase,
    PlayerId,
    PlayOfferAction,
    RecruitAction,
    apply_action,
    legal_actions,
    new_game,
    state_fingerprint,
    validate_state,
)
from agent_avenue.engine.model import GameState, Offer
from agent_avenue.engine.transitions import _legal_actions_unchecked


def _state_with_hand(hand: tuple[CardName, ...]) -> GameState:
    state = new_game(seed=1)
    pool = list(CANONICAL_DECK)
    for card in hand:
        pool.remove(card)
    other = tuple(pool[:4])
    return replace(state, deck=tuple(pool[4:]), hands=(hand, other))


def test_ordered_offers_are_complete_and_stable_for_duplicate_hand() -> None:
    state = new_game(seed=2)
    actions = legal_actions(state)
    assert len(actions) == 6
    pairs = {
        (action.face_up, action.face_down)
        for action in actions
        if isinstance(action, PlayOfferAction)
    }
    assert (CardName.CODEBREAKER, CardName.SENTINEL) in pairs
    assert (CardName.SENTINEL, CardName.CODEBREAKER) in pairs
    assert (CardName.CODEBREAKER, CardName.CODEBREAKER) not in pairs


def test_all_one_name_exception_allows_one_semantic_offer() -> None:
    state = _state_with_hand((CardName.ENFORCER,) * 4)
    assert _legal_actions_unchecked(state) == (
        PlayOfferAction(0, PlayerId.PLAYER_ONE, CardName.ENFORCER, CardName.ENFORCER),
    )


def test_same_name_offer_is_rejected_when_another_name_exists_without_mutation() -> None:
    state = new_game(seed=2)
    action = PlayOfferAction(0, PlayerId.PLAYER_ONE, CardName.CODEBREAKER, CardName.CODEBREAKER)
    before = state_fingerprint(state)
    with pytest.raises(EngineError) as error:
        apply_action(state, action)
    assert error.value.code is ErrorCode.ILLEGAL_ACTION
    assert state_fingerprint(state) == before


def test_play_removes_both_cards_then_draws_to_four() -> None:
    state = new_game(seed=7)
    action = legal_actions(state)[0]
    assert isinstance(action, PlayOfferAction)
    old_deck = state.deck
    next_state = apply_action(state, action)
    assert next_state.phase is Phase.RECRUIT
    assert next_state.offer == Offer(state.active_player, action.face_up, action.face_down)
    assert len(next_state.hands[0]) == 4
    assert next_state.deck == old_deck[2:]
    assert next_state.hands[0][-2:] == old_deck[:2]


def test_play_draws_partially_when_deck_is_short() -> None:
    state = new_game(seed=3)
    # Exercise the pure play primitive with a structurally conserved short-deck state; such a
    # position is normally reached only after many completed turns.
    state = replace(state, deck=state.deck[:1], recruited=(state.deck[1:], ()))
    action = legal_actions(replace(state, recruited=((), ()), deck=new_game(seed=3).deck))[0]
    assert isinstance(action, PlayOfferAction)
    from agent_avenue.engine.transitions import _apply_play

    next_state = _apply_play(state, action)
    assert len(next_state.hands[0]) == 3
    assert next_state.deck == ()


def test_recruit_applies_both_score_changes_before_terminal_adjudication() -> None:
    state = new_game(seed=237)
    played = apply_action(
        state,
        PlayOfferAction(0, PlayerId.PLAYER_ONE, CardName.SIDEKICK, CardName.MOLE),
    )
    choices = legal_actions(played)
    assert choices == (
        RecruitAction(1, PlayerId.PLAYER_TWO, OfferSlot.FACE_UP),
        RecruitAction(1, PlayerId.PLAYER_TWO, OfferSlot.FACE_DOWN),
    )
    resolved = apply_action(played, choices[0])
    assert resolved.scores == (-3, 4)
    assert resolved.phase is Phase.TERMINAL
    assert resolved.active_player is PlayerId.PLAYER_ONE
    assert resolved.turn == 1
    assert resolved.outcome is not None
    assert resolved.outcome.winner is PlayerId.PLAYER_TWO
    assert resolved.history[0].score_changes == (-3, 4)


def test_stale_wrong_actor_and_wrong_type_errors_are_structured() -> None:
    state = new_game(seed=9)
    legal = legal_actions(state)[0]
    assert isinstance(legal, PlayOfferAction)
    cases = (
        (replace(legal, revision=3), ErrorCode.STALE_ACTION),
        (replace(legal, actor=PlayerId.PLAYER_TWO), ErrorCode.WRONG_ACTOR),
        (RecruitAction(0, PlayerId.PLAYER_ONE, OfferSlot.FACE_UP), ErrorCode.WRONG_ACTION_TYPE),
    )
    for action, code in cases:
        with pytest.raises(EngineError) as error:
            apply_action(state, action)
        assert error.value.code is code


@given(seed=st.integers(min_value=0, max_value=2**64 - 1))
@settings(max_examples=30, deadline=None)
def test_seeded_games_preserve_cards_and_are_deterministic(seed: int) -> None:
    def run() -> GameState:
        state = new_game(seed=seed)
        for _ in range(200):
            validate_state(state)
            all_cards = (
                state.deck
                + state.hands[0]
                + state.hands[1]
                + state.recruited[0]
                + state.recruited[1]
            )
            cards = Counter(all_cards)
            if state.offer is not None:
                cards.update((state.offer.face_up, state.offer.face_down))
            assert cards == Counter(CANONICAL_DECK)
            actions = legal_actions(state)
            if not actions:
                return state
            state = apply_action(state, actions[0])
        raise AssertionError("game did not terminate")

    first = run()
    second = run()
    assert first.phase is Phase.TERMINAL
    assert first == second
    assert state_fingerprint(first) == state_fingerprint(second)


def test_malformed_enum_like_action_values_are_rejected() -> None:
    state = new_game(seed=12)
    legal = legal_actions(state)[0]
    assert isinstance(legal, PlayOfferAction)
    malformed_card = replace(legal, face_up=legal.face_up.value)  # type: ignore[arg-type]
    malformed_revision = replace(legal, revision=True)  # type: ignore[arg-type]
    for action in (malformed_card, malformed_revision):
        with pytest.raises(EngineError) as error:
            apply_action(state, action)
        assert error.value.code is ErrorCode.ILLEGAL_ACTION

    offered = apply_action(state, legal)
    malformed_slot = RecruitAction(
        offered.revision,
        PlayerId.PLAYER_TWO,
        "face_up",  # type: ignore[arg-type]
    )
    with pytest.raises(EngineError) as error:
        apply_action(offered, malformed_slot)
    assert error.value.code is ErrorCode.ILLEGAL_ACTION


def test_forged_terminal_condition_in_play_state_is_invalid() -> None:
    state = new_game(seed=13)
    forged = replace(state, scores=(7, 0))
    with pytest.raises(EngineError) as error:
        validate_state(forged)
    assert error.value.code is ErrorCode.INVALID_STATE


@pytest.mark.parametrize(
    ("seed", "has_instant_win", "has_instant_loss"),
    [
        (1, False, False),
        (6, True, False),
        (0, False, True),
    ],
)
def test_scripted_complete_games_cover_ordinary_and_instant_endings(
    seed: int, has_instant_win: bool, has_instant_loss: bool
) -> None:
    state = new_game(seed=seed)
    while state.phase is not Phase.TERMINAL:
        state = apply_action(state, legal_actions(state)[0])
    assert state.outcome is not None
    assert bool(state.outcome.facts.instant_winners) is has_instant_win
    assert bool(state.outcome.facts.instant_losers) is has_instant_loss


def test_state_must_match_seed_and_action_provenance() -> None:
    state = new_game(seed=21)
    opponent_hand = list(state.hands[1])
    deck = list(state.deck)
    opponent_hand[0], deck[0] = deck[0], opponent_hand[0]
    forged = replace(state, hands=(state.hands[0], tuple(opponent_hand)), deck=tuple(deck))
    with pytest.raises(EngineError) as error:
        validate_state(forged)
    assert error.value.code is ErrorCode.INVALID_STATE


def test_mutable_state_zones_are_rejected() -> None:
    state = new_game(seed=22)
    forged = replace(state, deck=list(state.deck))  # type: ignore[arg-type]
    with pytest.raises(EngineError) as error:
        validate_state(forged)
    assert error.value.code is ErrorCode.INVALID_STATE


def test_finished_game_rejects_further_actions() -> None:
    state = new_game(seed=1)
    while state.phase is not Phase.TERMINAL:
        state = apply_action(state, legal_actions(state)[0])
    action = PlayOfferAction(
        state.revision,
        state.active_player,
        CardName.ENFORCER,
        CardName.SENTINEL,
    )
    with pytest.raises(EngineError) as error:
        apply_action(state, action)
    assert error.value.code is ErrorCode.GAME_FINISHED

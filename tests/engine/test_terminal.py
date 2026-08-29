from dataclasses import replace

import pytest

from agent_avenue.engine import CardName, OutcomeReason, OutcomeResolution, PlayerId, new_game
from agent_avenue.engine.terminal import adjudicate


def _terminal_state(
    *,
    scores: tuple[int, int] = (0, 0),
    recruited: tuple[tuple[CardName, ...], tuple[CardName, ...]] = ((), ()),
    active: PlayerId = PlayerId.PLAYER_ONE,
):
    return replace(new_game(seed=1), scores=scores, recruited=recruited, active_player=active)


def test_score_gap_wins() -> None:
    outcome = adjudicate(_terminal_state(scores=(7, 0)))
    assert outcome is not None
    assert outcome.winner is PlayerId.PLAYER_ONE
    assert outcome.reason is OutcomeReason.CONDITION
    assert outcome.facts.score_gap_winners == (PlayerId.PLAYER_ONE,)


@pytest.mark.parametrize(
    ("card", "owner", "winner"),
    [
        (CardName.CODEBREAKER, PlayerId.PLAYER_ONE, PlayerId.PLAYER_ONE),
        (CardName.CODEBREAKER, PlayerId.PLAYER_TWO, PlayerId.PLAYER_TWO),
        (CardName.DAREDEVIL, PlayerId.PLAYER_ONE, PlayerId.PLAYER_TWO),
        (CardName.DAREDEVIL, PlayerId.PLAYER_TWO, PlayerId.PLAYER_ONE),
    ],
)
def test_instant_win_and_loss(card: CardName, owner: PlayerId, winner: PlayerId) -> None:
    recruited = ((card,) * 3, ()) if owner is PlayerId.PLAYER_ONE else ((), (card,) * 3)
    outcome = adjudicate(_terminal_state(recruited=recruited))
    assert outcome is not None
    assert outcome.winner is winner
    assert outcome.resolution is OutcomeResolution.SOLE_CANDIDATE


def test_both_players_winning_goes_to_active_player() -> None:
    recruited = ((CardName.CODEBREAKER,) * 3, (CardName.CODEBREAKER,) * 3)
    outcome = adjudicate(_terminal_state(recruited=recruited, active=PlayerId.PLAYER_TWO))
    assert outcome is not None
    assert outcome.winner is PlayerId.PLAYER_TWO
    assert outcome.resolution is OutcomeResolution.ACTIVE_CONDITION_TIE


def test_both_players_losing_goes_to_active_player() -> None:
    recruited = ((CardName.DAREDEVIL,) * 3, (CardName.DAREDEVIL,) * 3)
    outcome = adjudicate(_terminal_state(recruited=recruited, active=PlayerId.PLAYER_ONE))
    assert outcome is not None
    assert outcome.winner is PlayerId.PLAYER_ONE
    assert outcome.resolution is OutcomeResolution.ACTIVE_CONDITION_TIE


def test_win_and_opponent_loss_reinforce_one_candidate() -> None:
    recruited = (
        (CardName.CODEBREAKER,) * 3,
        (CardName.DAREDEVIL,) * 3,
    )
    outcome = adjudicate(_terminal_state(recruited=recruited, active=PlayerId.PLAYER_TWO))
    assert outcome is not None
    assert outcome.winner is PlayerId.PLAYER_ONE
    assert outcome.resolution is OutcomeResolution.SOLE_CANDIDATE


def test_same_player_win_and_loss_creates_active_player_tie() -> None:
    recruited = (
        (CardName.CODEBREAKER,) * 3 + (CardName.DAREDEVIL,) * 3,
        (),
    )
    outcome = adjudicate(_terminal_state(recruited=recruited, active=PlayerId.PLAYER_TWO))
    assert outcome is not None
    assert set(outcome.facts.candidate_winners) == {PlayerId.PLAYER_ONE, PlayerId.PLAYER_TWO}
    assert outcome.winner is PlayerId.PLAYER_TWO


def test_condition_precedes_deck_exhaustion() -> None:
    state = _terminal_state(scores=(9, 0))
    state = replace(state, deck=(), hands=(state.hands[0], ()))
    outcome = adjudicate(state)
    assert outcome is not None
    assert outcome.reason is OutcomeReason.CONDITION


@pytest.mark.parametrize(
    ("scores", "active", "winner", "resolution"),
    [
        ((3, 1), PlayerId.PLAYER_ONE, PlayerId.PLAYER_ONE, OutcomeResolution.HIGH_SCORE),
        ((1, 3), PlayerId.PLAYER_ONE, PlayerId.PLAYER_TWO, OutcomeResolution.HIGH_SCORE),
        ((2, 2), PlayerId.PLAYER_TWO, PlayerId.PLAYER_TWO, OutcomeResolution.ACTIVE_SCORE_TIE),
    ],
)
def test_deck_exhaustion_score_resolution(
    scores: tuple[int, int],
    active: PlayerId,
    winner: PlayerId,
    resolution: OutcomeResolution,
) -> None:
    state = _terminal_state(scores=scores, active=active)
    hands = list(state.hands)
    hands[1 if active is PlayerId.PLAYER_ONE else 0] = ()
    state = replace(state, deck=(), hands=(hands[0], hands[1]))
    outcome = adjudicate(state)
    assert outcome is not None
    assert outcome.reason is OutcomeReason.DECK_EXHAUSTION
    assert outcome.winner is winner
    assert outcome.resolution is resolution

"""Centralized end-of-turn terminal adjudication."""

from collections import Counter

from .cards import CardName
from .model import (
    GameState,
    OutcomeReason,
    OutcomeResolution,
    PlayerId,
    TerminalFacts,
    TerminalOutcome,
    player_index,
)


def adjudicate(state: GameState) -> TerminalOutcome | None:
    """Resolve all conditions from one post-recruit state.

    All direct win and loss conditions are converted to winner candidates. If both players are
    candidates, the outgoing active player wins as required by every condition tie rule. Deck
    exhaustion is considered only when no instant or score condition applies.
    """
    p1, p2 = PlayerId.PLAYER_ONE, PlayerId.PLAYER_TWO
    players = (p1, p2)
    score_gap_winners = tuple(
        player
        for player in players
        if state.scores[player_index(player)] >= state.scores[player_index(player.other())] + 7
    )
    counts = tuple(Counter(cards) for cards in state.recruited)
    instant_winners = tuple(
        player for player in players if counts[player_index(player)][CardName.CODEBREAKER] >= 3
    )
    instant_losers = tuple(
        player for player in players if counts[player_index(player)][CardName.DAREDEVIL] >= 3
    )
    candidate_set = set(score_gap_winners) | set(instant_winners)
    candidate_set.update(player.other() for player in instant_losers)
    candidate_winners = tuple(player for player in players if player in candidate_set)
    next_player = state.active_player.other()
    facts = TerminalFacts(
        scores=state.scores,
        codebreaker_counts=(
            counts[0][CardName.CODEBREAKER],
            counts[1][CardName.CODEBREAKER],
        ),
        daredevil_counts=(
            counts[0][CardName.DAREDEVIL],
            counts[1][CardName.DAREDEVIL],
        ),
        score_gap_winners=score_gap_winners,
        instant_winners=instant_winners,
        instant_losers=instant_losers,
        candidate_winners=candidate_winners,
        deck_empty=not state.deck,
        next_player_hand_size=len(state.hands[player_index(next_player)]),
    )
    if candidate_winners:
        if len(candidate_winners) == 1:
            winner = candidate_winners[0]
            resolution = OutcomeResolution.SOLE_CANDIDATE
        else:
            winner = state.active_player
            resolution = OutcomeResolution.ACTIVE_CONDITION_TIE
        return TerminalOutcome(
            winner=winner,
            reason=OutcomeReason.CONDITION,
            resolution=resolution,
            active_player=state.active_player,
            turn=state.turn,
            facts=facts,
        )
    if not state.deck and facts.next_player_hand_size < 2:
        active_score = state.scores[player_index(state.active_player)]
        next_score = state.scores[player_index(next_player)]
        if active_score == next_score:
            winner = state.active_player
            resolution = OutcomeResolution.ACTIVE_SCORE_TIE
        else:
            winner = state.active_player if active_score > next_score else next_player
            resolution = OutcomeResolution.HIGH_SCORE
        return TerminalOutcome(
            winner=winner,
            reason=OutcomeReason.DECK_EXHAUSTION,
            resolution=resolution,
            active_player=state.active_player,
            turn=state.turn,
            facts=facts,
        )
    return None

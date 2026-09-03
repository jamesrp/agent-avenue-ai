"""Centralized end-of-turn terminal adjudication."""

from collections import Counter
from collections.abc import Sequence

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

TERMINAL_EVALUATOR_VERSION = "engine-terminal-v1"


def adjudicate_position(
    *,
    scores: tuple[int, int],
    recruited: tuple[Sequence[CardName], Sequence[CardName]],
    active_player: PlayerId,
    turn: int,
    deck_empty: bool,
    next_player_hand_size: int,
) -> TerminalOutcome | None:
    """Adjudicate one public post-recruit position using the authoritative tie rules.

    The arguments are exactly the public material facts needed by terminal resolution. Keeping this
    pure entry point beside :func:`adjudicate` lets information-safe policies reuse the engine's
    evaluator without constructing or receiving an authoritative ``GameState``.
    """
    p1, p2 = PlayerId.PLAYER_ONE, PlayerId.PLAYER_TWO
    players = (p1, p2)
    score_gap_winners = tuple(
        player
        for player in players
        if scores[player_index(player)] >= scores[player_index(player.other())] + 7
    )
    counts = tuple(Counter(cards) for cards in recruited)
    instant_winners = tuple(
        player for player in players if counts[player_index(player)][CardName.CODEBREAKER] >= 3
    )
    instant_losers = tuple(
        player for player in players if counts[player_index(player)][CardName.DAREDEVIL] >= 3
    )
    candidate_set = set(score_gap_winners) | set(instant_winners)
    candidate_set.update(player.other() for player in instant_losers)
    candidate_winners = tuple(player for player in players if player in candidate_set)
    facts = TerminalFacts(
        scores=scores,
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
        deck_empty=deck_empty,
        next_player_hand_size=next_player_hand_size,
    )
    if candidate_winners:
        if len(candidate_winners) == 1:
            winner = candidate_winners[0]
            resolution = OutcomeResolution.SOLE_CANDIDATE
        else:
            winner = active_player
            resolution = OutcomeResolution.ACTIVE_CONDITION_TIE
        return TerminalOutcome(
            winner=winner,
            reason=OutcomeReason.CONDITION,
            resolution=resolution,
            active_player=active_player,
            turn=turn,
            facts=facts,
        )
    if deck_empty and next_player_hand_size < 2:
        next_player = active_player.other()
        active_score = scores[player_index(active_player)]
        next_score = scores[player_index(next_player)]
        if active_score == next_score:
            winner = active_player
            resolution = OutcomeResolution.ACTIVE_SCORE_TIE
        else:
            winner = active_player if active_score > next_score else next_player
            resolution = OutcomeResolution.HIGH_SCORE
        return TerminalOutcome(
            winner=winner,
            reason=OutcomeReason.DECK_EXHAUSTION,
            resolution=resolution,
            active_player=active_player,
            turn=turn,
            facts=facts,
        )
    return None


def adjudicate(state: GameState) -> TerminalOutcome | None:
    """Resolve all conditions from one post-recruit authoritative state."""
    next_player = state.active_player.other()
    return adjudicate_position(
        scores=state.scores,
        recruited=state.recruited,
        active_player=state.active_player,
        turn=state.turn,
        deck_empty=not state.deck,
        next_player_hand_size=len(state.hands[player_index(next_player)]),
    )

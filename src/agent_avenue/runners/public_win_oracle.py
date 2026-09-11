"""Independent public current-turn win oracle for replay and policy audits.

This module intentionally does not import the production terminal-offense classifier. It derives
publicly possible assignments itself and delegates only final material adjudication to the engine's
central terminal evaluator.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from agent_avenue.engine.cards import CANONICAL_DECK, CardName, recruit_effect
from agent_avenue.engine.model import (
    Action,
    GameState,
    OfferSlot,
    Phase,
    PlayerId,
    PlayOfferAction,
    RecruitAction,
    TerminalOutcome,
    player_index,
)
from agent_avenue.engine.terminal import adjudicate_position
from agent_avenue.engine.transitions import apply_action, legal_actions
from agent_avenue.observation import observe
from agent_avenue.observation.model import PlayerObservation, PublicPlayer, RecruitContext


class PublicWinOracleError(ValueError):
    """Raised when a public decision cannot be audited consistently."""


@dataclass(frozen=True, slots=True)
class OracleActionOutcomes:
    """All public-consistent terminal outcomes for one candidate action."""

    action: Action
    outcomes: tuple[TerminalOutcome | None, ...]
    exact_transition_cross_checks: int

    @property
    def guaranteed_winner(self) -> PlayerId | None:
        """Return the common winner when every assignment terminates for that player."""
        if not self.outcomes or any(outcome is None for outcome in self.outcomes):
            return None
        winners = {outcome.winner for outcome in self.outcomes if outcome is not None}
        return next(iter(winners)) if len(winners) == 1 else None


@dataclass(frozen=True, slots=True)
class PublicForcedWinOracle:
    """Independent forced-win classification for one complete public decision."""

    action_outcomes: tuple[OracleActionOutcomes, ...]
    forced_win_actions: tuple[Action, ...]
    hidden_identities: tuple[CardName, ...]
    exact_transition_cross_checks: int


def _public_player(observation: PlayerObservation, player: PlayerId) -> PublicPlayer:
    matches = tuple(item for item in observation.players if item.player is player)
    if len(matches) != 1:
        raise PublicWinOracleError("observation must contain each public player exactly once")
    return matches[0]


def public_consistent_hidden_identities(
    observation: PlayerObservation, face_up: CardName
) -> tuple[CardName, ...]:
    """Enumerate every face-down name with positive viewer-visible remaining multiplicity."""
    remaining = Counter(CANONICAL_DECK)
    for player in observation.players:
        remaining.subtract(player.recruited)
    remaining.subtract(observation.own_hand)
    remaining[face_up] -= 1
    if any(count < 0 for count in remaining.values()):
        raise PublicWinOracleError("observation contains impossible public card counts")
    identities = tuple(card for card in CardName if remaining[card] > 0)
    if not identities:
        raise PublicWinOracleError("no public-consistent face-down identity remains")
    return identities


def _deck_empty_after_action(observation: PlayerObservation, action: Action) -> bool:
    if isinstance(action, RecruitAction):
        return observation.remaining_deck_count == 0
    remaining_hand = len(observation.own_hand) - 2
    draws = min(max(4 - remaining_hand, 0), observation.remaining_deck_count)
    return draws == observation.remaining_deck_count


def _outcome_from_public_material(
    observation: PlayerObservation,
    *,
    viewer_card: CardName,
    opponent_card: CardName,
    deck_empty: bool,
) -> TerminalOutcome | None:
    viewer = _public_player(observation, observation.viewer)
    opponent = _public_player(observation, observation.viewer.other())
    scores = [0, 0]
    recruited: list[tuple[CardName, ...]] = [(), ()]
    for public, card in ((viewer, viewer_card), (opponent, opponent_card)):
        index = player_index(public.player)
        cards = (*public.recruited, card)
        recruited[index] = cards
        effect = recruit_effect(card, cards.count(card))
        scores[index] = public.score + (effect.points if effect.kind == "score" else 0)
    next_player = observation.active_player.other()
    return adjudicate_position(
        scores=(scores[0], scores[1]),
        recruited=(recruited[0], recruited[1]),
        active_player=observation.active_player,
        turn=observation.turn,
        deck_empty=deck_empty,
        next_player_hand_size=_public_player(observation, next_player).hand_size,
    )


def _play_material_outcomes(
    observation: PlayerObservation, action: PlayOfferAction
) -> tuple[TerminalOutcome | None, TerminalOutcome | None]:
    deck_empty = _deck_empty_after_action(observation, action)
    # FACE_UP means the opponent recruits the visible card and the viewer gets face down.
    face_up_choice = _outcome_from_public_material(
        observation,
        viewer_card=action.face_down,
        opponent_card=action.face_up,
        deck_empty=deck_empty,
    )
    face_down_choice = _outcome_from_public_material(
        observation,
        viewer_card=action.face_up,
        opponent_card=action.face_down,
        deck_empty=deck_empty,
    )
    return (face_up_choice, face_down_choice)


def _exact_play_outcomes(
    state: GameState, observation: PlayerObservation, action: PlayOfferAction
) -> tuple[TerminalOutcome | None, TerminalOutcome | None]:
    if observe(state, observation.viewer) != observation:
        raise PublicWinOracleError(
            "authoritative state does not match the supplied public decision"
        )
    after_play = apply_action(state, action)
    recruits = legal_actions(after_play)
    if len(recruits) != 2 or any(not isinstance(item, RecruitAction) for item in recruits):
        raise PublicWinOracleError("play transition did not produce both recruit assignments")
    by_slot = {item.slot: item for item in recruits if isinstance(item, RecruitAction)}
    if set(by_slot) != set(OfferSlot):
        raise PublicWinOracleError("play transition recruit assignments are incomplete")
    outcomes: list[TerminalOutcome | None] = []
    for slot in OfferSlot:
        resolved = apply_action(after_play, by_slot[slot])
        outcomes.append(resolved.outcome if resolved.phase is Phase.TERMINAL else None)
    return (outcomes[0], outcomes[1])


def _recruit_outcomes(
    observation: PlayerObservation, action: RecruitAction
) -> tuple[tuple[TerminalOutcome | None, ...], tuple[CardName, ...]]:
    decision = observation.decision
    if not isinstance(decision, RecruitContext):
        raise PublicWinOracleError("recruit action requires public recruit context")
    identities = (
        (decision.known_face_down,)
        if decision.known_face_down is not None
        else public_consistent_hidden_identities(observation, decision.face_up)
    )
    deck_empty = _deck_empty_after_action(observation, action)
    outcomes = tuple(
        _outcome_from_public_material(
            observation,
            viewer_card=decision.face_up if action.slot is OfferSlot.FACE_UP else hidden,
            opponent_card=hidden if action.slot is OfferSlot.FACE_UP else decision.face_up,
            deck_empty=deck_empty,
        )
        for hidden in identities
    )
    return outcomes, identities


def independent_public_forced_win_oracle(
    observation: PlayerObservation,
    legal_action_set: tuple[Action, ...],
    *,
    authoritative_state: GameState | None = None,
    exact_play_actions: tuple[PlayOfferAction, ...] | None = None,
) -> PublicForcedWinOracle:
    """Classify public guaranteed wins without calling the production offense implementation.

    When an authoritative replay state is supplied, exact play cross-checks default to every
    candidate. Audits may provide ``exact_play_actions`` to cross-check the selected action plus
    every action classified as a guaranteed win; either way, each checked offer resolves both
    recruit assignments through engine transitions. Recruit decisions deliberately remain
    information-set calculations over all public-consistent hidden identities.
    """
    if not legal_action_set:
        raise PublicWinOracleError("forced-win oracle requires at least one legal action")
    if any(action not in observation.legal_actions for action in legal_action_set):
        raise PublicWinOracleError("oracle actions must belong to the public legal action set")

    if exact_play_actions is not None and (
        authoritative_state is None
        or any(action not in legal_action_set for action in exact_play_actions)
    ):
        raise PublicWinOracleError("exact play cross-check actions must be legal and state-backed")

    rows: list[OracleActionOutcomes] = []
    all_hidden: tuple[CardName, ...] = ()
    cross_checks = 0
    for action in legal_action_set:
        outcomes: tuple[TerminalOutcome | None, ...]
        if isinstance(action, PlayOfferAction):
            outcomes = _play_material_outcomes(observation, action)
            checks = 0
        elif isinstance(action, RecruitAction):
            outcomes, identities = _recruit_outcomes(observation, action)
            if all_hidden and identities != all_hidden:
                raise PublicWinOracleError("recruit candidates used inconsistent hidden identities")
            all_hidden = identities
            checks = 0
        else:  # pragma: no cover - closed Action union defensive guard
            raise TypeError(f"unsupported action type: {type(action)!r}")
        rows.append(OracleActionOutcomes(action, outcomes, checks))
        cross_checks += checks
    forced = tuple(
        row.action
        for row in rows
        if row.outcomes
        and all(
            outcome is not None and outcome.winner is observation.viewer for outcome in row.outcomes
        )
    )
    if authoritative_state is not None:
        requested = (
            tuple(action for action in legal_action_set if isinstance(action, PlayOfferAction))
            if exact_play_actions is None
            else exact_play_actions
        )
        cross_check_actions = {
            *requested,
            *(action for action in forced if isinstance(action, PlayOfferAction)),
        }
        checked_rows: list[OracleActionOutcomes] = []
        for row in rows:
            if isinstance(row.action, PlayOfferAction) and row.action in cross_check_actions:
                exact = _exact_play_outcomes(authoritative_state, observation, row.action)
                if exact != row.outcomes:
                    raise PublicWinOracleError(
                        "public play adjudication disagrees with exact engine transitions"
                    )
                checks = len(exact)
                checked_rows.append(OracleActionOutcomes(row.action, row.outcomes, checks))
                cross_checks += checks
            else:
                checked_rows.append(row)
        rows = checked_rows
    return PublicForcedWinOracle(tuple(rows), forced, all_hidden, cross_checks)

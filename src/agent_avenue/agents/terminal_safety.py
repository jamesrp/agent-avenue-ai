"""Information-safe immediate terminal-loss policy shield."""

from collections import Counter
from dataclasses import dataclass, replace
from typing import Final

from agent_avenue.engine.cards import CANONICAL_DECK, CardName, recruit_effect
from agent_avenue.engine.model import (
    Action,
    OfferSlot,
    PlayerId,
    PlayOfferAction,
    RecruitAction,
    TerminalOutcome,
    player_index,
)
from agent_avenue.engine.terminal import TERMINAL_EVALUATOR_VERSION, adjudicate_position
from agent_avenue.observation.model import PlayerObservation, PublicPlayer, RecruitContext

from .base import Agent, PublicDecision
from .random_source import RandomSource

TERMINAL_SAFETY_VERSION: Final = "terminal-safety-v1"
FALLBACK_VERSION: Final = "all-losing-preserve-complete-v1"
UNCERTAINTY_VERSION: Final = "public-remaining-multiset-enumeration-v1"
RESOLUTION_SCOPE: Final = "current-offer-resolution-only-v1"


@dataclass(frozen=True, slots=True)
class TerminalSafetyFilter:
    """One deterministic filtering decision, including forced-loss fallback metadata."""

    allowed_actions: tuple[Action, ...]
    provable_loss_actions: tuple[Action, ...]
    vetoed_actions: tuple[Action, ...]
    forced_loss_fallback: bool


def _public_player(observation: PlayerObservation, player: PlayerId) -> PublicPlayer:
    matches = tuple(item for item in observation.players if item.player is player)
    if len(matches) != 1:
        raise ValueError("observation must contain each public player exactly once")
    return matches[0]


def information_consistent_face_down_cards(
    observation: PlayerObservation, face_up: CardName
) -> tuple[CardName, ...]:
    """Enumerate identities with positive viewer-visible remaining counts."""
    remaining = Counter(CANONICAL_DECK)
    for player in observation.players:
        remaining.subtract(player.recruited)
    remaining.subtract(observation.own_hand)
    remaining[face_up] -= 1
    if any(count < 0 for count in remaining.values()):
        raise ValueError("observation contains impossible public card counts")
    cards = tuple(card for card in CardName if remaining[card] > 0)
    if not cards:
        raise ValueError("no information-consistent face-down card remains")
    return cards


def _deck_empty_after_action(
    observation: PlayerObservation, action: PlayOfferAction | RecruitAction
) -> bool:
    if isinstance(action, RecruitAction):
        return observation.remaining_deck_count == 0
    remaining_hand_size = len(observation.own_hand) - 2
    draw_count = min(max(4 - remaining_hand_size, 0), observation.remaining_deck_count)
    return observation.remaining_deck_count == draw_count


def terminal_outcomes_for_action(
    observation: PlayerObservation, action: PlayOfferAction | RecruitAction
) -> tuple[TerminalOutcome | None, ...]:
    """Resolve every public-information-consistent current-turn assignment."""
    deck_empty = _deck_empty_after_action(observation, action)
    assignments: tuple[tuple[CardName, CardName], ...]
    if isinstance(action, PlayOfferAction):
        # The opponent may recruit either slot; the offerer receives the other card.
        assignments = (
            (action.face_down, action.face_up),
            (action.face_up, action.face_down),
        )
    else:
        decision = observation.decision
        if not isinstance(decision, RecruitContext):
            raise ValueError("recruit action requires public recruit context")
        hidden_cards = (
            (decision.known_face_down,)
            if decision.known_face_down is not None
            else information_consistent_face_down_cards(observation, decision.face_up)
        )
        assignments = tuple(
            (decision.face_up, hidden)
            if action.slot is OfferSlot.FACE_UP
            else (hidden, decision.face_up)
            for hidden in hidden_cards
        )
    return tuple(
        _outcome_after_assignment(
            observation,
            viewer_card=viewer_card,
            opponent_card=opponent_card,
            deck_empty=deck_empty,
        )
        for viewer_card, opponent_card in assignments
    )


def _outcome_after_assignment(
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


def _is_viewer_loss(outcome: TerminalOutcome | None, viewer: PlayerId) -> bool:
    return outcome is not None and outcome.winner is not viewer


def _play_is_unsafe(observation: PlayerObservation, action: PlayOfferAction) -> bool:
    return any(
        _is_viewer_loss(outcome, observation.viewer)
        for outcome in terminal_outcomes_for_action(observation, action)
    )


def _recruit_is_unsafe(observation: PlayerObservation, action: RecruitAction) -> bool:
    outcomes = terminal_outcomes_for_action(observation, action)
    # A recruit action is provably losing only across the entire public information set.
    return all(_is_viewer_loss(outcome, observation.viewer) for outcome in outcomes)


def filter_terminal_actions(
    observation: PlayerObservation, legal_actions: tuple[Action, ...]
) -> TerminalSafetyFilter:
    """Veto only publicly provable immediate losses, preserving semantic action order."""
    if not legal_actions:
        raise ValueError("terminal safety requires at least one legal action")
    unsafe: list[Action] = []
    for action in legal_actions:
        if isinstance(action, PlayOfferAction):
            is_unsafe = _play_is_unsafe(observation, action)
        elif isinstance(action, RecruitAction):
            is_unsafe = _recruit_is_unsafe(observation, action)
        else:  # pragma: no cover - closed Action union defensive guard
            raise TypeError(f"unsupported action type: {type(action)!r}")
        if is_unsafe:
            unsafe.append(action)
    vetoed = tuple(unsafe)
    if len(vetoed) == len(legal_actions):
        return TerminalSafetyFilter(legal_actions, vetoed, (), True)
    return TerminalSafetyFilter(
        tuple(action for action in legal_actions if action not in vetoed), vetoed, vetoed, False
    )


def _base_config(agent: Agent) -> dict[str, object]:
    config = getattr(agent, "config", None)
    to_data = getattr(config, "to_data", None)
    if callable(to_data):
        data = to_data()
    else:
        config_to_data = getattr(agent, "config_to_data", None)
        if not callable(config_to_data):
            raise ValueError("wrapped agent must expose normalized configuration")
        data = config_to_data()
    if not isinstance(data, dict):
        raise ValueError("wrapped agent configuration must be a JSON object")
    return data


@dataclass(frozen=True, slots=True)
class TerminalSafetyAgent:
    """Filter a public turn before invoking any base policy or epsilon exploration."""

    base: Agent

    def config_to_data(self) -> dict[str, object]:
        return {
            "type": "terminal_safety",
            "version": TERMINAL_SAFETY_VERSION,
            "base": _base_config(self.base),
            "fallback": FALLBACK_VERSION,
            "public_uncertainty": UNCERTAINTY_VERSION,
            "resolution_scope": RESOLUTION_SCOPE,
            "terminal_evaluator": TERMINAL_EVALUATOR_VERSION,
        }

    def choose_action(
        self,
        observation: PlayerObservation,
        decision: PublicDecision,
        legal_actions: tuple[Action, ...],
        rng: RandomSource,
    ) -> Action:
        if decision != observation.decision or legal_actions != observation.legal_actions:
            raise ValueError("agent inputs must describe one consistent public turn")
        result = filter_terminal_actions(observation, legal_actions)
        filtered_observation = replace(observation, legal_actions=result.allowed_actions)
        action = self.base.choose_action(
            filtered_observation,
            filtered_observation.decision,
            result.allowed_actions,
            rng,
        )
        if action not in result.allowed_actions:
            raise ValueError("wrapped agent returned an action vetoed by terminal safety")
        return action

"""Information-safe versioned greedy baseline policy.

The policy evaluates only ``PlayerObservation``, its public decision, semantic legal actions, and
public card definitions. Unknown cards are expectations over the viewer-visible remaining multiset;
no authoritative deck order, opposing hand, or true face-down card is accepted by this module.
"""

from collections import Counter
from dataclasses import dataclass, fields
from fractions import Fraction

from agent_avenue.engine.cards import CANONICAL_DECK, CardName, recruit_effect
from agent_avenue.engine.model import Action, OfferSlot, PlayerId, PlayOfferAction, RecruitAction
from agent_avenue.observation.model import PlayerObservation, PublicPlayer, RecruitContext

from .base import PublicDecision
from .random_source import RandomSource

HEURISTIC_VERSION = "greedy-public-v1"


@dataclass(frozen=True, slots=True)
class GreedyHeuristicConfig:
    """Normalized integer weights for the v1 public-information heuristic."""

    version: str = HEURISTIC_VERSION
    immediate_score_weight: int = 100
    score_gap_weight: int = 10
    threshold_win_weight: int = 100_000
    codebreaker_progress_weight: int = 300
    daredevil_risk_weight: int = 350

    def __post_init__(self) -> None:
        if self.version != HEURISTIC_VERSION:
            raise ValueError(f"unsupported heuristic version: {self.version!r}")
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name != "version" and (type(value) is not int or value < 0):
                raise ValueError(f"{item.name} must be a non-negative integer")

    def to_data(self) -> dict[str, object]:
        return {
            "codebreaker_progress_weight": self.codebreaker_progress_weight,
            "daredevil_risk_weight": self.daredevil_risk_weight,
            "immediate_score_weight": self.immediate_score_weight,
            "score_gap_weight": self.score_gap_weight,
            "threshold_win_weight": self.threshold_win_weight,
            "type": "greedy_heuristic",
            "version": self.version,
        }

    @classmethod
    def from_data(cls, data: object) -> "GreedyHeuristicConfig":
        if not isinstance(data, dict) or data.get("type") != "greedy_heuristic":
            raise ValueError("malformed greedy heuristic configuration")
        expected = {item.name for item in fields(cls)} | {"type"}
        if set(data) != expected:
            raise ValueError("greedy heuristic configuration fields do not match its version")
        return cls(
            version=data["version"],
            immediate_score_weight=data["immediate_score_weight"],
            score_gap_weight=data["score_gap_weight"],
            threshold_win_weight=data["threshold_win_weight"],
            codebreaker_progress_weight=data["codebreaker_progress_weight"],
            daredevil_risk_weight=data["daredevil_risk_weight"],
        )


def _public_player(observation: PlayerObservation, player: PlayerId) -> PublicPlayer:
    return next(item for item in observation.players if item.player is player)


def _effect(card: CardName, recruited: tuple[CardName, ...]) -> tuple[int, bool, bool]:
    effect = recruit_effect(card, recruited.count(card) + 1)
    return (
        effect.points if effect.kind == "score" else 0,
        effect.kind == "win",
        effect.kind == "lose",
    )


def _assignment_utility(
    observation: PlayerObservation,
    viewer_card: CardName,
    opponent_card: CardName,
    config: GreedyHeuristicConfig,
) -> int:
    viewer = _public_player(observation, observation.viewer)
    opponent = _public_player(observation, observation.viewer.other())
    viewer_points, viewer_instant_win, viewer_instant_lose = _effect(viewer_card, viewer.recruited)
    opponent_points, opponent_instant_win, opponent_instant_lose = _effect(
        opponent_card, opponent.recruited
    )
    viewer_score = viewer.score + viewer_points
    opponent_score = opponent.score + opponent_points
    score_swing = viewer_points - opponent_points
    utility = score_swing * config.immediate_score_weight
    utility += (viewer_score - opponent_score) * config.score_gap_weight

    viewer_candidate = (
        viewer_score >= opponent_score + 7 or viewer_instant_win or opponent_instant_lose
    )
    opponent_candidate = (
        opponent_score >= viewer_score + 7 or opponent_instant_win or viewer_instant_lose
    )
    if viewer_candidate or opponent_candidate:
        if viewer_candidate and opponent_candidate:
            winner = observation.active_player
        else:
            winner = observation.viewer if viewer_candidate else observation.viewer.other()
        utility += (
            config.threshold_win_weight
            if winner is observation.viewer
            else -config.threshold_win_weight
        )

    viewer_cb = viewer.recruited.count(CardName.CODEBREAKER) + (viewer_card is CardName.CODEBREAKER)
    opponent_cb = opponent.recruited.count(CardName.CODEBREAKER) + (
        opponent_card is CardName.CODEBREAKER
    )
    viewer_dd = viewer.recruited.count(CardName.DAREDEVIL) + (viewer_card is CardName.DAREDEVIL)
    opponent_dd = opponent.recruited.count(CardName.DAREDEVIL) + (
        opponent_card is CardName.DAREDEVIL
    )
    utility += (viewer_cb - opponent_cb) * config.codebreaker_progress_weight
    utility += (opponent_dd - viewer_dd) * config.daredevil_risk_weight
    return utility


def visible_unknown_multiset(
    observation: PlayerObservation, face_up: CardName
) -> Counter[CardName]:
    """Return cards not identified by the viewer, including opposing hand and deck.

    During a hidden recruit choice this information set also includes the actual face-down card,
    but only as one indistinguishable member of the multiset.
    """
    remaining = Counter(CANONICAL_DECK)
    for player in observation.players:
        remaining.subtract(player.recruited)
    remaining.subtract(observation.own_hand)
    remaining[face_up] -= 1
    if any(count < 0 for count in remaining.values()):
        raise ValueError("observation contains impossible public card counts")
    return +remaining


def _expected_unknown_utility(
    observation: PlayerObservation,
    known_card: CardName,
    unknown_goes_to_viewer: bool,
    config: GreedyHeuristicConfig,
) -> Fraction:
    unknown = visible_unknown_multiset(observation, known_card)
    total = sum(unknown.values())
    if total <= 0:
        raise ValueError("no information-visible cards remain for a face-down recruit")
    weighted = 0
    for card, copies in unknown.items():
        if unknown_goes_to_viewer:
            utility = _assignment_utility(observation, card, known_card, config)
        else:
            utility = _assignment_utility(observation, known_card, card, config)
        weighted += utility * copies
    return Fraction(weighted, total)


def score_actions(
    observation: PlayerObservation,
    legal_actions: tuple[Action, ...],
    config: GreedyHeuristicConfig,
) -> tuple[Fraction, ...]:
    """Score legal actions solely from one safe observation."""
    scores: list[Fraction] = []
    for action in legal_actions:
        if isinstance(action, PlayOfferAction):
            face_up_choice = _assignment_utility(
                observation, action.face_down, action.face_up, config
            )
            face_down_choice = _assignment_utility(
                observation, action.face_up, action.face_down, config
            )
            # The opponent recruits adversarially; the offerer receives the other card.
            scores.append(Fraction(min(face_up_choice, face_down_choice)))
        elif isinstance(action, RecruitAction):
            decision = observation.decision
            if not isinstance(decision, RecruitContext):
                raise ValueError("recruit action requires public recruit context")
            if decision.known_face_down is not None:
                if action.slot is OfferSlot.FACE_UP:
                    value = _assignment_utility(
                        observation, decision.face_up, decision.known_face_down, config
                    )
                else:
                    value = _assignment_utility(
                        observation, decision.known_face_down, decision.face_up, config
                    )
                scores.append(Fraction(value))
            else:
                scores.append(
                    _expected_unknown_utility(
                        observation,
                        decision.face_up,
                        action.slot is OfferSlot.FACE_DOWN,
                        config,
                    )
                )
        else:  # pragma: no cover - closed Action union defensive guard
            raise TypeError(f"unsupported action type: {type(action)!r}")
    return tuple(scores)


@dataclass(frozen=True, slots=True)
class GreedyHeuristicAgent:
    """Choose the highest public-information score, randomly breaking exact ties."""

    config: GreedyHeuristicConfig = GreedyHeuristicConfig()

    def choose_action(
        self,
        observation: PlayerObservation,
        decision: PublicDecision,
        legal_actions: tuple[Action, ...],
        rng: RandomSource,
    ) -> Action:
        if decision != observation.decision or legal_actions != observation.legal_actions:
            raise ValueError("agent inputs must describe one consistent public turn")
        if not legal_actions:
            raise ValueError("heuristic agent requires at least one legal action")
        scores = score_actions(observation, legal_actions, self.config)
        best = max(scores)
        candidates = tuple(index for index, score in enumerate(scores) if score == best)
        return legal_actions[candidates[rng.randbelow(len(candidates))]]

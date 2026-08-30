"""Stable semantic ordering for public game actions."""

from agent_avenue.engine.cards import CardName
from agent_avenue.engine.model import Action, OfferSlot, PlayOfferAction, RecruitAction

_CARD_ORDER = tuple(CardName)


def semantic_action_key(action: Action) -> tuple[int, int, int]:
    """Return an ordering independent of transient legal-action tuple positions."""
    if isinstance(action, PlayOfferAction):
        return (0, _CARD_ORDER.index(action.face_up), _CARD_ORDER.index(action.face_down))
    if isinstance(action, RecruitAction):
        return (1, 0 if action.slot is OfferSlot.FACE_UP else 1, 0)
    raise TypeError("unsupported semantic action")

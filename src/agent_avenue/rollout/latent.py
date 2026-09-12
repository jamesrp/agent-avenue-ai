"""Trusted-offline latent worlds for information-safe Step-4 rollouts.

``LatentRolloutState`` is deliberately not an authoritative live-state representation.  It has
no setup seed, configuration, replay provenance, or route into live agents.  It projects only a
fresh allowlisted ``PlayerObservation`` for the current decision actor.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Final

from agent_avenue.agents.ordering import semantic_action_key
from agent_avenue.agents.random_source import RandomSource
from agent_avenue.engine.cards import CardName, recruit_effect
from agent_avenue.engine.model import (
    Action,
    CompletedTurn,
    Offer,
    OfferSlot,
    Phase,
    PlayDecision,
    PlayerId,
    PlayOfferAction,
    RecruitAction,
    RecruitDecision,
    TerminalOutcome,
    player_index,
)
from agent_avenue.engine.terminal import adjudicate_position
from agent_avenue.observation.model import (
    PlayContext,
    PlayerObservation,
    PublicPlayer,
    RecruitContext,
)

EXPANDED_CANONICAL_DECK: Final[tuple[CardName, ...]] = (
    CardName.DOUBLE_AGENT,
    CardName.DOUBLE_AGENT,
    CardName.DOUBLE_AGENT,
    CardName.DOUBLE_AGENT,
    CardName.DOUBLE_AGENT,
    CardName.DOUBLE_AGENT,
    CardName.ENFORCER,
    CardName.ENFORCER,
    CardName.ENFORCER,
    CardName.ENFORCER,
    CardName.ENFORCER,
    CardName.ENFORCER,
    CardName.CODEBREAKER,
    CardName.CODEBREAKER,
    CardName.CODEBREAKER,
    CardName.CODEBREAKER,
    CardName.CODEBREAKER,
    CardName.CODEBREAKER,
    CardName.DAREDEVIL,
    CardName.DAREDEVIL,
    CardName.DAREDEVIL,
    CardName.DAREDEVIL,
    CardName.DAREDEVIL,
    CardName.DAREDEVIL,
    CardName.SABOTEUR,
    CardName.SABOTEUR,
    CardName.SABOTEUR,
    CardName.SABOTEUR,
    CardName.SABOTEUR,
    CardName.SABOTEUR,
    CardName.SENTINEL,
    CardName.SENTINEL,
    CardName.SENTINEL,
    CardName.SENTINEL,
    CardName.SENTINEL,
    CardName.SENTINEL,
    CardName.SIDEKICK,
    CardName.MOLE,
)
LATENT_STATE_VERSION: Final[str] = "latent-rollout-state-v1"
LATENT_SAMPLER_VERSION: Final[str] = "expanded-copy-exchangeable-sampler-v1"
ROLLOUT_TRANSITION_VERSION: Final[str] = "rollout-transition-v1"


class LatentRolloutError(ValueError):
    """Raised for an invalid synthetic allocation or mechanical rollout transition."""


def _is_exact_int(value: object) -> bool:
    return type(value) is int


def _canonical_cards(cards: Sequence[CardName]) -> tuple[CardName, ...]:
    return tuple(sorted(cards, key=lambda card: tuple(CardName).index(card)))


def _replace_player_tuple[T](values: tuple[T, T], index: int, value: T) -> tuple[T, T]:
    return (value, values[1]) if index == 0 else (values[0], value)


@dataclass(frozen=True, slots=True)
class LatentRolloutState:
    """A synthetic allocation containing public material and sampled hidden zones only."""

    deck: tuple[CardName, ...]
    hands: tuple[tuple[CardName, ...], tuple[CardName, ...]]
    recruited: tuple[tuple[CardName, ...], tuple[CardName, ...]]
    scores: tuple[int, int]
    active_player: PlayerId
    turn: int
    phase: Phase
    revision: int
    offer: Offer | None = None
    history: tuple[CompletedTurn, ...] = ()
    outcome: TerminalOutcome | None = None


def _expected_active(state: LatentRolloutState) -> PlayerId | None:
    if not state.history:
        return None
    if state.phase is Phase.TERMINAL:
        return state.history[-1].active_player
    return state.history[-1].active_player.other()


def _validate_history(state: LatentRolloutState) -> None:
    rebuilt_recruited: list[list[CardName]] = [[], []]
    rebuilt_scores = [0, 0]
    prior_active: PlayerId | None = None
    for index, item in enumerate(state.history):
        if type(item) is not CompletedTurn or item.turn != index + 1:
            raise LatentRolloutError("history turns must be contiguous CompletedTurn values")
        if prior_active is not None and item.active_player is prior_active:
            raise LatentRolloutError("history active players must alternate")
        prior_active = item.active_player
        expected_opponent = (
            item.face_up if item.chosen_slot is OfferSlot.FACE_UP else item.face_down
        )
        expected_active = item.face_down if item.chosen_slot is OfferSlot.FACE_UP else item.face_up
        if (
            item.opponent_recruited is not expected_opponent
            or item.active_recruited is not expected_active
        ):
            raise LatentRolloutError("history recruit assignments are inconsistent")
        active_index = player_index(item.active_player)
        opponent_index = player_index(item.active_player.other())
        rebuilt_recruited[opponent_index].append(expected_opponent)
        rebuilt_recruited[active_index].append(expected_active)
        score_changes = [0, 0]
        for player, card in ((opponent_index, expected_opponent), (active_index, expected_active)):
            effect = recruit_effect(card, rebuilt_recruited[player].count(card))
            if effect.kind == "score":
                score_changes[player] = effect.points
        if item.score_changes != (score_changes[0], score_changes[1]):
            raise LatentRolloutError("history score changes are inconsistent")
        rebuilt_scores[0] += score_changes[0]
        rebuilt_scores[1] += score_changes[1]
    if state.recruited != (tuple(rebuilt_recruited[0]), tuple(rebuilt_recruited[1])):
        raise LatentRolloutError("recruited cards do not match public history")
    if state.scores != (rebuilt_scores[0], rebuilt_scores[1]):
        raise LatentRolloutError("scores do not match public history")


def validate_latent_state(state: LatentRolloutState) -> None:
    """Validate synthetic structure/card conservation without provenance reconstruction."""
    if type(state) is not LatentRolloutState:
        raise LatentRolloutError("state must be a LatentRolloutState")
    if (
        not isinstance(state.active_player, PlayerId)
        or not isinstance(state.phase, Phase)
        or state.phase is Phase.END
    ):
        raise LatentRolloutError("state has an invalid active player or phase")
    if any(not _is_exact_int(value) for value in (state.turn, state.revision)):
        raise LatentRolloutError("turn and revision must be exact integers")
    if not 1 <= state.turn <= 19 or state.revision < 0:
        raise LatentRolloutError("state is outside the mechanical turn/revision range")
    if (
        type(state.deck) is not tuple
        or type(state.hands) is not tuple
        or type(state.recruited) is not tuple
        or type(state.history) is not tuple
        or len(state.hands) != 2
        or len(state.recruited) != 2
        or len(state.scores) != 2
    ):
        raise LatentRolloutError("latent state zones must be immutable two-player tuples")
    if any(type(zone) is not tuple for zone in (*state.hands, *state.recruited)):
        raise LatentRolloutError("latent card zones must be tuples")
    if any(not _is_exact_int(score) for score in state.scores):
        raise LatentRolloutError("scores must be exact integers")
    if any(len(hand) > 4 for hand in state.hands):
        raise LatentRolloutError("hands cannot contain more than four cards")
    if state.phase is Phase.PLAY and (state.offer is not None or state.outcome is not None):
        raise LatentRolloutError("play positions may not contain an offer or terminal outcome")
    if state.phase is Phase.RECRUIT and (state.offer is None or state.outcome is not None):
        raise LatentRolloutError("recruit positions require one unresolved offer")
    if state.phase is Phase.TERMINAL and (state.offer is not None or state.outcome is None):
        raise LatentRolloutError("terminal positions require one exact terminal outcome")
    if state.offer is not None and state.offer.offered_by is not state.active_player:
        raise LatentRolloutError("an offer must be owned by the active player")
    if state.offer is not None and type(state.offer) is not Offer:
        raise LatentRolloutError("latent offers must use the immutable offer type")
    expected_revision = len(state.history) * 2 + (1 if state.phase is Phase.RECRUIT else 0)
    if state.revision != expected_revision:
        raise LatentRolloutError("revision does not agree with phase and completed history")
    expected_turn = len(state.history) if state.phase is Phase.TERMINAL else len(state.history) + 1
    if state.turn != max(expected_turn, 1):
        raise LatentRolloutError("turn does not agree with completed history")
    expected_active = _expected_active(state)
    if expected_active is not None and state.active_player is not expected_active:
        raise LatentRolloutError("active player does not alternate from public history")
    cards = list(state.deck)
    cards.extend(state.hands[0])
    cards.extend(state.hands[1])
    cards.extend(state.recruited[0])
    cards.extend(state.recruited[1])
    if state.offer is not None:
        cards.extend((state.offer.face_up, state.offer.face_down))
    if any(not isinstance(card, CardName) for card in cards) or Counter(cards) != Counter(
        EXPANDED_CANONICAL_DECK
    ):
        raise LatentRolloutError("all 38 expanded copies must be conserved across latent zones")
    _validate_history(state)
    if state.phase is Phase.PLAY:
        public_outcome = adjudicate_position(
            scores=state.scores,
            recruited=state.recruited,
            active_player=state.active_player,
            turn=state.turn,
            deck_empty=not state.deck,
            next_player_hand_size=len(state.hands[player_index(state.active_player.other())]),
        )
        if public_outcome is not None:
            raise LatentRolloutError("nonterminal play state already has an outcome")
    if state.phase is Phase.RECRUIT:
        assert state.offer is not None
        if state.offer.offered_by is not state.active_player:
            raise LatentRolloutError("recruit offer owner is inconsistent")
    if state.phase is Phase.TERMINAL:
        assert state.outcome is not None
        expected_outcome = adjudicate_position(
            scores=state.scores,
            recruited=state.recruited,
            active_player=state.active_player,
            turn=state.turn,
            deck_empty=not state.deck,
            next_player_hand_size=len(state.hands[player_index(state.active_player.other())]),
        )
        if state.outcome != expected_outcome:
            raise LatentRolloutError("terminal outcome does not match pure adjudication")


def latent_decision(state: LatentRolloutState) -> PlayDecision | RecruitDecision | TerminalOutcome:
    """Return the current semantic decision without entering the authoritative engine boundary."""
    validate_latent_state(state)
    if state.phase is Phase.TERMINAL:
        assert state.outcome is not None
        return state.outcome
    if state.phase is Phase.PLAY:
        return PlayDecision(state.revision, state.active_player, state.turn)
    assert state.offer is not None
    return RecruitDecision(
        state.revision, state.active_player.other(), state.turn, state.offer.face_up
    )


def latent_legal_actions(state: LatentRolloutState) -> tuple[Action, ...]:
    """Generate the complete semantic legal set for a synthetic state."""
    latent_decision(state)
    if state.phase is Phase.TERMINAL:
        return ()
    if state.phase is Phase.PLAY:
        hand = state.hands[player_index(state.active_player)]
        names = tuple(dict.fromkeys(hand))
        if len(hand) < 2:
            return ()
        if len(names) == 1:
            return (PlayOfferAction(state.revision, state.active_player, names[0], names[0]),)
        return tuple(
            PlayOfferAction(state.revision, state.active_player, face_up, face_down)
            for face_up in names
            for face_down in names
            if face_up is not face_down
        )
    actor = state.active_player.other()
    return tuple(RecruitAction(state.revision, actor, slot) for slot in OfferSlot)


def _validate_action_shape(action: object) -> Action:
    if type(action) not in (PlayOfferAction, RecruitAction):
        raise LatentRolloutError("action has an unsupported runtime type")
    assert isinstance(action, PlayOfferAction | RecruitAction)
    if not _is_exact_int(action.revision) or not isinstance(action.actor, PlayerId):
        raise LatentRolloutError("action has malformed common fields")
    if isinstance(action, PlayOfferAction) and (
        not isinstance(action.face_up, CardName) or not isinstance(action.face_down, CardName)
    ):
        raise LatentRolloutError("play action cards must be card names")
    if isinstance(action, RecruitAction) and not isinstance(action.slot, OfferSlot):
        raise LatentRolloutError("recruit action slot must be an offer slot")
    return action


def _remove_cards(
    hand: tuple[CardName, ...], first: CardName, second: CardName
) -> tuple[CardName, ...]:
    cards = list(hand)
    try:
        cards.remove(first)
        cards.remove(second)
    except ValueError as exc:
        raise LatentRolloutError("offered cards are absent from the acting hand") from exc
    return tuple(cards)


def _apply_play(state: LatentRolloutState, action: PlayOfferAction) -> LatentRolloutState:
    active_index = player_index(state.active_player)
    remaining = _remove_cards(state.hands[active_index], action.face_up, action.face_down)
    draw_count = min(4 - len(remaining), len(state.deck))
    hands = _replace_player_tuple(state.hands, active_index, remaining + state.deck[:draw_count])
    return replace(
        state,
        deck=state.deck[draw_count:],
        hands=hands,
        phase=Phase.RECRUIT,
        revision=state.revision + 1,
        offer=Offer(state.active_player, action.face_up, action.face_down),
    )


def _apply_recruit(state: LatentRolloutState, action: RecruitAction) -> LatentRolloutState:
    assert state.offer is not None
    active = state.active_player
    opponent = active.other()
    if action.slot is OfferSlot.FACE_UP:
        opponent_card, active_card = state.offer.face_up, state.offer.face_down
    else:
        opponent_card, active_card = state.offer.face_down, state.offer.face_up
    active_index, opponent_index = player_index(active), player_index(opponent)
    recruited_lists = [list(state.recruited[0]), list(state.recruited[1])]
    recruited_lists[opponent_index].append(opponent_card)
    recruited_lists[active_index].append(active_card)
    recruited = (tuple(recruited_lists[0]), tuple(recruited_lists[1]))
    score_changes = [0, 0]
    for index, card in ((opponent_index, opponent_card), (active_index, active_card)):
        effect = recruit_effect(card, recruited[index].count(card))
        if effect.kind == "score":
            score_changes[index] = effect.points
    scores = (state.scores[0] + score_changes[0], state.scores[1] + score_changes[1])
    completed = CompletedTurn(
        state.turn,
        active,
        state.offer.face_up,
        state.offer.face_down,
        action.slot,
        opponent_card,
        active_card,
        (score_changes[0], score_changes[1]),
    )
    resolved = replace(
        state,
        recruited=recruited,
        scores=scores,
        revision=state.revision + 1,
        history=(*state.history, completed),
        offer=None,
        phase=Phase.END,
    )
    outcome = adjudicate_position(
        scores=resolved.scores,
        recruited=resolved.recruited,
        active_player=resolved.active_player,
        turn=resolved.turn,
        deck_empty=not resolved.deck,
        next_player_hand_size=len(resolved.hands[player_index(opponent)]),
    )
    if outcome is not None:
        return replace(resolved, phase=Phase.TERMINAL, outcome=outcome)
    if resolved.turn >= 19:
        raise LatentRolloutError("turn 19 completed without a terminal outcome")
    return replace(
        resolved,
        active_player=opponent,
        turn=resolved.turn + 1,
        phase=Phase.PLAY,
    )


def rollout_transition_v1(state: LatentRolloutState, action: Action) -> LatentRolloutState:
    """Purely apply one semantic action using engine-equivalent rules and draw ordering."""
    validate_latent_state(state)
    action = _validate_action_shape(action)
    if state.phase is Phase.TERMINAL:
        raise LatentRolloutError("the synthetic game is already terminal")
    expected_actor = (
        state.active_player if state.phase is Phase.PLAY else state.active_player.other()
    )
    if action.revision != state.revision or action.actor is not expected_actor:
        raise LatentRolloutError("action revision or actor does not own the latent decision")
    if action not in latent_legal_actions(state):
        raise LatentRolloutError("action is not legal in the latent state")
    if isinstance(action, PlayOfferAction):
        next_state = _apply_play(state, action)
    else:
        next_state = _apply_recruit(state, action)
    validate_latent_state(next_state)
    return next_state


def _validate_public_observation(observation: PlayerObservation) -> None:
    if type(observation) is not PlayerObservation:
        raise LatentRolloutError("sampler requires a PlayerObservation")
    if observation.phase not in (Phase.PLAY, Phase.RECRUIT):
        raise LatentRolloutError("sampler requires a nonterminal play or recruit observation")
    if observation.viewer is not getattr(observation.decision, "actor", None):
        raise LatentRolloutError("sampler observation must belong to the decision actor")
    if observation.turn < 1 or observation.turn > 19 or observation.remaining_deck_count < 0:
        raise LatentRolloutError("observation has an invalid turn or deck count")
    if len(observation.players) != 2 or tuple(
        player.player for player in observation.players
    ) != tuple(PlayerId):
        raise LatentRolloutError("observation players must be ordered player one/player two")
    own_public = observation.players[player_index(observation.viewer)]
    if own_public.hand_size != len(observation.own_hand) or own_public.hand_size > 4:
        raise LatentRolloutError("observation own hand does not agree with public hand size")
    if any(player.hand_size < 0 or player.hand_size > 4 for player in observation.players):
        raise LatentRolloutError("observation hand sizes must be in [0, 4]")
    if observation.phase is Phase.PLAY:
        if (
            not isinstance(observation.decision, PlayContext)
            or observation.active_player is not observation.viewer
        ):
            raise LatentRolloutError("play observation decision is inconsistent")
    else:
        decision = observation.decision
        if (
            not isinstance(decision, RecruitContext)
            or decision.known_face_down is not None
            or observation.active_player is not decision.offered_by
            or observation.viewer is not observation.active_player.other()
        ):
            raise LatentRolloutError("recruit sampler may only use the recruiter safe view")


def _remaining_copies(
    *,
    recruited: tuple[tuple[CardName, ...], tuple[CardName, ...]],
    own_hand: tuple[CardName, ...],
    known_face_up: CardName | None,
) -> list[CardName]:
    remaining = list(EXPANDED_CANONICAL_DECK)
    for card in (*recruited[0], *recruited[1], *own_hand):
        try:
            remaining.remove(card)
        except ValueError as exc:
            raise LatentRolloutError("public cards exceed expanded deck copies") from exc
    if known_face_up is not None:
        try:
            remaining.remove(known_face_up)
        except ValueError as exc:
            raise LatentRolloutError("known face-up card exceeds expanded deck copies") from exc
    return remaining


def _draw_without_replacement(
    pool: list[CardName], count: int, rng: RandomSource
) -> tuple[CardName, ...]:
    if count < 0 or count > len(pool):
        raise LatentRolloutError("latent allocation requests an invalid number of copies")
    chosen: list[CardName] = []
    for _ in range(count):
        chosen.append(pool.pop(rng.randbelow(len(pool))))
    return tuple(chosen)


def _reverse_fisher_yates(pool: list[CardName], rng: RandomSource) -> tuple[CardName, ...]:
    for upper in range(len(pool) - 1, 0, -1):
        lower = rng.randbelow(upper + 1)
        pool[upper], pool[lower] = pool[lower], pool[upper]
    return tuple(pool)


def sample_latent_world(observation: PlayerObservation, rng: RandomSource) -> LatentRolloutState:
    """Sample an exchangeable expanded-copy world from a safe public observation only."""
    _validate_public_observation(observation)
    own_hand = _canonical_cards(observation.own_hand)
    recruited = (observation.players[0].recruited, observation.players[1].recruited)
    scores = (observation.players[0].score, observation.players[1].score)
    actor = observation.viewer
    opponent = actor.other()
    if observation.phase is Phase.PLAY:
        pool = _remaining_copies(recruited=recruited, own_hand=own_hand, known_face_up=None)
        opponent_hand = _draw_without_replacement(
            pool, observation.players[player_index(opponent)].hand_size, rng
        )
        deck = _reverse_fisher_yates(pool, rng)
        if len(deck) != observation.remaining_deck_count:
            raise LatentRolloutError("play public counts do not match residual hidden deck size")
        hands = (
            own_hand if actor is PlayerId.PLAYER_ONE else opponent_hand,
            opponent_hand if actor is PlayerId.PLAYER_ONE else own_hand,
        )
        play_decision = observation.decision
        assert isinstance(play_decision, PlayContext)
        state = LatentRolloutState(
            deck,
            hands,
            recruited,
            scores,
            observation.active_player,
            observation.turn,
            Phase.PLAY,
            play_decision.revision,
            None,
            observation.history,
        )
    else:
        decision = observation.decision
        assert isinstance(decision, RecruitContext)
        pool = _remaining_copies(
            recruited=recruited, own_hand=own_hand, known_face_up=decision.face_up
        )
        face_down = _draw_without_replacement(pool, 1, rng)[0]
        offerer_hand = _draw_without_replacement(
            pool, observation.players[player_index(decision.offered_by)].hand_size, rng
        )
        deck = _reverse_fisher_yates(pool, rng)
        if len(deck) != observation.remaining_deck_count:
            raise LatentRolloutError("recruit public counts do not match residual hidden deck size")
        hands = (
            own_hand if actor is PlayerId.PLAYER_ONE else offerer_hand,
            offerer_hand if actor is PlayerId.PLAYER_ONE else own_hand,
        )
        state = LatentRolloutState(
            deck,
            hands,
            recruited,
            scores,
            decision.offered_by,
            observation.turn,
            Phase.RECRUIT,
            decision.revision,
            Offer(decision.offered_by, decision.face_up, face_down),
            observation.history,
        )
    validate_latent_state(state)
    return state


def safe_latent_observation(state: LatentRolloutState, viewer: PlayerId) -> PlayerObservation:
    """Project only the decision actor's safe, canonicalized view of a latent world.

    In particular, a recruiter always receives ``known_face_down=None``.  This function does not
    call the production observer and never accepts/constructs an authoritative game state.
    """
    decision = latent_decision(state)
    if isinstance(decision, TerminalOutcome):
        raise LatentRolloutError("terminal latent states cannot be sent to policies or encoders")
    if viewer is not decision.actor:
        raise LatentRolloutError("only the current decision actor may observe a latent rollout")
    legal = tuple(sorted(latent_legal_actions(state), key=semantic_action_key))
    context: PlayContext | RecruitContext
    if state.phase is Phase.PLAY:
        context = PlayContext("play", decision.revision, decision.actor)
    else:
        assert state.offer is not None
        context = RecruitContext(
            "recruit",
            decision.revision,
            decision.actor,
            state.offer.offered_by,
            state.offer.face_up,
            None,
        )
    players = tuple(
        PublicPlayer(
            player,
            state.scores[player_index(player)],
            state.recruited[player_index(player)],
            len(state.hands[player_index(player)]),
        )
        for player in PlayerId
    )
    return PlayerObservation(
        viewer,
        _canonical_cards(state.hands[player_index(viewer)]),
        (players[0], players[1]),
        state.active_player,
        state.turn,
        state.phase,
        len(state.deck),
        state.history,
        context,
        legal,
    )


__all__ = [
    "EXPANDED_CANONICAL_DECK",
    "LATENT_SAMPLER_VERSION",
    "LATENT_STATE_VERSION",
    "ROLLOUT_TRANSITION_VERSION",
    "LatentRolloutError",
    "LatentRolloutState",
    "latent_decision",
    "latent_legal_actions",
    "rollout_transition_v1",
    "safe_latent_observation",
    "sample_latent_world",
    "validate_latent_state",
]

"""Pure legal-action generation and deterministic state transitions."""

from collections import Counter
from dataclasses import replace

from .cards import CANONICAL_DECK, CardName, recruit_effect
from .errors import EngineError, ErrorCode
from .model import (
    Action,
    CompletedTurn,
    Decision,
    GameState,
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
from .setup import validate_config
from .terminal import adjudicate


def _is_exact_int(value: object) -> bool:
    return type(value) is int


def _validate_action_shape(action: object) -> None:
    if type(action) not in (PlayOfferAction, RecruitAction):
        raise EngineError(ErrorCode.WRONG_ACTION_TYPE, "action has an unsupported runtime type")
    assert isinstance(action, PlayOfferAction | RecruitAction)
    if not _is_exact_int(action.revision) or not isinstance(action.actor, PlayerId):
        raise EngineError(ErrorCode.ILLEGAL_ACTION, "action contains malformed common fields")
    if isinstance(action, PlayOfferAction):
        if not isinstance(action.face_up, CardName) or not isinstance(action.face_down, CardName):
            raise EngineError(
                ErrorCode.ILLEGAL_ACTION, "play action card names must be CardName values"
            )
    elif not isinstance(action.slot, OfferSlot):
        raise EngineError(ErrorCode.ILLEGAL_ACTION, "recruit action slot must be an OfferSlot")


def _expected_active(state: GameState) -> PlayerId:
    completed = len(state.history)
    flips = completed if state.phase not in (Phase.END, Phase.TERMINAL) else completed - 1
    player = state.config.starting_player
    for _ in range(max(flips, 0)):
        player = player.other()
    return player


def _validate_history(state: GameState) -> None:
    expected_action_count = len(state.history) * 2 + (1 if state.phase is Phase.RECRUIT else 0)
    if len(state.actions) != expected_action_count:
        raise EngineError(
            ErrorCode.INVALID_STATE, "action count is inconsistent with phase/history"
        )
    rebuilt_recruited: list[list[CardName]] = [[], []]
    rebuilt_scores = [0, 0]
    for index, item in enumerate(state.history):
        if item.turn != index + 1:
            raise EngineError(ErrorCode.INVALID_STATE, "history turn numbers must be contiguous")
        play = state.actions[index * 2]
        recruit = state.actions[index * 2 + 1]
        if not isinstance(play, PlayOfferAction) or not isinstance(recruit, RecruitAction):
            raise EngineError(
                ErrorCode.INVALID_STATE, "completed turns require play/recruit action pairs"
            )
        expected_player = state.config.starting_player
        if index % 2 == 1:
            expected_player = expected_player.other()
        if item.active_player is not expected_player or play.actor is not expected_player:
            raise EngineError(ErrorCode.INVALID_STATE, "history active players must alternate")
        if (
            item.active_player is not play.actor
            or item.face_up is not play.face_up
            or item.face_down is not play.face_down
            or item.chosen_slot is not recruit.slot
            or recruit.actor is not play.actor.other()
        ):
            raise EngineError(ErrorCode.INVALID_STATE, "history does not match semantic actions")
        opponent_card = item.face_up if item.chosen_slot is OfferSlot.FACE_UP else item.face_down
        active_card = item.face_down if item.chosen_slot is OfferSlot.FACE_UP else item.face_up
        if item.opponent_recruited is not opponent_card or item.active_recruited is not active_card:
            raise EngineError(
                ErrorCode.INVALID_STATE, "history recruit assignments are inconsistent"
            )
        active_index = player_index(item.active_player)
        opponent_index = player_index(item.active_player.other())
        rebuilt_recruited[opponent_index].append(opponent_card)
        rebuilt_recruited[active_index].append(active_card)
        changes = [0, 0]
        for player_idx, card in ((opponent_index, opponent_card), (active_index, active_card)):
            effect = recruit_effect(card, rebuilt_recruited[player_idx].count(card))
            if effect.kind == "score":
                changes[player_idx] = effect.points
        if item.score_changes != (changes[0], changes[1]):
            raise EngineError(ErrorCode.INVALID_STATE, "history score changes are inconsistent")
        rebuilt_scores[0] += changes[0]
        rebuilt_scores[1] += changes[1]
    if state.recruited != (tuple(rebuilt_recruited[0]), tuple(rebuilt_recruited[1])):
        raise EngineError(ErrorCode.INVALID_STATE, "recruited cards do not match public history")
    if state.scores != (rebuilt_scores[0], rebuilt_scores[1]):
        raise EngineError(ErrorCode.INVALID_STATE, "scores do not match public history")
    if state.phase is Phase.RECRUIT:
        assert state.offer is not None
        play = state.actions[-1]
        if not isinstance(play, PlayOfferAction) or state.offer != Offer(
            play.actor, play.face_up, play.face_down
        ):
            raise EngineError(ErrorCode.INVALID_STATE, "current offer does not match latest play")


def validate_state(state: GameState) -> None:
    """Check structural and card-conservation invariants."""
    validate_config(state.config)
    if not isinstance(state.active_player, PlayerId) or not isinstance(state.phase, Phase):
        raise EngineError(ErrorCode.INVALID_STATE, "state player and phase must use domain enums")
    if (
        not _is_exact_int(state.seed)
        or not _is_exact_int(state.turn)
        or not _is_exact_int(state.revision)
    ):
        raise EngineError(
            ErrorCode.INVALID_STATE, "state seed, turn, and revision must be integers"
        )
    if state.turn < 1 or state.revision < 0 or state.revision != len(state.actions):
        raise EngineError(ErrorCode.INVALID_STATE, "invalid turn or decision revision")
    if len(state.hands) != 2 or len(state.recruited) != 2 or len(state.scores) != 2:
        raise EngineError(ErrorCode.INVALID_STATE, "player tuple fields must have length two")
    if state.phase is Phase.PLAY and (state.offer is not None or state.outcome is not None):
        raise EngineError(ErrorCode.INVALID_STATE, "play phase cannot contain offer or outcome")
    if state.phase is Phase.RECRUIT and (state.offer is None or state.outcome is not None):
        raise EngineError(ErrorCode.INVALID_STATE, "recruit phase requires exactly one offer")
    if state.offer is not None and state.offer.offered_by is not state.active_player:
        raise EngineError(ErrorCode.INVALID_STATE, "offer must belong to the active player")
    if state.phase is Phase.END:
        raise EngineError(
            ErrorCode.INVALID_STATE, "end phase is internal and cannot escape a transition"
        )
    if state.phase is Phase.TERMINAL and (state.offer is not None or state.outcome is None):
        raise EngineError(ErrorCode.INVALID_STATE, "terminal phase requires exactly one outcome")
    if (
        type(state.deck) is not tuple
        or type(state.hands) is not tuple
        or type(state.recruited) is not tuple
    ):
        raise EngineError(ErrorCode.INVALID_STATE, "authoritative card zones must be tuples")
    if any(type(zone) is not tuple for zone in (*state.hands, *state.recruited)):
        raise EngineError(ErrorCode.INVALID_STATE, "player card zones must be tuples")
    if type(state.actions) is not tuple or type(state.history) is not tuple:
        raise EngineError(ErrorCode.INVALID_STATE, "actions and history must be tuples")
    cards = list(state.deck)
    cards.extend(state.hands[0])
    cards.extend(state.hands[1])
    cards.extend(state.recruited[0])
    cards.extend(state.recruited[1])
    if state.offer is not None:
        cards.extend((state.offer.face_up, state.offer.face_down))
    if any(not isinstance(card, CardName) for card in cards):
        raise EngineError(ErrorCode.INVALID_STATE, "all cards must be CardName values")
    if any(len(hand) > 4 for hand in state.hands):
        raise EngineError(ErrorCode.INVALID_STATE, "hands cannot contain more than four cards")
    if Counter(cards) != Counter(CANONICAL_DECK):
        raise EngineError(
            ErrorCode.INVALID_STATE,
            "cards must be conserved across deck, hands, offer, and recruited zones",
            card_count=len(cards),
        )
    for action_index, action in enumerate(state.actions):
        try:
            _validate_action_shape(action)
        except EngineError as exc:
            raise EngineError(ErrorCode.INVALID_STATE, "state contains a malformed action") from exc
        if action.revision != action_index:
            raise EngineError(ErrorCode.INVALID_STATE, "action revisions must be contiguous")
    expected_turn = (
        len(state.history) if state.phase in (Phase.END, Phase.TERMINAL) else len(state.history) + 1
    )
    if state.turn != max(expected_turn, 1) or state.active_player is not _expected_active(state):
        raise EngineError(
            ErrorCode.INVALID_STATE, "turn or active player is inconsistent with history"
        )
    _validate_history(state)
    if state.phase is Phase.PLAY and adjudicate(state) is not None:
        raise EngineError(ErrorCode.INVALID_STATE, "nonterminal play state already has an outcome")
    if state.phase is Phase.TERMINAL:
        assert state.outcome is not None
        if adjudicate(replace(state, outcome=None)) != state.outcome:
            raise EngineError(
                ErrorCode.INVALID_STATE, "terminal outcome does not match adjudication"
            )
    _validate_provenance(state)


def current_decision(state: GameState) -> Decision | TerminalOutcome:
    """Return the sole semantic decision, or the terminal outcome."""
    validate_state(state)
    if state.phase is Phase.TERMINAL:
        assert state.outcome is not None
        return state.outcome
    if state.phase is Phase.END:
        raise EngineError(ErrorCode.INVALID_STATE, "end-of-turn resolution is automatic")
    if state.phase is Phase.PLAY:
        return PlayDecision(state.revision, state.active_player, state.turn)
    assert state.offer is not None
    return RecruitDecision(
        state.revision,
        state.active_player.other(),
        state.turn,
        state.offer.face_up,
    )


def _legal_actions_unchecked(state: GameState) -> tuple[Action, ...]:
    if state.phase is Phase.TERMINAL:
        return ()
    if state.phase is Phase.PLAY:
        actor = state.active_player
        hand = state.hands[player_index(actor)]
        names = tuple(dict.fromkeys(hand))
        if len(hand) < 2:
            return ()
        if len(names) == 1:
            card = names[0]
            return (PlayOfferAction(state.revision, actor, card, card),)
        return tuple(
            PlayOfferAction(state.revision, actor, face_up, face_down)
            for face_up in names
            for face_down in names
            if face_up is not face_down
        )
    if state.phase is Phase.RECRUIT:
        actor = state.active_player.other()
        return tuple(RecruitAction(state.revision, actor, slot) for slot in OfferSlot)
    return ()


def legal_actions(state: GameState) -> tuple[Action, ...]:
    """Return stable, complete semantic actions for the current decision."""
    current_decision(state)
    return _legal_actions_unchecked(state)


def _remove_cards(
    hand: tuple[CardName, ...], first: CardName, second: CardName
) -> tuple[CardName, ...]:
    mutable = list(hand)
    try:
        mutable.remove(first)
        mutable.remove(second)
    except ValueError as exc:  # guarded by legal action membership
        raise EngineError(ErrorCode.ILLEGAL_ACTION, "offered cards are not in hand") from exc
    return tuple(mutable)


def _replace_player_tuple[T](values: tuple[T, T], index: int, value: T) -> tuple[T, T]:
    return (value, values[1]) if index == 0 else (values[0], value)


def _apply_play(state: GameState, action: PlayOfferAction) -> GameState:
    active_index = player_index(state.active_player)
    remaining = _remove_cards(state.hands[active_index], action.face_up, action.face_down)
    draw_count = min(4 - len(remaining), len(state.deck))
    drawn = state.deck[:draw_count]
    hands = _replace_player_tuple(state.hands, active_index, remaining + drawn)
    return replace(
        state,
        deck=state.deck[draw_count:],
        hands=hands,
        phase=Phase.RECRUIT,
        revision=state.revision + 1,
        offer=Offer(state.active_player, action.face_up, action.face_down),
        actions=(*state.actions, action),
    )


def _apply_recruit(state: GameState, action: RecruitAction) -> GameState:
    assert state.offer is not None
    active = state.active_player
    opponent = active.other()
    if action.slot is OfferSlot.FACE_UP:
        opponent_card, active_card = state.offer.face_up, state.offer.face_down
    else:
        opponent_card, active_card = state.offer.face_down, state.offer.face_up
    recruited_lists = [list(state.recruited[0]), list(state.recruited[1])]
    active_index, opponent_index = player_index(active), player_index(opponent)
    recruited_lists[opponent_index].append(opponent_card)
    recruited_lists[active_index].append(active_card)
    recruited = (tuple(recruited_lists[0]), tuple(recruited_lists[1]))
    opponent_effect = recruit_effect(opponent_card, recruited[opponent_index].count(opponent_card))
    active_effect = recruit_effect(active_card, recruited[active_index].count(active_card))
    score_changes = [0, 0]
    if opponent_effect.kind == "score":
        score_changes[opponent_index] = opponent_effect.points
    if active_effect.kind == "score":
        score_changes[active_index] = active_effect.points
    scores = (
        state.scores[0] + score_changes[0],
        state.scores[1] + score_changes[1],
    )
    completed = CompletedTurn(
        turn=state.turn,
        active_player=active,
        face_up=state.offer.face_up,
        face_down=state.offer.face_down,
        chosen_slot=action.slot,
        opponent_recruited=opponent_card,
        active_recruited=active_card,
        score_changes=(score_changes[0], score_changes[1]),
    )
    resolved = replace(
        state,
        recruited=recruited,
        scores=scores,
        revision=state.revision + 1,
        actions=(*state.actions, action),
        history=(*state.history, completed),
        offer=None,
        phase=Phase.END,
    )
    outcome = adjudicate(resolved)
    if outcome is not None:
        return replace(resolved, phase=Phase.TERMINAL, outcome=outcome)
    return replace(
        resolved,
        active_player=opponent,
        turn=state.turn + 1,
        phase=Phase.PLAY,
    )


def _apply_action_unchecked(state: GameState, action: Action) -> GameState:
    _validate_action_shape(action)
    if state.phase is Phase.TERMINAL:
        raise EngineError(ErrorCode.GAME_FINISHED, "the game is already finished")
    if action.revision != state.revision:
        raise EngineError(ErrorCode.STALE_ACTION, "action has an unexpected revision")
    expected_actor = (
        state.active_player if state.phase is Phase.PLAY else state.active_player.other()
    )
    if action.actor is not expected_actor:
        raise EngineError(ErrorCode.WRONG_ACTOR, "action has an unexpected actor")
    if action not in _legal_actions_unchecked(state):
        raise EngineError(ErrorCode.ILLEGAL_ACTION, "action is not legal")
    if isinstance(action, PlayOfferAction):
        return _apply_play(state, action)
    return _apply_recruit(state, action)


def _validate_provenance(state: GameState) -> None:
    from .setup import new_game

    reconstructed = new_game(state.config, state.seed)
    try:
        for action in state.actions:
            reconstructed = _apply_action_unchecked(reconstructed, action)
    except EngineError as exc:
        raise EngineError(ErrorCode.INVALID_STATE, "state actions cannot be replayed") from exc
    if reconstructed != state:
        raise EngineError(
            ErrorCode.INVALID_STATE,
            "state does not match deterministic setup and semantic actions",
        )


def apply_action(state: GameState, action: Action) -> GameState:
    """Validate and apply one action without mutating the input state."""
    validate_state(state)
    _validate_action_shape(action)
    if state.phase is Phase.TERMINAL:
        raise EngineError(ErrorCode.GAME_FINISHED, "the game is already finished")
    if action.revision != state.revision:
        raise EngineError(
            ErrorCode.STALE_ACTION,
            "action decision revision does not match current decision",
            expected_revision=state.revision,
            actual_revision=action.revision,
        )
    expected_actor = (
        state.active_player if state.phase is Phase.PLAY else state.active_player.other()
    )
    if action.actor is not expected_actor:
        raise EngineError(
            ErrorCode.WRONG_ACTOR,
            "action actor does not own the current decision",
            expected_actor=expected_actor.value,
            actual_actor=action.actor.value,
        )
    if state.phase is Phase.PLAY and not isinstance(action, PlayOfferAction):
        raise EngineError(ErrorCode.WRONG_ACTION_TYPE, "play decision requires a play action")
    if state.phase is Phase.RECRUIT and not isinstance(action, RecruitAction):
        raise EngineError(ErrorCode.WRONG_ACTION_TYPE, "recruit decision requires a recruit action")
    if action not in legal_actions(state):
        raise EngineError(ErrorCode.ILLEGAL_ACTION, "action is not legal for the current decision")
    if isinstance(action, PlayOfferAction):
        next_state = _apply_play(state, action)
    else:
        next_state = _apply_recruit(state, action)
    validate_state(next_state)
    return next_state

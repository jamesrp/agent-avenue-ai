"""Construction and explicit serialization of player-safe observations."""

from agent_avenue.engine.model import (
    CompletedTurn,
    GameState,
    PlayDecision,
    PlayerId,
    RecruitDecision,
    player_index,
)
from agent_avenue.engine.transitions import current_decision, legal_actions

from .model import (
    ObservationDecision,
    PlayContext,
    PlayerObservation,
    PublicPlayer,
    RecruitContext,
    TerminalContext,
)


def observe(state: GameState, viewer: PlayerId) -> PlayerObservation:
    """Build a safe observation for one player."""
    if not isinstance(viewer, PlayerId):
        raise ValueError("viewer must be a PlayerId")
    decision = current_decision(state)
    if isinstance(decision, PlayDecision):
        context: ObservationDecision = PlayContext("play", decision.revision, decision.actor)
    elif isinstance(decision, RecruitDecision):
        assert state.offer is not None
        known = state.offer.face_down if viewer is state.active_player else None
        context = RecruitContext(
            "recruit",
            decision.revision,
            decision.actor,
            state.active_player,
            decision.face_up,
            known,
        )
    else:
        context = TerminalContext("terminal", decision)
    actions = legal_actions(state) if getattr(decision, "actor", None) is viewer else ()
    players = tuple(
        PublicPlayer(
            player=player,
            score=state.scores[player_index(player)],
            recruited=state.recruited[player_index(player)],
            hand_size=len(state.hands[player_index(player)]),
        )
        for player in PlayerId
    )
    return PlayerObservation(
        viewer=viewer,
        own_hand=state.hands[player_index(viewer)],
        players=(players[0], players[1]),
        active_player=state.active_player,
        turn=state.turn,
        phase=state.phase,
        remaining_deck_count=len(state.deck),
        history=state.history,
        decision=context,
        legal_actions=actions,
    )


def _completed_turn_data(item: CompletedTurn) -> dict[str, object]:
    return {
        "turn": item.turn,
        "active_player": item.active_player.value,
        "face_up": item.face_up.value,
        "face_down": item.face_down.value,
        "chosen_slot": item.chosen_slot.value,
        "opponent_recruited": item.opponent_recruited.value,
        "active_recruited": item.active_recruited.value,
        "score_changes": list(item.score_changes),
    }


def observation_to_data(observation: PlayerObservation) -> dict[str, object]:
    """Serialize only explicitly allowlisted public fields."""
    decision = observation.decision
    if isinstance(decision, PlayContext):
        decision_data: dict[str, object] = {
            "kind": decision.kind,
            "revision": decision.revision,
            "actor": decision.actor.value,
        }
    elif isinstance(decision, RecruitContext):
        decision_data = {
            "kind": decision.kind,
            "revision": decision.revision,
            "actor": decision.actor.value,
            "offered_by": decision.offered_by.value,
            "face_up": decision.face_up.value,
            "known_face_down": (
                None if decision.known_face_down is None else decision.known_face_down.value
            ),
            "slots": [slot.value for slot in decision.slots],
        }
    else:
        outcome = decision.outcome
        facts = outcome.facts
        decision_data = {
            "kind": decision.kind,
            "winner": outcome.winner.value,
            "reason": outcome.reason.value,
            "resolution": outcome.resolution.value,
            "active_player": outcome.active_player.value,
            "turn": outcome.turn,
            "facts": {
                "scores": list(facts.scores),
                "codebreaker_counts": list(facts.codebreaker_counts),
                "daredevil_counts": list(facts.daredevil_counts),
                "score_gap_winners": [player.value for player in facts.score_gap_winners],
                "instant_winners": [player.value for player in facts.instant_winners],
                "instant_losers": [player.value for player in facts.instant_losers],
                "candidate_winners": [player.value for player in facts.candidate_winners],
                "deck_empty": facts.deck_empty,
                "next_player_hand_size": facts.next_player_hand_size,
            },
        }
    action_data: list[dict[str, object]] = []
    for action in observation.legal_actions:
        if hasattr(action, "slot"):
            action_data.append(
                {
                    "type": "recruit",
                    "revision": action.revision,
                    "actor": action.actor.value,
                    "slot": action.slot.value,
                }
            )
        else:
            action_data.append(
                {
                    "type": "play_offer",
                    "revision": action.revision,
                    "actor": action.actor.value,
                    "face_up": action.face_up.value,
                    "face_down": action.face_down.value,
                }
            )
    return {
        "viewer": observation.viewer.value,
        "own_hand": [card.value for card in observation.own_hand],
        "players": [
            {
                "player": player.player.value,
                "score": player.score,
                "recruited": [card.value for card in player.recruited],
                "hand_size": player.hand_size,
            }
            for player in observation.players
        ],
        "active_player": observation.active_player.value,
        "turn": observation.turn,
        "phase": observation.phase.value,
        "remaining_deck_count": observation.remaining_deck_count,
        "history": [_completed_turn_data(item) for item in observation.history],
        "decision": decision_data,
        "legal_actions": action_data,
    }

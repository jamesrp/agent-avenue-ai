"""Safe synchronous agent protocol and turn-boundary validation."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agent_avenue.engine.model import Action
from agent_avenue.observation.model import ObservationDecision, PlayerObservation

from .random_source import RandomSource

PublicDecision = ObservationDecision


@runtime_checkable
class Agent(Protocol):
    """An action chooser with no access to authoritative game state."""

    def choose_action(
        self,
        observation: PlayerObservation,
        decision: PublicDecision,
        legal_actions: tuple[Action, ...],
        rng: RandomSource,
    ) -> Action:
        """Choose one supplied semantic action without advancing engine state."""


@dataclass(frozen=True, slots=True)
class AgentTurn:
    """A viewer-safe turn payload suitable for a runner-to-agent boundary."""

    observation: PlayerObservation
    decision: PublicDecision
    legal_actions: tuple[Action, ...]

    @classmethod
    def from_observation(cls, observation: PlayerObservation) -> "AgentTurn":
        return cls(observation, observation.decision, observation.legal_actions)

    def __post_init__(self) -> None:
        if self.decision != self.observation.decision:
            raise ValueError("decision must match observation.decision")
        if self.legal_actions != self.observation.legal_actions:
            raise ValueError("legal_actions must match observation.legal_actions")
        if not self.legal_actions:
            raise ValueError("an agent turn requires at least one legal action")


def choose_agent_action(agent: Agent, turn: AgentTurn, rng: RandomSource) -> Action:
    """Invoke an agent and fail loudly if it returns an action outside the turn."""
    action = agent.choose_action(turn.observation, turn.decision, turn.legal_actions, rng)
    if action not in turn.legal_actions:
        raise ValueError("agent returned an action that is not legal for this turn")
    return action

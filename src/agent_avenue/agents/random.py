"""Uniform random baseline agent."""

from dataclasses import dataclass

from agent_avenue.engine.model import Action
from agent_avenue.observation.model import PlayerObservation

from .base import PublicDecision
from .random_source import RandomSource


@dataclass(frozen=True, slots=True)
class RandomAgentConfig:
    version: str = "random-agent-v1"

    def __post_init__(self) -> None:
        if self.version != "random-agent-v1":
            raise ValueError(f"unsupported random agent version: {self.version!r}")

    def to_data(self) -> dict[str, object]:
        return {"type": "random", "version": self.version}


@dataclass(frozen=True, slots=True)
class RandomAgent:
    """Select every supplied legal action with equal probability."""

    config: RandomAgentConfig = RandomAgentConfig()

    def choose_action(
        self,
        observation: PlayerObservation,
        decision: PublicDecision,
        legal_actions: tuple[Action, ...],
        rng: RandomSource,
    ) -> Action:
        del observation, decision
        if not legal_actions:
            raise ValueError("random agent requires at least one legal action")
        return legal_actions[rng.randbelow(len(legal_actions))]

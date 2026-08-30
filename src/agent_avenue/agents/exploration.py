"""Seeded rational epsilon exploration for any safe agent."""

from dataclasses import dataclass

from agent_avenue.engine.model import Action
from agent_avenue.observation.model import ObservationDecision, PlayerObservation

from .base import Agent
from .ordering import semantic_action_key
from .random_source import RandomSource


@dataclass(frozen=True, slots=True)
class EpsilonConfig:
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if type(self.numerator) is not int or type(self.denominator) is not int:
            raise ValueError("epsilon numerator and denominator must be integers")
        if self.denominator <= 0 or not 0 <= self.numerator <= self.denominator:
            raise ValueError("epsilon must be a rational value between zero and one")

    def to_data(self) -> dict[str, object]:
        return {
            "version": "epsilon-rational-v1",
            "numerator": self.numerator,
            "denominator": self.denominator,
            "epsilon_zero_consumes_rng": False,
            "epsilon_one_skips_base_policy": True,
            "exploration_selection": "uniform-semantic-order-v1",
        }


@dataclass(slots=True)
class EpsilonGreedyAgent:
    """Use exact rational epsilon exploration without access to hidden state."""

    base: Agent
    epsilon: EpsilonConfig

    def config_to_data(self) -> dict[str, object]:
        base_config = getattr(self.base, "config", None)
        to_data = getattr(base_config, "to_data", None)
        if not callable(to_data):
            raise ValueError("wrapped agent must expose normalized configuration")
        return {
            "version": "epsilon-wrapper-v1",
            "base": to_data(),
            "epsilon": self.epsilon.to_data(),
        }

    def choose_action(
        self,
        observation: PlayerObservation,
        decision: ObservationDecision,
        legal_actions: tuple[Action, ...],
        rng: RandomSource,
    ) -> Action:
        if not legal_actions:
            raise ValueError("epsilon agent requires legal actions")
        candidates = tuple(sorted(legal_actions, key=semantic_action_key))
        if self.epsilon.numerator == self.epsilon.denominator:
            return candidates[rng.randbelow(len(candidates))]
        if self.epsilon.numerator > 0 and (
            rng.randbelow(self.epsilon.denominator) < self.epsilon.numerator
        ):
            return candidates[rng.randbelow(len(candidates))]
        return self.base.choose_action(observation, decision, legal_actions, rng)

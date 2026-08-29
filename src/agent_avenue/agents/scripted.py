"""Strict deterministic scripted agent for tests and replay regressions."""

from collections.abc import Callable
from dataclasses import dataclass, field

from agent_avenue.engine.model import Action
from agent_avenue.observation.model import PlayerObservation

from .base import AgentTurn, PublicDecision
from .random_source import RandomSource

ScriptCallback = Callable[[AgentTurn], Action]


@dataclass(slots=True)
class ScriptedAgent:
    """Consume exact semantic actions, or delegate each turn to a safe callback.

    Script divergence is always an error: actions are never skipped, repaired, or replaced.
    """

    actions: tuple[Action, ...] = ()
    callback: ScriptCallback | None = None
    name: str = "scripted"
    _position: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.callback is not None and self.actions:
            raise ValueError("provide either scripted actions or a callback, not both")
        if not self.name:
            raise ValueError("script name must be non-empty")

    @property
    def consumed(self) -> int:
        return self._position

    @property
    def remaining(self) -> int:
        return len(self.actions) - self._position

    def choose_action(
        self,
        observation: PlayerObservation,
        decision: PublicDecision,
        legal_actions: tuple[Action, ...],
        rng: RandomSource,
    ) -> Action:
        del rng
        turn = AgentTurn(observation, decision, legal_actions)
        if self.callback is not None:
            action = self.callback(turn)
        else:
            if self._position >= len(self.actions):
                raise ValueError(f"script {self.name!r} exhausted at decision {decision!r}")
            action = self.actions[self._position]
        if action not in legal_actions:
            raise ValueError(
                f"script {self.name!r} diverged at position {self._position}: {action!r} is illegal"
            )
        self._position += 1
        return action

    def assert_exhausted(self) -> None:
        """Fail if a completed scenario left unused semantic actions."""
        if self.callback is None and self.remaining:
            raise ValueError(f"script {self.name!r} has {self.remaining} unused action(s)")

    def config_to_data(self) -> dict[str, object]:
        """Return normalized non-action metadata; semantic actions belong in replay data."""
        return {
            "callback": self.callback is not None,
            "name": self.name,
            "type": "scripted",
            "version": "scripted-agent-v1",
        }

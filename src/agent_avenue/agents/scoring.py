"""Model-agnostic candidate-score diagnostics in stable semantic action order."""

from __future__ import annotations

import math
from dataclasses import dataclass

from agent_avenue.engine.model import Action

from .ordering import semantic_action_key


@dataclass(frozen=True, slots=True)
class LearnedCandidateScores:
    """Raw learned-policy scores and their exact maximum set.

    The action tuple is always in semantic order rather than transient engine order. Exact
    floating-point equality is intentional: it is the learned policy's declared tie rule.
    """

    actions: tuple[Action, ...]
    logits: tuple[float, ...]
    maximum_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.actions or len(self.actions) != len(self.logits):
            raise ValueError("candidate scores require aligned non-empty actions and logits")
        if len(set(self.actions)) != len(self.actions):
            raise ValueError("candidate score actions must be unique")
        if self.actions != tuple(sorted(self.actions, key=semantic_action_key)):
            raise ValueError("candidate score actions must use semantic order")
        if any(type(value) is not float or not math.isfinite(value) for value in self.logits):
            raise ValueError("candidate logits must be finite raw floats")
        maximum = max(self.logits)
        expected = tuple(index for index, value in enumerate(self.logits) if value == maximum)
        if self.maximum_indices != expected:
            raise ValueError("maximum indices do not match the exact maximum logits")

    @classmethod
    def from_logits(
        cls, actions: tuple[Action, ...], logits: tuple[float, ...]
    ) -> LearnedCandidateScores:
        """Build a validated diagnostic from semantic-order actions and raw float logits."""
        if not logits:
            raise ValueError("candidate logits cannot be empty")
        maximum = max(logits)
        maxima = tuple(index for index, value in enumerate(logits) if value == maximum)
        return cls(actions, logits, maxima)

    @property
    def tie_count(self) -> int:
        """Return the number of candidates sharing the exact maximum logit."""
        return len(self.maximum_indices)

    @property
    def maximum_actions(self) -> tuple[Action, ...]:
        """Return the exact maximum actions in semantic order."""
        return tuple(self.actions[index] for index in self.maximum_indices)

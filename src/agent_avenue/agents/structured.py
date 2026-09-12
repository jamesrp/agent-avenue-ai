"""Optional checkpoint-backed inference agent for structured value model v2.

It is intentionally not exported by :mod:`agent_avenue.agents`, keeping ordinary engine and baseline
agent imports free of PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Final, cast

import torch

from agent_avenue.encoding.candidate_structured_v2 import (
    ENCODER_FINGERPRINT,
    ENCODER_VERSION,
    FEATURE_WIDTH,
    encode_candidate,
)
from agent_avenue.engine.model import Action
from agent_avenue.learning.structured_checkpoint import LoadedStructuredCheckpoint
from agent_avenue.learning.structured_model import MODEL_VERSION, OUTPUT_SEMANTICS
from agent_avenue.observation.model import ObservationDecision, PlayerObservation

from .ordering import semantic_action_key
from .random_source import RandomSource
from .scoring import LearnedCandidateScores

STRUCTURED_AGENT_VERSION: Final[str] = "structured-value-agent-v2"
STRUCTURED_INFERENCE_POLICY: Final[str] = "batched-structured-residual-mlp-v2"
STRUCTURED_TIE_POLICY: Final[str] = "uniform-exact-max-logit-semantic-order-v1"


def _digest(value: object, label: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class StructuredValueConfig:
    """Immutable normalized policy identity suitable for run/arena provenance."""

    checkpoint_fingerprint: str
    tensor_digest: str
    version: str = STRUCTURED_AGENT_VERSION
    encoder_version: str = ENCODER_VERSION
    encoder_fingerprint: str = ENCODER_FINGERPRINT
    encoder_width: int = FEATURE_WIDTH
    model_version: str = MODEL_VERSION
    output_semantics: str = OUTPUT_SEMANTICS
    device: str = "cpu"
    inference: str = STRUCTURED_INFERENCE_POLICY
    tie_breaking: str = STRUCTURED_TIE_POLICY

    def __post_init__(self) -> None:
        _digest(self.checkpoint_fingerprint, "checkpoint_fingerprint")
        _digest(self.tensor_digest, "tensor_digest")
        if (
            self.version != STRUCTURED_AGENT_VERSION
            or self.encoder_version != ENCODER_VERSION
            or self.encoder_fingerprint != ENCODER_FINGERPRINT
            or self.encoder_width != FEATURE_WIDTH
            or self.model_version != MODEL_VERSION
            or self.output_semantics != OUTPUT_SEMANTICS
            or self.device != "cpu"
            or self.inference != STRUCTURED_INFERENCE_POLICY
            or self.tie_breaking != STRUCTURED_TIE_POLICY
        ):
            raise ValueError("structured agent configuration is incompatible with this runtime")

    def to_data(self) -> dict[str, object]:
        return {
            "type": "structured_value",
            "version": self.version,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "tensor_digest": self.tensor_digest,
            "encoder_version": self.encoder_version,
            "encoder_fingerprint": self.encoder_fingerprint,
            "encoder_width": self.encoder_width,
            "model_version": self.model_version,
            "output_semantics": self.output_semantics,
            "device": self.device,
            "inference": self.inference,
            "tie_breaking": self.tie_breaking,
        }

    @classmethod
    def from_data(cls, data: object) -> StructuredValueConfig:
        if not isinstance(data, dict) or data.get("type") != "structured_value":
            raise ValueError("malformed structured value configuration")
        expected = {field.name for field in fields(cls)} | {"type"}
        if set(data) != expected:
            raise ValueError("structured value configuration fields do not match its version")
        return cls(
            checkpoint_fingerprint=cast(str, data["checkpoint_fingerprint"]),
            tensor_digest=cast(str, data["tensor_digest"]),
            version=cast(str, data["version"]),
            encoder_version=cast(str, data["encoder_version"]),
            encoder_fingerprint=cast(str, data["encoder_fingerprint"]),
            encoder_width=cast(int, data["encoder_width"]),
            model_version=cast(str, data["model_version"]),
            output_semantics=cast(str, data["output_semantics"]),
            device=cast(str, data["device"]),
            inference=cast(str, data["inference"]),
            tie_breaking=cast(str, data["tie_breaking"]),
        )

    @classmethod
    def from_checkpoint(cls, checkpoint: LoadedStructuredCheckpoint) -> StructuredValueConfig:
        return cls(
            checkpoint_fingerprint=checkpoint.checkpoint_fingerprint,
            tensor_digest=_digest(
                checkpoint.manifest.get("tensor_digest"), "checkpoint tensor_digest"
            ),
        )


@dataclass(frozen=True, slots=True)
class StructuredValueAgent:
    """Score every legal candidate in semantic order using one safe v2 batch."""

    checkpoint: LoadedStructuredCheckpoint
    config: StructuredValueConfig

    @classmethod
    def from_checkpoint(cls, checkpoint: LoadedStructuredCheckpoint) -> StructuredValueAgent:
        return cls(checkpoint, StructuredValueConfig.from_checkpoint(checkpoint))

    def __post_init__(self) -> None:
        if self.config != StructuredValueConfig.from_checkpoint(self.checkpoint):
            raise ValueError("structured agent configuration does not match checkpoint")
        if (
            self.checkpoint.model.training
            or next(self.checkpoint.model.parameters()).device.type != "cpu"
        ):
            raise ValueError("structured agent requires an eval-mode CPU checkpoint")
        if any(parameter.requires_grad for parameter in self.checkpoint.model.parameters()):
            raise ValueError("structured agent requires inference-only model parameters")

    def score_candidates(
        self,
        observation: PlayerObservation,
        decision: ObservationDecision,
        legal_actions: tuple[Action, ...],
    ) -> LearnedCandidateScores:
        """Return logits for the complete legal set without consuming policy randomness."""
        if decision != observation.decision or legal_actions != observation.legal_actions:
            raise ValueError("agent inputs must describe one consistent public turn")
        if not legal_actions:
            raise ValueError("structured agent requires at least one legal action")
        candidates = tuple(sorted(legal_actions, key=semantic_action_key))
        features = torch.tensor(
            [encode_candidate(observation, action).vector for action in candidates],
            dtype=torch.float32,
            device="cpu",
        )
        with torch.inference_mode():
            logits = self.checkpoint.model(features)
        if logits.shape != (len(candidates),):
            raise ValueError("structured model returned an incompatible candidate batch")
        return LearnedCandidateScores.from_logits(
            candidates, tuple(float(value) for value in logits.tolist())
        )

    def choose_action(
        self,
        observation: PlayerObservation,
        decision: ObservationDecision,
        legal_actions: tuple[Action, ...],
        rng: RandomSource,
    ) -> Action:
        scores = self.score_candidates(observation, decision, legal_actions)
        maxima = scores.maximum_indices
        selected = maxima[0] if len(maxima) == 1 else maxima[rng.randbelow(len(maxima))]
        return scores.actions[selected]


__all__ = [
    "STRUCTURED_AGENT_VERSION",
    "STRUCTURED_INFERENCE_POLICY",
    "STRUCTURED_TIE_POLICY",
    "StructuredValueAgent",
    "StructuredValueConfig",
]

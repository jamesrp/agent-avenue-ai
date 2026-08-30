"""Checkpoint-backed information-safe candidate-value agent.

This optional module imports PyTorch and is deliberately not re-exported from
``agent_avenue.agents`` so core engine and baseline-agent imports remain RL-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Final, cast

import torch

from agent_avenue.encoding import (
    ENCODER_FINGERPRINT,
    ENCODER_VERSION,
    FEATURE_WIDTH,
    encode_candidate,
)
from agent_avenue.engine.model import Action
from agent_avenue.learning.model import MODEL_VERSION, OUTPUT_SEMANTICS
from agent_avenue.observation.model import ObservationDecision, PlayerObservation

from .ordering import semantic_action_key
from .random_source import RandomSource

if TYPE_CHECKING:
    from agent_avenue.learning.checkpoint import LoadedCheckpoint

LEARNED_AGENT_VERSION: Final = "learned-value-agent-v1"
TIE_POLICY: Final = "uniform-exact-max-logit-semantic-order-v1"
INFERENCE_POLICY: Final = "batched-candidate-mlp-v1"


def _digest(value: object, label: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class LearnedValueConfig:
    """Normalized identity and inference contract for a learned policy."""

    checkpoint_fingerprint: str
    tensor_digest: str
    version: str = LEARNED_AGENT_VERSION
    encoder_version: str = ENCODER_VERSION
    encoder_fingerprint: str = ENCODER_FINGERPRINT
    encoder_width: int = FEATURE_WIDTH
    model_version: str = MODEL_VERSION
    output_semantics: str = OUTPUT_SEMANTICS
    device: str = "cpu"
    inference: str = INFERENCE_POLICY
    tie_breaking: str = TIE_POLICY

    def __post_init__(self) -> None:
        _digest(self.checkpoint_fingerprint, "checkpoint_fingerprint")
        _digest(self.tensor_digest, "tensor_digest")
        expected = (
            self.version == LEARNED_AGENT_VERSION
            and self.encoder_version == ENCODER_VERSION
            and self.encoder_fingerprint == ENCODER_FINGERPRINT
            and self.encoder_width == FEATURE_WIDTH
            and self.model_version == MODEL_VERSION
            and self.output_semantics == OUTPUT_SEMANTICS
            and self.device == "cpu"
            and self.inference == INFERENCE_POLICY
            and self.tie_breaking == TIE_POLICY
        )
        if not expected:
            raise ValueError("learned agent configuration is incompatible with this runtime")

    def to_data(self) -> dict[str, object]:
        return {
            "type": "learned_value",
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
    def from_data(cls, data: object) -> LearnedValueConfig:
        if not isinstance(data, dict) or data.get("type") != "learned_value":
            raise ValueError("malformed learned value configuration")
        expected = {item.name for item in fields(cls)} | {"type"}
        if set(data) != expected:
            raise ValueError("learned value configuration fields do not match its version")
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
    def from_checkpoint(cls, checkpoint: LoadedCheckpoint) -> LearnedValueConfig:
        tensor_digest = checkpoint.manifest.get("tensor_digest")
        return cls(
            checkpoint_fingerprint=checkpoint.checkpoint_fingerprint,
            tensor_digest=_digest(tensor_digest, "checkpoint tensor_digest"),
        )


@dataclass(frozen=True, slots=True)
class LearnedValueAgent:
    """Rank all supplied legal actions in one safe, batched model call."""

    checkpoint: LoadedCheckpoint
    config: LearnedValueConfig

    @classmethod
    def from_checkpoint(cls, checkpoint: LoadedCheckpoint) -> LearnedValueAgent:
        return cls(checkpoint, LearnedValueConfig.from_checkpoint(checkpoint))

    def __post_init__(self) -> None:
        if self.config != LearnedValueConfig.from_checkpoint(self.checkpoint):
            raise ValueError("learned agent configuration does not match checkpoint")
        model = self.checkpoint.model
        if next(model.parameters()).device.type != "cpu" or model.training:
            raise ValueError("learned agent requires an eval-mode CPU checkpoint")
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("learned agent requires inference-only model parameters")

    def choose_action(
        self,
        observation: PlayerObservation,
        decision: ObservationDecision,
        legal_actions: tuple[Action, ...],
        rng: RandomSource,
    ) -> Action:
        if decision != observation.decision or legal_actions != observation.legal_actions:
            raise ValueError("agent inputs must describe one consistent public turn")
        if not legal_actions:
            raise ValueError("learned agent requires at least one legal action")

        candidates = tuple(sorted(legal_actions, key=semantic_action_key))
        vectors = [encode_candidate(observation, action).vector for action in candidates]
        features = torch.tensor(vectors, dtype=torch.float32, device="cpu")
        with torch.inference_mode():
            logits = self.checkpoint.model(features)
        if logits.shape != (len(candidates),):
            raise ValueError("learned model returned an incompatible candidate batch")
        values = tuple(float(value) for value in logits.tolist())
        if any(not math.isfinite(value) for value in values):
            raise ValueError("learned model returned a non-finite candidate logit")
        best = max(values)
        maxima = tuple(index for index, value in enumerate(values) if value == best)
        selected = maxima[0] if len(maxima) == 1 else maxima[rng.randbelow(len(maxima))]
        return candidates[selected]

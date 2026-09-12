"""Q0-embedded structured residual candidate value model v2."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import torch
from torch import Tensor, nn

from .model import (
    CandidateMLP,
)
from .model import (
    ModelError as V1ModelError,
)
from .model import (
    validate_state_dict as validate_v1_state_dict,
)

MODEL_VERSION: Final[str] = "candidate-structured-residual-mlp-v2"
INPUT_WIDTH: Final[int] = 519
V1_PREFIX_WIDTH: Final[int] = 87
BASE_HIDDEN_WIDTH: Final[int] = 128
HISTORY_WIDTH: Final[int] = 314
PLAY_CONSEQUENCE_WIDTH: Final[int] = 64
RECRUIT_CONSEQUENCE_WIDTH: Final[int] = 54
STRUCTURED_WIDTH: Final[int] = 32
RESIDUAL_INPUT_WIDTH: Final[int] = 192
OUTPUT_WIDTH: Final[int] = 1
PARAMETER_COUNT: Final[int] = 35_779
OUTPUT_SEMANTICS: Final[str] = "acting-player-eventual-win-logit"
INITIALIZATION_VERSION: Final[str] = "q0-base+xavier-projections+zero-residual-v1"


class StructuredModelError(ValueError):
    """Raised when the v2 model cannot safely consume a vector or q0 tensor."""


def _validate_seed(seed: int) -> int:
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise StructuredModelError("initialization seed must be an integer in [0, 2**63)")
    return seed


def _q0_state(value: CandidateMLP | Mapping[str, object]) -> Mapping[str, object]:
    state: Mapping[str, object]
    if isinstance(value, CandidateMLP):
        state = value.state_dict()
    elif isinstance(value, Mapping):
        state = value
    else:
        raise StructuredModelError(
            "q0 initialization requires a candidate-mlp-v1 model or state dict"
        )
    try:
        validate_v1_state_dict(state)
    except V1ModelError as exc:
        raise StructuredModelError("q0 state is incompatible with candidate-mlp-v1") from exc
    return state


class StructuredResidualMLP(nn.Module):
    """A trainable q0 base plus phase-specific zero-initialized residual paths."""

    model_version: Final[str] = MODEL_VERSION
    input_width: Final[int] = INPUT_WIDTH
    output_width: Final[int] = OUTPUT_WIDTH
    output_semantics: Final[str] = OUTPUT_SEMANTICS

    def __init__(
        self,
        q0_state_dict: CandidateMLP | Mapping[str, object],
        *,
        projection_seed: int = 0,
    ) -> None:
        super().__init__()
        state = _q0_state(q0_state_dict)
        projection_seed = _validate_seed(projection_seed)
        # ``fork_rng`` prevents construction from advancing process-global random state.
        with torch.random.fork_rng(devices=[]):
            self.base_hidden = nn.Linear(
                V1_PREFIX_WIDTH, BASE_HIDDEN_WIDTH, device="cpu", dtype=torch.float32
            )
            self.base_value = nn.Linear(
                BASE_HIDDEN_WIDTH, OUTPUT_WIDTH, device="cpu", dtype=torch.float32
            )
            self.play_history = nn.Linear(
                HISTORY_WIDTH, STRUCTURED_WIDTH, device="cpu", dtype=torch.float32
            )
            self.recruit_history = nn.Linear(
                HISTORY_WIDTH, STRUCTURED_WIDTH, device="cpu", dtype=torch.float32
            )
            self.play_consequence = nn.Linear(
                PLAY_CONSEQUENCE_WIDTH, STRUCTURED_WIDTH, device="cpu", dtype=torch.float32
            )
            self.recruit_consequence = nn.Linear(
                RECRUIT_CONSEQUENCE_WIDTH, STRUCTURED_WIDTH, device="cpu", dtype=torch.float32
            )
            self.play_residual = nn.Linear(
                RESIDUAL_INPUT_WIDTH, OUTPUT_WIDTH, device="cpu", dtype=torch.float32
            )
            self.recruit_residual = nn.Linear(
                RESIDUAL_INPUT_WIDTH, OUTPUT_WIDTH, device="cpu", dtype=torch.float32
            )
        self.initialize_from_q0(state, projection_seed=projection_seed)
        if count_parameters(self) != PARAMETER_COUNT:  # pragma: no cover - architecture guard
            raise RuntimeError(
                f"{MODEL_VERSION} has {count_parameters(self)} parameters, "
                f"expected {PARAMETER_COUNT}"
            )

    def initialize_from_q0(
        self, q0_state_dict: Mapping[str, object], *, projection_seed: int
    ) -> None:
        """Copy q0's exact trunk and deterministically reset only v2-specific layers."""
        state = _q0_state(q0_state_dict)
        projection_seed = _validate_seed(projection_seed)
        with torch.no_grad():
            self.base_hidden.weight.copy_(state["hidden.weight"])  # type: ignore[arg-type]
            self.base_hidden.bias.copy_(state["hidden.bias"])  # type: ignore[arg-type]
            self.base_value.weight.copy_(state["output.weight"])  # type: ignore[arg-type]
            self.base_value.bias.copy_(state["output.bias"])  # type: ignore[arg-type]
            generator = torch.Generator(device="cpu")
            generator.manual_seed(projection_seed)
            for layer in (
                self.play_history,
                self.recruit_history,
                self.play_consequence,
                self.recruit_consequence,
            ):
                nn.init.xavier_uniform_(layer.weight, generator=generator)
                nn.init.zeros_(layer.bias)
            for layer in (self.play_residual, self.recruit_residual):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, features: Tensor) -> Tensor:
        """Return one logit per candidate and reject malformed phase routing."""
        if features.shape[-1:] != (INPUT_WIDTH,):
            raise StructuredModelError(
                f"expected final feature dimension {INPUT_WIDTH}, got shape {tuple(features.shape)}"
            )
        if features.dtype != torch.float32:
            raise StructuredModelError("structured v2 features must use torch.float32")
        if features.device.type != "cpu":
            raise StructuredModelError("structured v2 inference is CPU-only")
        if not torch.isfinite(features).all().item():
            raise StructuredModelError("structured v2 features must be finite")
        phase = features[..., :2]
        valid_phase = (phase == 0.0) | (phase == 1.0)
        if not valid_phase.all().item() or not (phase.sum(dim=-1) == 1.0).all().item():
            raise StructuredModelError("structured v2 requires an exact one-hot play/recruit phase")
        base_hidden = torch.tanh(self.base_hidden(features[..., :V1_PREFIX_WIDTH]))
        base_logit = self.base_value(base_hidden).squeeze(-1)
        history = features[..., V1_PREFIX_WIDTH : V1_PREFIX_WIDTH + HISTORY_WIDTH]
        play_start = V1_PREFIX_WIDTH + HISTORY_WIDTH
        recruit_start = play_start + PLAY_CONSEQUENCE_WIDTH
        play_features = features[..., play_start:recruit_start]
        recruit_features = features[..., recruit_start:]
        play_residual = self.play_residual(
            torch.cat(
                (
                    base_hidden,
                    torch.tanh(self.play_history(history)),
                    torch.tanh(self.play_consequence(play_features)),
                ),
                dim=-1,
            )
        ).squeeze(-1)
        recruit_residual = self.recruit_residual(
            torch.cat(
                (
                    base_hidden,
                    torch.tanh(self.recruit_history(history)),
                    torch.tanh(self.recruit_consequence(recruit_features)),
                ),
                dim=-1,
            )
        ).squeeze(-1)
        result: Tensor = base_logit + torch.where(
            phase[..., 0] == 1.0, play_residual, recruit_residual
        )
        return result


# Names that make the model's candidate-value role explicit without creating a second architecture.
StructuredCandidateMLP = StructuredResidualMLP
StructuredValueModel = StructuredResidualMLP


def create_structured_model(
    q0_state_dict: CandidateMLP | Mapping[str, object], *, projection_seed: int = 0
) -> StructuredResidualMLP:
    """Construct the v2 model from a validated v1 q0 state tensor."""
    return StructuredResidualMLP(q0_state_dict, projection_seed=projection_seed)


def count_parameters(model: nn.Module) -> int:
    """Return the count of trainable scalar parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def model_spec() -> dict[str, object]:
    """Return normalized JSON-safe architecture metadata for immutable checkpoints."""
    return {
        "version": MODEL_VERSION,
        "input_width": INPUT_WIDTH,
        "v1_prefix_width": V1_PREFIX_WIDTH,
        "base_hidden_width": BASE_HIDDEN_WIDTH,
        "history_width": HISTORY_WIDTH,
        "play_consequence_width": PLAY_CONSEQUENCE_WIDTH,
        "recruit_consequence_width": RECRUIT_CONSEQUENCE_WIDTH,
        "structured_width": STRUCTURED_WIDTH,
        "residual_input_width": RESIDUAL_INPUT_WIDTH,
        "output_width": OUTPUT_WIDTH,
        "parameter_count": PARAMETER_COUNT,
        "output_semantics": OUTPUT_SEMANTICS,
        "activation": "tanh",
        "forward_output": "logit",
        "initialization": INITIALIZATION_VERSION,
    }


def validate_state_dict(state_dict: Mapping[str, object]) -> None:
    """Reject state mappings that are not exactly compatible with v2's architecture."""
    placeholder = CandidateMLP(seed=0).state_dict()
    expected = StructuredResidualMLP(placeholder, projection_seed=0).state_dict()
    if set(state_dict) != set(expected):
        missing = sorted(set(expected).difference(state_dict))
        unexpected = sorted(set(state_dict).difference(expected))
        raise StructuredModelError(
            f"incompatible structured model state keys; missing={missing}, unexpected={unexpected}"
        )
    for name, expected_tensor in expected.items():
        value = state_dict[name]
        if not isinstance(value, Tensor):
            raise StructuredModelError(f"model state value {name!r} is not a tensor")
        if value.layout != torch.strided or value.device.type != "cpu":
            raise StructuredModelError(f"model state tensor {name!r} must be a dense CPU tensor")
        if value.dtype != expected_tensor.dtype or value.shape != expected_tensor.shape:
            raise StructuredModelError(
                f"incompatible model state tensor {name!r}: "
                f"expected {tuple(expected_tensor.shape)} {expected_tensor.dtype}, "
                f"got {tuple(value.shape)} {value.dtype}"
            )
        if not torch.isfinite(value).all().item():
            raise StructuredModelError(f"model state tensor {name!r} contains non-finite values")


__all__ = [
    "BASE_HIDDEN_WIDTH",
    "HISTORY_WIDTH",
    "INITIALIZATION_VERSION",
    "INPUT_WIDTH",
    "MODEL_VERSION",
    "OUTPUT_SEMANTICS",
    "OUTPUT_WIDTH",
    "PARAMETER_COUNT",
    "PLAY_CONSEQUENCE_WIDTH",
    "RECRUIT_CONSEQUENCE_WIDTH",
    "RESIDUAL_INPUT_WIDTH",
    "STRUCTURED_WIDTH",
    "V1_PREFIX_WIDTH",
    "StructuredCandidateMLP",
    "StructuredModelError",
    "StructuredResidualMLP",
    "StructuredValueModel",
    "count_parameters",
    "create_structured_model",
    "model_spec",
    "validate_state_dict",
]

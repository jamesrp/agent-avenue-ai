"""The fixed candidate-action value network used by the first learned agent.

This module intentionally contains no game or observation types.  It consumes only already-safe,
materialized encoder vectors and returns logits suitable for ``BCEWithLogitsLoss``.
"""

from collections.abc import Mapping
from typing import Final

import torch
from torch import Tensor, nn

MODEL_VERSION: Final = "candidate-mlp-v1"
INPUT_WIDTH: Final = 87
HIDDEN_WIDTH: Final = 128
OUTPUT_WIDTH: Final = 1
PARAMETER_COUNT: Final = 11_393
OUTPUT_SEMANTICS: Final = "acting-player-eventual-win-logit"


class ModelError(ValueError):
    """Raised when model construction or input validation fails."""


def _validate_seed(seed: int) -> int:
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ModelError("initialization seed must be an integer in [0, 2**63)")
    return seed


class CandidateMLP(nn.Module):
    """An 87 -> 128 -> 1 action-conditioned value network returning logits.

    Parameters are initialized explicitly and solely from ``seed``.  Construction does not depend
    on, or advance, PyTorch's process-global random generator.
    """

    model_version: Final = MODEL_VERSION
    input_width: Final = INPUT_WIDTH
    hidden_width: Final = HIDDEN_WIDTH
    output_width: Final = OUTPUT_WIDTH
    output_semantics: Final = OUTPUT_SEMANTICS

    def __init__(self, *, seed: int = 0) -> None:
        super().__init__()
        seed = _validate_seed(seed)

        # nn.Linear initializes itself from the global generator.  fork_rng restores that state;
        # reset_parameters below then uses a private generator for the declared initialization.
        with torch.random.fork_rng(devices=[]):
            self.hidden = nn.Linear(INPUT_WIDTH, HIDDEN_WIDTH, device="cpu", dtype=torch.float32)
            self.output = nn.Linear(HIDDEN_WIDTH, OUTPUT_WIDTH, device="cpu", dtype=torch.float32)
        self.reset_parameters(seed)

        actual = count_parameters(self)
        if actual != PARAMETER_COUNT:  # pragma: no cover - protects future architecture edits
            raise RuntimeError(
                f"{MODEL_VERSION} has {actual} parameters, expected {PARAMETER_COUNT}"
            )

    def reset_parameters(self, seed: int) -> None:
        """Apply deterministic Xavier-uniform weights and zero biases from ``seed``."""
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_validate_seed(seed))
        nn.init.xavier_uniform_(self.hidden.weight, generator=generator)
        nn.init.zeros_(self.hidden.bias)
        nn.init.xavier_uniform_(self.output.weight, generator=generator)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: Tensor) -> Tensor:
        """Return one logit per candidate, preserving all leading dimensions."""
        if features.shape[-1:] != (INPUT_WIDTH,):
            raise ModelError(
                f"expected final feature dimension {INPUT_WIDTH}, got shape {tuple(features.shape)}"
            )
        if features.dtype != torch.float32:
            raise ModelError("candidate-mlp-v1 features must use torch.float32")
        if features.device.type != "cpu":
            raise ModelError("candidate-mlp-v1 inference is CPU-only")
        hidden: Tensor = self.hidden(features)
        output: Tensor = self.output(torch.tanh(hidden))
        return output.squeeze(-1)


# A descriptive alias for callers that prefer the semantic name.
CandidateValueModel = CandidateMLP


def create_model(seed: int = 0) -> CandidateMLP:
    """Construct the versioned baseline model on CPU."""
    return CandidateMLP(seed=seed)


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable scalar parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def model_spec() -> dict[str, object]:
    """Return normalized, JSON-safe architecture metadata."""
    return {
        "version": MODEL_VERSION,
        "input_width": INPUT_WIDTH,
        "hidden_width": HIDDEN_WIDTH,
        "output_width": OUTPUT_WIDTH,
        "parameter_count": PARAMETER_COUNT,
        "output_semantics": OUTPUT_SEMANTICS,
        "activation": "tanh",
        "forward_output": "logit",
        "initialization": "seeded-xavier-uniform-weights-zero-bias-v1",
    }


def validate_state_dict(state_dict: Mapping[str, object]) -> None:
    """Reject a state mapping that is not exactly compatible with ``candidate-mlp-v1``."""
    expected_model = CandidateMLP(seed=0)
    expected = expected_model.state_dict()
    if set(state_dict) != set(expected):
        missing = sorted(set(expected).difference(state_dict))
        unexpected = sorted(set(state_dict).difference(expected))
        raise ModelError(
            f"incompatible model state keys; missing={missing}, unexpected={unexpected}"
        )
    for name, expected_tensor in expected.items():
        value = state_dict[name]
        if not isinstance(value, Tensor):
            raise ModelError(f"model state value {name!r} is not a tensor")
        if value.layout != torch.strided:
            raise ModelError(f"model state tensor {name!r} must be strided")
        if value.device.type != "cpu":
            raise ModelError(f"model state tensor {name!r} must be on CPU")
        if value.dtype != expected_tensor.dtype or value.shape != expected_tensor.shape:
            raise ModelError(
                f"incompatible model state tensor {name!r}: "
                f"expected {tuple(expected_tensor.shape)} {expected_tensor.dtype}, "
                f"got {tuple(value.shape)} {value.dtype}"
            )
        if not torch.isfinite(value).all().item():
            raise ModelError(f"model state tensor {name!r} contains non-finite values")

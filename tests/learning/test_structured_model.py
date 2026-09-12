import pytest
import torch

from agent_avenue.learning.model import CandidateMLP
from agent_avenue.learning.structured_model import (
    PARAMETER_COUNT,
    StructuredModelError,
    count_parameters,
    create_structured_model,
)


def _features() -> torch.Tensor:
    features = torch.zeros((5, 519), dtype=torch.float32)
    features[:, :87] = torch.linspace(-0.5, 0.5, 5 * 87, dtype=torch.float32).reshape(5, 87)
    features[:, :2] = 0.0
    features[0::2, 0] = 1.0
    features[1::2, 1] = 1.0
    return features


def test_structured_model_has_exact_q0_embedding_parameter_count_and_seeded_digest() -> None:
    q0 = CandidateMLP(seed=31)
    first = create_structured_model(q0, projection_seed=99)
    second = create_structured_model(q0.state_dict(), projection_seed=99)
    features = _features()

    assert count_parameters(first) == PARAMETER_COUNT == 35_779
    assert torch.equal(first.base_hidden.weight, q0.hidden.weight)
    assert torch.equal(first.base_value.weight, q0.output.weight)
    assert torch.equal(first.play_residual.weight, torch.zeros_like(first.play_residual.weight))
    assert torch.equal(first.recruit_residual.bias, torch.zeros_like(first.recruit_residual.bias))
    assert torch.allclose(first(features), q0(features[:, :87]), atol=1e-7, rtol=0.0)
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            first.state_dict().values(), second.state_dict().values(), strict=True
        )
    )


def test_structured_model_rejects_invalid_phase_encodings() -> None:
    model = create_structured_model(CandidateMLP(seed=1), projection_seed=2)
    invalid = _features()
    invalid[0, :2] = 0.0
    with pytest.raises(StructuredModelError, match="one-hot"):
        model(invalid)

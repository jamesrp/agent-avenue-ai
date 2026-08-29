import torch

from agent_avenue.learning.model import (
    INPUT_WIDTH,
    PARAMETER_COUNT,
    CandidateMLP,
    count_parameters,
    model_spec,
)


def test_candidate_mlp_has_frozen_architecture_and_returns_logits() -> None:
    model = CandidateMLP(seed=17)
    assert count_parameters(model) == PARAMETER_COUNT == 11_393
    assert model_spec()["version"] == "candidate-mlp-v1"

    batch = torch.zeros((12, INPUT_WIDTH), dtype=torch.float32)
    logits = model(batch)
    assert logits.shape == (12,)
    assert torch.equal(logits, torch.zeros(12))


def test_seeded_xavier_is_reproducible_without_advancing_global_rng() -> None:
    torch.manual_seed(91)
    before = torch.random.get_rng_state().clone()
    first = CandidateMLP(seed=123)
    after = torch.random.get_rng_state()
    second = CandidateMLP(seed=123)
    different = CandidateMLP(seed=124)

    assert torch.equal(before, after)
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            first.state_dict().values(), second.state_dict().values(), strict=True
        )
    )
    assert not torch.equal(first.hidden.weight, different.hidden.weight)
    assert torch.count_nonzero(first.hidden.bias) == 0
    assert torch.count_nonzero(first.output.bias) == 0

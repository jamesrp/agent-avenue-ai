from __future__ import annotations

from agent_avenue.runners import (
    BootstrapMeanInterval,
    PromotionEvidence,
    assess_attempt,
    bootstrap_mean_interval,
    detect_plateau,
    evaluate_promotion,
)


def _interval(low: float, high: float, point: float = 0.5) -> BootstrapMeanInterval:
    return BootstrapMeanInterval((low, high), point, 200, 0.95, 1, "fixture", (499, 19_499))


def test_generic_bootstrap_uses_declared_familywise_indices() -> None:
    ordinary = bootstrap_mean_interval((0.0, 0.5, 1.0), master_seed=9, domain="ordinary")
    adjusted = bootstrap_mean_interval(
        (0.5, 0.5), master_seed=9, domain="adjusted", confidence_level=0.9875
    )
    assert ordinary.order_statistic_indices == (499, 19_499)
    assert adjusted.order_statistic_indices == (124, 19_874)
    assert adjusted.interval == (0.5, 0.5)


def test_promotion_requires_every_gate_and_one_borderline_confirmation() -> None:
    passing = PromotionEvidence(
        primary=_interval(0.51, 0.60, 0.55),
        player_one_win_rate=0.55,
        player_two_win_rate=0.55,
        versus_random=_interval(0.60, 0.75, 0.68),
        heuristic_difference=_interval(-0.04, 0.03, -0.01),
        compatibility_checks_passed=True,
    )
    assert evaluate_promotion(passing).status == "promote"

    borderline = PromotionEvidence(
        primary=_interval(0.49, 0.55),
        player_one_win_rate=0.55,
        player_two_win_rate=0.55,
        versus_random=_interval(0.60, 0.75),
        heuristic_difference=_interval(-0.04, 0.03),
        compatibility_checks_passed=True,
    )
    assert evaluate_promotion(borderline).status == "confirmation_required"
    failed_confirmation = PromotionEvidence(
        primary=borderline.primary,
        confirmation=_interval(0.50, 0.56),
        player_one_win_rate=0.55,
        player_two_win_rate=0.55,
        versus_random=_interval(0.60, 0.75),
        heuristic_difference=_interval(-0.04, 0.03),
        compatibility_checks_passed=True,
    )
    assert evaluate_promotion(failed_confirmation).status == "retain"


def test_promotion_reports_all_guardrail_failures() -> None:
    decision = evaluate_promotion(
        PromotionEvidence(
            primary=_interval(0.48, 0.55),
            confirmation=_interval(0.48, 0.55),
            player_one_win_rate=0.44,
            player_two_win_rate=0.43,
            versus_random=_interval(0.49, 0.60),
            heuristic_difference=_interval(-0.06, 0.01),
            compatibility_checks_passed=False,
        )
    )
    assert decision.status == "retain"
    assert set(decision.reasons) == {
        "primary_lower_bound_not_above_half",
        "player_one_guardrail",
        "player_two_guardrail",
        "random_guardrail",
        "heuristic_non_regression_guardrail",
        "compatibility_or_reproduction_guardrail",
    }


def test_plateau_requires_two_equivalent_attempts_against_same_incumbent() -> None:
    first = assess_attempt(
        attempt_id="g1-a1",
        incumbent_checkpoint="a" * 64,
        candidate_checkpoint="b" * 64,
        pair_scores=(0.5,) * 8,
        master_seed=1,
    )
    second = assess_attempt(
        attempt_id="g1-a2",
        incumbent_checkpoint="a" * 64,
        candidate_checkpoint="c" * 64,
        pair_scores=(0.5,) * 8,
        master_seed=2,
    )
    assert first.status == second.status == "practical_equivalence"
    assert detect_plateau((first,)).status == "continue"
    assert detect_plateau((first, second)).status == "plateaued"

    changed_incumbent = assess_attempt(
        attempt_id="g2-a1",
        incumbent_checkpoint="d" * 64,
        candidate_checkpoint="e" * 64,
        pair_scores=(0.5,) * 8,
        master_seed=3,
    )
    assert detect_plateau((second, changed_incumbent), budget_exhausted=True).status == (
        "budget_exhausted_inconclusive"
    )


def test_plateau_classifies_regression_and_inconclusive_without_stopping() -> None:
    regression = assess_attempt(
        attempt_id="regression",
        incumbent_checkpoint="a" * 64,
        candidate_checkpoint="b" * 64,
        pair_scores=(0.0,) * 8,
        master_seed=4,
    )
    inconclusive = assess_attempt(
        attempt_id="wide",
        incumbent_checkpoint="a" * 64,
        candidate_checkpoint="c" * 64,
        pair_scores=(0.0, 1.0) * 4,
        master_seed=5,
    )
    assert regression.status == "regression"
    assert inconclusive.status == "inconclusive"
    assert not detect_plateau((regression, inconclusive)).plateaued

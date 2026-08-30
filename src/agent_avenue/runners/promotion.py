"""Predeclared promotion gates and recipe-level plateau detection."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from agent_avenue.agents import RNG_ALGORITHM, SEED_DERIVATION, DeterministicRandom, derive_seed

BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_METHOD = "deterministic-block-percentile-bootstrap-v1"


@dataclass(frozen=True, slots=True)
class BootstrapMeanInterval:
    interval: tuple[float, float]
    point_estimate: float
    block_count: int
    confidence_level: float
    bootstrap_seed: int
    domain: str
    order_statistic_indices: tuple[int, int]

    def to_data(self) -> dict[str, object]:
        return {
            "method": BOOTSTRAP_METHOD,
            "statistic": "block_mean",
            "confidence_level": self.confidence_level,
            "point_estimate": self.point_estimate,
            "block_count": self.block_count,
            "resample_count": BOOTSTRAP_RESAMPLES,
            "order_statistic_indices": {
                "lower": self.order_statistic_indices[0],
                "upper": self.order_statistic_indices[1],
            },
            "interval": list(self.interval),
            "bootstrap_seed": self.bootstrap_seed,
            "bootstrap_rng_domain": self.domain,
            "rng_algorithm": RNG_ALGORITHM,
            "seed_derivation": SEED_DERIVATION,
        }


def _percentile_indices(confidence_level: float) -> tuple[int, int]:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between zero and one")
    tail = (1.0 - confidence_level) / 2.0
    lower = math.ceil(tail * BOOTSTRAP_RESAMPLES - 1e-12) - 1
    upper = math.ceil((1.0 - tail) * BOOTSTRAP_RESAMPLES - 1e-12) - 1
    return (max(0, lower), min(BOOTSTRAP_RESAMPLES - 1, upper))


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    master_seed: int,
    domain: str,
    confidence_level: float = 0.95,
) -> BootstrapMeanInterval:
    """Bootstrap arbitrary independent block scores with deterministic project RNG."""
    normalized = tuple(float(value) for value in values)
    if not normalized or any(not math.isfinite(value) for value in normalized):
        raise ValueError("bootstrap values must be a non-empty finite sequence")
    if not domain:
        raise ValueError("bootstrap domain must be non-empty")
    bootstrap_seed = derive_seed(master_seed, domain)
    rng = DeterministicRandom(bootstrap_seed, domain)
    count = len(normalized)
    means = [
        sum(normalized[rng.randbelow(count)] for _ in range(count)) / count
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    means.sort()
    indices = _percentile_indices(confidence_level)
    return BootstrapMeanInterval(
        interval=(means[indices[0]], means[indices[1]]),
        point_estimate=sum(normalized) / count,
        block_count=count,
        confidence_level=confidence_level,
        bootstrap_seed=bootstrap_seed,
        domain=domain,
        order_statistic_indices=indices,
    )


@dataclass(frozen=True, slots=True)
class PromotionPolicy:
    primary_threshold: float = 0.50
    minimum_seat_win_rate: float = 0.45
    random_threshold: float = 0.50
    heuristic_max_regression: float = 0.05
    confirmation_lower_minimum: float = 0.49


@dataclass(frozen=True, slots=True)
class PromotionEvidence:
    primary: BootstrapMeanInterval
    player_one_win_rate: float
    player_two_win_rate: float
    versus_random: BootstrapMeanInterval
    heuristic_difference: BootstrapMeanInterval
    compatibility_checks_passed: bool
    confirmation: BootstrapMeanInterval | None = None


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    status: Literal["promote", "retain", "confirmation_required"]
    reasons: tuple[str, ...]

    @property
    def promoted(self) -> bool:
        return self.status == "promote"

    def to_data(self) -> dict[str, object]:
        return {"status": self.status, "promoted": self.promoted, "reasons": list(self.reasons)}


def evaluate_promotion(
    evidence: PromotionEvidence, policy: PromotionPolicy | None = None
) -> PromotionDecision:
    """Apply the complete Milestone 6 gate without weakening it after seeing results."""
    policy = policy or PromotionPolicy()
    primary_lower = evidence.primary.interval[0]
    effective_primary = evidence.confirmation or evidence.primary
    if (
        evidence.confirmation is None
        and policy.confirmation_lower_minimum <= primary_lower <= policy.primary_threshold
    ):
        return PromotionDecision("confirmation_required", ("primary_borderline",))

    failures: list[str] = []
    if effective_primary.interval[0] <= policy.primary_threshold:
        failures.append("primary_lower_bound_not_above_half")
    if evidence.player_one_win_rate < policy.minimum_seat_win_rate:
        failures.append("player_one_guardrail")
    if evidence.player_two_win_rate < policy.minimum_seat_win_rate:
        failures.append("player_two_guardrail")
    if evidence.versus_random.interval[0] <= policy.random_threshold:
        failures.append("random_guardrail")
    if evidence.heuristic_difference.interval[0] <= -policy.heuristic_max_regression:
        failures.append("heuristic_non_regression_guardrail")
    if not evidence.compatibility_checks_passed:
        failures.append("compatibility_or_reproduction_guardrail")
    if failures:
        return PromotionDecision("retain", tuple(failures))
    return PromotionDecision("promote", ("all_predeclared_gates_passed",))


AttemptStatus = Literal["practical_equivalence", "regression", "inconclusive"]


@dataclass(frozen=True, slots=True)
class AttemptAssessment:
    attempt_id: str
    incumbent_checkpoint: str
    candidate_checkpoint: str
    interval: BootstrapMeanInterval
    minimum_practical_effect: float
    status: AttemptStatus

    def to_data(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "incumbent_checkpoint": self.incumbent_checkpoint,
            "candidate_checkpoint": self.candidate_checkpoint,
            "minimum_practical_effect": self.minimum_practical_effect,
            "status": self.status,
            "interval": self.interval.to_data(),
        }


def assess_attempt(
    *,
    attempt_id: str,
    incumbent_checkpoint: str,
    candidate_checkpoint: str,
    pair_scores: Sequence[float],
    master_seed: int,
    minimum_practical_effect: float = 0.05,
    confidence_level: float = 0.9875,
) -> AttemptAssessment:
    """Classify one failed proposal using a familywise interval over four planned looks."""
    if not attempt_id or incumbent_checkpoint == candidate_checkpoint:
        raise ValueError("attempt and distinct checkpoint identities are required")
    if not 0.0 < minimum_practical_effect < 0.5:
        raise ValueError("minimum_practical_effect must be between zero and one half")
    interval = bootstrap_mean_interval(
        pair_scores,
        master_seed=master_seed,
        domain=f"promotion:plateau:{attempt_id}:v1",
        confidence_level=confidence_level,
    )
    lower, upper = interval.interval
    lower_equivalence = 0.5 - minimum_practical_effect
    upper_equivalence = 0.5 + minimum_practical_effect
    if lower >= lower_equivalence and upper <= upper_equivalence:
        status: AttemptStatus = "practical_equivalence"
    elif upper < lower_equivalence:
        status = "regression"
    else:
        status = "inconclusive"
    return AttemptAssessment(
        attempt_id,
        incumbent_checkpoint,
        candidate_checkpoint,
        interval,
        minimum_practical_effect,
        status,
    )


@dataclass(frozen=True, slots=True)
class PlateauDecision:
    plateaued: bool
    status: Literal["plateaued", "continue", "budget_exhausted_inconclusive"]
    reason: str

    def to_data(self) -> dict[str, object]:
        return {"plateaued": self.plateaued, "status": self.status, "reason": self.reason}


def detect_plateau(
    attempts: Sequence[AttemptAssessment],
    *,
    consecutive_equivalent_attempts: int = 2,
    budget_exhausted: bool = False,
) -> PlateauDecision:
    """Detect no practical improvement for one unchanged recipe/incumbent, not global optimality."""
    if consecutive_equivalent_attempts < 2:
        raise ValueError("plateau detection requires at least two equivalent attempts")
    if len(attempts) >= consecutive_equivalent_attempts:
        tail = tuple(attempts[-consecutive_equivalent_attempts:])
        incumbent = tail[0].incumbent_checkpoint
        effect = tail[0].minimum_practical_effect
        if all(
            attempt.incumbent_checkpoint == incumbent
            and attempt.minimum_practical_effect == effect
            and attempt.status == "practical_equivalence"
            for attempt in tail
        ):
            return PlateauDecision(
                True,
                "plateaued",
                "no_detected_practical_improvement_for_unchanged_recipe_and_incumbent",
            )
    if budget_exhausted:
        return PlateauDecision(
            False,
            "budget_exhausted_inconclusive",
            "generation_budget_ended_without_plateau_evidence",
        )
    return PlateauDecision(False, "continue", "insufficient_equivalence_evidence")

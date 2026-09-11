#!/usr/bin/env python3
"""Run the approved step-1 terminal-offense confirmation or a clearly labeled smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_avenue.agents import (
    Agent,
    GreedyHeuristicAgent,
    LearnedCandidateScores,
    RandomAgent,
    TerminalOffenseAgent,
    TerminalSafetyAgent,
)
from agent_avenue.agents.ordering import semantic_action_key
from agent_avenue.engine import Action
from agent_avenue.observation.model import PlayerObservation
from agent_avenue.runners import (
    ANCHOR_OPPONENTS,
    CONTROL_ID,
    CYCLE_ID,
    DEFAULT_EXCLUDED_SETUP_ROOTS,
    DEFAULT_PAIRS_PER_FAMILY,
    FAMILY_IDS,
    FIELD_OPPONENTS,
    Q0_RNG_IDENTITY,
    ROOT_SEED,
    SETUP_HOLDOUT_SCOPE_VERSION,
    TREATMENT_ID,
    AgentSpec,
    ArenaConfig,
    ArenaReport,
    CandidateScorer,
    audit_terminal_confirmation,
    confirmation_cells,
    confirmation_statistics,
    family_master_seed,
    family_seed_domain,
    family_setup_seeds,
    run_resumable_arena,
    scan_prior_setup_blocks,
    setup_seed_fingerprint,
    validate_confirmation_corpora,
    validate_seed_families,
)
from agent_avenue.storage import code_fingerprint, inspect_source_identity, rules_fingerprint

Q_IDS = tuple(f"q{generation}-terminal-safety-v1" for generation in range(1, 5))
HISTORICAL_ID = "historical-q0"
HEURISTIC_ID = "greedy-public-v1"
RANDOM_ID = "random"
CLAIM_RUN_CUTOFF_SECONDS = 7 * 60 * 60 + 45 * 60
WHOLE_STEP_COMPUTE_BUDGET_SECONDS = 8 * 60 * 60
ALLOWED_RESUME_RETRIES = 1

CHECKPOINT_PATHS = {
    CONTROL_ID: Path("runs/terminal-safety-v1/q0-a1/checkpoint"),
    TREATMENT_ID: Path("runs/terminal-safety-v1/q0-a1/checkpoint"),
    Q_IDS[0]: Path("runs/terminal-safety-v1/q1-a1/candidate"),
    Q_IDS[1]: Path("runs/terminal-safety-v1/q2-a1/candidate"),
    Q_IDS[2]: Path("runs/terminal-safety-v1/q3-a1/candidate"),
    Q_IDS[3]: Path("runs/terminal-safety-v1/q4-a1/candidate"),
    HISTORICAL_ID: Path("runs/terminal-safety-v1/inputs/historical-q0"),
}


class ClaimRunCutoffError(RuntimeError):
    """Raised when the frozen claim-run cutoff prevents further execution."""


def _ensure_within_claim_cutoff(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise ClaimRunCutoffError(
            "claim-run cutoff reached; no result was emitted and only one resume/retry is allowed"
        )


def _begin_execution_attempt(output: Path, plan_fingerprint: object) -> Path:
    """Record one initial execution or the sole permitted resume/retry."""
    if not isinstance(plan_fingerprint, str):
        raise RuntimeError("plan fingerprint is malformed")
    path = output / "execution-state.json"
    if path.exists():
        state = _read(path)
        if state.get("plan_fingerprint") != plan_fingerprint:
            raise RuntimeError("execution state belongs to another immutable plan")
        attempts = state.get("attempt_count")
        if type(attempts) is not int or attempts < 1:
            raise RuntimeError("execution state attempt count is malformed")
        if attempts > ALLOWED_RESUME_RETRIES:
            raise RuntimeError("the one allowed confirmation resume/retry has already been used")
        attempt_count = attempts + 1
    else:
        attempt_count = 1
    data = {
        "version": "terminal-offense-confirmation-execution-state-v1",
        "plan_fingerprint": plan_fingerprint,
        "attempt_count": attempt_count,
        "allowed_resume_retries": ALLOWED_RESUME_RETRIES,
        "status": "running",
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)
    return path


def _complete_execution_attempt(path: Path) -> None:
    data = _read(path)
    data["status"] = "completed"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class PolicyBundle:
    specs: dict[str, AgentSpec]
    scorers: dict[str, CandidateScorer]
    checkpoint_metadata: dict[str, dict[str, object]]
    mode: str


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _write_immutable(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read(path) != value:
            raise RuntimeError(f"immutable artifact differs from recomputation: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def _agent_config(agent: Agent) -> dict[str, object]:
    config = getattr(agent, "config", None)
    to_data = getattr(config, "to_data", None)
    if callable(to_data):
        value = to_data()
    else:
        config_to_data = getattr(agent, "config_to_data", None)
        if not callable(config_to_data):
            raise RuntimeError("agent has no normalized configuration")
        value = config_to_data()
    if not isinstance(value, dict):
        raise RuntimeError("agent configuration is not a JSON object")
    return value


def _toy_scorer(
    observation: PlayerObservation, actions: tuple[Action, ...]
) -> LearnedCandidateScores:
    if observation.legal_actions != actions:
        raise ValueError("toy scorer requires one consistent candidate set")
    semantic = tuple(sorted(actions, key=semantic_action_key))
    return LearnedCandidateScores.from_logits(semantic, tuple(0.0 for _ in semantic))


def _toy_bundle() -> PolicyBundle:
    specs: dict[str, AgentSpec] = {}
    control = TerminalSafetyAgent(RandomAgent())
    treatment = TerminalOffenseAgent(TerminalSafetyAgent(RandomAgent()))
    specs[CONTROL_ID] = AgentSpec(
        CONTROL_ID,
        _agent_config(control),
        lambda: TerminalSafetyAgent(RandomAgent()),
        rng_identity=Q0_RNG_IDENTITY,
    )
    specs[TREATMENT_ID] = AgentSpec(
        TREATMENT_ID,
        _agent_config(treatment),
        lambda: TerminalOffenseAgent(TerminalSafetyAgent(RandomAgent())),
        rng_identity=Q0_RNG_IDENTITY,
    )
    for opponent in (*Q_IDS, HISTORICAL_ID, RANDOM_ID):
        specs[opponent] = AgentSpec(
            opponent,
            _agent_config(RandomAgent()),
            RandomAgent,
            rng_identity=opponent,
        )
    heuristic = GreedyHeuristicAgent()
    specs[HEURISTIC_ID] = AgentSpec(
        HEURISTIC_ID,
        _agent_config(heuristic),
        GreedyHeuristicAgent,
        rng_identity=HEURISTIC_ID,
    )
    return PolicyBundle(
        specs,
        {CONTROL_ID: _toy_scorer, TREATMENT_ID: _toy_scorer},
        {},
        "toy-random-smoke",
    )


def _learned_bundle() -> PolicyBundle:
    # Optional RL imports stay behind the explicit learned-policy execution path.
    from agent_avenue.agents.learned import LearnedValueAgent
    from agent_avenue.learning import LoadedCheckpoint, load_checkpoint

    checkpoints = {
        agent_id: load_checkpoint(path)
        for agent_id, path in CHECKPOINT_PATHS.items()
        if agent_id not in {TREATMENT_ID}
    }
    q0 = checkpoints[CONTROL_ID]
    specs: dict[str, AgentSpec] = {}

    def control_factory(checkpoint: LoadedCheckpoint = q0) -> Agent:
        return TerminalSafetyAgent(LearnedValueAgent.from_checkpoint(checkpoint))

    def treatment_factory(checkpoint: LoadedCheckpoint = q0) -> Agent:
        return TerminalOffenseAgent(
            TerminalSafetyAgent(LearnedValueAgent.from_checkpoint(checkpoint))
        )

    specs[CONTROL_ID] = AgentSpec(
        CONTROL_ID,
        _agent_config(control_factory()),
        control_factory,
        rng_identity=Q0_RNG_IDENTITY,
    )
    specs[TREATMENT_ID] = AgentSpec(
        TREATMENT_ID,
        _agent_config(treatment_factory()),
        treatment_factory,
        rng_identity=Q0_RNG_IDENTITY,
    )
    for agent_id in (*Q_IDS, HISTORICAL_ID):
        checkpoint = checkpoints[agent_id]
        shielded = agent_id != HISTORICAL_ID

        def factory(
            checkpoint: LoadedCheckpoint = checkpoint,
            shielded: bool = shielded,
        ) -> Agent:
            learned: Agent = LearnedValueAgent.from_checkpoint(checkpoint)
            return TerminalSafetyAgent(learned) if shielded else learned

        specs[agent_id] = AgentSpec(
            agent_id,
            _agent_config(factory()),
            factory,
            rng_identity=agent_id,
        )
    heuristic = GreedyHeuristicAgent()
    random = RandomAgent()
    specs[HEURISTIC_ID] = AgentSpec(
        HEURISTIC_ID,
        _agent_config(heuristic),
        GreedyHeuristicAgent,
        rng_identity=HEURISTIC_ID,
    )
    specs[RANDOM_ID] = AgentSpec(
        RANDOM_ID,
        _agent_config(random),
        RandomAgent,
        rng_identity=RANDOM_ID,
    )
    scoring_agent = LearnedValueAgent.from_checkpoint(q0)

    def scorer(
        observation: PlayerObservation, actions: tuple[Action, ...]
    ) -> LearnedCandidateScores:
        return scoring_agent.score_candidates(observation, observation.decision, actions)

    metadata: dict[str, dict[str, object]] = {}
    for agent_id, checkpoint in checkpoints.items():
        path = CHECKPOINT_PATHS[agent_id]
        metadata[agent_id] = {
            "path": str(path),
            "checkpoint_fingerprint": checkpoint.checkpoint_fingerprint,
            "tensor_digest": checkpoint.manifest["tensor_digest"],
        }
    metadata[TREATMENT_ID] = dict(metadata[CONTROL_ID])
    return PolicyBundle(
        specs,
        {CONTROL_ID: scorer, TREATMENT_ID: scorer},
        metadata,
        "learned-checkpoints",
    )


def build_plan(
    output: Path,
    bundle: PolicyBundle,
    *,
    pairs_per_family: int,
    setup_holdout: dict[str, object],
) -> dict[str, object]:
    source = inspect_source_identity()
    default_execution = (
        pairs_per_family == DEFAULT_PAIRS_PER_FAMILY and bundle.mode == "learned-checkpoints"
    )
    if default_execution and not source.tracked_tree_clean:
        raise RuntimeError("the default claim design requires a tracked-clean source tree")
    seed_validation = validate_seed_families(pairs_per_family)
    policies = {
        agent_id: {
            "config": dict(spec.config),
            "rng_identity": spec.rng_identity,
            "checkpoint": bundle.checkpoint_metadata.get(agent_id),
        }
        for agent_id, spec in sorted(bundle.specs.items())
    }
    cells = []
    for cell in confirmation_cells():
        cells.append(
            {
                **cell.to_data(),
                "family_seed_domain": family_seed_domain(cell.family),
                "family_master_seed": family_master_seed(cell.family),
                "setup_seed_fingerprint": setup_seed_fingerprint(
                    family_setup_seeds(cell.family, pairs_per_family)
                ),
                "pairs": pairs_per_family,
                "games": pairs_per_family * 2,
            }
        )
    payload: dict[str, object] = {
        "version": "terminal-offense-confirmation-plan-v1",
        "cycle_id": CYCLE_ID,
        "source": {
            **source.to_data(),
            "rules_fingerprint": rules_fingerprint(),
            "code_fingerprint": code_fingerprint(),
        },
        "frozen_default_design": {
            "root_seed": ROOT_SEED,
            "seed_families": list(FAMILY_IDS),
            "pairs_per_family": DEFAULT_PAIRS_PER_FAMILY,
            "cells_per_family": 15,
            "total_cells": 30,
            "total_games": 60_000,
            "control_id": CONTROL_ID,
            "treatment_id": TREATMENT_ID,
            "shared_q0_rng_identity": Q0_RNG_IDENTITY,
            "field_opponents": list(FIELD_OPPONENTS),
            "anchor_opponents": list(ANCHOR_OPPONENTS),
            "bootstrap_resamples": 20_000,
        },
        "execution": {
            "mode": bundle.mode,
            "evidence_class": (
                "claim-eligible-default" if default_execution else "smoke-only-nondefault"
            ),
            "uses_frozen_default_design": default_execution,
            "pairs_per_family": pairs_per_family,
            "total_cells": len(cells),
            "total_games": len(cells) * pairs_per_family * 2,
            "output": str(output.resolve()),
        },
        "execution_budget": {
            "claim_run_cutoff_seconds": CLAIM_RUN_CUTOFF_SECONDS,
            "claim_run_cutoff": "7h45m",
            "whole_step_compute_budget_seconds": WHOLE_STEP_COMPUTE_BUDGET_SECONDS,
            "whole_step_compute_budget": "8h",
            "incomplete_execution": "no result is emitted before every cell and audit completes",
            "allowed_resume_retries": ALLOWED_RESUME_RETRIES,
        },
        "root_seed": ROOT_SEED,
        "seed_validation": seed_validation,
        "setup_holdout_scope": {
            "version": SETUP_HOLDOUT_SCOPE_VERSION,
            "declared_roots": [str(path) for path in DEFAULT_EXCLUDED_SETUP_ROOTS],
            "recursive": True,
            "excludes_current_output_subtree": True,
        },
        "setup_holdout": {
            "artifact_fingerprint": setup_holdout["artifact_fingerprint"],
            "status": setup_holdout["status"],
            "overlap_count": setup_holdout["overlap_count"],
            "current_setup_fingerprint": setup_holdout["current_setup_fingerprint"],
            "prior_setup_fingerprint": setup_holdout["prior_setup_fingerprint"],
        },
        "direct_treatment_vs_control": {
            "interpretation": (
                "descriptive only: both arms intentionally share one RNG identity within "
                "a direct game"
            ),
            "causal_matched_evidence": False,
            "used_for_structural_adoption": False,
            "used_for_practical_lift_claim": False,
        },
        "policies": policies,
        "cells": cells,
        "tactical_oracles": {
            "production": "public-guaranteed-current-turn-win-v1",
            "independent": "independent-public-forced-win-oracle-v1",
            "required_agreement": "every replay decision",
            "play_transition_cross_checks": (
                "both recruit assignments for every selected or independently guaranteed play"
            ),
        },
        "adoption": {
            "structural_criteria": [
                "frozen and disjoint seed families",
                "zero overlap with declared prior setup blocks",
                "record/replay/schedule integrity",
                "independent-production oracle agreement",
                "zero treatment guaranteed-win misses and false wins",
                "zero q0 avoidable immediate losses",
                "all learned selections belong to exact maximum logits",
                "zero pre-endpoint control/treatment prefix mismatches and treatment failures",
            ],
            "practical_lift_claim": ("anchor-macro 95% lower bound > +0.25 percentage points"),
            "failed_practical_lift_blocks_structural_adoption": False,
        },
        "plan_fingerprint": "",
    }
    payload["plan_fingerprint"] = _fingerprint(
        {key: value for key, value in payload.items() if key != "plan_fingerprint"}
    )
    return payload


def _report_semantics(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in report.items()
        if key not in {"elapsed_seconds", "games_per_second"}
    }


def _run_cells(
    output: Path,
    plan: dict[str, Any],
    bundle: PolicyBundle,
    *,
    deadline: float,
) -> tuple[dict[str, ArenaReport], dict[str, Path], dict[str, str]]:
    reports: dict[str, ArenaReport] = {}
    directories: dict[str, Path] = {}
    artifact_fingerprints: dict[str, str] = {}
    cells = confirmation_cells()
    for index, cell in enumerate(cells, start=1):
        _ensure_within_claim_cutoff(deadline)
        print(f"[{index:02d}/{len(cells)}] {cell.key}", flush=True)
        config = ArenaConfig(
            run_id=cell.run_id,
            agent_a=bundle.specs[cell.agent_a_id],
            agent_b=bundle.specs[cell.agent_b_id],
            pair_count=int(plan["execution"]["pairs_per_family"]),
            master_seed=family_master_seed(cell.family),
        )
        directory = output / "arena-records" / cell.family / cell.cell_id
        retained = run_resumable_arena(
            directory,
            config,
            generation=None,
            corpus_configuration={
                "version": "terminal-offense-confirmation-cell-v1",
                "cycle_id": CYCLE_ID,
                "plan_fingerprint": plan["plan_fingerprint"],
                "cell": cell.to_data(),
                "family_seed_domain": family_seed_domain(cell.family),
                "common_setup_block_within_family": True,
            },
        )
        report_data = retained.report.to_data()
        artifact: dict[str, object] = {
            "version": "terminal-offense-confirmation-cell-artifact-v1",
            "plan_fingerprint": plan["plan_fingerprint"],
            "cell": cell.to_data(),
            "records_corpus_fingerprint": retained.records_manifest.corpus_fingerprint,
            "report": report_data,
            "artifact_fingerprint": "",
        }
        artifact["artifact_fingerprint"] = _fingerprint(
            {key: value for key, value in artifact.items() if key != "artifact_fingerprint"}
        )
        artifact_path = output / "arenas" / cell.family / f"{cell.cell_id}.json"
        if artifact_path.exists():
            existing = _read(artifact_path)
            existing_payload = {
                key: value for key, value in existing.items() if key != "artifact_fingerprint"
            }
            if _fingerprint(existing_payload) != existing.get("artifact_fingerprint"):
                raise RuntimeError(f"cell artifact fingerprint mismatch: {cell.key}")
            if (
                existing.get("plan_fingerprint") != plan["plan_fingerprint"]
                or existing.get("records_corpus_fingerprint")
                != retained.records_manifest.corpus_fingerprint
                or _report_semantics(existing["report"]) != _report_semantics(report_data)
            ):
                raise RuntimeError(f"immutable cell artifact changed: {cell.key}")
            artifact_fingerprints[cell.key] = str(existing["artifact_fingerprint"])
        else:
            _write_immutable(artifact_path, artifact)
            artifact_fingerprints[cell.key] = str(artifact["artifact_fingerprint"])
        reports[cell.key] = retained.report
        directories[cell.key] = directory
    return reports, directories, artifact_fingerprints


def _criterion(name: str, passed: bool, evidence: object) -> dict[str, object]:
    return {"criterion": name, "passed": passed, "evidence": evidence}


def _result(
    plan: dict[str, Any],
    *,
    statistics: dict[str, Any],
    audit: dict[str, Any],
    integrity: dict[str, Any],
    setup_holdout: dict[str, Any],
    cell_artifacts: dict[str, str],
) -> dict[str, object]:
    tactics = audit["guaranteed_win_and_safety"]
    ties = audit["exact_max_logit_ties"]["by_arm"]
    agreement = audit["independent_production_oracle_agreement"]["counts"]
    prefixes = audit["matched_action_prefixes"]["counts"]
    control_counts = tactics["control"]["counts"]
    treatment_counts = tactics["treatment"]["counts"]
    default_design = bool(plan["execution"]["uses_frozen_default_design"])
    criteria = [
        _criterion(
            "frozen and disjoint seed families",
            default_design
            and plan["seed_validation"]["status"] == "passed"
            and integrity["family_blocks_disjoint"],
            plan["seed_validation"],
        ),
        _criterion(
            "zero overlap with declared prior setup blocks",
            setup_holdout["status"] == "passed" and setup_holdout["overlap_count"] == 0,
            {
                "artifact_fingerprint": setup_holdout["artifact_fingerprint"],
                "scanned_corpus_count": setup_holdout["scanned_corpus_count"],
                "scanned_record_count": setup_holdout["scanned_record_count"],
                "overlap_count": setup_holdout["overlap_count"],
            },
        ),
        _criterion(
            "record/replay/schedule integrity",
            integrity["status"] == "passed",
            {
                "record_count": integrity["record_count"],
                "decision_count": integrity["decision_count"],
            },
        ),
        _criterion(
            "independent-production oracle agreement",
            agreement["disagreement_decisions"] == 0,
            agreement,
        ),
        _criterion(
            "zero treatment guaranteed-win misses and false wins",
            treatment_counts["guaranteed_win_misses"] == 0
            and treatment_counts["false_guaranteed_wins"] == 0,
            treatment_counts,
        ),
        _criterion(
            "zero q0 avoidable immediate losses",
            control_counts["q0_avoidable_immediate_loss_violations"] == 0
            and treatment_counts["q0_avoidable_immediate_loss_violations"] == 0,
            {
                "control": control_counts["q0_avoidable_immediate_loss_violations"],
                "treatment": treatment_counts["q0_avoidable_immediate_loss_violations"],
            },
        ),
        _criterion(
            "all selected actions belong to exact maximum logits",
            ties["control"]["counts"]["selected_outside_exact_maxima"] == 0
            and ties["treatment"]["counts"]["selected_outside_exact_maxima"] == 0,
            {
                "control": ties["control"]["counts"],
                "treatment": ties["treatment"]["counts"],
            },
        ),
        _criterion(
            "zero pre-endpoint prefix mismatches and treatment failures",
            prefixes["metadata_mismatches"] == 0
            and prefixes["pre_endpoint_prefix_mismatches"] == 0
            and prefixes["treatment_endpoint_failures"] == 0,
            prefixes,
        ),
    ]
    structural_passed = all(bool(row["passed"]) for row in criteria)
    result: dict[str, object] = {
        "version": "terminal-offense-confirmation-result-v1",
        "cycle_id": CYCLE_ID,
        "status": "completed",
        "evidence_class": plan["execution"]["evidence_class"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "source": plan["source"],
        "games": {
            "total": integrity["record_count"],
            "decisions": integrity["decision_count"],
            "cells": integrity["cell_count"],
            "pairs_per_family": plan["execution"]["pairs_per_family"],
        },
        "statistics": statistics,
        "direct_treatment_vs_control": statistics["direct_treatment_vs_control_interpretation"],
        "replay_audit": audit,
        "integrity": integrity,
        "setup_block_holdout": setup_holdout,
        "cell_artifact_fingerprints": dict(sorted(cell_artifacts.items())),
        "structural_adoption": {
            "eligible_default_design": default_design,
            "passed": structural_passed,
            "criteria": criteria,
            "practical_lift_is_separate": True,
        },
        "practical_lift_claim": statistics["practical_lift_claim"],
        "result_fingerprint": "",
    }
    result["result_fingerprint"] = _fingerprint(
        {key: value for key, value in result.items() if key != "result_fingerprint"}
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/m7-terminal-offense-confirm-v1"),
    )
    parser.add_argument(
        "--pairs-per-family",
        type=int,
        default=DEFAULT_PAIRS_PER_FAMILY,
        help=(
            "paired setups in each of two seed families "
            f"(frozen claim default: {DEFAULT_PAIRS_PER_FAMILY}; 60,000 games total)"
        ),
    )
    parser.add_argument(
        "--toy-agents",
        action="store_true",
        help="run a non-claim smoke without loading PyTorch checkpoints",
    )
    args = parser.parse_args()
    if args.pairs_per_family < 1:
        parser.error("--pairs-per-family must be positive")

    bundle = _toy_bundle() if args.toy_agents else _learned_bundle()
    setup_holdout = scan_prior_setup_blocks(
        pair_count=args.pairs_per_family,
        excluded_roots=DEFAULT_EXCLUDED_SETUP_ROOTS,
        current_output=args.output,
    )
    plan = build_plan(
        args.output,
        bundle,
        pairs_per_family=args.pairs_per_family,
        setup_holdout=setup_holdout,
    )
    plan_path = args.output / "plan.json"
    _write_immutable(plan_path, plan)
    _write_immutable(args.output / "analysis" / "setup-block-holdout.json", setup_holdout)
    if setup_holdout["overlap_count"] != 0:
        raise RuntimeError(
            "confirmation setup block overlaps declared prior corpora; no games were started"
        )

    result_path = args.output / "result.json"
    if result_path.exists():
        result = _read(result_path)
        if result.get("plan_fingerprint") != plan["plan_fingerprint"]:
            raise RuntimeError("existing result belongs to a different immutable plan")
        payload = {key: value for key, value in result.items() if key != "result_fingerprint"}
        if _fingerprint(payload) != result.get("result_fingerprint"):
            raise RuntimeError("existing result fingerprint mismatch")
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0

    execution_state = _begin_execution_attempt(args.output, plan["plan_fingerprint"])
    deadline = time.monotonic() + CLAIM_RUN_CUTOFF_SECONDS
    reports, directories, cell_artifacts = _run_cells(
        args.output,
        plan,
        bundle,
        deadline=deadline,
    )
    _ensure_within_claim_cutoff(deadline)
    rng_identities = {agent_id: spec.rng_identity for agent_id, spec in bundle.specs.items()}
    integrity = validate_confirmation_corpora(
        directories,
        pair_count=args.pairs_per_family,
        rng_identities=rng_identities,
        verify_replays=False,
    )
    _write_immutable(args.output / "analysis" / "integrity.json", integrity)
    statistics = confirmation_statistics(reports)
    _write_immutable(args.output / "analysis" / "statistics.json", statistics)
    audit = audit_terminal_confirmation(
        directories,
        candidate_scorers=bundle.scorers,
        source_label=str(plan["plan_fingerprint"]),
        verify_replays=False,
    )
    _write_immutable(args.output / "analysis" / "replay-audit.json", audit)
    _ensure_within_claim_cutoff(deadline)
    result = _result(
        plan,
        statistics=statistics,
        audit=audit,
        integrity=integrity,
        setup_holdout=setup_holdout,
        cell_artifacts=cell_artifacts,
    )
    _write_immutable(result_path, result)
    _complete_execution_attempt(execution_state)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - command boundary
        print(f"terminal-offense confirmation failed: {exc}", file=sys.stderr)
        raise

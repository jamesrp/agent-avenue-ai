#!/usr/bin/env python3
"""Deterministically recompute retained step-1 terminal-offense confirmation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from agent_avenue.agents import LearnedCandidateScores, RandomAgent
from agent_avenue.agents.ordering import semantic_action_key
from agent_avenue.engine import Action
from agent_avenue.observation.model import PlayerObservation
from agent_avenue.runners import (
    CONTROL_ID,
    CYCLE_ID,
    DEFAULT_PAIRS_PER_FAMILY,
    Q0_RNG_IDENTITY,
    ROOT_SEED,
    TREATMENT_ID,
    AgentSpec,
    ArenaConfig,
    CandidateScorer,
    arena_report_from_records,
    audit_terminal_confirmation,
    confirmation_cells,
    confirmation_statistics,
    family_master_seed,
    validate_confirmation_corpora,
    validate_seed_families,
)
from agent_avenue.storage import load_corpus


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


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def _toy_scorer(
    observation: PlayerObservation, actions: tuple[Action, ...]
) -> LearnedCandidateScores:
    semantic = tuple(sorted(actions, key=semantic_action_key))
    return LearnedCandidateScores.from_logits(semantic, tuple(0.0 for _ in semantic))


def _scorers(plan: dict[str, Any]) -> dict[str, CandidateScorer]:
    mode = plan["execution"]["mode"]
    if mode == "toy-random-smoke":
        return {CONTROL_ID: _toy_scorer, TREATMENT_ID: _toy_scorer}
    if mode != "learned-checkpoints":
        raise RuntimeError("unsupported confirmation policy mode")
    from agent_avenue.agents.learned import LearnedValueAgent
    from agent_avenue.learning import load_checkpoint

    checkpoint_data = plan["policies"][CONTROL_ID]["checkpoint"]
    checkpoint = load_checkpoint(Path(str(checkpoint_data["path"])))
    if checkpoint.checkpoint_fingerprint != checkpoint_data["checkpoint_fingerprint"]:
        raise RuntimeError("q0 checkpoint fingerprint differs from frozen plan")
    agent = LearnedValueAgent.from_checkpoint(checkpoint)

    def score(
        observation: PlayerObservation, actions: tuple[Action, ...]
    ) -> LearnedCandidateScores:
        return agent.score_candidates(observation, observation.decision, actions)

    return {CONTROL_ID: score, TREATMENT_ID: score}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root
    output = args.output or root / "validation.json"
    plan = _read(root / "plan.json")
    result = _read(root / "result.json")
    plan_payload = {key: value for key, value in plan.items() if key != "plan_fingerprint"}
    if _fingerprint(plan_payload) != plan["plan_fingerprint"]:
        raise RuntimeError("plan fingerprint mismatch")
    result_payload = {key: value for key, value in result.items() if key != "result_fingerprint"}
    if _fingerprint(result_payload) != result["result_fingerprint"]:
        raise RuntimeError("result fingerprint mismatch")
    if plan["cycle_id"] != CYCLE_ID or plan["root_seed"] != ROOT_SEED:
        raise RuntimeError("plan cycle or root seed differs from frozen design")
    if plan["frozen_default_design"]["pairs_per_family"] != DEFAULT_PAIRS_PER_FAMILY:
        raise RuntimeError("plan default pair count differs from frozen design")

    policy_data = plan["policies"]
    specs: dict[str, AgentSpec] = {}
    rng_identities: dict[str, str] = {}
    for agent_id, raw in policy_data.items():
        config = raw["config"]
        identity = str(raw["rng_identity"])
        specs[agent_id] = AgentSpec(agent_id, config, RandomAgent, rng_identity=identity)
        rng_identities[agent_id] = identity
    if (
        rng_identities[CONTROL_ID] != Q0_RNG_IDENTITY
        or rng_identities[TREATMENT_ID] != Q0_RNG_IDENTITY
    ):
        raise RuntimeError("control/treatment RNG identity is not the frozen shared identity")

    pair_count = int(plan["execution"]["pairs_per_family"])
    reports = {}
    directories = {}
    cell_artifacts: dict[str, str] = {}
    for cell in confirmation_cells():
        directory = root / "arena-records" / cell.family / cell.cell_id
        artifact_path = root / "arenas" / cell.family / f"{cell.cell_id}.json"
        artifact = _read(artifact_path)
        artifact_payload = {
            key: value for key, value in artifact.items() if key != "artifact_fingerprint"
        }
        if _fingerprint(artifact_payload) != artifact["artifact_fingerprint"]:
            raise RuntimeError(f"cell artifact fingerprint mismatch: {cell.key}")
        manifest, records = load_corpus(directory)
        if manifest.corpus_fingerprint != artifact["records_corpus_fingerprint"]:
            raise RuntimeError(f"cell corpus fingerprint mismatch: {cell.key}")
        retained_report = artifact["report"]
        recomputed = arena_report_from_records(
            ArenaConfig(
                run_id=cell.run_id,
                agent_a=specs[cell.agent_a_id],
                agent_b=specs[cell.agent_b_id],
                pair_count=pair_count,
                master_seed=family_master_seed(cell.family),
            ),
            records,
            elapsed_seconds=float(retained_report["elapsed_seconds"]),
        )
        if recomputed.to_data() != retained_report:
            raise RuntimeError(f"cell arena report mismatch: {cell.key}")
        reports[cell.key] = recomputed
        directories[cell.key] = directory
        cell_artifacts[cell.key] = str(artifact["artifact_fingerprint"])

    integrity = validate_confirmation_corpora(
        directories,
        pair_count=pair_count,
        rng_identities=rng_identities,
        verify_replays=False,
    )
    if integrity != _read(root / "analysis" / "integrity.json"):
        raise RuntimeError("integrity analysis mismatch")
    statistics = confirmation_statistics(reports)
    if statistics != _read(root / "analysis" / "statistics.json"):
        raise RuntimeError("statistical recomputation mismatch")
    audit = audit_terminal_confirmation(
        directories,
        candidate_scorers=_scorers(plan),
        source_label=str(plan["plan_fingerprint"]),
        verify_replays=False,
    )
    if audit != _read(root / "analysis" / "replay-audit.json"):
        raise RuntimeError("replay-derived audit mismatch")
    if result["statistics"] != statistics or result["replay_audit"] != audit:
        raise RuntimeError("result does not embed recomputed analysis")
    if result["integrity"] != integrity:
        raise RuntimeError("result does not embed recomputed integrity")
    if result["practical_lift_claim"] != statistics["practical_lift_claim"]:
        raise RuntimeError("result practical-lift claim differs from recomputation")
    tactics = audit["guaranteed_win_and_safety"]
    ties = audit["exact_max_logit_ties"]["by_arm"]
    agreement = audit["independent_production_oracle_agreement"]["counts"]
    prefixes = audit["matched_action_prefixes"]["counts"]
    control_counts = tactics["control"]["counts"]
    treatment_counts = tactics["treatment"]["counts"]
    expected_structural_passes = [
        bool(plan["execution"]["uses_frozen_default_design"])
        and plan["seed_validation"]["status"] == "passed"
        and integrity["family_blocks_disjoint"],
        integrity["status"] == "passed",
        agreement["disagreement_decisions"] == 0,
        treatment_counts["guaranteed_win_misses"] == 0
        and treatment_counts["false_guaranteed_wins"] == 0,
        control_counts["q0_avoidable_immediate_loss_violations"] == 0
        and treatment_counts["q0_avoidable_immediate_loss_violations"] == 0,
        ties["control"]["counts"]["selected_outside_exact_maxima"] == 0
        and ties["treatment"]["counts"]["selected_outside_exact_maxima"] == 0,
        prefixes["metadata_mismatches"] == 0
        and prefixes["pre_intervention_prefix_mismatches"] == 0
        and prefixes["treatment_intervention_conversion_failures"] == 0,
    ]
    retained_structural = result["structural_adoption"]
    retained_passes = [bool(row["passed"]) for row in retained_structural["criteria"]]
    if (
        retained_passes != expected_structural_passes
        or bool(retained_structural["passed"]) != all(expected_structural_passes)
        or bool(retained_structural["eligible_default_design"])
        != bool(plan["execution"]["uses_frozen_default_design"])
    ):
        raise RuntimeError("structural adoption decision differs from recomputation")
    if result["cell_artifact_fingerprints"] != dict(sorted(cell_artifacts.items())):
        raise RuntimeError("result cell artifact fingerprints mismatch")
    if plan["seed_validation"] != validate_seed_families(pair_count):
        raise RuntimeError("plan seed validation mismatch")

    validation: dict[str, object] = {
        "version": "terminal-offense-confirmation-validation-v1",
        "status": "passed",
        "validation_source_revision": subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "plan_fingerprint": plan["plan_fingerprint"],
        "result_fingerprint": result["result_fingerprint"],
        "cell_count": len(reports),
        "record_count": integrity["record_count"],
        "decision_count": integrity["decision_count"],
        "integrity_status": integrity["status"],
        "oracle_disagreement_decisions": audit["independent_production_oracle_agreement"]["counts"][
            "disagreement_decisions"
        ],
        "artifact_fingerprint": "",
    }
    validation["artifact_fingerprint"] = _fingerprint(
        {key: value for key, value in validation.items() if key != "artifact_fingerprint"}
    )
    _write(output, validation)
    print(json.dumps(validation, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

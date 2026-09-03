#!/usr/bin/env python3
"""Run the predeclared terminal-safety-v1 q0-through-q4 experiment chain."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_avenue.agents import SEED_DERIVATION, derive_seed
from agent_avenue.engine import GameConfig
from agent_avenue.engine.setup import normalize_config
from agent_avenue.learning import inspect_checkpoint
from agent_avenue.storage import inspect_source_identity, load_corpus

EXPERIMENT_ID = "terminal-safety-v1"
EXPERIMENT_VERSION = "terminal-safety-chain-v1"
ROOT_SEED = 2026090301
DEVELOPMENT_PAIRS = 400
CROSSPLAY_PAIRS = 200
FINAL_PAIRS = 500
ROOT = Path("runs") / EXPERIMENT_ID
ARCHIVE = Path("artifacts/archive") / "terminal-safety-v1-artifacts-2026-09-03.tar.gz"
HISTORICAL_Q0 = Path("checkpoints/q0")
FROZEN_HISTORICAL_Q0 = ROOT / "inputs/historical-q0"
EXPECTED_SOURCE: dict[str, object] | None = None


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def fingerprint(value: dict[str, object], *excluded: str) -> str:
    return hashlib.sha256(
        canonical_json({key: item for key, item in value.items() if key not in excluded})
    ).hexdigest()


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(canonical_json(value) + b"\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def derived_seed(name: str) -> int:
    return derive_seed(ROOT_SEED, f"experiment:{EXPERIMENT_ID}:{name}") & ((1 << 63) - 1)


def assert_frozen_source() -> None:
    if EXPECTED_SOURCE is None:
        return
    if inspect_source_identity().to_data() != EXPECTED_SOURCE:
        raise RuntimeError("Git revision, tracked tree, or uv.lock changed during the experiment")


def command(label: str, *arguments: str) -> dict[str, Any]:
    assert_frozen_source()
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    print(f"[{datetime.now(UTC).isoformat()}] start {label}", flush=True)
    completed = subprocess.run(
        [sys.executable, "-m", "agent_avenue", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    (logs / f"{label}.stdout").write_text(completed.stdout)
    (logs / f"{label}.stderr").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit {completed.returncode}; see {logs / f'{label}.stderr'}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{label} produced no JSON result")
    value = json.loads(lines[-1])
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} result must be a JSON object")
    print(f"[{datetime.now(UTC).isoformat()}] done {label}", flush=True)
    return value


def arena(
    *,
    label: str,
    agent_a: str,
    agent_b: str,
    checkpoint_a: Path | None,
    checkpoint_b: Path | None,
    shield_a: bool,
    shield_b: bool,
    pairs: int,
    seed: int,
    output: Path,
    records_root: Path,
) -> dict[str, Any]:
    arguments = [
        "arena",
        "--agent-a",
        agent_a,
        "--agent-b",
        agent_b,
        "--pairs",
        str(pairs),
        "--seed",
        str(seed),
        "--run-id",
        label,
        "--output",
        str(output),
        "--records-dir",
        str(records_root),
    ]
    if checkpoint_a is not None:
        arguments.extend(("--agent-a-checkpoint", str(checkpoint_a)))
    if checkpoint_b is not None:
        arguments.extend(("--agent-b-checkpoint", str(checkpoint_b)))
    if shield_a:
        arguments.append("--agent-a-terminal-safety")
    if shield_b:
        arguments.append("--agent-b-terminal-safety")
    return command(label, *arguments)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object at {path}")
    return value


def write_chain_plan() -> dict[str, object]:
    global EXPECTED_SOURCE

    source = inspect_source_identity()
    if not source.tracked_tree_clean:
        raise RuntimeError("tracked source must be clean before the controlled experiment starts")
    if FROZEN_HISTORICAL_Q0.exists():
        frozen_historical = inspect_checkpoint(FROZEN_HISTORICAL_Q0)
    else:
        if not HISTORICAL_Q0.is_dir():
            raise RuntimeError(f"historical q0 checkpoint is missing: {HISTORICAL_Q0}")
        FROZEN_HISTORICAL_Q0.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(HISTORICAL_Q0, FROZEN_HISTORICAL_Q0)
        frozen_historical = inspect_checkpoint(FROZEN_HISTORICAL_Q0)
    seed_names = [
        "q0-bootstrap",
        "q0-development",
        "q0-crossplay",
        *(f"q{generation}-iteration" for generation in range(1, 5)),
        *(f"q{generation}-crossplay" for generation in range(1, 5)),
        "locked-final",
    ]
    data: dict[str, object] = {
        "version": EXPERIMENT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "declaration_date": "2026-09-03",
        "source": source.to_data(),
        "historical_q0_input": {
            "source_path": str(HISTORICAL_Q0),
            "frozen_path": str(FROZEN_HISTORICAL_Q0),
            "checkpoint_fingerprint": frozen_historical.checkpoint_fingerprint,
            "tensor_digest": frozen_historical.tensor_digest,
        },
        "root_seed": ROOT_SEED,
        "seed_derivation": SEED_DERIVATION,
        "seeds": {name: derived_seed(name) for name in seed_names},
        "q0": {
            "games": 4_000,
            "epsilon": {"numerator": 1, "denominator": 5},
            "policy_shield": "terminal-safety-v1",
            "development_pairs_per_matchup": DEVELOPMENT_PAIRS,
            "development_opponents": ["random", "heuristic", "historical-q0"],
            "gate": "paired-bootstrap lower bound above 0.50 versus random",
        },
        "generations": {
            "values": [1, 2, 3, 4],
            "games_each": 4_000,
            "epsilon_schedule": ["1/10", "3/40", "1/20", "1/40"],
            "promotion_gate": "unchanged Milestone 6 gate",
            "policy_shield": "terminal-safety-v1",
        },
        "crossplay_diagnostic": {
            "diagnostic_only": True,
            "promotion_evidence": False,
            "pairs_per_matchup": CROSSPLAY_PAIRS,
            "pool_for_qn": "(heuristic, q0, ..., q(n-1))",
            "matchups": "all unordered pairs with replacement",
            "q_models": "all generated proposals, including rejected candidates",
            "learned_behavior_policy": "greedy checkpoint wrapped by terminal-safety-v1",
            "metric": "chosen-action actor-relative terminal-outcome prediction",
            "training_use": "none",
        },
        "locked_final": {
            "pairs_per_unique_matchup": FINAL_PAIRS,
            "opponents": [
                "random",
                "heuristic",
                "historical-q0",
                "shielded-q0-if-distinct",
                "immediate-parent-if-distinct",
            ],
        },
        "artifact_root": str(ROOT),
        "archive": str(ARCHIVE),
        "plan_fingerprint": "",
    }
    data["plan_fingerprint"] = fingerprint(data, "plan_fingerprint")
    path = ROOT / "experiment-plan.json"
    if path.exists():
        if read_json(path) != data:
            raise RuntimeError("existing experiment plan does not match the frozen declaration")
    else:
        atomic_json(path, data)
    EXPECTED_SOURCE = source.to_data()
    return data


def run_q0(plan: dict[str, object]) -> Path:
    seeds = plan["seeds"]
    assert isinstance(seeds, dict)
    q0_root = ROOT / "q0-a1"
    q0_checkpoint = q0_root / "checkpoint"
    command(
        "q0-bootstrap",
        "bootstrap",
        str(q0_root),
        "--attempt-id",
        "q0-terminal-safety-v1-a1",
        "--experiment-id",
        EXPERIMENT_ID,
        "--seed",
        str(seeds["q0-bootstrap"]),
        "--terminal-safety",
    )

    development = q0_root / "development"
    records = q0_root / "development-records"
    shared_seed = int(seeds["q0-development"])
    reports = {
        "random": arena(
            label="q0-vs-random",
            agent_a="learned",
            agent_b="random",
            checkpoint_a=q0_checkpoint,
            checkpoint_b=None,
            shield_a=True,
            shield_b=False,
            pairs=DEVELOPMENT_PAIRS,
            seed=shared_seed,
            output=development / "versus-random.json",
            records_root=records,
        ),
        "heuristic": arena(
            label="q0-vs-heuristic",
            agent_a="learned",
            agent_b="heuristic",
            checkpoint_a=q0_checkpoint,
            checkpoint_b=None,
            shield_a=True,
            shield_b=False,
            pairs=DEVELOPMENT_PAIRS,
            seed=shared_seed,
            output=development / "versus-heuristic.json",
            records_root=records,
        ),
        "historical_q0": arena(
            label="q0-vs-historical-q0",
            agent_a="learned",
            agent_b="learned",
            checkpoint_a=q0_checkpoint,
            checkpoint_b=FROZEN_HISTORICAL_Q0,
            shield_a=True,
            shield_b=False,
            pairs=DEVELOPMENT_PAIRS,
            seed=shared_seed,
            output=development / "versus-historical-q0.json",
            records_root=records,
        ),
    }
    lower = reports["random"]["paired_bootstrap_confidence_interval_95"]["interval"][0]
    decision: dict[str, object] = {
        "version": "terminal-safety-q0-gate-v1",
        "experiment_plan_fingerprint": plan["plan_fingerprint"],
        "checkpoint_fingerprint": read_json(q0_root / "result.json")["checkpoint_fingerprint"],
        "development_seed": shared_seed,
        "paired_seed_count": DEVELOPMENT_PAIRS,
        "versus_random_lower_bound": lower,
        "proceed": float(lower) > 0.5,
        "reports": {
            name: report["records"]["corpus_fingerprint"] for name, report in reports.items()
        },
        "artifact_fingerprint": "",
    }
    decision["artifact_fingerprint"] = fingerprint(decision, "artifact_fingerprint")
    decision_path = q0_root / "development-decision.json"
    if decision_path.exists() and read_json(decision_path) != decision:
        raise RuntimeError("existing q0 development decision does not match regenerated evidence")
    if not decision_path.exists():
        atomic_json(decision_path, decision)

    command(
        "q0-crossplay",
        "crossplay-evaluate",
        str(q0_root / "crossplay"),
        "--candidate",
        str(q0_checkpoint),
        "--candidate-label",
        "q0",
        "--generation",
        "0",
        "--exclude-corpus",
        str(q0_root / "corpus"),
        "--pairs",
        str(CROSSPLAY_PAIRS),
        "--seed",
        str(seeds["q0-crossplay"]),
        "--terminal-safety",
    )
    if not decision["proceed"]:
        raise RuntimeError(
            f"shielded q0 failed its random gate with paired-bootstrap lower bound {lower}"
        )
    return q0_checkpoint


def run_generations(
    plan: dict[str, object], q0_checkpoint: Path
) -> tuple[Path, str, list[Path], Path | None, str | None]:
    seeds = plan["seeds"]
    assert isinstance(seeds, dict)
    incumbent = q0_checkpoint
    incumbent_label = "q0"
    proposals: list[Path] = [q0_checkpoint]
    training_corpora: list[Path] = [ROOT / "q0-a1/corpus"]
    champion_parent: Path | None = None
    champion_parent_label: str | None = None

    for generation in range(1, 5):
        q_root = ROOT / f"q{generation}-a1"
        candidate_parent = incumbent
        candidate_parent_label = incumbent_label
        result = command(
            f"q{generation}-iteration",
            "iterate",
            str(q_root),
            "--incumbent",
            str(incumbent),
            "--generation",
            str(generation),
            "--attempt-id",
            f"q{generation}-terminal-safety-v1-a1",
            "--seed",
            str(seeds[f"q{generation}-iteration"]),
            "--terminal-safety",
        )
        candidate = q_root / "candidate"
        proposals.append(candidate)
        training_corpora.append(q_root / "corpus")

        arguments = [
            "crossplay-evaluate",
            str(q_root / "crossplay"),
            "--candidate",
            str(candidate),
            "--candidate-label",
            f"q{generation}",
            "--generation",
            str(generation),
            "--pairs",
            str(CROSSPLAY_PAIRS),
            "--seed",
            str(seeds[f"q{generation}-crossplay"]),
            "--terminal-safety",
        ]
        for index, prior in enumerate(proposals[:-1]):
            arguments.extend(("--prior", f"q{index}={prior}"))
        for corpus in training_corpora:
            arguments.extend(("--exclude-corpus", str(corpus)))
        command(f"q{generation}-crossplay", *arguments)

        if result["selected_role"] == "candidate":
            champion_parent = candidate_parent
            champion_parent_label = candidate_parent_label
            incumbent = candidate
            incumbent_label = f"q{generation}"

    return incumbent, incumbent_label, proposals, champion_parent, champion_parent_label


def _setup_key(config: GameConfig, seed: int) -> tuple[bytes, int]:
    return canonical_json(normalize_config(config)), seed


def verify_locked_final_holdout(plan: dict[str, object]) -> dict[str, object]:
    seeds = plan["seeds"]
    assert isinstance(seeds, dict)
    final_seed = int(seeds["locked-final"])
    final_keys = {
        _setup_key(
            GameConfig(),
            derive_seed(final_seed, f"arena:pair:{index}:setup") & ((1 << 64) - 1),
        )
        for index in range(FINAL_PAIRS)
    }

    prior_keys: set[tuple[bytes, int]] = set()
    checked_corpora: list[dict[str, object]] = []
    manifests: set[Path] = set()
    for base in (ROOT, Path("runs/milestone6"), Path("runs/q0-corpus")):
        if not base.exists():
            continue
        candidates = [base / "manifest.json"] if (base / "manifest.json").is_file() else []
        candidates.extend(base.rglob("manifest.json"))
        for manifest_path in candidates:
            directory = manifest_path.parent
            if manifest_path in manifests or not (directory / "games.jsonl.gz").is_file():
                continue
            manifests.add(manifest_path)
            if directory == ROOT / "final" or (ROOT / "final") in directory.parents:
                continue
            manifest, records = load_corpus(directory, verify_code=False)
            prior_keys.update(
                _setup_key(record.replay.config, record.replay.seed) for record in records
            )
            checked_corpora.append(
                {
                    "path": str(directory),
                    "corpus_fingerprint": manifest.corpus_fingerprint,
                    "game_count": manifest.record_count,
                }
            )

    reserved_historical = (
        ("historical-q0-vs-random", 2026083001, 400),
        ("historical-q0-vs-heuristic", 2026083002, 400),
    )
    for _label, master_seed, pair_count in reserved_historical:
        prior_keys.update(
            _setup_key(
                GameConfig(),
                derive_seed(master_seed, f"arena:pair:{index}:setup") & ((1 << 64) - 1),
            )
            for index in range(pair_count)
        )

    overlaps = final_keys & prior_keys
    if overlaps:
        raise RuntimeError(f"locked-final setup block overlaps {len(overlaps)} prior setups")
    data: dict[str, object] = {
        "version": "terminal-safety-locked-final-holdout-v1",
        "experiment_plan_fingerprint": plan["plan_fingerprint"],
        "master_seed": final_seed,
        "paired_seed_count": FINAL_PAIRS,
        "normalized_game_config": normalize_config(GameConfig()),
        "checked_corpora": sorted(checked_corpora, key=lambda item: str(item["path"])),
        "reserved_historical_schedules": [
            {"label": label, "master_seed": seed, "paired_seed_count": count}
            for label, seed, count in reserved_historical
        ],
        "prior_unique_setup_count": len(prior_keys),
        "overlap_count": 0,
        "artifact_fingerprint": "",
    }
    data["artifact_fingerprint"] = fingerprint(data, "artifact_fingerprint")
    path = ROOT / "final/holdout-validation.json"
    if path.exists() and read_json(path) != data:
        raise RuntimeError("existing locked-final holdout validation does not match prior data")
    if not path.exists():
        atomic_json(path, data)
    return data


def run_locked_final(
    plan: dict[str, object],
    champion: Path,
    champion_label: str,
    q0_checkpoint: Path,
    immediate_parent: Path | None,
    parent_label: str | None,
) -> dict[str, Any]:
    seeds = plan["seeds"]
    assert isinstance(seeds, dict)
    verify_locked_final_holdout(plan)

    opponents: list[tuple[str, str, Path | None, bool]] = [
        ("random", "random", None, False),
        ("heuristic", "heuristic", None, False),
        ("historical-q0", "learned", FROZEN_HISTORICAL_Q0, False),
    ]
    if q0_checkpoint != champion:
        opponents.append(("shielded-q0", "learned", q0_checkpoint, True))
    if immediate_parent is not None and immediate_parent not in {champion, q0_checkpoint}:
        assert parent_label is not None
        opponents.append(
            (
                f"immediate-parent-{parent_label}",
                "learned",
                immediate_parent,
                True,
            )
        )

    reports: dict[str, Any] = {}
    seen: set[tuple[str, str | None, bool]] = set()
    for label, kind, checkpoint, shield in opponents:
        key = (kind, str(checkpoint) if checkpoint is not None else None, shield)
        if key in seen:
            continue
        seen.add(key)
        reports[label] = arena(
            label=f"final-{champion_label}-vs-{label}",
            agent_a="learned",
            agent_b=kind,
            checkpoint_a=champion,
            checkpoint_b=checkpoint,
            shield_a=True,
            shield_b=shield,
            pairs=FINAL_PAIRS,
            seed=int(seeds["locked-final"]),
            output=ROOT / "final/arenas" / f"versus-{label}.json",
            records_root=ROOT / "final/arena-records",
        )
    return reports


def summarize(
    plan: dict[str, object],
    champion: Path,
    champion_label: str,
    final_reports: dict[str, Any],
) -> dict[str, object]:
    generations = []
    for generation in range(1, 5):
        decision = read_json(ROOT / f"q{generation}-a1/promotion-decision.json")
        crossplay = read_json(ROOT / f"q{generation}-a1/crossplay/report.json")
        generations.append(
            {
                "generation": generation,
                "candidate_checkpoint": decision["candidate_checkpoint"],
                "selected_checkpoint": decision["selected_checkpoint"],
                "selected_role": decision["selected_role"],
                "decision": decision["decision"],
                "evidence": decision["evidence"],
                "crossplay_macro_log_loss": crossplay["macro_paired_block_log_loss"],
                "crossplay_worst_matchup": crossplay["worst_matchup_log_loss"],
            }
        )
    q0_result = read_json(ROOT / "q0-a1/result.json")
    q0_gate = read_json(ROOT / "q0-a1/development-decision.json")
    q0_crossplay = read_json(ROOT / "q0-a1/crossplay/report.json")
    data: dict[str, object] = {
        "version": "terminal-safety-experiment-result-v1",
        "experiment_plan_fingerprint": plan["plan_fingerprint"],
        "source": plan["source"],
        "q0": {
            "checkpoint_fingerprint": q0_result["checkpoint_fingerprint"],
            "development_gate": q0_gate,
            "crossplay_macro_log_loss": q0_crossplay["macro_paired_block_log_loss"],
        },
        "generations": generations,
        "champion": {"label": champion_label, "path": str(champion)},
        "locked_final_holdout_validation": read_json(ROOT / "final/holdout-validation.json"),
        "locked_final": {
            label: {
                "win_rate": report["agent_a_win_rate"],
                "paired_bootstrap_confidence_interval_95": report[
                    "paired_bootstrap_confidence_interval_95"
                ],
                "by_seat": report["agent_a_by_seat"],
                "average_score_margin": report["average_score_margin"],
                "average_turns": report["average_turns"],
                "terminal_reasons": report["terminal_reasons"],
                "records": report["records"],
                "safety_diagnostics": report["safety_diagnostics"],
            }
            for label, report in final_reports.items()
        },
        "artifact_fingerprint": "",
    }
    data["artifact_fingerprint"] = fingerprint(data, "artifact_fingerprint")
    result_path = ROOT / "result.json"
    if result_path.exists() and read_json(result_path) != data:
        raise RuntimeError("existing experiment result does not match regenerated artifacts")
    if not result_path.exists():
        atomic_json(result_path, data)
    return data


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _embedded_archive_manifest(
    plan: dict[str, object], result: dict[str, object]
) -> dict[str, object]:
    members = []
    for path in sorted(ROOT.rglob("*")):
        if (
            not path.is_file()
            or path.name.endswith(".lock")
            or path == ROOT / "archive-manifest.json"
        ):
            continue
        members.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    data: dict[str, object] = {
        "version": "terminal-safety-embedded-archive-manifest-v1",
        "experiment_plan_fingerprint": plan["plan_fingerprint"],
        "result_fingerprint": result["artifact_fingerprint"],
        "members": members,
        "artifact_fingerprint": "",
    }
    data["artifact_fingerprint"] = fingerprint(data, "artifact_fingerprint")
    return data


def verify_archive(result: dict[str, object]) -> None:
    sidecar_path = ARCHIVE.with_suffix(ARCHIVE.suffix + ".json")
    sidecar = read_json(sidecar_path)
    if sidecar.get("sha256") != _sha256_file(ARCHIVE):
        raise RuntimeError("terminal-safety archive checksum mismatch")
    if sidecar.get("size") != ARCHIVE.stat().st_size:
        raise RuntimeError("terminal-safety archive size mismatch")
    if sidecar.get("result_fingerprint") != result.get("artifact_fingerprint"):
        raise RuntimeError("terminal-safety archive points to a different result")
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        manifest_member = archive.extractfile(f"{EXPERIMENT_ID}/archive-manifest.json")
        if manifest_member is None:
            raise RuntimeError("terminal-safety archive has no embedded manifest")
        embedded = json.loads(manifest_member.read())
        if not isinstance(embedded, dict):
            raise RuntimeError("terminal-safety embedded archive manifest is malformed")
        if embedded.get("artifact_fingerprint") != fingerprint(embedded, "artifact_fingerprint"):
            raise RuntimeError("terminal-safety embedded archive manifest fingerprint mismatch")
        if embedded.get("result_fingerprint") != result.get("artifact_fingerprint"):
            raise RuntimeError("terminal-safety embedded archive result mismatch")
        members = embedded.get("members")
        if not isinstance(members, list):
            raise RuntimeError("terminal-safety embedded archive member list is malformed")
        for item in members:
            if not isinstance(item, dict):
                raise RuntimeError("terminal-safety embedded archive member is malformed")
            path = item.get("path")
            if not isinstance(path, str):
                raise RuntimeError("terminal-safety embedded archive member path is malformed")
            member = archive.extractfile(f"{EXPERIMENT_ID}/{path}")
            if member is None:
                raise RuntimeError(f"terminal-safety archive is missing {path}")
            data = member.read()
            if (
                item.get("size") != len(data)
                or item.get("sha256") != hashlib.sha256(data).hexdigest()
            ):
                raise RuntimeError(f"terminal-safety archive member checksum mismatch: {path}")


def completed_result() -> dict[str, object] | None:
    sidecar = ARCHIVE.with_suffix(ARCHIVE.suffix + ".json")
    if ARCHIVE.exists() != sidecar.exists():
        raise RuntimeError("terminal-safety archive and sidecar must exist together")
    if not ARCHIVE.exists():
        return None
    result = read_json(ROOT / "result.json")
    verify_archive(result)
    return result


def archive_artifacts(plan: dict[str, object], result: dict[str, object]) -> None:
    sidecar_path = ARCHIVE.with_suffix(ARCHIVE.suffix + ".json")
    if ARCHIVE.exists() or sidecar_path.exists():
        if not ARCHIVE.exists() or not sidecar_path.exists():
            raise RuntimeError("terminal-safety archive and sidecar must exist together")
        verify_archive(result)
        return
    embedded = _embedded_archive_manifest(plan, result)
    atomic_json(ROOT / "archive-manifest.json", embedded)
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    temporary = ARCHIVE.with_suffix(ARCHIVE.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    with tarfile.open(temporary, "w:gz") as archive:
        archive.add(ROOT, arcname=EXPERIMENT_ID)
    os.replace(temporary, ARCHIVE)
    digest = _sha256_file(ARCHIVE)
    manifest: dict[str, object] = {
        "version": "terminal-safety-archive-manifest-v1",
        "experiment_plan_fingerprint": plan["plan_fingerprint"],
        "result_fingerprint": result["artifact_fingerprint"],
        "archive": str(ARCHIVE),
        "sha256": digest,
        "size": ARCHIVE.stat().st_size,
        "created_at": datetime.now(UTC).isoformat(),
    }
    atomic_json(sidecar_path, manifest)
    verify_archive(result)


def main() -> None:
    existing = completed_result()
    if existing is not None:
        print(json.dumps(existing, sort_keys=True))
        return
    plan = write_chain_plan()
    q0_checkpoint = run_q0(plan)
    champion, champion_label, _proposals, immediate_parent, parent_label = run_generations(
        plan, q0_checkpoint
    )
    final_reports = run_locked_final(
        plan,
        champion,
        champion_label,
        q0_checkpoint,
        immediate_parent,
        parent_label,
    )
    result = summarize(plan, champion, champion_label, final_reports)
    archive_artifacts(plan, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

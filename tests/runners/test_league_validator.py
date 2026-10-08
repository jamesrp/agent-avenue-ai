"""Adversarial and agreement tests for the independent Step-5 league validator."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agent_avenue.agents import RandomAgent
from agent_avenue.runners import AgentSpec, ArenaConfig, run_game, schedule_arena
from agent_avenue.runners.league import (
    CHALLENGER_ID,
    INCUMBENT_ID,
    M_IDS,
    RANDOM_ID,
    REGISTRY_PATH,
    LeagueDesign,
    LeagueRunConfig,
    _anchor_contrasts,
    _m_family,
    _Summaries,
    build_policy_bundle,
    cell_arena_config,
    compare_aligned_games,
    fingerprint,
    load_registry,
    record_tactical_counts,
    run_league,
)

SCRIPT = Path("scripts/validate_independent_league_promotion_v1.py")


def _load_validator() -> Any:
    spec = importlib.util.spec_from_file_location("league_validator_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def validator() -> Any:
    return _load_validator()


@pytest.fixture(scope="module")
def registry() -> dict[str, Any]:
    return json.loads(REGISTRY_PATH.read_text())


@pytest.fixture(scope="module")
def specs() -> Any:
    return build_policy_bundle(load_registry(), Path("."), mode="toy-random").specs


def _design(validator: Any, registry: dict[str, Any], namespace: str, root: int, pairs: int) -> Any:
    return validator.Design(
        namespace, root, pairs, validator.Roles(registry), registry["schedule"]["family_order"]
    )


def _reseal(value: dict[str, Any]) -> dict[str, Any]:
    payload = {key: item for key, item in value.items() if key != "artifact_fingerprint"}
    return {**payload, "artifact_fingerprint": fingerprint(payload)}


# ---------------------------------------------------------------------------------------------
# Independence


def test_validator_imports_no_production_league_code() -> None:
    tree = ast.parse(SCRIPT.read_text())
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not any(name.startswith("agent_avenue.runners.league") for name in imported)
    assert "agent_avenue.runners" not in imported  # no package-level re-exports either
    allowed_runner_modules = {"agent_avenue.runners.public_win_oracle"}
    assert {name for name in imported if name.startswith("agent_avenue.runners")} <= (
        allowed_runner_modules
    )


# ---------------------------------------------------------------------------------------------
# Schedule and declarations


def test_validator_reconstructs_the_committed_claim_schedule(
    validator: Any, registry: dict[str, Any]
) -> None:
    design = _design(validator, registry, validator.CYCLE, registry["root_seed"], 200)
    assert validator.check_registry_schedule(registry, design) == []
    tampered = copy.deepcopy(registry)
    tampered["schedule"]["cells"][5]["right"], tampered["schedule"]["cells"][6]["right"] = (
        tampered["schedule"]["cells"][6]["right"],
        tampered["schedule"]["cells"][5]["right"],
    )
    tampered["schedule"]["families"]["family-a"]["setup_seed_fingerprint"] = "0" * 64
    problems = validator.check_registry_schedule(tampered, design)
    assert "registry cells" in problems and "registry family family-a" in problems
    short = _design(validator, registry, validator.CYCLE, registry["root_seed"], 199)
    assert any(
        "cardinality" in problem for problem in validator.check_registry_schedule(registry, short)
    )


def test_validator_declaration_check_rejects_wrapping_mutations(
    validator: Any, registry: dict[str, Any], specs: Any
) -> None:
    check = validator._checkpoint_check(registry, "toy-random")
    roles = validator.Roles(registry)
    for policy in roles.order:
        assert validator.realizes(roles.declared[policy], dict(specs[policy].config), check)
    q1 = dict(specs["q1-terminal-safety-v1"].config)
    wrapped = {**dict(specs[CHALLENGER_ID].config), "base": q1}
    assert not validator.realizes(roles.declared["q1-terminal-safety-v1"], wrapped, check)
    unshielded = {**dict(specs[CHALLENGER_ID].config)}
    unshielded["base"] = unshielded["base"]["base"]
    assert not validator.realizes(roles.declared[CHALLENGER_ID], unshielded, check)
    learned = validator._checkpoint_check(registry, "learned-checkpoints")
    assert not validator.realizes(
        roles.declared[INCUMBENT_ID], dict(specs[INCUMBENT_ID].config), learned
    )


# ---------------------------------------------------------------------------------------------
# Cell schedule mutations


def _cell(specs: Any, validator: Any, registry: dict[str, Any]) -> tuple[Any, Any, list[Any]]:
    design = LeagueDesign("validator-cell-test", 41, 2)
    cell = next(c for c in design.cells() if (c.left, c.right) == (INCUMBENT_ID, RANDOM_ID))
    records = [run_game(spec) for spec in schedule_arena(cell_arena_config(cell, specs))]
    vdesign = _design(validator, registry, design.namespace, design.root_seed, design.pair_count)
    return cell, vdesign, records


def test_cell_schedule_mutations_are_detected(
    validator: Any, registry: dict[str, Any], specs: Any
) -> None:
    cell, design, records = _cell(specs, validator, registry)
    configs = {policy: dict(spec.config) for policy, spec in specs.items()}

    def problems(rows: list[Any]) -> list[str]:
        return validator.check_cell_schedule(
            design, cell.family, cell.left, cell.right, rows, configs
        )

    assert problems(records) == []
    inverted = [records[1], records[0], *records[2:]]
    assert any("seat order" in item or "identity" in item for item in problems(inverted))
    first = records[0]
    moved = replace(first, replay=replace(first.replay, seed=first.replay.seed + 1))
    assert any("setup" in item for item in problems([moved, *records[1:]]))
    seats = list(first.seats)
    seats[1] = replace(seats[1], seed=seats[1].seed + 1)
    assert any(
        "seat RNG" in item for item in problems([replace(first, seats=tuple(seats)), *records[1:]])
    )
    seats = list(first.seats)
    seats[0] = replace(seats[0], rng_domain="agent:q0-terminal-safety-v1")
    assert any(
        "seat RNG" in item for item in problems([replace(first, seats=tuple(seats)), *records[1:]])
    )
    wrong_config = {**configs, RANDOM_ID: {"type": "random", "version": "other"}}
    assert validator.check_cell_schedule(
        design, cell.family, cell.left, cell.right, records, wrong_config
    )


def test_validator_tactical_oracle_agrees_with_production_counts(validator: Any) -> None:
    raw = AgentSpec(CHALLENGER_ID, {"type": "random", "version": "random-agent-v1"}, RandomAgent)
    other = AgentSpec(RANDOM_ID, {"type": "random", "version": "random-agent-v1"}, RandomAgent)
    records = [run_game(spec) for spec in schedule_arena(ArenaConfig("t", raw, other, 20, 5))]
    production = record_tactical_counts(records)
    independent = validator.tactical_counts(records)
    assert production == independent
    assert independent[CHALLENGER_ID]["missed_guaranteed_wins"] > 0
    assert independent[CHALLENGER_ID]["executed_avoidable_provable_losses"] > 0


def test_validator_prefix_audit_agrees_and_detects_misalignment(
    validator: Any, registry: dict[str, Any], specs: Any
) -> None:
    roles = validator.Roles(registry)
    design = LeagueDesign("validator-prefix-test", 43, 4)
    cells = {(c.left, c.right): c for c in design.cells() if c.family == "family-a"}
    inc = [
        run_game(s)
        for s in schedule_arena(cell_arena_config(cells[(INCUMBENT_ID, RANDOM_ID)], specs))
    ]
    ch = [
        run_game(s)
        for s in schedule_arena(cell_arena_config(cells[(CHALLENGER_ID, RANDOM_ID)], specs))
    ]
    rows = [validator.aligned_pair(roles, a, b) for a, b in zip(inc, ch, strict=True)]
    production = [compare_aligned_games(a, b) for a, b in zip(inc, ch, strict=True)]
    for row, other in zip(rows, production, strict=True):
        assert row["metadata"] == other["metadata_aligned"]
        assert row["prefix"] == other["prefix_aligned"]
        assert row["classification"] == other["endpoint_classification"]
        assert row["terminated"] == other["terminated_in_guaranteed_win"]
    totals = validator.prefix_totals(rows, [record.pair_id for record in inc])
    assert totals["metadata_failures"] == totals["prefix_failures"] == 0
    shifted = validator.prefix_totals(
        [validator.aligned_pair(roles, a, b) for a, b in zip(inc, ch[1:] + ch[:1], strict=True)],
        [record.pair_id for record in inc],
    )
    assert shifted["metadata_failures"] == len(inc)


# ---------------------------------------------------------------------------------------------
# Statistics agreement and bootstrap alignment


def _summaries(design: LeagueDesign, score: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    production: dict[str, Any] = {}
    independent: dict[str, Any] = {}
    for cell in design.cells():
        wins = [
            score(cell.family, cell.left, cell.right, block) for block in range(cell.pair_count)
        ]
        production[cell.key] = {
            "key": cell.key,
            "left": cell.left,
            "right": cell.right,
            "left_pair_wins": wins,
        }
        independent[cell.key] = {"left": cell.left, "right": cell.right, "pair_scores": wins}
    return production, independent


def _score(family: str, left: str, right: str, block: int) -> int:
    return (block * 5 + len(left) * 3 + len(right) + (family == "family-b") * 2) % 3


def test_validator_anchor_and_m_family_bootstraps_match_production(
    validator: Any, registry: dict[str, Any]
) -> None:
    design = LeagueDesign("validator-stats-test", 13, 9)
    production, independent = _summaries(design, _score)
    vdesign = _design(validator, registry, design.namespace, design.root_seed, design.pair_count)
    scores = validator.Scores(vdesign, independent)
    assert validator._anchors(vdesign, scores) == _anchor_contrasts(_Summaries(design, production))
    assert validator._nested_m(vdesign, scores) == _m_family(_Summaries(design, production))


def test_bootstrap_index_misalignment_would_be_visible(
    validator: Any, registry: dict[str, Any]
) -> None:
    design = LeagueDesign("validator-stats-test", 14, 10)

    def same_arms(family: str, left: str, right: str, block: int) -> int:
        return (block + len(right)) % 3

    _, independent = _summaries(design, same_arms)
    vdesign = _design(validator, registry, design.namespace, design.root_seed, design.pair_count)
    anchors = validator._anchors(vdesign, validator.Scores(vdesign, independent))
    assert anchors["macro"]["interval_95"] == [0.0, 0.0]


@pytest.mark.parametrize(
    ("lift", "random_lower", "seat", "expected"),
    [
        (0.0025 + 1e-9, 0.5 + 1e-9, 0.45, "promote_q0_terminal_offense_v1"),
        (0.0025, 0.6, 0.6, "retain_q0_terminal_safety_v1"),
        (0.01, 0.5, 0.6, "retain_q0_terminal_safety_v1"),
        (0.01, 0.6, 0.4499, "retain_q0_terminal_safety_v1"),
    ],
)
def test_validator_threshold_boundaries(
    validator: Any,
    registry: dict[str, Any],
    lift: float,
    random_lower: float,
    seat: float,
    expected: str,
) -> None:
    roles = validator.Roles(registry)
    seats = {
        "player_one": {"combined": {"win_rate": seat}},
        "player_two": {"combined": {"win_rate": 0.9}},
    }
    statistics = {
        "anchor_contrasts": {"macro": {"interval_95": [lift, 0.1]}},
        "matchups": {
            f"{INCUMBENT_ID}--vs--{CHALLENGER_ID}": {"seats": {CHALLENGER_ID: seats}},
            f"{CHALLENGER_ID}--vs--greedy-public-v1": {"seats": {CHALLENGER_ID: seats}},
            f"{CHALLENGER_ID}--vs--{RANDOM_ID}": {
                "seats": {CHALLENGER_ID: seats},
                "combined": {"paired_block_bootstrap_95": [random_lower, 1.0]},
            },
        },
    }
    zero = dict.fromkeys(validator.TACTICAL_KEYS, 0)
    prefix = dict.fromkeys(
        (
            "incumbent_win_challenger_loss_blocks",
            "incumbent_win_challenger_loss_games",
            "metadata_failures",
            "prefix_failures",
            "post_intervention_termination_failures",
            "challenger_failed_guaranteed_win",
        ),
        0,
    )
    tactical = {INCUMBENT_ID: zero, CHALLENGER_ID: zero}
    decision, _ = validator.decide(roles, statistics, tactical, prefix, True)
    assert decision == expected
    assert validator.decide(roles, statistics, tactical, prefix, False)[0] == "blocked_no_decision"
    bad_m = {**tactical, M_IDS[1]: {**zero, "false_forced_wins": 1}}
    assert validator.decide(roles, statistics, bad_m, prefix, True)[0] == "blocked_no_decision"


# ---------------------------------------------------------------------------------------------
# End to end: one tiny toy league, independent validation, and artifact mutations


@pytest.fixture(scope="module")
def league_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("league")
    output = root / "smoke"
    run_league(
        LeagueRunConfig(
            output=output,
            policy_mode="toy-random",
            smoke_pairs=1,
            holdout_roots=(root / "empty-runs",),
        )
    )
    return output


def test_toy_league_validates_independently(validator: Any, league_output: Path) -> None:
    outcome = validator.validate(
        league_output,
        registry_path=REGISTRY_PATH,
        inputs_root=Path("."),
        holdout_roots=(league_output.parent / "empty-runs",),
    )
    report = outcome["validation"]
    assert report["status"] == "passed", report["problems"]
    assert report["final_decision"] == report["retained_decision"]
    assert report["replayed_games"] == 264
    assert report["evidence_class"] == "nonclaim-smoke"


def _design_for(validator: Any, output: Path) -> Any:
    plan = json.loads((output / "plan.json").read_text())
    registry = json.loads(REGISTRY_PATH.read_text())
    return validator._design_from_plan(plan, registry, validator.Roles(registry))


def test_checksum_scope_and_tree_mutations(
    validator: Any, league_output: Path, tmp_path: Path
) -> None:
    copy_root = tmp_path / "copy"
    shutil.copytree(league_output, copy_root)
    design = _design_for(validator, copy_root)
    clean: list[str] = []
    validator.check_tree_and_checksums(copy_root, design, clean)
    assert clean == []

    (copy_root / "notes.json").write_text("{}")
    problems: list[str] = []
    validator.check_tree_and_checksums(copy_root, design, problems)
    assert any("output tree differs" in item for item in problems)
    (copy_root / "notes.json").unlink()

    statistics = copy_root / "statistics.json"
    statistics.write_text(statistics.read_text().replace('"matrix"', '"matrix" ', 1))
    problems = []
    validator.check_tree_and_checksums(copy_root, design, problems)
    assert "checksum mismatch: statistics.json" in problems

    manifest = json.loads((copy_root / "checksums.json").read_text())
    manifest["files"].pop("statistics.json")
    (copy_root / "checksums.json").write_text(json.dumps(_reseal(manifest)))
    problems = []
    validator.check_tree_and_checksums(copy_root, design, problems)
    assert "checksum scope differs from the retained artifact set" in problems


def test_decision_and_cross_reference_mutations(validator: Any, league_output: Path) -> None:
    def load(name: str) -> dict[str, Any]:
        return json.loads((league_output / name).read_text())

    decision, result = load("promotion-decision.json"), load("result.json")
    artifacts = {
        name: load(f"{file}.json")
        for name, file in (
            ("statistics", "statistics"),
            ("prefix", "prefix-audit"),
            ("tactical", "tactical"),
            ("holdout", "holdout"),
        )
    }
    plan_fp = load("plan.json")["plan_fingerprint"]
    problems: list[str] = []
    validator.check_cross_references(decision, result, artifacts, plan_fp, problems)
    assert problems == []

    promoted = _reseal({**decision, "decision": "promote_q0_terminal_offense_v1"})
    problems = []
    validator.check_cross_references(promoted, result, artifacts, plan_fp, problems)
    assert "result cross-references" in problems
    checks = [condition["passed"] for condition in decision["conditions"]]
    validator.check_decision_reproduces(promoted, decision["decision"], checks, problems)
    assert any("decision does not reproduce" in item for item in problems)

    flipped = copy.deepcopy(decision)
    flipped["conditions"][0]["passed"] = not flipped["conditions"][0]["passed"]
    problems = []
    validator.check_decision_reproduces(flipped, decision["decision"], checks, problems)
    assert problems

    unsealed = {**decision, "decision": "promote_q0_terminal_offense_v1"}
    problems = []
    validator.check_cross_references(unsealed, result, artifacts, plan_fp, problems)
    assert "decision seal" in problems

    authorized = _reseal({**decision, "deployment_change_authorized_by_runner": True})
    problems = []
    validator.check_cross_references(authorized, result, artifacts, plan_fp, problems)
    assert "runner decision claims deployment authority" in problems


def test_tampered_plan_is_rejected_before_any_replay(
    validator: Any, league_output: Path, tmp_path: Path
) -> None:
    copy_root = tmp_path / "copy"
    shutil.copytree(league_output, copy_root)
    plan = json.loads((copy_root / "plan.json").read_text())
    plan["execution"]["pair_count"] = 2
    (copy_root / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(validator.ValidationFailure, match="plan fingerprint"):
        validator.validate(
            copy_root,
            registry_path=REGISTRY_PATH,
            inputs_root=Path("."),
            holdout_roots=(tmp_path / "none",),
        )

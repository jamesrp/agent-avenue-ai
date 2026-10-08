"""Step-5 independent league: schedule, declarations, audits, statistics, and decision rule."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from agent_avenue.agents import GreedyHeuristicAgent, RandomAgent
from agent_avenue.engine import PlayerId
from agent_avenue.runners import AgentSpec, GameSpec, run_game, schedule_arena
from agent_avenue.runners.league import (
    ANCHOR_IDS,
    CHALLENGER_ID,
    DECISION_BLOCKED,
    DECISION_PROMOTE,
    DECISION_RETAIN,
    HEURISTIC_ID,
    INCUMBENT_ID,
    M_IDS,
    POLICY_ORDER,
    PROMOTION_LIFT_THRESHOLD,
    Q0_CORE_RNG_IDENTITY,
    RANDOM_ID,
    REGISTRY_FINGERPRINT,
    REGISTRY_PATH,
    LeagueDesign,
    LeagueError,
    _anchor_contrasts,
    _m_family,
    _Summaries,
    aggregate_tactical,
    build_policy_bundle,
    cell_arena_config,
    claim_design,
    compare_aligned_games,
    declared_config_matches,
    empty_tactical,
    fingerprint,
    load_registry,
    promotion_decision,
    record_tactical_counts,
    run_league_cell,
    scan_setup_holdout,
    schedule_data,
    smoke_design,
    verify_league_inputs,
    verify_registry_schedule,
)


def _reseal(registry: dict[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in registry.items() if key != "artifact_fingerprint"}
    return {**payload, "artifact_fingerprint": fingerprint(payload)}


def _write_registry(path: Path, registry: Mapping[str, Any]) -> Path:
    path.write_text(json.dumps(registry, sort_keys=True, indent=2))
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _toy_specs() -> Mapping[str, AgentSpec]:
    return build_policy_bundle(load_registry(), Path("."), mode="toy-random").specs


# ---------------------------------------------------------------------------------------------
# Registry and schedule


def test_committed_registry_authenticates_and_reconstructs_the_frozen_schedule() -> None:
    registry = load_registry()
    assert registry["artifact_fingerprint"] == REGISTRY_FINGERPRINT
    audit = verify_registry_schedule(registry)
    assert audit["status"] == "passed", audit["failures"]
    assert (audit["total_cells"], audit["total_games"]) == (132, 52_800)
    design = claim_design()
    assert tuple(registry["schedule"]["policy_order"]) == POLICY_ORDER
    cells = design.cells()
    assert len({cell.key for cell in cells}) == 132
    assert all(cell.games == 400 for cell in cells)
    unordered = {(cell.left, cell.right) for cell in cells}
    assert len(unordered) == 66


def test_registry_tampering_is_rejected_by_seal_and_by_schedule_reconstruction(
    tmp_path: Path,
) -> None:
    registry = json.loads(REGISTRY_PATH.read_text())
    tampered = copy.deepcopy(registry)
    tampered["schedule"]["cells"][0]["games"] = 398
    with pytest.raises(LeagueError, match="self-fingerprint"):
        load_registry(_write_registry(tmp_path / "unsealed.json", tampered))

    resealed = load_registry(_write_registry(tmp_path / "resealed.json", _reseal(tampered)))
    audit = verify_registry_schedule(resealed)
    assert audit["status"] == "failed"
    assert "cells" in audit["failures"]
    assert "registry_is_not_the_committed_claim_registry" in audit["failures"]

    overlapping = copy.deepcopy(registry)
    family_b = overlapping["schedule"]["families"]["family-b"]
    family_b["master_seed"] = overlapping["schedule"]["families"]["family-a"]["master_seed"]
    audit = verify_registry_schedule(_reseal(overlapping))
    assert "family:family-b" in audit["failures"]


def test_two_promotion_eligible_policies_are_rejected(tmp_path: Path) -> None:
    registry = json.loads(REGISTRY_PATH.read_text())
    registry["policies"][2]["promotion_eligible"] = True
    with pytest.raises(LeagueError, match="only promotion-eligible"):
        load_registry(_write_registry(tmp_path / "registry.json", _reseal(registry)))


def test_families_are_disjoint_and_smoke_uses_separate_domains() -> None:
    claim = schedule_data(claim_design())
    assert claim["families_disjoint"] is True
    claim_families = claim["families"]
    assert isinstance(claim_families, dict)
    smoke = smoke_design(200)
    assert smoke.namespace != claim_design().namespace
    claim_seeds = {
        seed
        for family in claim_design().family_order
        for seed in claim_design().family_setup_seeds(family)
    }
    smoke_seeds = {
        seed for family in smoke.family_order for seed in smoke.family_setup_seeds(family)
    }
    assert claim_seeds.isdisjoint(smoke_seeds)
    assert not smoke.is_claim_design and claim_design().is_claim_design


# ---------------------------------------------------------------------------------------------
# Policy declarations and RNG identity


def test_toy_bundle_realizes_declarations_and_aligns_only_the_q0_variants() -> None:
    specs = _toy_specs()
    assert tuple(specs) == POLICY_ORDER
    assert specs[INCUMBENT_ID].rng_identity == specs[CHALLENGER_ID].rng_identity
    assert specs[INCUMBENT_ID].rng_identity == Q0_CORE_RNG_IDENTITY
    others = [
        spec.rng_identity for key, spec in specs.items() if key not in {INCUMBENT_ID, CHALLENGER_ID}
    ]
    assert len(set(others)) == len(others)
    assert specs[CHALLENGER_ID].config["type"] == "terminal_offense"
    assert specs[CHALLENGER_ID].config["base"]["type"] == "terminal_safety"  # type: ignore[index]
    assert specs[M_IDS[0]].config["type"] == "terminal_offense"
    assert specs["q1-terminal-safety-v1"].config["type"] == "terminal_safety"
    assert specs["historical-q0"].config["type"] == "random"  # toy stand-in for the raw checkpoint


@pytest.mark.parametrize(
    ("policy_id", "mutate"),
    [
        # q1-q4 must stay safety-only: adding the offense wrapper is a policy-wrapping mutation.
        (
            "q1-terminal-safety-v1",
            lambda config: {
                "type": "terminal_offense",
                "version": "terminal-offense-v1",
                "base": config,
                "selection": "base-policy-on-forced-win-set-v1",
                "win_definition": "public-guaranteed-current-turn-win-v1",
            },
        ),
        # The challenger without its safety layer.
        (CHALLENGER_ID, lambda config: {**config, "base": config["base"]["base"]}),
        # A changed heuristic weight.
        (HEURISTIC_ID, lambda config: {**config, "score_gap_weight": 11}),
        # An unexpected extra runtime field.
        (RANDOM_ID, lambda config: {**config, "temperature": 1}),
    ],
)
def test_declaration_check_rejects_policy_wrapping_and_config_mutations(
    policy_id: str, mutate: Any
) -> None:
    registry = load_registry()
    declared = next(
        policy["config"] for policy in registry["policies"] if policy["id"] == policy_id
    )
    runtime = dict(_toy_specs()[policy_id].config)
    resolver = lambda name, value: value == {"type": "random", "version": "random-agent-v1"}  # noqa: E731
    assert declared_config_matches(declared, runtime, resolver)
    assert not declared_config_matches(declared, mutate(copy.deepcopy(runtime)), resolver)


def test_challenger_without_shared_core_rng_identity_is_rejected(tmp_path: Path) -> None:
    registry = json.loads(REGISTRY_PATH.read_text())
    registry["policies"][1]["rng_identity"] = "q0-terminal-offense-v1"
    loaded = load_registry(_write_registry(tmp_path / "registry.json", _reseal(registry)))
    with pytest.raises(LeagueError, match="share the q0 core RNG identity"):
        build_policy_bundle(loaded, Path("."), mode="toy-random")


# ---------------------------------------------------------------------------------------------
# Learned checkpoints (synthetic fixtures standing in for retained inputs)


def build_learned_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Write tiny real checkpoints and a resealed registry that binds their identities."""
    pytest.importorskip("torch")
    from agent_avenue.learning import create_model, load_checkpoint, save_checkpoint
    from agent_avenue.learning.model import CandidateMLP
    from agent_avenue.learning.structured_checkpoint import (
        load_structured_checkpoint,
        save_structured_checkpoint,
    )
    from agent_avenue.learning.structured_model import create_structured_model

    registry = json.loads(REGISTRY_PATH.read_text())
    root = tmp_path / "inputs"
    for index, entry in enumerate(registry["checkpoints"]):
        path = root / entry["path"]
        loaded: Any
        if entry["agent_kind"] == "learned_value":
            save_checkpoint(path, create_model(seed=101 + index), metrics={"fixture": index})
            loaded = load_checkpoint(path)
        else:
            save_structured_checkpoint(
                path,
                create_structured_model(
                    CandidateMLP(seed=201 + index), projection_seed=301 + index
                ),
                metrics={},
                q0_parent_checkpoint_fingerprint="a" * 64,
                q0_parent_tensor_digest="b" * 64,
                dataset_fingerprint="c" * 64,
                source_corpus_fingerprints=("d" * 64,),
                split_identities={"train_pair_ids": "d" * 64},
                training_config={"batch_size": 1024},
                training_seeds={"structured_init": index},
                created_at="2026-09-12T00:00:00+00:00",
            )
            loaded = load_structured_checkpoint(path)
        entry["checkpoint_fingerprint"] = loaded.checkpoint_fingerprint
        entry["tensor_digest"] = loaded.manifest["tensor_digest"]
        entry["files"] = {
            item.name: _sha256(item) for item in sorted(path.iterdir()) if item.is_file()
        }
    for name, entry in registry["source_evidence"].items():
        path = root / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        content: dict[str, object] = {"fixture": name}
        for key in ("artifact_fingerprint", "plan_fingerprint"):
            if entry[key] is not None:
                content[key] = entry[key]
        path.write_text(json.dumps(content))
        entry["sha256"] = _sha256(path)
    registry_path = _write_registry(tmp_path / "fixture-registry.json", _reseal(registry))
    return registry_path, root


def test_learned_bundle_binds_checkpoint_identities_and_plays_a_cell(tmp_path: Path) -> None:
    registry_path, root = build_learned_fixture(tmp_path)
    registry = load_registry(registry_path)
    audit = verify_league_inputs(registry, root, require_archives=False)
    assert audit["status"] == "passed", audit["failures"]
    assert verify_league_inputs(registry, root, require_archives=True)["status"] == "failed"
    bundle = build_policy_bundle(registry, root, mode="learned-checkpoints")
    q0 = next(entry for entry in registry["checkpoints"] if entry["name"] == "q0-core")
    incumbent = bundle.specs[INCUMBENT_ID].config
    assert incumbent["base"]["checkpoint_fingerprint"] == q0["checkpoint_fingerprint"]  # type: ignore[index]
    m1 = bundle.specs[M_IDS[0]].config["base"]["base"]  # type: ignore[index]
    assert m1["type"] == "structured_value"
    assert bundle.specs["historical-q0"].config["type"] == "learned_value"

    design = LeagueDesign("league-test", 17, 1)
    cell = next(c for c in design.cells() if (c.left, c.right) == (CHALLENGER_ID, M_IDS[0]))
    artifact = run_league_cell(tmp_path / "out", {"plan_fingerprint": "p"}, cell, bundle.specs)
    tactical = artifact["tactical"]
    assert isinstance(tactical, dict)
    for policy in (CHALLENGER_ID, M_IDS[0]):
        assert tactical[policy]["missed_guaranteed_wins"] == 0
        assert tactical[policy]["executed_avoidable_provable_losses"] == 0
    assert (
        run_league_cell(tmp_path / "out", {"plan_fingerprint": "p"}, cell, bundle.specs) == artifact
    )


def test_checkpoint_substitution_is_rejected(tmp_path: Path) -> None:
    registry_path, root = build_learned_fixture(tmp_path)
    registry = load_registry(registry_path)
    by_name = {entry["name"]: entry for entry in registry["checkpoints"]}
    q1, q2 = root / by_name["q1"]["path"], root / by_name["q2"]["path"]
    (q1 / "weights.pt").write_bytes((q2 / "weights.pt").read_bytes())
    audit = verify_league_inputs(registry, root, require_archives=False)
    assert "checkpoint:q1" in audit["failures"]

    swapped = json.loads(registry_path.read_text())
    for entry in swapped["checkpoints"]:
        if entry["name"] == "q3":
            entry["path"] = by_name["q4"]["path"]
    with pytest.raises((LeagueError, ValueError)):
        build_policy_bundle(_reseal(swapped), root, mode="learned-checkpoints")


# ---------------------------------------------------------------------------------------------
# Holdout


def test_setup_holdout_blocks_any_prior_corpus_overlap(tmp_path: Path) -> None:
    from agent_avenue.runners import ArenaConfig, run_resumable_arena

    design = LeagueDesign("league-holdout-test", 5, 2)
    clean = scan_setup_holdout(design, roots=(tmp_path / "runs",), current_output=None)
    assert clean["status"] == "passed" and clean["overlap_count"] == 0

    family = design.family_order[1]
    arena = ArenaConfig(
        "prior-run",
        AgentSpec("a", {"type": "random", "version": "random-agent-v1"}, RandomAgent),
        AgentSpec("b", {"type": "random", "version": "random-agent-v1"}, RandomAgent),
        1,
        design.family_master_seed(family),
    )
    run_resumable_arena(
        tmp_path / "runs" / "prior" / "records", arena, generation=None, corpus_configuration={}
    )
    blocked = scan_setup_holdout(design, roots=(tmp_path / "runs",), current_output=None)
    assert blocked["status"] == "failed" and blocked["overlap_count"] == 1
    excluded = scan_setup_holdout(
        design, roots=(tmp_path / "runs",), current_output=tmp_path / "runs" / "prior"
    )
    assert excluded["overlap_count"] == 0


# ---------------------------------------------------------------------------------------------
# Aligned prefixes and tactical counts


def _cell_records(
    opponent: str, specs: Mapping[str, AgentSpec], pairs: int = 3
) -> tuple[list[Any], list[Any]]:
    design = LeagueDesign("league-prefix-test", 23, pairs)
    cells = {(cell.family, cell.left, cell.right): cell for cell in design.cells()}
    family = design.family_order[0]
    incumbent = cell_arena_config(cells[(family, INCUMBENT_ID, opponent)], specs)
    challenger = cell_arena_config(cells[(family, CHALLENGER_ID, opponent)], specs)
    return (
        [run_game(spec) for spec in schedule_arena(incumbent)],
        [run_game(spec) for spec in schedule_arena(challenger)],
    )


def test_aligned_games_share_prefixes_until_the_guaranteed_win_endpoint() -> None:
    incumbent, challenger = _cell_records(RANDOM_ID, _toy_specs(), pairs=4)
    for inc, ch in zip(incumbent, challenger, strict=True):
        row = compare_aligned_games(inc, ch)
        assert row["metadata_aligned"] and row["prefix_aligned"], row
        if row["endpoint_index"] is not None:
            assert row["terminated_in_guaranteed_win"] and row["challenger_won"]
        assert not (row["incumbent_won"] and not row["challenger_won"])


def test_prefix_audit_detects_opponent_stream_mismatch_and_divergence() -> None:
    specs = dict(_toy_specs())
    incumbent, challenger = _cell_records(RANDOM_ID, specs, pairs=1)
    first, second = incumbent[0], challenger[0]

    # Opponent stream mismatch: a different seed for the shared opponent.
    from dataclasses import replace

    opponent_index = 1 - [seat.agent_id for seat in first.seats].index(INCUMBENT_ID)
    seats = list(second.seats)
    seats[opponent_index] = replace(seats[opponent_index], seed=seats[opponent_index].seed + 1)
    row = compare_aligned_games(first, replace(second, seats=tuple(seats)))
    assert not row["metadata_aligned"]

    # Seat inversion: the b-first challenger game against the a-first incumbent game.
    assert not compare_aligned_games(first, challenger[1])["metadata_aligned"]

    # Prefix divergence: the "challenger" is secretly the heuristic under the same identity.
    specs[CHALLENGER_ID] = AgentSpec(
        CHALLENGER_ID,
        GreedyHeuristicAgent().config.to_data(),
        GreedyHeuristicAgent,
        rng_identity=Q0_CORE_RNG_IDENTITY,
    )
    _, impostor = _cell_records(RANDOM_ID, specs, pairs=3)
    rows = [compare_aligned_games(a, b) for a, b in zip(incumbent * 3, impostor, strict=False)]
    assert any(not row["prefix_aligned"] for row in rows)


def test_tactical_counts_expose_missed_wins_and_avoidable_losses_of_unshielded_play() -> None:
    raw = AgentSpec(CHALLENGER_ID, {"type": "random", "version": "random-agent-v1"}, RandomAgent)
    other = AgentSpec(RANDOM_ID, {"type": "random", "version": "random-agent-v1"}, RandomAgent)
    from agent_avenue.runners import ArenaConfig

    records = [run_game(spec) for spec in schedule_arena(ArenaConfig("t", raw, other, 30, 99))]
    counts = record_tactical_counts(records)[CHALLENGER_ID]
    assert counts["missed_guaranteed_wins"] > 0
    assert counts["executed_avoidable_provable_losses"] > 0
    assert counts["false_forced_wins"] == 0
    tactical = aggregate_tactical({"cell": {"tactical": {CHALLENGER_ID: counts, M_IDS[0]: counts}}})
    assert tactical["status"] == "failed"
    assert f"offense_envelope:{M_IDS[0]}" in tactical["nonchallenger_envelope_violations"]  # type: ignore[operator]


# ---------------------------------------------------------------------------------------------
# Statistics


def _synthetic_summaries(design: LeagueDesign, score: Any) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for cell in design.cells():
        wins = [
            score(cell.family, cell.left, cell.right, block) for block in range(cell.pair_count)
        ]
        summaries[cell.key] = {
            "key": cell.key,
            "family": cell.family,
            "left": cell.left,
            "right": cell.right,
            "left_pair_wins": wins,
        }
    return summaries


def test_anchor_bootstrap_uses_one_index_draw_for_both_arms_and_all_anchors() -> None:
    design = LeagueDesign("league-stats-test", 3, 12)

    def identical_arms(family: str, left: str, right: str, block: int) -> int:
        # Challenger and incumbent get identical, block-varying anchor outcomes.
        return (block * 7 + len(right) + (family == "family-b")) % 3

    summaries = _Summaries(design, _synthetic_summaries(design, identical_arms))
    contrasts = _anchor_contrasts(summaries)
    # Aligned resampling of identical arms has zero spread; misaligned draws would not.
    assert contrasts["macro"]["interval_95"] == [0.0, 0.0]  # type: ignore[index]
    for anchor in ANCHOR_IDS:
        assert contrasts["by_anchor"][anchor]["interval_95"] == [0.0, 0.0]  # type: ignore[index]


def test_m_family_is_nested_over_all_replicates_and_never_selects_one() -> None:
    design = LeagueDesign("league-stats-test", 4, 6)

    def score(family: str, left: str, right: str, block: int) -> int:
        if left in M_IDS or right in M_IDS:
            replicate = left if left in M_IDS else right
            value = M_IDS.index(replicate)  # replicate-specific strength 0, 1, 2
            return value if left == replicate else 2 - value
        return 1

    family = _m_family(_Summaries(design, _synthetic_summaries(design, score)))
    assert family["selects_replicate"] is False and family["excludes_m_vs_m_cells"] is True
    for row in family["by_opponent"].values():  # type: ignore[union-attr]
        assert row["point_estimate"] == pytest.approx(0.5)  # mean of 0, 0.5, 1
        low, high = row["interval_95"]
        assert low < 0.5 < high  # replicate identity resampling contributes spread


# ---------------------------------------------------------------------------------------------
# Decision rule boundaries


def _decision_inputs(*, lift_lower: float, random_lower: float, seat: float) -> dict[str, Any]:
    seats = {
        "player_one": {"combined": {"win_rate": seat}},
        "player_two": {"combined": {"win_rate": 0.9}},
    }
    matchups = {
        f"{INCUMBENT_ID}--vs--{CHALLENGER_ID}": {"seats": {CHALLENGER_ID: seats}},
        f"{CHALLENGER_ID}--vs--{HEURISTIC_ID}": {"seats": {CHALLENGER_ID: seats}},
        f"{CHALLENGER_ID}--vs--{RANDOM_ID}": {
            "seats": {CHALLENGER_ID: seats},
            "combined": {"paired_block_bootstrap_95": [random_lower, 1.0]},
        },
    }
    clean_prefix = {
        "incumbent_win_challenger_loss_blocks": 0,
        "incumbent_win_challenger_loss_games": 0,
        "metadata_failures": 0,
        "prefix_failures": 0,
        "post_intervention_termination_failures": 0,
        "challenger_failed_guaranteed_win": 0,
    }
    return {
        "statistics": {
            "artifact_fingerprint": "s",
            "anchor_contrasts": {"macro": {"interval_95": [lift_lower, 0.05]}},
            "matchups": matchups,
        },
        "tactical": {
            "artifact_fingerprint": "t",
            "status": "passed",
            "by_policy": {CHALLENGER_ID: empty_tactical(), INCUMBENT_ID: empty_tactical()},
        },
        "prefix": {"artifact_fingerprint": "p", "totals": clean_prefix},
        "integrity": {"holdout": True},
    }


@pytest.mark.parametrize(
    ("lift", "random_lower", "seat", "expected"),
    [
        (PROMOTION_LIFT_THRESHOLD + 1e-9, 0.5 + 1e-9, 0.45, DECISION_PROMOTE),
        (PROMOTION_LIFT_THRESHOLD, 0.6, 0.6, DECISION_RETAIN),  # strict lift bound
        (0.01, 0.5, 0.6, DECISION_RETAIN),  # strict random bound
        (0.01, 0.6, 0.45 - 1e-9, DECISION_RETAIN),  # inclusive seat floor
    ],
)
def test_promotion_thresholds_are_strict_where_frozen(
    lift: float, random_lower: float, seat: float, expected: str
) -> None:
    inputs = _decision_inputs(lift_lower=lift, random_lower=random_lower, seat=seat)
    assert promotion_decision(**inputs)["decision"] == expected


def test_any_integrity_failure_blocks_rather_than_retains() -> None:
    inputs = _decision_inputs(lift_lower=0.01, random_lower=0.6, seat=0.6)
    inputs["integrity"] = {"holdout": True, "within_claim_cutoff": False}
    assert promotion_decision(**inputs)["decision"] == DECISION_BLOCKED
    inputs = _decision_inputs(lift_lower=0.01, random_lower=0.6, seat=0.6)
    inputs["tactical"]["status"] = "failed"
    assert promotion_decision(**inputs)["decision"] == DECISION_BLOCKED


def test_regression_block_or_challenger_miss_retains() -> None:
    inputs = _decision_inputs(lift_lower=0.01, random_lower=0.6, seat=0.6)
    inputs["prefix"]["totals"]["incumbent_win_challenger_loss_games"] = 1
    inputs["prefix"]["totals"]["incumbent_win_challenger_loss_blocks"] = 1
    assert promotion_decision(**inputs)["decision"] == DECISION_RETAIN
    inputs = _decision_inputs(lift_lower=0.01, random_lower=0.6, seat=0.6)
    inputs["tactical"]["by_policy"][CHALLENGER_ID]["missed_guaranteed_wins"] = 1
    assert promotion_decision(**inputs)["decision"] == DECISION_RETAIN


def test_game_spec_seat_orders_are_physical_seats() -> None:
    specs = _toy_specs()
    cell = claim_design().cells()[0]
    spec: GameSpec = next(iter(schedule_arena(cell_arena_config(cell, specs))))
    assert tuple(agent.agent_id for agent in spec.seats) == (INCUMBENT_ID, CHALLENGER_ID)
    assert tuple(PlayerId) == (PlayerId.PLAYER_ONE, PlayerId.PLAYER_TWO)

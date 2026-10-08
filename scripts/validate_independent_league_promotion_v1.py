#!/usr/bin/env python3
"""Independently validate retained Step-5 independent-league evidence.

This validator deliberately imports none of ``agent_avenue.runners.league``. It reads roles and
schedule declarations from the committed registry, reconstructs seeds and cells itself, replays
every semantic record, classifies guaranteed wins with the independent public oracle, recomputes
all summaries, bootstraps, prefix audits, and the promotion rule, and verifies checksum scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from agent_avenue.agents import (
    FALLBACK_VERSION,
    RESOLUTION_SCOPE,
    RNG_ALGORITHM,
    SEED_DERIVATION,
    UNCERTAINTY_VERSION,
    DeterministicRandom,
    derive_seed,
    filter_terminal_actions,
)
from agent_avenue.engine import GameState, Phase, PlayerId, PlayOfferAction, apply_action, new_game
from agent_avenue.engine.model import player_index
from agent_avenue.engine.setup import normalize_config
from agent_avenue.engine.terminal import TERMINAL_EVALUATOR_VERSION
from agent_avenue.observation import observe
from agent_avenue.runners.public_win_oracle import independent_public_forced_win_oracle
from agent_avenue.storage import (
    GameRecord,
    code_fingerprint,
    inspect_source_identity,
    load_corpus,
    rules_fingerprint,
)

VALIDATION_VERSION = "m7-independent-league-validation-v1"
CYCLE = "m7-independent-league-promotion-v1"
COMMITTED_REGISTRY = Path("research/cycles/m7-independent-league-inputs.json")
COMMITTED_REGISTRY_FINGERPRINT = "df238b7d9c948563ee15e4e2cfc59fe1525a9721b10e24362a7d251c1dfb7d48"
SMOKE_NAMESPACE = f"{CYCLE}:nonclaim-smoke"
SMOKE_ROOT_DOMAIN = f"{CYCLE}:nonclaim-smoke-root:v1"
CLAIM_PAIRS = 200
CLAIM_GAMES = 52_800
RESAMPLES = 20_000
LOWER = 499
UPPER = 19_499
ARENA_PAIRED_DOMAIN = "arena:paired-bootstrap:agent-a-win-rate:v1"
LIFT_THRESHOLD = 0.0025
RANDOM_THRESHOLD = 0.5
SEAT_THRESHOLD = 0.45
CUTOFF_SECONDS = 465 * 60
HARD_LIMIT_SECONDS = 480 * 60
WILSON_Z = 1.959963984540054
SAFETY_EXTRAS = {
    "fallback": FALLBACK_VERSION,
    "public_uncertainty": UNCERTAINTY_VERSION,
    "resolution_scope": RESOLUTION_SCOPE,
    "terminal_evaluator": TERMINAL_EVALUATOR_VERSION,
}
ARCHIVES = {
    "terminal_safety": "artifacts/archive/terminal-safety-v1-artifacts-2026-09-03.tar.gz",
    "terminal_offense": "artifacts/archive/m7-terminal-offense-confirm-v1-2026-09-11.tar.gz",
    "structured_v2": "artifacts/archive/m7-structured-model-v2-2026-09-12.tar.gz",
    "rollout_step4": "artifacts/archive/m7-counterfactual-rollout-supervision-v1-2026-09-17.tar.gz",
}
NOT_CHECKSUMMED = {
    "checksums.json",
    "execution-state.json",
    "validation.json",
    "validator-runtime.json",
    "driver.stdout",
    "driver.stderr",
    "validation.stdout",
    "validation.stderr",
    "wrapper-exit.json",
    ".lock",
}
TOP_LEVEL_ARTIFACTS = {
    "plan.json",
    "input-registry.json",
    "input-audit.json",
    "source.json",
    "schedule.json",
    "holdout.json",
    "prefix-audit.json",
    "tactical.json",
    "statistics.json",
    "promotion-decision.json",
    "result.json",
    "runtime.json",
    "checksums.json",
    "execution-state.json",
}
TACTICAL_KEYS = (
    "decisions",
    "guaranteed_win_opportunities",
    "guaranteed_win_conversions",
    "missed_guaranteed_wins",
    "false_forced_wins",
    "executed_provable_losses",
    "executed_avoidable_provable_losses",
)


class ValidationFailure(RuntimeError):
    """A retained artifact does not reproduce independently."""


# ---------------------------------------------------------------------------------------------
# Primitive helpers (validator-local)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationFailure(f"unreadable artifact: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValidationFailure(f"artifact is not an object: {path.name}")
    return value


def _self_sealed(value: Mapping[str, Any]) -> bool:
    return value.get("artifact_fingerprint") == _digest(
        {key: item for key, item in value.items() if key != "artifact_fingerprint"}
    )


def _same(retained: object, recomputed: object, where: str, problems: list[str]) -> None:
    """Strict structural equality with a tight tolerance only for non-integer floats."""
    if isinstance(recomputed, Mapping):
        if not isinstance(retained, Mapping) or set(retained) != set(recomputed):
            problems.append(f"{where}: keys differ")
            return
        for key in recomputed:
            _same(retained[key], recomputed[key], f"{where}.{key}", problems)
    elif isinstance(recomputed, list | tuple):
        if not isinstance(retained, list) or len(retained) != len(recomputed):
            problems.append(f"{where}: sequence differs")
            return
        for index, (left, right) in enumerate(zip(retained, recomputed, strict=True)):
            _same(left, right, f"{where}[{index}]", problems)
    elif isinstance(recomputed, float) and not isinstance(recomputed, bool):
        if (
            not isinstance(retained, int | float)
            or isinstance(retained, bool)
            or not math.isclose(float(retained), recomputed, rel_tol=0.0, abs_tol=1e-12)
        ):
            problems.append(f"{where}: {retained!r} != {recomputed!r}")
    elif type(retained) is not type(recomputed) or retained != recomputed:
        problems.append(f"{where}: {retained!r} != {recomputed!r}")


def _wilson(wins: int, games: int) -> list[float]:
    p = wins / games
    z2 = WILSON_Z * WILSON_Z
    scale = 1 + z2 / games
    middle = (p + z2 / (2 * games)) / scale
    half = WILSON_Z * math.sqrt(p * (1 - p) / games + z2 / (4 * games * games)) / scale
    return [max(0.0, middle - half), min(1.0, middle + half)]


def _percentiles(samples: list[int], denominator: int) -> list[float]:
    ordered = sorted(samples)
    return [ordered[LOWER] / denominator, ordered[UPPER] / denominator]


# ---------------------------------------------------------------------------------------------
# Registry roles, design, and schedule


class Roles:
    """Policy roles read from the registry rather than from production constants."""

    def __init__(self, registry: Mapping[str, Any]) -> None:
        policies = registry["policies"]
        self.order: tuple[str, ...] = tuple(policy["id"] for policy in policies)

        def ids(role: str) -> tuple[str, ...]:
            return tuple(policy["id"] for policy in policies if policy["role"] == role)

        incumbent = ids("incumbent-promotion-reference")
        challenger = ids("sole-promotion-eligible-challenger")
        if len(incumbent) != 1 or len(challenger) != 1:
            raise ValidationFailure("registry must declare one incumbent and one challenger")
        self.incumbent, self.challenger = incumbent[0], challenger[0]
        self.descendants = ids("historical-descendant-descriptive")
        self.replicates = ids("structured-replicate-family-descriptive")
        self.anchors = ids("anchor-descriptive")
        eligible = [policy["id"] for policy in policies if policy["promotion_eligible"] is True]
        if eligible != [self.challenger] or len(self.replicates) != 3 or len(self.anchors) != 3:
            raise ValidationFailure("registry roles do not match the frozen agreement")
        self.random = next(
            policy["id"] for policy in policies if policy["config"].get("type") == "random"
        )
        self.heuristic = next(
            policy["id"]
            for policy in policies
            if policy["config"].get("type") == "greedy_heuristic"
        )
        self.rng = {policy["id"]: policy["rng_identity"] for policy in policies}
        self.declared = {policy["id"]: policy["config"] for policy in policies}
        # Incumbent/challenger failures are promotion conditions; the rest are integrity failures.
        self.integrity_enveloped = (*self.descendants, *self.replicates)


class Design:
    def __init__(
        self, namespace: str, root: int, pairs: int, roles: Roles, families: Sequence[str]
    ):
        self.namespace = namespace
        self.root = root
        self.pairs = pairs
        self.roles = roles
        self.families = tuple(families)

    def master(self, family: str) -> int:
        return derive_seed(self.root, f"{self.namespace}:seed-family:{family}") & ((1 << 63) - 1)

    def setups(self, family: str) -> list[int]:
        master = self.master(family)
        return [
            derive_seed(master, f"arena:pair:{block}:setup") & ((1 << 64) - 1)
            for block in range(self.pairs)
        ]

    def cells(self) -> list[tuple[str, str, str]]:
        order = self.roles.order
        return [
            (family, order[i], order[j])
            for family in self.families
            for i in range(len(order))
            for j in range(i + 1, len(order))
        ]

    def boot_rng(self, domain: str) -> DeterministicRandom:
        name = f"{self.namespace}:{domain}"
        return DeterministicRandom(derive_seed(self.root, name), name)

    def boot_seed(self, domain: str) -> int:
        return derive_seed(self.root, f"{self.namespace}:{domain}")


def _cell_key(family: str, left: str, right: str) -> str:
    return f"{family}:{left}--vs--{right}"


def _setup_fp(seeds: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps(list(seeds), separators=(",", ":")).encode()).hexdigest()


def check_registry_schedule(registry: Mapping[str, Any], design: Design) -> list[str]:
    """Rebuild the claim schedule and compare every registry field."""
    problems: list[str] = []
    schedule = registry["schedule"]
    if schedule["policy_order"] != list(design.roles.order):
        problems.append("registry policy order")
    if schedule["family_order"] != list(design.families):
        problems.append("registry family order")
    if schedule["seed_derivation"] != SEED_DERIVATION or schedule["rng_algorithm"] != RNG_ALGORITHM:
        problems.append("registry RNG declaration")
    seen: set[int] = set()
    for family in design.families:
        seeds = design.setups(family)
        if len(set(seeds)) != len(seeds) or seen & set(seeds):
            problems.append(f"setup overlap in {family}")
        seen |= set(seeds)
        if schedule["families"].get(family) != {
            "master_seed": design.master(family),
            "pair_count": design.pairs,
            "seed_domain": f"{design.namespace}:seed-family:{family}",
            "setup_seed_fingerprint": _setup_fp(seeds),
        }:
            problems.append(f"registry family {family}")
    expected_cells = [
        {
            "family": family,
            "games": 2 * design.pairs,
            "key": _cell_key(family, left, right),
            "left": left,
            "pair_count": design.pairs,
            "right": right,
            "run_id": f"{design.namespace}:{family}:{left}:vs:{right}",
        }
        for family, left, right in design.cells()
    ]
    if schedule["cells"] != expected_cells:
        problems.append("registry cells")
    total = sum(cell["games"] for cell in expected_cells)
    if schedule["total_games"] != total or schedule["total_cells"] != len(expected_cells):
        problems.append("registry totals")
    if total != CLAIM_GAMES or len(expected_cells) != 132:
        problems.append("frozen 132-cell / 52,800-game cardinality")
    return problems


def expected_schedule(design: Design) -> dict[str, Any]:
    """Independently rebuild the plan's serialized schedule for one design."""
    families: dict[str, Any] = {}
    every: list[int] = []
    for family in design.families:
        seeds = design.setups(family)
        every += seeds
        families[family] = {
            "seed_domain": f"{design.namespace}:seed-family:{family}",
            "master_seed": design.master(family),
            "pair_count": design.pairs,
            "setup_seed_fingerprint": _setup_fp(seeds),
            "unique_setups": len(set(seeds)) == len(seeds),
        }
    cells = [
        {
            "family": family,
            "games": 2 * design.pairs,
            "key": _cell_key(family, left, right),
            "left": left,
            "pair_count": design.pairs,
            "right": right,
            "run_id": f"{design.namespace}:{family}:{left}:vs:{right}",
        }
        for family, left, right in design.cells()
    ]
    return {
        "design": {
            "namespace": design.namespace,
            "root_seed": design.root,
            "pair_count": design.pairs,
            "policy_order": list(design.roles.order),
            "family_order": list(design.families),
            "claim_design": design.namespace == CYCLE and design.pairs == CLAIM_PAIRS,
        },
        "families": families,
        "families_disjoint": len(set(every)) == len(every),
        "cells": cells,
        "total_cells": len(cells),
        "total_games": sum(cell["games"] for cell in cells),
        "seed_derivation": SEED_DERIVATION,
        "rng_algorithm": RNG_ALGORITHM,
    }


# ---------------------------------------------------------------------------------------------
# Declarations and inputs


def realizes(
    declared: Mapping[str, Any],
    runtime: Mapping[str, Any],
    checkpoint: Callable[[str, Mapping[str, Any]], bool],
) -> bool:
    """Independent check that one runtime config realizes a registry declaration."""
    if declared.get("type") == "checkpoint_ref":
        return set(declared) == {"name", "type"} and checkpoint(declared["name"], runtime)
    for key, expected in declared.items():
        if key not in runtime:
            return False
        actual = runtime[key]
        if isinstance(expected, dict):
            if not isinstance(actual, dict) or not realizes(expected, actual, checkpoint):
                return False
        elif actual != expected:
            return False
    extra = {key: runtime[key] for key in runtime if key not in declared}
    return not extra or (declared.get("type") == "terminal_safety" and extra == SAFETY_EXTRAS)


def _checkpoint_check(
    registry: Mapping[str, Any], mode: str
) -> Callable[[str, Mapping[str, Any]], bool]:
    entries = {entry["name"]: entry for entry in registry["checkpoints"]}

    def learned(name: str, runtime: Mapping[str, Any]) -> bool:
        entry = entries.get(name)
        return bool(entry) and all(
            runtime.get(key) == entry[source]
            for key, source in (
                ("type", "agent_kind"),
                ("checkpoint_fingerprint", "checkpoint_fingerprint"),
                ("tensor_digest", "tensor_digest"),
                ("encoder_version", "encoder_version"),
            )
        )

    def toy(name: str, runtime: Mapping[str, Any]) -> bool:
        return name in entries and dict(runtime) == {"type": "random", "version": "random-agent-v1"}

    return learned if mode == "learned-checkpoints" else toy


def check_inputs(registry: Mapping[str, Any], root: Path, *, archives_required: bool) -> list[str]:
    problems: list[str] = []
    for entry in registry["checkpoints"]:
        directory = root / entry["path"]
        if not directory.is_dir():
            problems.append(f"checkpoint missing: {entry['name']}")
            continue
        actual = {item.name: _file_digest(item) for item in directory.iterdir() if item.is_file()}
        if actual != dict(entry["files"]):
            problems.append(f"checkpoint files differ: {entry['name']}")
    for name, entry in registry["source_evidence"].items():
        path = root / entry["path"]
        if not path.is_file() or _file_digest(path) != entry["sha256"]:
            problems.append(f"source evidence differs: {name}")
            continue
        content = _load(path)
        # Results from Steps 1, 3, and 4 seal themselves under result_fingerprint.
        seal = content.get("result_fingerprint", content.get("artifact_fingerprint"))
        if entry["artifact_fingerprint"] is not None and seal != entry["artifact_fingerprint"]:
            problems.append(f"source evidence seal differs: {name}")
        if entry["plan_fingerprint"] is not None and (
            content.get("plan_fingerprint") != entry["plan_fingerprint"]
        ):
            problems.append(f"source evidence plan_fingerprint differs: {name}")
    for name, digest in registry["archive_sha256"].items():
        path = root / ARCHIVES[name]
        if path.is_file():
            if _file_digest(path) != digest:
                problems.append(f"archive differs: {name}")
        elif archives_required:
            problems.append(f"archive missing: {name}")
    return problems


# ---------------------------------------------------------------------------------------------
# Holdout


def _setup_key(config: Mapping[str, Any], seed: int) -> str:
    return _digest({"game_config": dict(config), "setup_seed": seed})


def independent_holdout(design: Design, roots: Sequence[Path], output: Path) -> dict[str, Any]:
    from agent_avenue.engine import GameConfig

    config = normalize_config(GameConfig())
    proposed = {
        _setup_key(config, seed) for family in design.families for seed in design.setups(family)
    }
    prior: set[str] = set()
    corpora = 0
    fingerprints: dict[str, str] = {}
    excluded = output.resolve()
    for declared in roots:
        root = declared.resolve()
        if not root.is_dir():
            continue
        for manifest in sorted(root.rglob("manifest.json")):
            if manifest.resolve().is_relative_to(excluded):
                continue
            try:
                declared_corpus = "records_file" in json.loads(manifest.read_text())
            except (OSError, json.JSONDecodeError):
                declared_corpus = False
            if not (manifest.parent / "games.jsonl.gz").exists() and not declared_corpus:
                continue
            loaded, records = load_corpus(manifest.parent, verify_code=False, verify_replays=False)
            corpora += 1
            fingerprints[str(manifest.parent)] = loaded.corpus_fingerprint
            prior |= {_setup_key(normalize_config(r.replay.config), r.replay.seed) for r in records}
    return {
        "proposed_setup_count": len(proposed),
        "proposed_setup_fingerprint": _digest(sorted(proposed)),
        "prior_unique_setup_count": len(prior),
        "prior_setup_fingerprint": _digest(sorted(prior)),
        "corpus_count": corpora,
        "corpus_fingerprints": fingerprints,
        "overlap_count": len(proposed & prior),
    }


# ---------------------------------------------------------------------------------------------
# Per-cell schedule, summary, and tactical recomputation


def check_cell_schedule(
    design: Design,
    family: str,
    left: str,
    right: str,
    records: Sequence[GameRecord],
    runtime_configs: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    problems: list[str] = []
    if len(records) != 2 * design.pairs:
        return ["record count"]
    master = design.master(family)
    setups = design.setups(family)
    run_id = f"{design.namespace}:{family}:{left}:vs:{right}"
    for block in range(design.pairs):
        pair_id = f"pair-{block:06d}"
        for offset, order in ((0, (left, right)), (1, (right, left))):
            record = records[2 * block + offset]
            suffix = "a-first" if offset == 0 else "b-first"
            if (
                record.run_id != run_id
                or record.pair_id != pair_id
                or record.game_id != f"{pair_id}-{suffix}"
                or record.replay.seed != setups[block]
            ):
                problems.append(f"setup/schedule identity at {record.game_id}")
            if tuple(seat.agent_id for seat in record.seats) != order:
                problems.append(f"seat order at {record.game_id}")
                continue
            for position, seat in enumerate(record.seats):
                identity = design.roles.rng[seat.agent_id]
                if (
                    seat.player is not tuple(PlayerId)[position]
                    or seat.seed != derive_seed(master, f"arena:pair:{block}:agent:{identity}")
                    or seat.rng_domain != f"agent:{identity}"
                    or seat.rng_algorithm != RNG_ALGORITHM
                    or json.loads(json.dumps(dict(seat.config), sort_keys=True))
                    != runtime_configs[seat.agent_id]
                ):
                    problems.append(f"seat RNG/config at {record.game_id}:{seat.agent_id}")
    return problems


def summarize_cell(left: str, right: str, records: Sequence[GameRecord]) -> dict[str, Any]:
    pair_scores: list[int] = []
    seat_rows = {
        policy: {player.value: {"games": 0, "wins": 0} for player in PlayerId}
        for policy in (left, right)
    }
    margin = turns = decisions = 0
    endings: Counter[str] = Counter()
    for start in range(0, len(records), 2):
        score = 0
        for record in records[start : start + 2]:
            won = player_index(record.winner)
            for position, seat in enumerate(record.seats):
                seat_rows[seat.agent_id][seat.player.value]["games"] += 1
                seat_rows[seat.agent_id][seat.player.value]["wins"] += int(position == won)
                if seat.agent_id == left:
                    score += int(position == won)
                    margin += record.final_scores[position] - record.final_scores[1 - position]
            turns += record.turn_count
            decisions += record.decision_count
            endings[record.terminal_reason] += 1
        pair_scores.append(score)
    return {
        "left": left,
        "right": right,
        "pair_scores": pair_scores,
        "seats": seat_rows,
        "margin": margin,
        "turns": turns,
        "decisions": decisions,
        "endings": dict(sorted(endings.items())),
    }


def compare_summary(
    retained: Mapping[str, Any], independent: Mapping[str, Any], key: str, problems: list[str]
) -> None:
    family = key.split(":", 1)[0]
    expected = {
        "key": key,
        "family": family,
        "left": independent["left"],
        "right": independent["right"],
        "pair_count": len(independent["pair_scores"]),
        "games": 2 * len(independent["pair_scores"]),
        "left_pair_wins": independent["pair_scores"],
        "left_wins": sum(independent["pair_scores"]),
        "seats": independent["seats"],
        "left_score_margin_total": independent["margin"],
        "turns_total": independent["turns"],
        "decisions_total": independent["decisions"],
        "terminal_reasons": independent["endings"],
    }
    _same(retained, expected, f"{key}.summary", problems)


def _block_multiplicities(rng: DeterministicRandom, blocks: int, strata: int) -> Any:
    """Yield, for each of the RESAMPLES resamples, one index-multiplicity Counter per stratum.

    Draws are consumed in the stream order of sequential per-stratum ``randbelow(blocks)`` calls;
    they are fetched 250 resamples at a time because every stratum shares one bound.
    """
    width = blocks * strata
    produced = 0
    while produced < RESAMPLES:
        take = min(250, RESAMPLES - produced)
        flat = rng.randbelow_batch(blocks, width * take)
        for offset in range(0, width * take, width):
            yield [
                Counter(flat[offset + stratum * blocks : offset + (stratum + 1) * blocks])
                for stratum in range(strata)
            ]
        produced += take


def arena_interval(pair_scores: Sequence[int], master: int) -> tuple[list[float], int]:
    seed = derive_seed(master, ARENA_PAIRED_DOMAIN)
    rng = DeterministicRandom(seed, ARENA_PAIRED_DOMAIN)
    n = len(pair_scores)
    sums = [
        sum(pair_scores[index] * count for index, count in drawn.items())
        for (drawn,) in _block_multiplicities(rng, n, 1)
    ]
    return _percentiles(sums, 2 * n), seed


def _resolution(state: GameState, actions: Sequence[Any], index: int) -> GameState:
    after = apply_action(state, actions[index])
    if isinstance(actions[index], PlayOfferAction) and after.phase is not Phase.TERMINAL:
        after = apply_action(after, actions[index + 1])
    return after


def tactical_counts(records: Sequence[GameRecord]) -> dict[str, dict[str, int]]:
    """Independent oracle replay of guaranteed-win and provable-loss behavior."""
    rows: dict[str, Counter[str]] = {}
    for record in records:
        state = new_game(record.replay.config, record.replay.seed)
        actions = record.replay.actions
        for index, action in enumerate(actions):
            actor = (
                state.active_player if state.phase is Phase.PLAY else state.active_player.other()
            )
            observation = observe(state, actor)
            oracle = independent_public_forced_win_oracle(
                observation,
                observation.legal_actions,
                authoritative_state=state,
                exact_play_actions=(action,) if isinstance(action, PlayOfferAction) else (),
            )
            shield = filter_terminal_actions(observation, observation.legal_actions)
            after = _resolution(state, actions, index)
            immediate = (
                after.phase is Phase.TERMINAL
                and after.outcome is not None
                and (after.outcome.winner is actor)
            )
            hit = action in oracle.forced_win_actions
            loss = action in shield.provable_loss_actions
            row = rows.setdefault(record.seats[player_index(actor)].agent_id, Counter())
            row["decisions"] += 1
            row["guaranteed_win_opportunities"] += int(bool(oracle.forced_win_actions))
            row["guaranteed_win_conversions"] += int(hit)
            row["missed_guaranteed_wins"] += int(bool(oracle.forced_win_actions) and not hit)
            row["false_forced_wins"] += int(hit and not immediate)
            row["executed_provable_losses"] += int(loss)
            row["executed_avoidable_provable_losses"] += int(
                loss and not shield.forced_loss_fallback
            )
            state = apply_action(state, action)
    return {
        agent: {key: counts[key] for key in TACTICAL_KEYS} for agent, counts in sorted(rows.items())
    }


# ---------------------------------------------------------------------------------------------
# Aligned prefix audit


def aligned_pair(roles: Roles, inc: GameRecord, ch: GameRecord) -> dict[str, Any]:
    """Independent prefix/endpoint check for one aligned incumbent/challenger game pair."""
    seat = [s.agent_id for s in inc.seats].index(roles.incumbent)
    opp = 1 - seat
    row = {
        "metadata": False,
        "prefix": False,
        "endpoint": False,
        "classification": None,
        "terminated": False,
        "incumbent_won": player_index(inc.winner) == seat,
        "challenger_won": player_index(ch.winner) == seat,
    }
    aligned = (
        ch.seats[seat].agent_id == roles.challenger
        and (inc.game_id, inc.pair_id, inc.replay.seed, inc.replay.config)
        == (ch.game_id, ch.pair_id, ch.replay.seed, ch.replay.config)
        and inc.seats[seat].seed == ch.seats[seat].seed
        and inc.seats[seat].rng_domain == ch.seats[seat].rng_domain
        and roles.rng[roles.incumbent] == roles.rng[roles.challenger]
        and inc.seats[seat].rng_domain == f"agent:{roles.rng[roles.incumbent]}"
        and (inc.seats[opp].agent_id, inc.seats[opp].seed, inc.seats[opp].rng_domain)
        == (ch.seats[opp].agent_id, ch.seats[opp].seed, ch.seats[opp].rng_domain)
        and dict(inc.seats[opp].config) == dict(ch.seats[opp].config)
        and inc.seats[opp].player is ch.seats[opp].player
    )
    if not aligned:
        return row
    row["metadata"] = True
    a, b = inc.replay.actions, ch.replay.actions
    state = new_game(inc.replay.config, inc.replay.seed)
    for index in range(min(len(a), len(b))):
        actor = state.active_player if state.phase is Phase.PLAY else state.active_player.other()
        if player_index(actor) == seat:
            observation = observe(state, actor)
            forced = independent_public_forced_win_oracle(
                observation, observation.legal_actions, authoritative_state=state
            ).forced_win_actions
            if forced:
                ch_hit, inc_hit = b[index] in forced, a[index] in forced
                after = apply_action(state, b[index])
                used = index + 1
                if (
                    isinstance(b[index], PlayOfferAction)
                    and after.phase is not Phase.TERMINAL
                    and used < len(b)
                ):
                    after = apply_action(after, b[used])
                    used += 1
                row.update(
                    prefix=True,
                    endpoint=True,
                    classification=(
                        "both_converted"
                        if ch_hit and inc_hit
                        else "incumbent_missed_challenger_converted"
                        if ch_hit
                        else "challenger_failed_guaranteed_win"
                    ),
                    terminated=ch_hit
                    and after.phase is Phase.TERMINAL
                    and after.outcome is not None
                    and player_index(after.outcome.winner) == seat
                    and used == len(b),
                )
                return row
        if a[index] != b[index]:
            return row
        state = apply_action(state, a[index])
    row["prefix"] = len(a) == len(b) and inc.winner == ch.winner
    return row


def prefix_totals(
    rows: Sequence[Mapping[str, Any]], pair_ids: Sequence[str | None]
) -> dict[str, int]:
    bad_blocks = {
        pair
        for row, pair in zip(rows, pair_ids, strict=True)
        if row["incumbent_won"] and not row["challenger_won"]
    }
    return {
        "aligned_games": len(rows),
        "metadata_failures": sum(not row["metadata"] for row in rows),
        "prefix_failures": sum(not row["prefix"] for row in rows),
        "endpoints": sum(bool(row["endpoint"]) for row in rows),
        "both_converted": sum(row["classification"] == "both_converted" for row in rows),
        "incumbent_missed_challenger_converted": sum(
            row["classification"] == "incumbent_missed_challenger_converted" for row in rows
        ),
        "challenger_failed_guaranteed_win": sum(
            row["classification"] == "challenger_failed_guaranteed_win" for row in rows
        ),
        "post_intervention_termination_failures": sum(
            bool(row["endpoint"]) and not row["terminated"] for row in rows
        ),
        "games_without_endpoint": sum(row["metadata"] and not row["endpoint"] for row in rows),
        "incumbent_win_challenger_loss_games": sum(
            row["incumbent_won"] and not row["challenger_won"] for row in rows
        ),
        "incumbent_win_challenger_loss_blocks": len(bad_blocks),
        "challenger_win_incumbent_loss_games": sum(
            row["challenger_won"] and not row["incumbent_won"] for row in rows
        ),
    }


# ---------------------------------------------------------------------------------------------
# Statistics


class Scores:
    def __init__(self, design: Design, summaries: Mapping[str, Mapping[str, Any]]):
        self.design = design
        self.summaries = summaries
        self.position = {policy: index for index, policy in enumerate(design.roles.order)}

    def cell(self, family: str, a: str, b: str) -> Mapping[str, Any]:
        left, right = (a, b) if self.position[a] < self.position[b] else (b, a)
        return self.summaries[_cell_key(family, left, right)]

    def blocks(self, policy: str, opponent: str, family: str) -> list[int]:
        cell = self.cell(family, policy, opponent)
        scores = cell["pair_scores"]
        return list(scores) if cell["left"] == policy else [2 - value for value in scores]


def _seat_block(scores: Scores, policy: str, opponent: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for seat in sorted(player.value for player in PlayerId):
        per_family = {
            family: scores.cell(family, policy, opponent)["seats"][policy][seat]
            for family in scores.design.families
        }
        games = sum(row["games"] for row in per_family.values())
        wins = sum(row["wins"] for row in per_family.values())

        def entry(w: int, g: int) -> dict[str, Any]:
            return {"games": g, "wins": w, "win_rate": w / g, "wilson_95": _wilson(w, g)}

        out[seat] = {
            "combined": entry(wins, games),
            "by_family": {
                family: entry(row["wins"], row["games"]) for family, row in per_family.items()
            },
        }
    return out


def independent_statistics(
    design: Design,
    summaries: Mapping[str, Mapping[str, Any]],
    family_intervals: Mapping[str, tuple[list[float], int]],
    elapsed: Mapping[str, float],
    cell_tactical: Mapping[str, Mapping[str, Mapping[str, int]]],
    configs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    roles = design.roles
    scores = Scores(design, summaries)
    per_matchup_games = 2 * design.pairs * len(design.families)
    matchups: dict[str, Any] = {}
    wins_table: dict[str, dict[str, int]] = {policy: {} for policy in roles.order}
    for i, left in enumerate(roles.order):
        for right in roles.order[i + 1 :]:
            per_family = {family: scores.blocks(left, right, family) for family in design.families}
            total = sum(sum(values) for values in per_family.values())
            label = f"matchup-family-stratified-bootstrap:v1:{left}--vs--{right}"
            rng = design.boot_rng(label)
            samples = [
                sum(
                    per_family[family][index] * count
                    for family, drawn in zip(design.families, strata, strict=True)
                    for index, count in drawn.items()
                )
                for strata in _block_multiplicities(rng, design.pairs, len(design.families))
            ]
            margin = turns = decisions = 0
            endings: Counter[str] = Counter()
            seconds = 0.0
            by_family: dict[str, Any] = {}
            for family in design.families:
                cell = scores.cell(family, left, right)
                key = _cell_key(family, cell["left"], cell["right"])
                interval, seed = family_intervals[key]
                own = sum(per_family[family])
                by_family[family] = {
                    "left_wins": own,
                    "games": 2 * design.pairs,
                    "left_win_rate": own / (2 * design.pairs),
                    "paired_bootstrap_95": interval,
                    "paired_bootstrap_seed": seed,
                }
                margin += cell["margin"] if cell["left"] == left else -cell["margin"]
                turns += cell["turns"]
                decisions += cell["decisions"]
                endings.update(cell["endings"])
                seconds += elapsed[key]
            wins_table[left][right] = total
            wins_table[right][left] = per_matchup_games - total
            matchups[f"{left}--vs--{right}"] = {
                "left": left,
                "right": right,
                "combined": {
                    "left_wins": total,
                    "games": per_matchup_games,
                    "left_win_rate": total / per_matchup_games,
                    "paired_block_bootstrap_95": _percentiles(samples, per_matchup_games),
                    "bootstrap": {
                        "method": "family-stratified-paired-block-percentile-v1",
                        "resamples": RESAMPLES,
                        "order_statistic_indices": [LOWER, UPPER],
                        "seed": design.boot_seed(label),
                    },
                },
                "by_family": by_family,
                "seats": {
                    left: _seat_block(scores, left, right),
                    right: _seat_block(scores, right, left),
                },
                "left_average_score_margin": margin / per_matchup_games,
                "average_turns": turns / per_matchup_games,
                "average_decisions": decisions / per_matchup_games,
                "terminal_reasons": dict(sorted(endings.items())),
                "arena_elapsed_seconds": seconds,
                "games_per_second": per_matchup_games / seconds if seconds > 0 else None,
                "tactical": {
                    policy: {
                        name: sum(
                            cell_tactical[_cell_key(family, left, right)][policy][name]
                            for family in design.families
                        )
                        for name in TACTICAL_KEYS
                    }
                    for policy in (left, right)
                },
                "config_fingerprints": {
                    policy: _digest(configs[policy]) for policy in (left, right)
                },
            }
    matrix = {
        policy: {other: wins / per_matchup_games for other, wins in row.items()}
        for policy, row in wins_table.items()
    }
    policies: dict[str, Any] = {}
    for policy in roles.order:
        others = [other for other in roles.order if other != policy]
        lowest = min(wins_table[policy][other] for other in others)
        worst = next(other for other in others if wins_table[policy][other] == lowest)
        numerator = sum(wins_table[policy][other] for other in others)
        policies[policy] = {
            "equal_opponent_macro": numerator / (per_matchup_games * len(others)),
            "equal_opponent_macro_numerator": numerator,
            "worst_opponent": worst,
            "worst_opponent_win_rate": matrix[policy][worst],
            "copeland_descriptive": sum(
                2 * wins_table[policy][other] > per_matchup_games for other in others
            ),
        }
    return {
        "design": {
            "namespace": design.namespace,
            "root_seed": design.root,
            "pair_count": design.pairs,
            "policy_order": list(roles.order),
            "family_order": list(design.families),
            "claim_design": design.namespace == CYCLE and design.pairs == CLAIM_PAIRS,
        },
        "matchups": matchups,
        "matrix": matrix,
        "policies": policies,
        "scalar_rankings_descriptive_only": True,
        "m_family": _nested_m(design, scores),
        "anchor_contrasts": _anchors(design, scores),
    }


def _anchors(design: Design, scores: Scores) -> dict[str, Any]:
    roles = design.roles
    deltas = {
        anchor: {
            family: [
                c - i
                for c, i in zip(
                    scores.blocks(roles.challenger, anchor, family),
                    scores.blocks(roles.incumbent, anchor, family),
                    strict=True,
                )
            ]
            for family in design.families
        }
        for anchor in roles.anchors
    }
    n_total = design.pairs * len(design.families)
    denominator = 2 * n_total
    label = "anchor-joint-family-stratified-bootstrap:v1"
    rng = design.boot_rng(label)
    per_anchor: dict[str, list[int]] = {anchor: [] for anchor in roles.anchors}
    macro: list[int] = []
    for strata in _block_multiplicities(rng, design.pairs, len(design.families)):
        counts = dict(zip(design.families, strata, strict=True))
        values = {
            anchor: sum(
                deltas[anchor][family][index] * multiplicity
                for family in design.families
                for index, multiplicity in counts[family].items()
            )
            for anchor in roles.anchors
        }
        for anchor in roles.anchors:
            per_anchor[anchor].append(values[anchor])
        macro.append(sum(values.values()))
    points = {
        anchor: sum(sum(rows) for rows in deltas[anchor].values()) for anchor in roles.anchors
    }
    k = len(roles.anchors)
    return {
        "estimand": "score(challenger, O) - score(incumbent, O) on aligned blocks",
        "by_anchor": {
            anchor: {
                "numerator": points[anchor],
                "denominator": denominator,
                "point_estimate": points[anchor] / denominator,
                "interval_95": _percentiles(per_anchor[anchor], denominator),
            }
            for anchor in roles.anchors
        },
        "macro": {
            "numerator": sum(points.values()),
            "denominator": denominator * k,
            "point_estimate": sum(points.values()) / (denominator * k),
            "interval_95": _percentiles(macro, denominator * k),
        },
        "bootstrap": {
            "method": "joint-family-stratified-common-block-percentile-v1",
            "resamples": RESAMPLES,
            "order_statistic_indices": [LOWER, UPPER],
            "seed": design.boot_seed(label),
            "strata": list(design.families),
        },
    }


def _nested_m(design: Design, scores: Scores) -> dict[str, Any]:
    roles = design.roles
    reps = roles.replicates
    others = [policy for policy in roles.order if policy not in reps]
    table = {
        (rep, other, family): scores.blocks(rep, other, family)
        for rep in reps
        for other in others
        for family in design.families
    }
    denominator = 2 * design.pairs * len(design.families) * len(reps)
    label = "m-family-nested-bootstrap:v1"
    rng = design.boot_rng(label)
    draws: dict[str, list[int]] = {other: [] for other in others}
    macro: list[int] = []
    worst: list[int] = []
    for _ in range(RESAMPLES):
        picked = Counter(rng.randbelow_batch(len(reps), len(reps)))
        flat = rng.randbelow_batch(design.pairs, design.pairs * len(design.families))
        blocks = {
            family: Counter(flat[k * design.pairs : (k + 1) * design.pairs])
            for k, family in enumerate(design.families)
        }
        values = {
            other: sum(
                weight * multiplicity * table[(reps[rep_index], other, family)][block]
                for rep_index, weight in picked.items()
                for family in design.families
                for block, multiplicity in blocks[family].items()
            )
            for other in others
        }
        for other in others:
            draws[other].append(values[other])
        macro.append(sum(values.values()))
        worst.append(min(values.values()))
    points = {
        other: sum(sum(table[(rep, other, family)]) for rep in reps for family in design.families)
        for other in others
    }
    lowest = min(points.values())
    worst_name = next(other for other in others if points[other] == lowest)
    return {
        "replicates": list(reps),
        "opponents": others,
        "excludes_m_vs_m_cells": True,
        "selects_replicate": False,
        "by_opponent": {
            other: {
                "numerator": points[other],
                "denominator": denominator,
                "point_estimate": points[other] / denominator,
                "interval_95": _percentiles(draws[other], denominator),
            }
            for other in others
        },
        "versus_incumbent": points[roles.incumbent] / denominator,
        "versus_challenger": points[roles.challenger] / denominator,
        "equal_weight_macro": {
            "numerator": sum(points.values()),
            "denominator": denominator * len(others),
            "point_estimate": sum(points.values()) / (denominator * len(others)),
            "interval_95": _percentiles(macro, denominator * len(others)),
        },
        "worst_opponent": {
            "opponent": worst_name,
            "point_estimate": lowest / denominator,
            "resampled_minimum_interval_95": _percentiles(worst, denominator),
        },
        "bootstrap": {
            "method": "nested-replicate-outer-aligned-block-inner-percentile-v1",
            "resamples": RESAMPLES,
            "order_statistic_indices": [LOWER, UPPER],
            "seed": design.boot_seed(label),
        },
    }


# ---------------------------------------------------------------------------------------------
# Decision


def decide(
    roles: Roles,
    statistics: Mapping[str, Any],
    tactical: Mapping[str, Mapping[str, int]],
    prefix: Mapping[str, int],
    integrity_ok: bool,
) -> tuple[str, list[bool]]:
    """Independent statement of the immutable eight-condition promotion rule."""
    challenger = tactical[roles.challenger]
    incumbent = tactical[roles.incumbent]
    lift_lower = statistics["anchor_contrasts"]["macro"]["interval_95"][0]
    random_key = f"{roles.challenger}--vs--{roles.random}"
    random_lower = statistics["matchups"][random_key]["combined"]["paired_block_bootstrap_95"][0]
    seats: list[float] = []
    for opponent in (roles.incumbent, roles.heuristic, roles.random):
        pair = sorted((roles.challenger, opponent), key=roles.order.index)
        seat_rows = statistics["matchups"][f"{pair[0]}--vs--{pair[1]}"]["seats"][roles.challenger]
        seats += [row["combined"]["win_rate"] for row in seat_rows.values()]
    checks = [
        lift_lower > LIFT_THRESHOLD,
        prefix["incumbent_win_challenger_loss_blocks"] == 0
        and prefix["incumbent_win_challenger_loss_games"] == 0,
        prefix["metadata_failures"] == 0
        and prefix["prefix_failures"] == 0
        and prefix["post_intervention_termination_failures"] == 0
        and prefix["challenger_failed_guaranteed_win"] == 0,
        challenger["missed_guaranteed_wins"] == 0
        and challenger["false_forced_wins"] == 0
        and challenger["executed_avoidable_provable_losses"] == 0,
        incumbent["executed_avoidable_provable_losses"] == 0,
        random_lower > RANDOM_THRESHOLD,
        min(seats) >= SEAT_THRESHOLD,
        integrity_ok,
    ]
    envelope_ok = all(
        tactical.get(policy, {}).get("executed_avoidable_provable_losses", 0) == 0
        for policy in roles.integrity_enveloped
    ) and all(
        tactical.get(policy, {}).get("missed_guaranteed_wins", 0) == 0
        and tactical.get(policy, {}).get("false_forced_wins", 0) == 0
        for policy in roles.replicates
    )
    if not integrity_ok or not envelope_ok:
        return "blocked_no_decision", checks
    if all(checks):
        return "promote_q0_terminal_offense_v1", checks
    return "retain_q0_terminal_safety_v1", checks


def check_claim_preflight(
    output: Path, plan: Mapping[str, Any], registry: Mapping[str, Any], problems: list[str]
) -> None:
    """A claim must be bound to a sealed, passing measured runtime preflight."""
    path = output / "runtime-preflight.json"
    if not path.is_file():
        problems.append("claim runtime preflight is missing")
        return
    preflight = _load(path)
    if (
        not _self_sealed(preflight)
        or preflight.get("artifact_fingerprint") != plan.get("runtime_preflight_fingerprint")
        or preflight.get("claim_eligible") is not True
        or not preflight.get("projected_claim_minutes", math.inf) <= 420.0
        or preflight.get("code_fingerprint") != plan["source"]["code_fingerprint"]
        or preflight.get("registry_fingerprint") != registry["artifact_fingerprint"]
    ):
        problems.append("claim runtime preflight is invalid or unbound")


def check_cross_references(
    decision: Mapping[str, Any],
    result: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    plan_fingerprint: str,
    problems: list[str],
) -> None:
    """Require sealed artifacts and exact result/decision fingerprint cross-references."""
    for name, artifact in (("decision", decision), ("result", result), *artifacts.items()):
        if not _self_sealed(artifact):
            problems.append(f"{name} seal")
    expected_result = {
        "decision": decision["decision"],
        "decision_fingerprint": decision["artifact_fingerprint"],
        "plan_fingerprint": plan_fingerprint,
        **{f"{name}_fingerprint": item["artifact_fingerprint"] for name, item in artifacts.items()},
    }
    if any(result.get(key) != value for key, value in expected_result.items()):
        problems.append("result cross-references")
    if any(
        decision.get(f"{name}_fingerprint") != artifacts[name]["artifact_fingerprint"]
        for name in ("statistics", "prefix", "tactical")
    ):
        problems.append("decision cross-references")
    if decision.get("deployment_change_authorized_by_runner") is not False:
        problems.append("runner decision claims deployment authority")


def check_decision_reproduces(
    decision: Mapping[str, Any], expected: str, checks: Sequence[bool], problems: list[str]
) -> None:
    retained_checks = [condition["passed"] for condition in decision["conditions"]]
    if decision["decision"] != expected or retained_checks != list(checks):
        problems.append(f"decision does not reproduce: {decision['decision']} != {expected}")


# ---------------------------------------------------------------------------------------------
# Output-tree scope and checksums


def _unchecksummed(relative: str) -> bool:
    name = relative.rsplit("/", 1)[-1]
    return name == ".lock" or ("/" not in relative and name in NOT_CHECKSUMMED)


def expected_tree(design: Design, *, claim: bool = False) -> set[str]:
    files = {name for name in TOP_LEVEL_ARTIFACTS}
    if claim:
        files.add("runtime-preflight.json")
    for family, left, right in design.cells():
        base = f"cells/{family}/{left}--vs--{right}"
        files |= {f"{base}/cell.json", f"{base}/records/manifest.json"}
        files.add(f"{base}/records/games.jsonl.gz")
    return files


def check_tree_and_checksums(
    output: Path, design: Design, problems: list[str], *, claim: bool = False
) -> None:
    """Require the exact retained file set and exact checksum scope and digests."""
    present = {
        relative
        for path in output.rglob("*")
        if path.is_file()
        for relative in [path.relative_to(output).as_posix()]
        if not _unchecksummed(relative) or relative in {"checksums.json", "execution-state.json"}
    }
    expected = expected_tree(design, claim=claim)
    if present != expected:
        extra, missing = sorted(present - expected), sorted(expected - present)
        problems.append(f"output tree differs: extra={extra[:5]} missing={missing[:5]}")
    manifest = _load(output / "checksums.json")
    if not _self_sealed(manifest) or set(manifest) != {
        "version",
        "excluded_names",
        "files",
        "artifact_fingerprint",
    }:
        problems.append("checksum manifest is malformed")
        return
    if set(manifest["excluded_names"]) != NOT_CHECKSUMMED:
        problems.append("checksum exclusion scope differs")
    scoped = {name for name in expected if not _unchecksummed(name)}
    if set(manifest["files"]) != scoped:
        problems.append("checksum scope differs from the retained artifact set")
    for name, digest in manifest["files"].items():
        path = output / name
        if not path.is_file() or _file_digest(path) != digest:
            problems.append(f"checksum mismatch: {name}")


# ---------------------------------------------------------------------------------------------
# Main validation


def _design_from_plan(plan: Mapping[str, Any], registry: Mapping[str, Any], roles: Roles) -> Design:
    declared = plan["schedule"]["design"]
    claim = plan["execution"]["claim"] is True
    families = tuple(registry["schedule"]["family_order"])
    if claim:
        expected = (CYCLE, registry["root_seed"], CLAIM_PAIRS)
    else:
        smoke_root = derive_seed(registry["root_seed"], SMOKE_ROOT_DOMAIN) & ((1 << 63) - 1)
        expected = (SMOKE_NAMESPACE, smoke_root, declared["pair_count"])
    actual = (declared["namespace"], declared["root_seed"], declared["pair_count"])
    if actual != expected or declared["policy_order"] != list(roles.order):
        raise ValidationFailure("plan design is not the frozen claim or nonclaim smoke design")
    if declared["family_order"] != list(families) or type(declared["pair_count"]) is not int:
        raise ValidationFailure("plan family order or block count is malformed")
    return Design(expected[0], expected[1], expected[2], roles, families)


def validate(
    output: Path, *, registry_path: Path, inputs_root: Path, holdout_roots: Sequence[Path]
) -> dict[str, Any]:
    started = time.perf_counter()
    phases: dict[str, float] = {}
    problems: list[str] = []

    def lap(name: str, since: float) -> float:
        now = time.perf_counter()
        phases[name] = phases.get(name, 0.0) + now - since
        return now

    plan = _load(output / "plan.json")
    payload = {
        key: value for key, value in plan.items() if key not in {"plan_fingerprint", "version"}
    }
    if plan.get("plan_fingerprint") != _digest(payload):
        raise ValidationFailure("plan fingerprint does not match its content")
    claim = plan["execution"]["claim"] is True
    registry = _load(registry_path)
    if not _self_sealed(registry) or _load(output / "input-registry.json") != registry:
        raise ValidationFailure("registry copy or self-fingerprint differs")
    if plan["registry_fingerprint"] != registry["artifact_fingerprint"]:
        raise ValidationFailure("plan names another registry")
    if claim and (
        registry["artifact_fingerprint"] != COMMITTED_REGISTRY_FINGERPRINT
        or registry_path.resolve() != COMMITTED_REGISTRY.resolve()
    ):
        raise ValidationFailure("claim evidence must use the committed registry")
    roles = Roles(registry)
    design = _design_from_plan(plan, registry, roles)

    # Source identity: validation runs at the exact source that produced the evidence.
    source = inspect_source_identity()
    frozen = plan["source"]
    if (
        source.git_revision != frozen["git_revision"]
        or source.uv_lock_sha256 != frozen["uv_lock_sha256"]
        or code_fingerprint() != frozen["code_fingerprint"]
        or rules_fingerprint() != frozen["rules_fingerprint"]
    ):
        problems.append("validator source differs from the frozen plan source")
    if claim and not (source.tracked_tree_clean and frozen["tracked_tree_clean"] is True):
        problems.append("claim validation requires a tracked-clean frozen source")
    if _load(output / "source.json") != frozen:
        problems.append("retained source identity differs from plan")
    _same(plan["schedule"], expected_schedule(design), "plan.schedule", problems)
    if _load(output / "schedule.json") != plan["schedule"]:
        problems.append("retained schedule differs from plan")
    input_audit = _load(output / "input-audit.json")
    if not _self_sealed(input_audit) or (
        input_audit["artifact_fingerprint"] != plan["input_audit_fingerprint"]
    ):
        problems.append("input audit seal or plan binding")
    runtime = _load(output / "runtime.json")
    if not _self_sealed(runtime) or runtime["plan_fingerprint"] != plan["plan_fingerprint"]:
        problems.append("runtime report seal or plan binding")
    if plan.get("deployment_mutation") != "never-performed-by-runner":
        problems.append("plan does not forbid deployment mutation")
    mark = lap("authentication", started)

    if claim:
        problems += [f"schedule: {item}" for item in check_registry_schedule(registry, design)]
    mode = plan["execution"]["policy_mode"]
    if claim and mode != "learned-checkpoints":
        problems.append("claim policy mode")
    check = _checkpoint_check(registry, mode)
    runtime_configs = {policy: plan["policies"][policy]["config"] for policy in roles.order}
    for policy in roles.order:
        entry = plan["policies"][policy]
        if (
            entry["declared"] != roles.declared[policy]
            or entry["rng_identity"] != roles.rng[policy]
        ):
            problems.append(f"plan declaration or RNG identity differs: {policy}")
        if not realizes(roles.declared[policy], runtime_configs[policy], check):
            problems.append(f"runtime config does not realize declaration: {policy}")
    if roles.rng[roles.incumbent] != roles.rng[roles.challenger]:
        problems.append("incumbent/challenger core RNG identities differ")
    if mode == "learned-checkpoints":
        input_problems = check_inputs(registry, inputs_root, archives_required=claim)
        problems += [f"inputs: {item}" for item in input_problems]
    mark = lap("inputs", mark)

    holdout = _load(output / "holdout.json")
    independent = independent_holdout(design, holdout_roots, output)
    if (
        not _self_sealed(holdout)
        or holdout["status"] != "passed"
        or independent["overlap_count"] != 0
        or holdout["overlap_count"] != 0
    ):
        problems.append("setup holdout overlap or malformed holdout inventory")
    for key in ("proposed_setup_count", "proposed_setup_fingerprint"):
        if holdout.get(key) != independent[key]:
            problems.append(f"holdout {key} does not reproduce")
    # runs/ may legitimately change after the claim started (for example, a moved smoke tree), so
    # the fresh scan must show zero overlap and every retained corpus still present must be
    # byte-identical; exact equality of the whole prior inventory is reported, not required.
    for row in holdout.get("prior_inventory", []):
        current = independent["corpus_fingerprints"].get(row["path"])
        if current is not None and current != row["corpus_fingerprint"]:
            problems.append(f"retained holdout corpus changed: {row['path']}")
    holdout_inventory_identical = (
        holdout.get("prior_setup_fingerprint") == independent["prior_setup_fingerprint"]
    )
    mark = lap("holdout", mark)

    summaries: dict[str, dict[str, Any]] = {}
    intervals: dict[str, tuple[list[float], int]] = {}
    elapsed: dict[str, float] = {}
    tactical_total: dict[str, Counter[str]] = {}
    cell_tactical: dict[str, dict[str, dict[str, int]]] = {}
    records_by_key: dict[str, tuple[GameRecord, ...]] = {}
    result = _load(output / "result.json")
    aligned_keys = {
        _cell_key(family, variant, other)
        for family in design.families
        for variant in (roles.incumbent, roles.challenger)
        for other in roles.order[2:]
    }
    for family, left, right in design.cells():
        key = _cell_key(family, left, right)
        directory = output / "cells" / family / f"{left}--vs--{right}"
        artifact = _load(directory / "cell.json")
        manifest, records = load_corpus(directory / "records")
        if (
            not _self_sealed(artifact)
            or artifact["corpus_fingerprint"] != manifest.corpus_fingerprint
        ):
            problems.append(f"{key}: cell artifact seal or corpus fingerprint")
        if artifact["plan_fingerprint"] != plan["plan_fingerprint"]:
            problems.append(f"{key}: stale plan fingerprint")
        if result["cells"].get(key) != {
            "artifact_fingerprint": artifact["artifact_fingerprint"],
            "corpus_fingerprint": manifest.corpus_fingerprint,
        }:
            problems.append(f"{key}: result cross-reference")
        problems += [
            f"{key}: {item}"
            for item in check_cell_schedule(design, family, left, right, records, runtime_configs)
        ]
        summary = summarize_cell(left, right, records)
        compare_summary(artifact["summary"], summary, key, problems)
        interval, seed = arena_interval(summary["pair_scores"], design.master(family))
        report = artifact["report"]
        paired = report["paired_bootstrap_confidence_interval_95"]
        if paired["interval"] != interval or paired["bootstrap_seed"] != seed:
            problems.append(f"{key}: arena paired interval does not reproduce")
        if (
            report["run_id"] != f"{design.namespace}:{family}:{left}:vs:{right}"
            or report["master_seed"] != design.master(family)
            or report["wins"]
            != {
                "a": sum(summary["pair_scores"]),
                "b": 2 * design.pairs - sum(summary["pair_scores"]),
            }
            or report["agents"]["a"]["id"] != left
            or report["agents"]["b"]["id"] != right
        ):
            problems.append(f"{key}: arena report identity or outcome")
        tactical = tactical_counts(records)
        _same(artifact["tactical"], tactical, f"{key}.tactical", problems)
        for policy, counts in tactical.items():
            tactical_total.setdefault(policy, Counter()).update(counts)
        cell_tactical[key] = tactical
        summaries[key] = summary
        intervals[key] = (interval, seed)
        elapsed[key] = float(report["elapsed_seconds"])
        if key in aligned_keys:
            records_by_key[key] = records
    mark = lap("replay_and_tactical", mark)

    prefix_rows: dict[str, dict[str, int]] = {}
    for family in design.families:
        for other in roles.order[2:]:
            inc = records_by_key[_cell_key(family, roles.incumbent, other)]
            ch = records_by_key[_cell_key(family, roles.challenger, other)]
            rows = [aligned_pair(roles, a, b) for a, b in zip(inc, ch, strict=True)]
            prefix_rows[f"{family}:{other}"] = prefix_totals(rows, [a.pair_id for a in inc])
    prefix_sum: Counter[str] = Counter()
    for row in prefix_rows.values():
        prefix_sum.update(row)
    retained_prefix = _load(output / "prefix-audit.json")
    if not _self_sealed(retained_prefix):
        problems.append("prefix audit seal")
    _same(retained_prefix["by_family_opponent"], prefix_rows, "prefix.by_family_opponent", problems)
    _same(retained_prefix["totals"], dict(prefix_sum), "prefix.totals", problems)
    mark = lap("prefix_audit", mark)

    tactical_by_policy = {
        policy: {key: counts[key] for key in TACTICAL_KEYS}
        for policy, counts in sorted(tactical_total.items())
    }
    retained_tactical = _load(output / "tactical.json")
    if not _self_sealed(retained_tactical):
        problems.append("tactical seal")
    _same(retained_tactical["by_policy"], tactical_by_policy, "tactical.by_policy", problems)

    statistics = independent_statistics(
        design, summaries, intervals, elapsed, cell_tactical, runtime_configs
    )
    retained_statistics = _load(output / "statistics.json")
    if not _self_sealed(retained_statistics):
        problems.append("statistics seal")
    _same(
        {
            k: v
            for k, v in retained_statistics.items()
            if k not in {"version", "artifact_fingerprint"}
        },
        statistics,
        "statistics",
        problems,
    )
    mark = lap("statistics", mark)

    state = _load(output / "execution-state.json")
    runner_seconds = float(state["active_seconds"])
    deadline_ok = (not claim) or runner_seconds < CUTOFF_SECONDS
    if state.get("completed") is not True or state["plan_fingerprint"] != plan["plan_fingerprint"]:
        problems.append("runner execution did not complete this plan")
    if len(state["attempts"]) > 2:
        problems.append("more than one resume attempt")
    if not deadline_ok:
        problems.append("claim cutoff exceeded")

    check_tree_and_checksums(output, design, problems, claim=claim)
    if claim:
        check_claim_preflight(output, plan, registry, problems)
    decision = _load(output / "promotion-decision.json")
    check_cross_references(
        decision,
        result,
        {
            "statistics": retained_statistics,
            "prefix": retained_prefix,
            "tactical": retained_tactical,
            "holdout": holdout,
        },
        plan["plan_fingerprint"],
        problems,
    )
    evidence_ok = not problems
    expected_decision, checks = decide(
        roles, statistics, tactical_by_policy, dict(prefix_sum), evidence_ok
    )
    check_decision_reproduces(decision, expected_decision, checks, problems)
    phases["checks_and_decision"] = time.perf_counter() - mark
    total = time.perf_counter() - started
    if claim and runner_seconds + total > HARD_LIMIT_SECONDS:
        problems.append("whole-step hard limit exceeded")
    status = "passed" if not problems else "failed"
    report = {
        "version": VALIDATION_VERSION,
        "status": status,
        "evidence_class": "claim-locked-final" if claim else "nonclaim-smoke",
        "plan_fingerprint": plan["plan_fingerprint"],
        "registry_fingerprint": registry["artifact_fingerprint"],
        "result_fingerprint": result["artifact_fingerprint"],
        "decision_fingerprint": decision["artifact_fingerprint"],
        "checksums_fingerprint": _load(output / "checksums.json")["artifact_fingerprint"],
        "retained_decision": decision["decision"],
        "independent_decision": expected_decision,
        "final_decision": expected_decision if status == "passed" else "blocked_no_decision",
        "independent_condition_results": checks,
        "replayed_games": sum(2 * design.pairs for _ in design.cells()),
        "holdout_prior_inventory_identical_at_validation": holdout_inventory_identical,
        "runner_active_seconds": runner_seconds,
        "validator_seconds": total,
        "problems": problems[:200],
        "problem_count": len(problems),
        "imports_production_league_module": "agent_avenue.runners.league" in sys.modules,
    }
    report["artifact_fingerprint"] = _digest(report)
    return {"validation": report, "runtime": {"elapsed_seconds": total, "phase_seconds": phases}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--registry", type=Path, default=COMMITTED_REGISTRY)
    parser.add_argument("--inputs-root", type=Path, default=Path("."))
    parser.add_argument("--holdout-root", type=Path, action="append", default=None)
    arguments = parser.parse_args()
    roots = tuple(arguments.holdout_root or (Path("runs"),))
    try:
        outcome = validate(
            arguments.output,
            registry_path=arguments.registry,
            inputs_root=arguments.inputs_root,
            holdout_roots=roots,
        )
    except (ValidationFailure, KeyError, TypeError, ValueError) as exc:
        report = {
            "version": VALIDATION_VERSION,
            "status": "failed",
            "final_decision": "blocked_no_decision",
            "problems": [f"{type(exc).__name__}: {exc}"],
            "problem_count": 1,
        }
        report["artifact_fingerprint"] = _digest(report)
        outcome = {"validation": report, "runtime": {"elapsed_seconds": None, "phase_seconds": {}}}
    for name, value in (
        ("validation.json", outcome["validation"]),
        ("validator-runtime.json", outcome["runtime"]),
    ):
        (arguments.output / name).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    print(json.dumps(outcome["validation"], sort_keys=True, indent=2))
    return 0 if outcome["validation"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

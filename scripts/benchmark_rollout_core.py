"""Timed small-panel Step-4 rollout-core preflight; never runs a claim panel or arena."""

from __future__ import annotations

import argparse
import gzip
import json
import time
from collections.abc import Sequence
from pathlib import Path

from agent_avenue.agents import Agent, GreedyHeuristicAgent, RandomAgent
from agent_avenue.agents.learned import LearnedValueAgent
from agent_avenue.learning import load_checkpoint, load_structured_dataset
from agent_avenue.rollout import (
    PANEL_STRATA,
    REQUIRED_POLICY_IDS,
    PanelCandidate,
    PanelPosition,
    canonical_observation,
    canonical_panel_order,
    extract_train_panel_candidates,
    rollout_candidate_world,
    sample_position_worlds,
)
from agent_avenue.storage import load_corpus


def _position_candidates(
    *, corpus: Path, dataset: Path, replicate_id: str, count: int
) -> tuple[PanelPosition, ...]:
    structured = load_structured_dataset(dataset)
    split = structured.manifest["split"]
    if not isinstance(split, dict):
        raise RuntimeError("structured dataset has no frozen split")
    train = split.get("train_game_fingerprints")
    if not isinstance(train, list) or len(train) < count:
        raise RuntimeError("structured dataset has too few train records for the requested smoke")
    train_ids = tuple(str(value) for value in train[: max(count * 4, count)])
    _, all_records = load_corpus(corpus, verify_code=False, verify_replays=True)
    # The core validates the supplied fingerprint set against record semantics; keep no non-train
    # records in memory after this filter.
    from agent_avenue.storage import game_record_fingerprint

    wanted = set(train_ids)
    records = tuple(record for record in all_records if game_record_fingerprint(record) in wanted)
    candidates = extract_train_panel_candidates(
        records,
        replicate_id=replicate_id,
        train_record_fingerprints=train_ids,
        verify_code=False,
    )
    by_identity: dict[str, PanelCandidate] = {}
    for candidate in candidates:
        prior = by_identity.get(candidate.safe_identity)
        if prior is None or candidate.source_record_fingerprint < prior.source_record_fingerprint:
            by_identity[candidate.safe_identity] = candidate
    stratum_index = {stratum: index for index, stratum in enumerate(PANEL_STRATA)}
    positions: list[PanelPosition] = []
    used_sources: set[str] = set()
    for candidate in sorted(
        by_identity.values(),
        key=lambda value: (
            stratum_index[str(value.stratum)],
            value.selection_hash,
            value.safe_identity,
        ),
    ):
        if candidate.source_record_fingerprint in used_sources:
            continue
        assert candidate.stratum is not None
        positions.append(
            PanelPosition(
                candidate.replicate_id,
                candidate.stratum,
                candidate.safe_identity,
                candidate.selection_hash,
                canonical_observation(candidate.observation),
                candidate.source_record_fingerprint,
            )
        )
        used_sources.add(candidate.source_record_fingerprint)
        if len(positions) == count:
            break
    if len(positions) != count:
        raise RuntimeError("small benchmark panel could not satisfy source-game uniqueness")
    return canonical_panel_order(positions)


def _policies(root: Path) -> tuple[dict[str, Agent], LearnedValueAgent]:
    q_paths = {
        "q0": root / "terminal-safety-v1/q0-a1/checkpoint",
        "q1": root / "terminal-safety-v1/q1-a1/candidate",
        "q2": root / "terminal-safety-v1/q2-a1/candidate",
        "q3": root / "terminal-safety-v1/q3-a1/candidate",
        "q4": root / "terminal-safety-v1/q4-a1/candidate",
    }
    learned = {
        policy_id: LearnedValueAgent.from_checkpoint(load_checkpoint(path))
        for policy_id, path in q_paths.items()
    }
    policies: dict[str, Agent] = {
        **learned,
        "heuristic": GreedyHeuristicAgent(),
        "random": RandomAgent(),
    }
    if set(policies) != REQUIRED_POLICY_IDS:
        raise RuntimeError("benchmark continuation population is incomplete")
    return policies, learned["q0"]


def run_benchmark(*, source_root: Path, replicate_id: str, positions: int) -> dict[str, object]:
    """Measure actual retained-M target work units, including compact sample serialization."""
    try:
        replicate_number = int(replicate_id.removeprefix("replicate-"))
    except ValueError as exc:
        raise RuntimeError("replicate_id must use the frozen replicate-N spelling") from exc
    dataset = source_root / f"m7-structured-model-v2/datasets/M{replicate_number}/dataset.npz"
    corpus = source_root / f"m7-population-replay-v1/corpora/replicate-{replicate_number}/treatment"
    panel = _position_candidates(
        corpus=corpus,
        dataset=dataset,
        replicate_id=replicate_id,
        count=positions,
    )
    policies, leaf = _policies(source_root)
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    units = 0
    leaves = 0
    for position in panel:
        worlds = sample_position_worlds(position)
        for action in position.observation.legal_actions:
            for world in worlds:
                sample = rollout_candidate_world(
                    position,
                    action,
                    world,
                    continuation_policies=policies,
                    leaf_scorer=leaf,
                )
                rows.append(sample.to_data())
                units += sample.actions_applied + int(sample.depth_leaf)
                leaves += int(sample.depth_leaf)
    serialized = gzip.compress(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode(), mtime=0
    )
    elapsed = time.perf_counter() - started
    rate = units / elapsed if elapsed else float("inf")
    return {
        "version": "step4-rollout-core-benchmark-v1",
        "scope": "small retained-M training-panel smoke only; no full panel, claim, or arena",
        "replicate_id": replicate_id,
        "positions": len(panel),
        "candidate_rows": sum(len(position.observation.legal_actions) for position in panel),
        "world_samples": len(rows),
        "transition_or_leaf_work_units": units,
        "depth_leaf_fraction": leaves / len(rows),
        "serialized_bytes": len(serialized),
        "elapsed_seconds": elapsed,
        "units_per_second": rate,
        "projects_at_least_67_units_per_second": rate >= 67.0,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--replicate-id", default="replicate-1")
    parser.add_argument("--positions", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.positions <= 0:
        raise SystemExit("--positions must be positive")
    print(json.dumps(run_benchmark(**vars(args)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

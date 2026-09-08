"""Independent deterministic interpretation of retained diagnostic outputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

ANALYSIS_VERSION = "m7-diagnostic-interpretation-v1"


class DiagnosticAnalysisError(ValueError):
    """Raised when retained evidence is incomplete or internally inconsistent."""


def _read(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticAnalysisError(f"unable to read diagnostic input: {path}") from exc
    if not isinstance(value, dict):
        raise DiagnosticAnalysisError(f"diagnostic input must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _terminal_guardrails(historical: Mapping[str, object]) -> list[dict[str, object]]:
    raw = historical.get("heuristic_guardrails")
    if not isinstance(raw, list):
        return []
    values = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        generation = item.get("generation")
        difference = item.get("heuristic_difference")
        if (
            isinstance(path, str)
            and path.startswith("runs/terminal-safety-v1/q")
            and path.endswith("promotion-decision.json")
            and type(generation) is int
            and 1 <= generation <= 4
            and isinstance(difference, dict)
        ):
            values.append(
                {
                    "generation": generation,
                    "point_estimate": difference.get("point_estimate"),
                    "interval": difference.get("interval"),
                    "block_count": difference.get("block_count"),
                    "decision": item.get("decision"),
                    "source": path,
                }
            )
    return sorted(values, key=lambda item: cast(int, item["generation"]))


def _terminal_transfer(historical: Mapping[str, object]) -> list[dict[str, object]]:
    raw = historical.get("prediction_transfer")
    if not isinstance(raw, list):
        return []
    values = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        candidate = item.get("candidate")
        macro = item.get("macro_log_loss")
        if (
            isinstance(path, str)
            and path.startswith("runs/terminal-safety-v1/q")
            and isinstance(candidate, dict)
            and isinstance(macro, dict)
            and type(candidate.get("generation")) is int
        ):
            values.append(
                {
                    "generation": candidate["generation"],
                    "label": candidate.get("label"),
                    "macro_log_loss": macro.get("point_estimate"),
                    "interval": macro.get("interval"),
                    "matchup_count": macro.get("matchup_count"),
                    "worst_matchup": item.get("worst_matchup"),
                    "source": path,
                }
            )
    return sorted(values, key=lambda item: cast(int, item["generation"]))


def _validate_safe_trace(position: Mapping[str, object]) -> None:
    observation = position.get("observation")
    if not isinstance(observation, dict):
        raise DiagnosticAnalysisError("safe trace lacks a serialized player observation")
    forbidden = {"deck", "hands", "opponent_hand", "authoritative_state", "rng_state"}

    def visit(value: object) -> None:
        if isinstance(value, dict):
            overlap = forbidden.intersection(value)
            if overlap:
                raise DiagnosticAnalysisError(f"safe trace contains forbidden keys: {overlap}")
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(observation)


def _safe_trace_markdown(traces: Mapping[str, object]) -> str:
    positions = traces.get("positions")
    if not isinstance(positions, list) or not positions:
        raise DiagnosticAnalysisError("no information-safe trace positions were produced")
    lines = [
        "# Hand-checkable information-safe positions",
        "",
        "Each position is serialized from the acting player's `PlayerObservation`. Own hand, "
        "public scores/tableaux/history, current public decision, and legal actions are visible. "
        "Opposing hand identities, deck order, and an unknown face-down identity are absent.",
        "",
    ]
    for index, raw in enumerate(positions, start=1):
        if not isinstance(raw, dict):
            raise DiagnosticAnalysisError("safe trace position is malformed")
        _validate_safe_trace(raw)
        observation = cast(dict[str, object], raw["observation"])
        players = observation.get("players")
        scores = {
            cast(str, player["player"]): player["score"]
            for player in cast(list[dict[str, object]], players)
        }
        decision = cast(dict[str, object], observation["decision"])
        legal = cast(list[object], observation["legal_actions"])
        lines.extend(
            (
                f"## Position {index}",
                "",
                f"- Trace: `{raw.get('record_fingerprint')}` decision {raw.get('decision_index')}",
                (
                    f"- Acting viewer: `{observation.get('viewer')}`; "
                    f"phase: `{observation.get('phase')}`; turn: {observation.get('turn')}"
                ),
                f"- Own hand: `{', '.join(cast(list[str], observation.get('own_hand', [])))}`",
                f"- Public scores: `{json.dumps(scores, sort_keys=True)}`",
                f"- Public decision: `{json.dumps(decision, sort_keys=True)}`",
                f"- Legal candidate count: {len(legal)}",
                (
                    "- Recorded chosen action: "
                    f"`{json.dumps(raw.get('chosen_action'), sort_keys=True)}`"
                ),
                "",
            )
        )
    return "\n".join(lines)


def _plot_svg(guardrails: list[dict[str, object]], transfer: list[dict[str, object]]) -> str:
    width, height = 760, 400
    left, top, plot_width, plot_height = 70, 40, 620, 270
    values = [
        float(cast(float, item["point_estimate"]))
        for item in guardrails
        if isinstance(item.get("point_estimate"), int | float)
    ] + [
        float(cast(float, item["macro_log_loss"])) - 0.55
        for item in transfer
        if isinstance(item.get("macro_log_loss"), int | float)
    ]
    bound = max((abs(value) for value in values), default=0.2)
    bound = max(bound, 0.15)

    def y(value: float) -> float:
        return top + plot_height / 2 - value / bound * (plot_height / 2)

    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            '<text x="20" y="22" font-family="sans-serif" font-size="16">'
            "Retained diagnostic signals (descriptive, not causal)</text>"
        ),
        (
            f'<line x1="{left}" y1="{y(0):.1f}" x2="{left + plot_width}" '
            f'y2="{y(0):.1f}" stroke="#555"/>'
        ),
    ]
    for index, item in enumerate(guardrails):
        x = left + 50 + index * 90
        point = float(cast(float, item["point_estimate"]))
        interval = cast(list[float], item["interval"])
        lines.extend(
            (
                (
                    f'<line x1="{x}" y1="{y(interval[0]):.1f}" x2="{x}" '
                    f'y2="{y(interval[1]):.1f}" stroke="#9c2f2f" stroke-width="3"/>'
                ),
                f'<circle cx="{x}" cy="{y(point):.1f}" r="5" fill="#9c2f2f"/>',
                (
                    f'<text x="{x - 12}" y="{top + plot_height + 22}" '
                    f'font-family="sans-serif" font-size="12">q{item["generation"]}</text>'
                ),
            )
        )
    lines.append(
        f'<text x="{left}" y="{height - 42}" font-family="sans-serif" '
        'font-size="12" fill="#9c2f2f">Heuristic win-rate difference with '
        "95% block interval</text>"
    )
    transfer_points = []
    for index, item in enumerate(transfer):
        x = left + 410 + index * 45
        value = float(cast(float, item["macro_log_loss"])) - 0.55
        transfer_points.append(f"{x},{y(value):.1f}")
        lines.append(f'<circle cx="{x}" cy="{y(value):.1f}" r="4" fill="#235c91"/>')
    if transfer_points:
        lines.append(
            f'<polyline points="{" ".join(transfer_points)}" fill="none" '
            'stroke="#235c91" stroke-width="2"/>'
        )
    lines.append(
        f'<text x="{left + 360}" y="{height - 42}" font-family="sans-serif" '
        'font-size="12" fill="#235c91">Crossplay log loss minus 0.55 '
        "(q0-q4)</text>"
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def analyze_diagnostic_pack(raw: Path, web_evidence: Path, output: Path) -> dict[str, object]:
    """Validate independent evidence and write deterministic human-facing summaries."""
    catalog_path = raw / "artifact-catalog.json"
    summary_path = raw / "diagnostic-summary.json"
    traces_path = raw / "safe-traces.json"
    catalog = _read(catalog_path)
    summary = _read(summary_path)
    traces = _read(traces_path)
    web = _read(web_evidence)

    archives = catalog.get("archives")
    corpora = catalog.get("corpora")
    recomputations = summary.get("arena_recomputations")
    restore_checks = summary.get("restore_semantic_checks")
    historical = summary.get("historical_evidence")
    if not all(
        isinstance(value, expected)
        for value, expected in (
            (archives, list),
            (corpora, list),
            (recomputations, list),
            (restore_checks, list),
            (historical, dict),
        )
    ):
        raise DiagnosticAnalysisError("diagnostic pack is missing required evidence sections")
    assert isinstance(archives, list)
    assert isinstance(corpora, list)
    assert isinstance(recomputations, list)
    assert isinstance(restore_checks, list)
    assert isinstance(historical, dict)

    invalid_corpora = [item for item in corpora if item.get("status") != "verified"]
    mismatches = [item for item in recomputations if item.get("status") != "matched"]
    restore_invalid = [
        corpus
        for restore in restore_checks
        if isinstance(restore, dict)
        for corpus in cast(list[dict[str, object]], restore.get("corpora", []))
        if corpus.get("status") != "verified"
    ]
    bad_archives = [
        item
        for item in archives
        if not isinstance(item, dict)
        or item.get("expected_sha256") != item.get("sha256")
        or not item.get("member_checksum_count")
    ]
    web_cases = web.get("cases")
    if (
        invalid_corpora
        or mismatches
        or restore_invalid
        or bad_archives
        or web.get("status") != "completed"
        or not isinstance(web_cases, list)
        or len(web_cases) != 4
    ):
        raise DiagnosticAnalysisError(
            "scientific-validity checks failed: "
            f"invalid_corpora={len(invalid_corpora)}, mismatches={len(mismatches)}, "
            f"restore_invalid={len(restore_invalid)}, bad_archives={len(bad_archives)}, "
            f"web_cases={len(web_cases) if isinstance(web_cases, list) else 'invalid'}"
        )

    guardrails = _terminal_guardrails(historical)
    transfer = _terminal_transfer(historical)
    if len(guardrails) != 4 or len(transfer) != 5:
        raise DiagnosticAnalysisError(
            "expected terminal-safety q1-q4 guardrails and q0-q4 transfer"
        )
    negative_guardrails = all(
        isinstance(item.get("point_estimate"), int | float)
        and float(cast(float, item["point_estimate"])) < 0
        for item in guardrails
    )
    transfer_values = [float(cast(float, item["macro_log_loss"])) for item in transfer]
    transfer_worsened = transfer_values[-1] > transfer_values[0]
    coverage = cast(dict[str, object], summary["coverage"])
    targets = cast(dict[str, int], coverage["targets"])
    decisions = int(cast(int, coverage["decision_count"]))
    win_fraction = targets.get("win", 0) / decisions

    recommendation = {
        "next_experiment": "counterfactual-candidate-ranking-supervision",
        "confidence": "moderate-directional-not-causal",
        "why_first": (
            "The repeated heuristic non-regression failures and worsening interaction-transfer "
            "diagnostic are more directly connected to selected-action supervision and candidate "
            "coverage than to raw network capacity. A fixed-corpus ranking comparison "
            "isolates that factor before paying for new mixed-opponent self-play."
        ),
        "runner_up": "mixed-opponent-replay",
        "alternative_explanations": [
            (
                "The 87-feature encoder omits public action history, so representation may limit "
                "belief-sensitive play."
            ),
            (
                "The all-pairs pools expand with generation, so rising macro log loss is not a "
                "controlled same-distribution trend."
            ),
            "One training replicate per generation does not measure between-training variability.",
            (
                "Heuristic guardrail losses may reflect policy tradeoffs rather than globally "
                "worse play."
            ),
        ],
        "reserved_for_user": True,
    }
    result = {
        "version": ANALYSIS_VERSION,
        "status": "completed",
        "evidence_class": "retrospective-historical-diagnostic",
        "input_sha256": {
            "artifact_catalog": _sha256(catalog_path),
            "diagnostic_summary": _sha256(summary_path),
            "safe_traces": _sha256(traces_path),
            "web_evidence": _sha256(web_evidence),
        },
        "validity": {
            "archive_count": len(archives),
            "verified_corpus_count": len(corpora),
            "matched_arena_recomputations": len(recomputations),
            "restored_archive_count": len(restore_checks),
            "web_checkpoint_cases": len(web_cases),
        },
        "coverage": {
            "retained_decisions": decisions,
            "target_win_fraction": win_fraction,
            "action_types": coverage.get("action_types"),
            "phases": coverage.get("phases"),
            "actor_seats": coverage.get("actor_seats"),
            "terminal_reasons": coverage.get("terminal_reasons"),
        },
        "terminal_safety_guardrails": guardrails,
        "prediction_transfer": transfer,
        "signals": {
            "all_q1_q4_heuristic_differences_negative": negative_guardrails,
            "q4_macro_log_loss_above_q0": transfer_worsened,
            "q0_macro_log_loss": transfer_values[0],
            "q4_macro_log_loss": transfer_values[-1],
        },
        "recommendation": recommendation,
        "scientific_limit": (
            "These retained diagnostics select a question to test; they do not establish causal "
            "improvement or authorize the recommended experiment."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "analysis.json", result)
    (output / "safe-traces.md").write_text(_safe_trace_markdown(traces))
    (output / "diagnostic-signals.svg").write_text(_plot_svg(guardrails, transfer))
    table_lines = [
        "# Diagnostic tables",
        "",
        "## Terminal-safety heuristic non-regression",
        "",
        "| Generation | Difference | 95% block interval | Decision |",
        "| ---: | ---: | ---: | --- |",
    ]
    for item in guardrails:
        interval = cast(list[float], item["interval"])
        decision = cast(dict[str, object], item["decision"])
        table_lines.append(
            f"| q{item['generation']} | {float(cast(float, item['point_estimate'])):.3f} | "
            f"[{interval[0]:.3f}, {interval[1]:.3f}] | {decision.get('status')} |"
        )
    table_lines.extend(
        (
            "",
            "## Held-out all-pairs chosen-action prediction",
            "",
            "| Candidate | Matchups | Macro log loss | 95% interval |",
            "| --- | ---: | ---: | ---: |",
        )
    )
    for item in transfer:
        interval = cast(list[float], item["interval"])
        table_lines.append(
            f"| {item['label']} | {item['matchup_count']} | "
            f"{float(cast(float, item['macro_log_loss'])):.4f} | "
            f"[{interval[0]:.4f}, {interval[1]:.4f}] |"
        )
    table_lines.extend(
        (
            "",
            "The matchup pool expands by generation, so this table is descriptive and is not a",
            "controlled monotonic trend test.",
            "",
        )
    )
    (output / "tables.md").write_text("\n".join(table_lines))
    return cast(dict[str, object], result)

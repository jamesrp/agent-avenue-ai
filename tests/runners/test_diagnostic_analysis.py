import hashlib
import json
from pathlib import Path

import pytest

from agent_avenue.runners.diagnostic_analysis import (
    DiagnosticAnalysisError,
    analyze_diagnostic_pack,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _pack(root: Path) -> tuple[Path, Path]:
    raw = root / "raw"
    archives = [
        {
            "sha256": str(index) * 64,
            "expected_sha256": str(index) * 64,
            "member_checksum_count": index,
        }
        for index in range(1, 4)
    ]
    _write(
        raw / "artifact-catalog.json",
        {
            "archives": archives,
            "corpora": [{"path": "a", "status": "verified"}],
        },
    )
    guardrails = [
        {
            "path": f"runs/terminal-safety-v1/q{generation}-a1/promotion-decision.json",
            "generation": generation,
            "decision": {"status": "retain"},
            "heuristic_difference": {
                "point_estimate": -0.02 * generation,
                "interval": [-0.1, 0.01 - 0.01 * generation],
                "block_count": 200,
            },
        }
        for generation in range(1, 5)
    ]
    transfer = [
        {
            "path": f"runs/terminal-safety-v1/q{generation}-a1/crossplay/report.json",
            "candidate": {"generation": generation, "label": f"q{generation}"},
            "macro_log_loss": {
                "point_estimate": 0.54 + 0.005 * generation,
                "interval": [0.52, 0.58],
                "matchup_count": max(1, generation + 1),
            },
            "worst_matchup": {"pair_id": "fixture", "point_estimate": 0.6},
        }
        for generation in range(5)
    ]
    _write(
        raw / "diagnostic-summary.json",
        {
            "coverage": {
                "decision_count": 100,
                "targets": {"win": 55, "loss": 45},
                "action_types": {"play_offer": 50, "recruit": 50},
                "phases": {"play": 50, "recruit": 50},
                "actor_seats": {"player_one": 50, "player_two": 50},
                "terminal_reasons": {"score": 10},
            },
            "arena_recomputations": [{"status": "matched"}],
            "restore_semantic_checks": [
                {"corpora": [{"status": "verified"}]},
                {"corpora": [{"status": "verified"}]},
            ],
            "historical_evidence": {
                "heuristic_guardrails": guardrails,
                "prediction_transfer": transfer,
            },
        },
    )
    _write(
        raw / "safe-traces.json",
        {
            "positions": [
                {
                    "record_fingerprint": "a" * 64,
                    "decision_index": 0,
                    "observation": {
                        "viewer": "player_one",
                        "phase": "play",
                        "turn": 1,
                        "own_hand": ["mole"],
                        "players": [
                            {"player": "player_one", "score": 0},
                            {"player": "player_two", "score": 0},
                        ],
                        "decision": {"kind": "play"},
                        "legal_actions": [{"type": "play_offer"}],
                    },
                    "chosen_action": {"type": "play_offer"},
                }
            ]
        },
    )
    web = root / "web.json"
    _write(web, {"status": "completed", "cases": [{}, {}, {}, {}]})
    return raw, web


def test_diagnostic_analysis_validates_and_writes_inspectable_outputs(tmp_path: Path) -> None:
    raw, web = _pack(tmp_path)
    output = tmp_path / "analysis"
    result = analyze_diagnostic_pack(raw, web, output)

    assert result["status"] == "completed"
    assert result["recommendation"]["next_experiment"] == (
        "counterfactual-candidate-ranking-supervision"
    )
    assert result["signals"]["all_q1_q4_heuristic_differences_negative"] is True
    assert (output / "tables.md").is_file()
    assert (output / "safe-traces.md").is_file()
    assert (output / "diagnostic-signals.svg").read_text().startswith("<svg")
    recorded = result["input_sha256"]["diagnostic_summary"]
    assert recorded == hashlib.sha256((raw / "diagnostic-summary.json").read_bytes()).hexdigest()


def test_diagnostic_analysis_blocks_mismatched_arena_evidence(tmp_path: Path) -> None:
    raw, web = _pack(tmp_path)
    summary_path = raw / "diagnostic-summary.json"
    summary = json.loads(summary_path.read_text())
    summary["arena_recomputations"][0]["status"] = "mismatch"
    _write(summary_path, summary)

    with pytest.raises(DiagnosticAnalysisError, match="scientific-validity checks failed"):
        analyze_diagnostic_pack(raw, web, tmp_path / "analysis")

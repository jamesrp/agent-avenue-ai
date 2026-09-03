"""Replay-derived diagnostics for the terminal-safety policy contract."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Final

from agent_avenue.agents import TERMINAL_SAFETY_VERSION, filter_terminal_actions
from agent_avenue.engine import Phase, PlayerId, apply_action, new_game
from agent_avenue.observation import observe
from agent_avenue.storage import (
    GameRecord,
    code_fingerprint,
    game_record_fingerprint,
    rules_fingerprint,
    verify_game_record,
)

SAFETY_AUDIT_VERSION: Final = "terminal-safety-audit-v1"


class SafetyAuditError(ValueError):
    """Raised when retained records violate the declared safety contract."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _empty_counts() -> Counter[str]:
    return Counter(
        {
            "decisions": 0,
            "decisions_with_provable_loss": 0,
            "provable_loss_actions": 0,
            "counterfactual_veto_actions": 0,
            "all_actions_provably_losing": 0,
            "vetoed_actions": 0,
            "forced_loss_fallbacks": 0,
            "executed_provable_losses": 0,
            "executed_avoidable_provable_losses": 0,
        }
    )


def _counts_data(counts: Mapping[str, int]) -> dict[str, int]:
    return {key: counts[key] for key in _empty_counts()}


def _is_shielded(config: Mapping[str, object]) -> bool:
    return config.get("version") == TERMINAL_SAFETY_VERSION


def audit_terminal_safety(
    records: Iterable[GameRecord],
    *,
    source_corpus_fingerprint: str,
    verify_code: bool = True,
) -> dict[str, object]:
    """Replay records and audit every decision using only its public observation."""
    buffered = tuple(records)
    if not buffered:
        raise SafetyAuditError("terminal-safety audit requires at least one game record")
    if not source_corpus_fingerprint:
        raise SafetyAuditError("terminal-safety audit requires a source corpus fingerprint")

    overall = _empty_counts()
    by_agent: dict[str, Counter[str]] = {}
    by_seat = {player.value: _empty_counts() for player in PlayerId}
    agent_configs: dict[str, dict[str, object]] = {}
    terminal_reasons = Counter[str]()

    for record in buffered:
        verify_game_record(record, verify_code=verify_code)
        terminal_reasons[record.terminal_reason] += 1
        state = new_game(record.replay.config, record.replay.seed)
        for action in record.replay.actions:
            if state.phase is Phase.TERMINAL:  # pragma: no cover - record verification guards this
                raise SafetyAuditError("record contains actions after terminal state")
            actor = (
                state.active_player if state.phase is Phase.PLAY else state.active_player.other()
            )
            seat = record.seats[0 if actor is PlayerId.PLAYER_ONE else 1]
            normalized_config = json.loads(
                json.dumps(dict(seat.config), sort_keys=True, separators=(",", ":"))
            )
            if not isinstance(normalized_config, dict):  # pragma: no cover - mapping guarantees it
                raise SafetyAuditError("agent configuration must be a JSON object")
            existing_config = agent_configs.setdefault(seat.agent_id, normalized_config)
            if existing_config != normalized_config:
                raise SafetyAuditError("one agent id has multiple configurations in an audit")
            agent_counts = by_agent.setdefault(seat.agent_id, _empty_counts())
            seat_counts = by_seat[actor.value]

            observation = observe(state, actor)
            filtered = filter_terminal_actions(observation, observation.legal_actions)
            shielded = _is_shielded(normalized_config)
            values = {
                "decisions": 1,
                "decisions_with_provable_loss": int(bool(filtered.provable_loss_actions)),
                "provable_loss_actions": len(filtered.provable_loss_actions),
                "counterfactual_veto_actions": len(filtered.vetoed_actions),
                "all_actions_provably_losing": int(filtered.forced_loss_fallback),
                "vetoed_actions": len(filtered.vetoed_actions) if shielded else 0,
                "forced_loss_fallbacks": int(filtered.forced_loss_fallback and shielded),
                "executed_provable_losses": int(action in filtered.provable_loss_actions),
                "executed_avoidable_provable_losses": int(
                    action in filtered.provable_loss_actions and not filtered.forced_loss_fallback
                ),
            }
            for key, value in values.items():
                overall[key] += value
                agent_counts[key] += value
                seat_counts[key] += value
            state = apply_action(state, action)

    agents: dict[str, object] = {}
    for agent_id in sorted(by_agent):
        counts = _counts_data(by_agent[agent_id])
        shielded = _is_shielded(agent_configs[agent_id])
        if shielded and counts["executed_avoidable_provable_losses"] != 0:
            raise SafetyAuditError(
                f"shielded agent {agent_id!r} executed an avoidable provable immediate loss"
            )
        agents[agent_id] = {
            "policy_shield": TERMINAL_SAFETY_VERSION if shielded else None,
            "config": agent_configs[agent_id],
            "counts": counts,
        }

    data: dict[str, object] = {
        "version": SAFETY_AUDIT_VERSION,
        "source_corpus_fingerprint": source_corpus_fingerprint,
        "record_count": len(buffered),
        "record_fingerprints": [game_record_fingerprint(record) for record in buffered],
        "counts": _counts_data(overall),
        "by_agent": agents,
        "by_seat": {player.value: _counts_data(by_seat[player.value]) for player in PlayerId},
        "terminal_reasons": dict(sorted(terminal_reasons.items())),
        "rules_fingerprint": rules_fingerprint(),
        "code_fingerprint": code_fingerprint(),
        "artifact_fingerprint": "",
    }
    payload = {key: value for key, value in data.items() if key != "artifact_fingerprint"}
    data["artifact_fingerprint"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return data

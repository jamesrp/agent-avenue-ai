"""Deterministic retention and retrospective diagnostics for retained M6 artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

from agent_avenue.engine import (
    Action,
    Phase,
    PlayerId,
    PlayOfferAction,
    RecruitAction,
    apply_action,
    new_game,
)
from agent_avenue.observation import observation_to_data, observe
from agent_avenue.runners.arena import wilson_interval
from agent_avenue.storage import (
    GameRecord,
    game_record_fingerprint,
    inspect_source_identity,
    load_corpus,
)

DIAGNOSTIC_VERSION = "m7-diagnostic-readiness-v1"


class DiagnosticError(ValueError):
    """Raised when retained diagnostic inputs are unsafe or malformed."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path, root: Path) -> dict[str, object]:
    return {
        "locator": path.resolve().as_uri(),
        "relative_path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"unable to read JSON input: {path}") from exc
    if not isinstance(value, dict):
        raise DiagnosticError(f"JSON input must be an object: {path}")
    return value


def _safe_tar_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    members: dict[str, tarfile.TarInfo] = {}
    for member in archive.getmembers():
        name = member.name
        pure = Path(name)
        if not name or pure.is_absolute() or ".." in pure.parts or member.issym() or member.islnk():
            raise DiagnosticError("archive contains an unsafe member")
        if name in members:
            raise DiagnosticError("archive contains duplicate members")
        if not (member.isfile() or member.isdir()):
            raise DiagnosticError("archive contains a non-regular member")
        members[name] = member
    return members


def _archive_expected_digest(path: Path) -> tuple[str | None, str | None]:
    sha_sidecar = path.with_name(path.name + ".sha256")
    json_sidecar = path.with_name(path.name + ".json")
    if sha_sidecar.is_file():
        fields = sha_sidecar.read_text().strip().split()
        if not fields or len(fields[0]) != 64:
            raise DiagnosticError(f"invalid SHA-256 sidecar: {sha_sidecar}")
        return fields[0], sha_sidecar.name
    if json_sidecar.is_file():
        expected = _read_json(json_sidecar).get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise DiagnosticError(f"invalid archive JSON sidecar: {json_sidecar}")
        return expected, json_sidecar.name
    return None, None


def inspect_archive(path: Path, *, extract_to: Path | None = None) -> dict[str, object]:
    """Validate an archive and, optionally, extract it to an empty disposable directory."""
    if path.is_symlink() or not path.is_file():
        raise DiagnosticError(f"archive must be a regular file: {path}")
    expected_digest, sidecar = _archive_expected_digest(path)
    actual_digest = _sha256(path)
    if expected_digest is not None and actual_digest != expected_digest:
        raise DiagnosticError(f"archive SHA-256 does not match its sidecar: {path}")

    with tarfile.open(path, "r:gz") as archive:
        members = _safe_tar_members(archive)
        manifest_names = [name for name in members if name.endswith("archive-manifest.json")]
        if len(manifest_names) != 1:
            raise DiagnosticError("archive must contain exactly one archive-manifest.json")
        manifest_member = members[manifest_names[0]]
        extracted = archive.extractfile(manifest_member)
        if extracted is None:  # pragma: no cover - regular member guarantees an object
            raise DiagnosticError("unable to read archive manifest")
        try:
            manifest_value = json.loads(extracted.read())
        except json.JSONDecodeError as exc:
            raise DiagnosticError("archive manifest is invalid JSON") from exc
        if not isinstance(manifest_value, dict):
            raise DiagnosticError("archive manifest must be an object")
        payloads = manifest_value.get("entries", manifest_value.get("members"))
        if payloads is None:
            payloads = manifest_value.get("files")
        if not isinstance(payloads, list):
            raise DiagnosticError("archive manifest has no entries, members, or files list")
        prefix = manifest_names[0].removesuffix("archive-manifest.json")
        checked = 0
        payload_members: list[tuple[tarfile.TarInfo, str, int]] = []
        for item in payloads:
            if not isinstance(item, dict):
                raise DiagnosticError("archive payload manifest item is malformed")
            relative = item.get("path")
            expected = item.get("sha256")
            size = item.get("size")
            if (
                not isinstance(relative, str)
                or not isinstance(expected, str)
                or type(size) is not int
            ):
                raise DiagnosticError("archive payload manifest item has invalid identity")
            direct = members.get(relative)
            prefixed = members.get(prefix + relative)
            if direct is not None and prefixed is not None and direct is not prefixed:
                raise DiagnosticError(f"archive payload member is ambiguous: {relative}")
            member = direct or prefixed
            if member is None or not member.isfile() or member.size != size:
                raise DiagnosticError(
                    f"archive payload member is missing or has a size mismatch: {relative}"
                )
            source = archive.extractfile(member)
            if source is None:  # pragma: no cover - regular member guarantees an object
                raise DiagnosticError("unable to read archive payload member")
            digest = hashlib.sha256()
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
            if digest.hexdigest() != expected:
                raise DiagnosticError(f"archive payload checksum mismatch: {relative}")
            payload_members.append((member, expected, size))
            checked += 1
        restore_reused = False
        if extract_to is not None:
            if extract_to.exists() and any(extract_to.iterdir()):
                for member, expected, size in payload_members:
                    destination = extract_to / member.name
                    if (
                        not destination.is_file()
                        or destination.stat().st_size != size
                        or _sha256(destination) != expected
                    ):
                        raise DiagnosticError(
                            f"existing restore payload is incomplete or changed: {member.name}"
                        )
                restore_reused = True
            else:
                stage = extract_to.with_name(f".{extract_to.name}.partial-{actual_digest[:12]}")
                if stage.exists():
                    shutil.rmtree(stage)
                stage.mkdir(parents=True)
                try:
                    for name, member in members.items():
                        destination = stage / name
                        if member.isdir():
                            destination.mkdir(parents=True, exist_ok=True)
                        else:
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            source = archive.extractfile(member)
                            if source is None:  # pragma: no cover
                                raise DiagnosticError("unable to extract archive member")
                            with destination.open("wb") as target:
                                shutil.copyfileobj(source, target)
                    if extract_to.exists():
                        extract_to.rmdir()
                    os.replace(stage, extract_to)
                except Exception:
                    shutil.rmtree(stage, ignore_errors=True)
                    raise
    return {
        "path": path.resolve().as_uri(),
        "sha256": actual_digest,
        "expected_sha256": expected_digest,
        "checksum_sidecar": sidecar,
        "member_checksum_count": checked,
        "archive_manifest": manifest_names[0],
        "extracted_to": extract_to.resolve().as_uri() if extract_to is not None else None,
        "restore_reused": restore_reused,
    }


def _is_below(path: Path, roots: tuple[Path, ...]) -> bool:
    resolved = path.resolve()
    return any(resolved == root or resolved.is_relative_to(root) for root in roots)


def _corpus_directories(root: Path, excluded: tuple[Path, ...] = ()) -> tuple[Path, ...]:
    return tuple(
        sorted(
            {
                path.parent
                for path in root.rglob("manifest.json")
                if (path.parent / "games.jsonl.gz").is_file()
                and not _is_below(path.parent, excluded)
            }
        )
    )


def _corpus_inventory(
    root: Path, excluded: tuple[Path, ...] = ()
) -> tuple[list[dict[str, object]], dict[str, tuple[GameRecord, ...]]]:
    items: list[dict[str, object]] = []
    records_by_path: dict[str, tuple[GameRecord, ...]] = {}
    for directory in _corpus_directories(root, excluded):
        relative = directory.relative_to(root).as_posix()
        try:
            manifest, records = load_corpus(directory, verify_code=False)
            records_by_path[relative] = records
            items.append(
                {
                    "path": relative,
                    "status": "verified",
                    "corpus_fingerprint": manifest.corpus_fingerprint,
                    "declaration_fingerprint": manifest.declaration_fingerprint,
                    "record_count": manifest.record_count,
                    "decision_count": manifest.decision_count,
                    "record_fingerprints": list(manifest.record_fingerprints),
                }
            )
        except Exception as exc:
            items.append({"path": relative, "status": "invalid", "error": str(exc)})
    return items, records_by_path


def _action_coverage(records: Iterable[GameRecord]) -> dict[str, object]:
    buffered = tuple(records)
    actions = Counter[str]()
    cards = Counter[str]()
    phases = Counter[str]()
    targets = Counter[str]()
    terminal_reasons = Counter[str]()
    seats = Counter[str]()
    total = 0
    for record in buffered:
        terminal_reasons[record.terminal_reason] += 1
        state = new_game(record.replay.config, record.replay.seed)
        for action in record.replay.actions:
            actor = (
                state.active_player if state.phase is Phase.PLAY else state.active_player.other()
            )
            phases[state.phase.value] += 1
            targets["win" if actor is record.winner else "loss"] += 1
            seats[actor.value] += 1
            if isinstance(action, RecruitAction):
                actions["recruit"] += 1
                cards[f"recruit:{action.slot.value}"] += 1
            else:
                actions["play_offer"] += 1
                cards[f"face_up:{action.face_up.value}"] += 1
                cards[f"face_down:{action.face_down.value}"] += 1
            total += 1
            state = apply_action(state, action)
    return {
        "record_count": len(buffered),
        "decision_count": total,
        "action_types": dict(sorted(actions.items())),
        "card_roles": dict(sorted(cards.items())),
        "phases": dict(sorted(phases.items())),
        "targets": dict(sorted(targets.items())),
        "actor_seats": dict(sorted(seats.items())),
        "terminal_reasons": dict(sorted(terminal_reasons.items())),
    }


def _action_data(action: Action) -> dict[str, object]:
    if isinstance(action, RecruitAction):
        return {
            "type": "recruit",
            "revision": action.revision,
            "actor": action.actor.value,
            "slot": action.slot.value,
        }
    assert isinstance(action, PlayOfferAction)
    return {
        "type": "play_offer",
        "revision": action.revision,
        "actor": action.actor.value,
        "face_up": action.face_up.value,
        "face_down": action.face_down.value,
    }


def _safe_traces(records: Iterable[GameRecord], limit: int) -> list[dict[str, object]]:
    traces: list[dict[str, object]] = []
    for record in records:
        state = new_game(record.replay.config, record.replay.seed)
        for index, action in enumerate(record.replay.actions):
            actor = (
                state.active_player if state.phase is Phase.PLAY else state.active_player.other()
            )
            traces.append(
                {
                    "record_fingerprint": game_record_fingerprint(record),
                    "game_id": record.game_id,
                    "decision_index": index,
                    "actor": actor.value,
                    "observation": observation_to_data(observe(state, actor)),
                    "chosen_action": _action_data(action),
                }
            )
            if len(traces) >= limit:
                return traces
            state = apply_action(state, action)
    return traces


def _arena_recomputation(
    root: Path,
    records_by_path: Mapping[str, tuple[GameRecord, ...]],
    excluded: tuple[Path, ...] = (),
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for report_path in sorted(root.rglob("*.json")):
        if _is_below(report_path, excluded):
            continue
        relative = report_path.relative_to(root).as_posix()
        if "/arenas/" not in f"/{relative}" and "/development/" not in f"/{relative}":
            continue
        try:
            outer = _read_json(report_path)
            nested_report = outer.get("report")
            report = nested_report if isinstance(nested_report, dict) else outer
            agents = report.get("agents")
            agent_a = outer.get("agent_a_id")
            if not isinstance(agent_a, str):
                agent_a_data = agents.get("a") if isinstance(agents, dict) else None
                agent_a = agent_a_data.get("id") if isinstance(agent_a_data, dict) else None
            if not isinstance(agent_a, str):
                continue
            records_data = outer.get("records")
            if not isinstance(records_data, dict):
                records_data = report.get("records")
            raw_records_path = records_data.get("path") if isinstance(records_data, dict) else None
            candidates: tuple[Path, ...]
            if isinstance(raw_records_path, str):
                candidate = Path(raw_records_path)
                if not candidate.is_absolute():
                    candidate = report_path.parent.parent / candidate
                candidates = (candidate,)
            else:
                name = report_path.stem
                candidates = (
                    report_path.parent.parent / "arena-records" / name,
                    report_path.parent.parent / "development-records" / name,
                )
            records_relative = None
            for candidate in candidates:
                try:
                    relative_candidate = candidate.resolve().relative_to(root.resolve()).as_posix()
                except ValueError:
                    continue
                if relative_candidate in records_by_path:
                    records_relative = relative_candidate
                    break
            if records_relative is None:
                continue
            records = records_by_path[records_relative]
            wins = 0
            margins = 0
            turns = 0
            decisions = 0
            terminal = Counter[str]()
            seat_games = Counter[str]()
            seat_wins = Counter[str]()
            for record in records:
                a_index = 0 if record.seats[0].agent_id == agent_a else 1
                a_seat = record.seats[a_index].player.value
                won = (
                    record.seats[0 if record.winner is PlayerId.PLAYER_ONE else 1].agent_id
                    == agent_a
                )
                wins += int(won)
                seat_games[a_seat] += 1
                seat_wins[a_seat] += int(won)
                margins += record.final_scores[a_index] - record.final_scores[1 - a_index]
                turns += record.turn_count
                decisions += record.decision_count
                terminal[record.terminal_reason] += 1
            total = len(records)
            recomputed: dict[str, object] = {
                "total_games": total,
                "wins": {"a": wins, "b": total - wins},
                "agent_a_win_rate": wins / total,
                "wilson_confidence_interval_95": list(wilson_interval(wins, total)),
                "agent_a_by_seat": {
                    seat: {
                        "games": seat_games[seat],
                        "wins": seat_wins[seat],
                        "win_rate": seat_wins[seat] / seat_games[seat],
                    }
                    for seat in sorted(seat_games)
                },
                "terminal_reasons": dict(sorted(terminal.items())),
                "average_score_margin": margins / total,
                "average_turns": turns / total,
                "average_decisions": decisions / total,
            }
            expected = {key: report.get(key) for key in recomputed if key in report}
            matches = all(
                expected[key] == value for key, value in recomputed.items() if key in expected
            )
            results.append(
                {
                    "report": relative,
                    "records": records_relative,
                    "status": "matched" if matches else "mismatch",
                    "recomputed": recomputed,
                }
            )
        except (DiagnosticError, ValueError, ZeroDivisionError) as exc:
            results.append({"report": relative, "status": "invalid", "error": str(exc)})
    return results


def _historical_reports(root: Path, excluded: tuple[Path, ...] = ()) -> dict[str, object]:
    guardrails: list[dict[str, object]] = []
    calibration: list[dict[str, object]] = []
    transfer: list[dict[str, object]] = []
    source_identities: list[dict[str, object]] = []
    for path in sorted(root.rglob("*.json")):
        if _is_below(path, excluded):
            continue
        try:
            data = _read_json(path)
        except DiagnosticError:
            continue
        relative = path.relative_to(root).as_posix()
        source = data.get("source")
        if isinstance(source, dict):
            source_identities.append({"path": relative, "source": source})
        decision = data.get("decision")
        evidence = data.get("evidence")
        if (
            isinstance(decision, dict)
            and isinstance(evidence, dict)
            and "heuristic_difference" in evidence
        ):
            guardrails.append(
                {
                    "path": relative,
                    "generation": data.get("generation"),
                    "decision": decision,
                    "heuristic_difference": evidence["heuristic_difference"],
                }
            )
        if data.get("diagnostic_only") is True and "macro_paired_block_log_loss" in data:
            cells: list[dict[str, object]] = []
            matchups = data.get("matchups")
            if isinstance(matchups, list):
                for matchup in matchups:
                    if not isinstance(matchup, dict):
                        continue
                    model = matchup.get("model")
                    metrics = (
                        model.get("chosen_action_outcome_metrics")
                        if isinstance(model, dict)
                        else None
                    )
                    if isinstance(metrics, dict) and "calibration" in metrics:
                        cells.append(
                            {
                                "pair_id": matchup.get("pair_id"),
                                "left": matchup.get("left"),
                                "right": matchup.get("right"),
                                "calibration": metrics["calibration"],
                            }
                        )
            transfer.append(
                {
                    "path": relative,
                    "candidate": data.get("candidate"),
                    "macro_log_loss": data.get("macro_paired_block_log_loss"),
                    "worst_matchup": data.get("worst_matchup_log_loss"),
                    "interpretation_limits": data.get("interpretation_limits"),
                    "calibration_by_cell": cells,
                }
            )
        if "calibration" in data:
            calibration.append({"path": relative, "calibration": data["calibration"]})
    return {
        "heuristic_guardrails": guardrails,
        "prediction_transfer": transfer,
        "calibration_artifacts": calibration,
        "source_identities": source_identities,
    }


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(data) + b"\n")


def run_diagnostics(
    *,
    artifact_root: Path,
    output: Path,
    archives: tuple[Path, ...] = (),
    restore_directory: Path | None = None,
    trace_limit: int = 4,
) -> dict[str, object]:
    """Inventory retained artifacts and write a non-claim-generating evidence pack."""
    if trace_limit < 1:
        raise DiagnosticError("trace_limit must be positive")
    if artifact_root.is_symlink() or not artifact_root.is_dir():
        raise DiagnosticError("artifact root must be a directory")
    if output.exists() and not output.is_dir():
        raise DiagnosticError("diagnostic output must be a directory")
    output.mkdir(parents=True, exist_ok=True)
    if restore_directory is not None and restore_directory.resolve() == output.resolve():
        raise DiagnosticError("restore directory must not be the output directory")
    excluded: tuple[Path, ...] = (output.resolve(),)
    if restore_directory is not None and restore_directory.resolve().is_relative_to(
        artifact_root.resolve()
    ):
        excluded = (*excluded, restore_directory.resolve())

    files = [
        path
        for base in (
            artifact_root / "runs",
            artifact_root / "artifacts",
            artifact_root / "checkpoints",
        )
        if base.is_dir()
        for path in sorted(base.rglob("*"))
        if path.is_file() and not path.is_symlink() and not _is_below(path, excluded)
    ]
    lock = artifact_root / "uv.lock"
    if lock.is_file() and not lock.is_symlink():
        files.append(lock)
    archive_reports = []
    restored_paths: list[Path] = []
    for index, archive in enumerate(archives):
        restore = None if restore_directory is None else restore_directory / f"archive-{index:02d}"
        archive_reports.append(inspect_archive(archive, extract_to=restore))
        if restore is not None:
            restored_paths.append(restore)
    corpus_items, records_by_path = _corpus_inventory(artifact_root, excluded)
    all_records = tuple(record for records in records_by_path.values() for record in records)
    catalog = {
        "version": DIAGNOSTIC_VERSION,
        "retrospective_historical_evidence": True,
        "producer_source": inspect_source_identity().to_data(),
        "artifact_root": artifact_root.resolve().as_uri(),
        "live_files": [_file_identity(path, artifact_root) for path in files],
        "archives": archive_reports,
        "corpora": corpus_items,
    }
    recomputations = _arena_recomputation(artifact_root, records_by_path, excluded)
    summary = {
        "version": DIAGNOSTIC_VERSION,
        "retrospective_historical_evidence": True,
        "interpretation_limit": (
            "This deterministic audit reports retained historical evidence only; it makes no new "
            "playing-strength or causal claim."
        ),
        "input_catalog_sha256": hashlib.sha256(_canonical_json(catalog)).hexdigest(),
        "coverage": _action_coverage(all_records),
        "arena_recomputations": recomputations,
        "historical_evidence": _historical_reports(artifact_root, excluded),
        "restore_semantic_checks": [
            {
                "restore": restored.resolve().as_uri(),
                "corpora": _corpus_inventory(restored)[0],
            }
            for restored in restored_paths
        ],
    }
    traces = {
        "version": DIAGNOSTIC_VERSION,
        "retrospective_historical_evidence": True,
        "information_safety": (
            "Each trace contains only PlayerObservation serialization, legal candidates, and the "
            "chosen action; no deck order, opposing hand, or facedown card outside the viewer's "
            "knowledge is emitted."
        ),
        "positions": _safe_traces(all_records, trace_limit),
    }
    note = "\n".join(
        (
            "# Retention gap note",
            "",
            "This is a retrospective, deterministic inventory of locally retained artifacts. "
            "It does not establish a new strength result.",
            f"- Verified corpora: {sum(item.get('status') == 'verified' for item in corpus_items)}",
            f"- Invalid corpora: {sum(item.get('status') == 'invalid' for item in corpus_items)}",
            f"- Archive checksum validations: {len(archive_reports)}",
            (
                "- Arena aggregate recomputations: "
                f"{sum(item.get('status') == 'matched' for item in recomputations)} matched, "
                f"{sum(item.get('status') == 'mismatch' for item in recomputations)} mismatched."
            ),
            (
                "- Durability limitation: local live trees and local archive copies "
                "are inventoried here; off-VM backup is not established by this command."
            ),
            (
                "- Scientific limitation: selected-action historical labels and fresh, expanding "
                "all-pairs matrices cannot identify a causal next intervention or prove "
                "counterfactual action quality."
            ),
            "",
        )
    )
    _write_json(output / "artifact-catalog.json", catalog)
    _write_json(output / "diagnostic-summary.json", summary)
    _write_json(output / "safe-traces.json", traces)
    (output / "retention-note.md").write_text(note)
    return {
        "output": output.resolve().as_uri(),
        "artifact_catalog": "artifact-catalog.json",
        "diagnostic_summary": "diagnostic-summary.json",
        "safe_traces": "safe-traces.json",
        "retention_note": "retention-note.md",
    }

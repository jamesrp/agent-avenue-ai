import hashlib
import io
import json
import tarfile
from pathlib import Path

from agent_avenue.agents import RandomAgent, RandomAgentConfig
from agent_avenue.runners import AgentSpec, ArenaConfig, run_resumable_arena
from agent_avenue.runners.diagnostics import inspect_archive, run_diagnostics


def _random_spec(agent_id: str) -> AgentSpec:
    config = RandomAgentConfig()
    return AgentSpec(agent_id, config.to_data(), RandomAgent)


def _archive_with_manifest(archive: Path, source_root: Path, payload: list[Path]) -> None:
    entries = []
    for path in payload:
        relative = path.relative_to(source_root).as_posix()
        entries.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size,
            }
        )
    manifest = json.dumps({"entries": entries}, sort_keys=True).encode()
    with tarfile.open(archive, "w:gz") as destination:
        for path in payload:
            destination.add(path, arcname=path.relative_to(source_root).as_posix())
        info = tarfile.TarInfo("archive-manifest.json")
        info.size = len(manifest)
        destination.addfile(info, io.BytesIO(manifest))
    archive.with_name(archive.name + ".sha256").write_text(
        f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n"
    )


def test_diagnostics_inventory_restore_and_safe_traces(tmp_path: Path) -> None:
    root = tmp_path / "retained"
    records = root / "runs" / "milestone6" / "q1-a1" / "arena-records" / "primary"
    retained = run_resumable_arena(
        records,
        ArenaConfig("fixture", _random_spec("a"), _random_spec("b"), 2, 17),
        generation=1,
        corpus_configuration={"fixture": True},
    )
    arena = root / "runs" / "milestone6" / "q1-a1" / "arenas"
    arena.mkdir()
    (arena / "primary.json").write_text(json.dumps(retained.report.to_data(), sort_keys=True))
    archive = root / "artifacts" / "milestone6" / "fixture.tar.gz"
    archive.parent.mkdir(parents=True)
    _archive_with_manifest(
        archive,
        root,
        [records / "games.jsonl.gz", records / "manifest.json", arena / "primary.json"],
    )

    output = tmp_path / "output"
    result = run_diagnostics(
        artifact_root=root,
        output=output,
        archives=(archive,),
        restore_directory=tmp_path / "restore",
        trace_limit=2,
    )

    assert result["diagnostic_summary"] == "diagnostic-summary.json"
    catalog = json.loads((output / "artifact-catalog.json").read_text())
    summary = json.loads((output / "diagnostic-summary.json").read_text())
    traces = json.loads((output / "safe-traces.json").read_text())
    assert catalog["archives"][0]["member_checksum_count"] == 3
    assert catalog["corpora"][0]["status"] == "verified"
    assert summary["arena_recomputations"][0]["status"] == "matched"
    assert summary["restore_semantic_checks"][0]["corpora"][0]["status"] == "verified"
    assert (
        inspect_archive(archive, extract_to=tmp_path / "restore" / "archive-00")["restore_reused"]
        is True
    )
    assert len(traces["positions"]) == 2
    assert "deck" not in traces["positions"][0]["observation"]
    assert "own_hand" in traces["positions"][0]["observation"]

    stable_output_a = root / "runs" / "diagnostic-output"
    run_diagnostics(artifact_root=root, output=stable_output_a, archives=(archive,))
    first_bytes = {
        name: (stable_output_a / name).read_bytes()
        for name in (
            "artifact-catalog.json",
            "diagnostic-summary.json",
            "safe-traces.json",
            "retention-note.md",
        )
    }
    run_diagnostics(artifact_root=root, output=stable_output_a, archives=(archive,))
    for name, expected in first_bytes.items():
        assert (stable_output_a / name).read_bytes() == expected


def test_archive_inspection_rejects_payload_checksum_mismatch(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar.gz"
    payload = b"not-the-declared-digest"
    manifest = json.dumps(
        {"entries": [{"path": "payload.txt", "sha256": "0" * 64, "size": len(payload)}]}
    ).encode()
    with tarfile.open(archive, "w:gz") as destination:
        payload_info = tarfile.TarInfo("payload.txt")
        payload_info.size = len(payload)
        destination.addfile(payload_info, io.BytesIO(payload))
        manifest_info = tarfile.TarInfo("archive-manifest.json")
        manifest_info.size = len(manifest)
        destination.addfile(manifest_info, io.BytesIO(manifest))

    try:
        inspect_archive(archive)
    except ValueError as error:
        assert "payload checksum mismatch" in str(error)
    else:  # pragma: no cover - assertion message is clearer than a bare failure
        raise AssertionError("checksum mismatch was accepted")

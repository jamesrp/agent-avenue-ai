# Research workflow inventory

**Scope:** current repository and this VM, inspected September 8, 2026.  “Verified” means
observed from the named source or command during this audit; “inferred” means a committed
report asserts a past result but its full inputs were not rerun here; “unverified durability” means
no surviving local artifact or backing service was confirmed. No secrets are recorded below.

## Status and gap

- **Verified (committed status):** `docs/STATUS.md` (dated September 3, 2026) and `README.md`
  mark milestones 1–6 complete, milestone 7 planned/not started, historical q0 as the pure-neural
  baseline, and `q0-terminal-safety-v1` as selected hybrid champion.
- **Verified (committed result reports):** q1–q4 are complete: the base chain is documented in
  `docs/MILESTONE6_RESULTS.md`, and the shielded q1–q4 rerun plus terminal-safety audit is in
  `docs/TERMINAL_SAFETY_RESULTS.md`. Both retained q0 after all four candidates failed the
  heuristic non-regression guardrail. Therefore **q1–q4 and terminal-safety are complete**.
- **Verified missing capability:** learned web QA remains missing. `src/agent_avenue/web/routes.py`
  and `web/templates/landing.html` expose only human, random, and heuristic modes; `docs/WEB_QA.md`
  says learned checkpoints are not selectable and lists their future checklist. Basic web QA
  (milestone 2) is complete, but not learned-checkpoint QA.
- **Inferred:** the recorded historical results are reproducible claims, not re-executed claims in
  this audit; current HEAD is `69b10647066ce5a97c4e5e805a007a7a7203ed11`, not the historical
  result revisions named in `docs/STATUS.md`.

## Repository workflow capabilities

| Area | Verified current implementation / interface | Evidence |
| --- | --- | --- |
| Seeded execution | Single game, paired seat-balanced arena, replay verification, corpus generation, dataset build, training, checkpoint inspection, bootstrap, safety audit, crossplay, and iteration | `uv run python -m agent_avenue --help`; `src/agent_avenue/cli.py` |
| Resumable experiments | `bootstrap OUTPUT ...` and `iterate OUTPUT ...` explicitly run or resume. Corpus runs stage gzip game shards under an exclusive lock, validate each scheduled record, then atomically finalize a manifest. Arena retention uses the same corpus mechanism. | `src/agent_avenue/runners/{bootstrap,iteration,corpus,arena}.py`; `tests/storage/test_corpus.py` |
| Provenance | Source identity records Git revision, `uv.lock` SHA-256, tracked-tree cleanliness, and tracked diff SHA-256. Records/manifests bind rules/code, configuration, seeds, schedule, and per-record hashes. | `src/agent_avenue/storage/{provenance,game_record,corpus,fingerprints}.py` |
| Artifacts | Corpus manifests and game records are replay-verified; datasets carry lineage/split fingerprints; immutable checkpoints contain manifest, weights, metrics, compatibility metadata, and digests. | `src/agent_avenue/learning/{dataset,checkpoint}.py`; `tests/learning/test_checkpoint.py` |
| Evaluation/reporting | Pair/seat arena reports; promotion gates and immutable decisions; held-out prior-policy crossplay; replay-derived terminal-safety audits. Human-readable result/protocol documents are committed. | `src/agent_avenue/runners/{arena,promotion,iteration,crossplay,safety_audit}.py`; `docs/EXPERIMENT_PROTOCOL.md` |
| Regression coverage | 25 test files / 147 `test_` functions found; the full optional-dependency suite collected 159 tests and passed 158 (1 skipped). Coverage includes corpus interruption/resume, bootstrap, iteration, crossplay, replay, checkpoints, safety, and web tests. | `tests/` search; full test run on September 8, 2026 |

Useful bounded commands (write generated outputs outside Git):

```bash
uv run python -m agent_avenue bootstrap RUN_DIR --attempt-id ID --experiment-id ID --seed N
uv run python -m agent_avenue iterate RUN_DIR --incumbent CHECKPOINT --generation N --attempt-id ID --seed N
uv run python -m agent_avenue safety-audit CORPUS --output REPORT.json
uv run python -m agent_avenue crossplay-evaluate OUT --candidate CHECKPOINT --candidate-label qN --generation N --seed N
uv run python -m agent_avenue replay RECORD_OR_REPLAY.json
uv run --extra web --extra rl ruff check .
uv run --extra web --extra rl mypy
uv run --extra web --extra rl pytest  # current fully provisioned check
```

## Artifact durability

- **Verified durable on this VM:**
  `~/.local/share/agent-avenue-ai-archives/milestone6-baseline.tar.gz` exists (21,097,384 bytes).
  Its SHA-256 is `9264aadbfdabb3045d805f96d898a76c8b180ea39250c9dbe69ce4370cab8b3a`, matching
  `docs/MILESTONE6_RESULTS.md`; its sidecar also exists. This verifies bytes at that path, not an
  off-VM backup or restoration test.
- **Verified absent from this worktree:** `artifacts/`, `runs/`, and `checkpoints/` do not exist;
  `.gitignore` excludes all three. Thus the checked-out source alone cannot rerun historical q0,
  q1–q4, or terminal-safety results.
- **Inferred from committed reports:** Milestone 6 and terminal-safety describe verified ignored
  archives, checksums, compressed records, checkpoints, and reports. The terminal-safety archive
  was not present in this worktree or the inspected local archive directory, so its current bytes
  were not verified.
- **Unverified durability:** object storage, LFS, remote backups, archive retention policy,
  restoration drills, and cross-VM persistence were not found or tested. Treat result artifacts as
  locally retained only until an explicit independent restore/checksum procedure is performed.

## VM and orchestration capabilities

| Item | Verified observation | Bounded-use implication |
| --- | --- | --- |
| CPU/RAM | `nproc`/`getconf` report 2 online CPUs; `free -h` reported 7.7 GiB RAM and no swap. `cpuset.cpus.effective` is `0-1`. | Use CPU-only, low-parallelism runs; retain `--cpu-threads 1` default unless a measured plan changes it. |
| Limits | This container exposes cgroup v2 but no visible `cpu.max`, `memory.max`, or `memory.current` at `/sys/fs/cgroup`; shell virtual-memory and CPU-time limits are unlimited. | A hard CPU/RAM quota is **unverified**; observed free memory is not a guarantee. |
| Shelley | `shelley version` returned `v0.1025.967052336` (`dc54de75c6ec5bea95e53fdc83b9bf7a673f78bd`). | The CLI client is explicitly experimental; pin no automation to undocumented behavior. |
| Conversations / completion | `shelley client chat -p ... [-c ID]`, `read [-wait] ID`, `list`, `search`, `tag`, and `archive` are present. New chats support `-disable-notifications`; help names push/email/Discord/ntfy end-of-turn notifications. | A long job can send completion into an existing conversation with `shelley client chat -c "$SHELLEY_CONVERSATION_ID" -p "..."`; receiver configuration/delivery is unverified. |
| Persistent work | `shelley dtach new -s SOCKET -- CMD` starts a detach/reattach terminal session; `shelley dtach attach -s SOCKET` reconnects. No native `shelley task` or `shelley conversation` command exists. | Suitable for a bounded process while the VM/service survives; survival across reboot/service restart is unverified. |
| Native Codex | `/usr/local/bin/codex` is installed: `codex-cli 0.149.1`; `codex login status` returned `Not logged in`. `codex resume`, `queue`, and `exec` are advertised. | Codex is available but unauthenticated; do not plan API-backed Codex work without an approved login/integration. |

## Audit commands and limitations

Commands run: `git status --short --branch`, `git ls-files`, source/test searches, project CLI
`--help`, `shelley version`, `shelley client help chat`, `shelley dtach -h`, `codex --version`,
`codex login status`, `nproc`, `getconf _NPROCESSORS_ONLN`, `free -h`, cgroup/ulimit inspection,
and local archive size/SHA-256 checks. `make check` was also run from the initial plain environment:
Ruff passed, but mypy failed because that Makefile target does not request the optional web/RL
dependencies. The explicit full check above then passed (158 passed, 1 skipped), and `make check`
passed after those extras were present. The web module was not started before optional dependencies
were installed; this does not alter the source-level learned-web finding. Historical training/evaluation
and archive extraction were intentionally not rerun.

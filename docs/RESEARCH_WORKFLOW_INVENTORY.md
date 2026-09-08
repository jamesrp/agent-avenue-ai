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
| Regression coverage | The full suite collected and passed 168 tests. Coverage includes corpus interruption/resume, bootstrap, iteration, crossplay, replay, checkpoints, safety, web, and the bounded workflow's retry/interruption/stop/budget/tamper/containment behavior. | `make check` on September 8, 2026 |

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

- **Verified present in the authoritative primary checkout:** ignored `artifacts/`, `runs/`, and
  `checkpoints/` trees exist (about 56 MB, 66 MB, and 404 KB respectively). The historical q0
  checkpoint and both completed run trees are locally inspectable; they remain intentionally absent
  from ordinary Git history.
- **Verified terminal-safety archive bytes:** the primary checkout contains
  `artifacts/archive/terminal-safety-v1-artifacts-2026-09-03.tar.gz` (30,036,123 bytes). Its SHA-256
  is `d33c3e4bc3fea107a4d7ffe93dc841d9af8d12687269b7da783f540167fb27b9`, matching its JSON
  sidecar and `docs/TERMINAL_SAFETY_RESULTS.md`.
- **Verified secondary local copies:**
  `~/.local/share/agent-avenue-ai-archives/` contains byte-identical Milestone 6 and terminal-safety
  archives. Their SHA-256 values are respectively
  `9264aadbfdabb3045d805f96d898a76c8b180ea39250c9dbe69ce4370cab8b3a` and
  `d33c3e4bc3fea107a4d7ffe93dc841d9af8d12687269b7da783f540167fb27b9`. The
  terminal-safety copy also passed a complete gzip/tar member listing. These are second paths on the
  same VM, not off-VM backups.
- **Unverified durability:** cross-VM/object-storage retention and recovery after VM/disk loss remain
  unconfigured. A full disposable extraction plus semantic restore validation is proposed for the
  first real cycle.

## VM and orchestration capabilities

| Item | Verified observation | Bounded-use implication |
| --- | --- | --- |
| CPU/RAM | `nproc`/`getconf` report 2 online CPUs; `free -h` reported 7.7 GiB RAM and no swap. `cpuset.cpus.effective` is `0-1`. | Use CPU-only, low-parallelism runs; retain `--cpu-threads 1` default unless a measured plan changes it. |
| Limits | This container exposes cgroup v2 but no visible `cpu.max`, `memory.max`, or `memory.current` at `/sys/fs/cgroup`; shell virtual-memory and CPU-time limits are unlimited. | A hard CPU/RAM quota is **unverified**; observed free memory is not a guarantee. |
| Shelley | `shelley version` returned `v0.1025.967052336` (`dc54de75c6ec5bea95e53fdc83b9bf7a673f78bd`). | The CLI client is explicitly experimental; pin no automation to undocumented behavior. |
| Conversations / completion | `shelley client chat -p ... [-c ID]`, `read [-wait] ID`, `list`, `search`, `tag`, and `archive` are present. New chats support `-disable-notifications`; help names push/email/Discord/ntfy end-of-turn notifications. | The setup sent one terminal workflow message into a disposable existing conversation and observed the subsequent agent turn. External push/email/etc. delivery was not tested. |
| Persistent work | `shelley dtach new -s SOCKET -- CMD` starts a detach/reattach terminal session; `shelley dtach attach -s SOCKET` reconnects. No native `shelley task` or `shelley conversation` command exists. | Suitable for a bounded process while the VM/service survives; survival across reboot/service restart is unverified. |
| Native Codex | `/usr/local/bin/codex` is installed: `codex-cli 0.149.1`; `codex login status` returned `Not logged in`. `codex resume`, `queue`, and `exec` are advertised. | Codex is available but unauthenticated; do not plan API-backed Codex work without an approved login/integration. |

## Audit commands and limitations

Commands run by the worker and lead: `git status --short --branch`, `git ls-files`, source/test
searches, project CLI `--help`, `shelley version`, `shelley client help`, `shelley dtach -h`,
`codex --help`, `codex --version`, `codex login status`, `nproc`, `getconf _NPROCESSORS_ONLN`,
`free -h`, cgroup/ulimit inspection, and primary/secondary archive size/SHA-256 checks. The worker's
isolated worktree correctly lacked ignored run artifacts; the lead rechecked those paths in the
authoritative primary checkout before integration. Historical training/evaluation and a full
archive extraction were intentionally not rerun.

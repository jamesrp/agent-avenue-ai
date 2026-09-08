# AGENTS.md

## Mission

Build a faithful, deterministic implementation of the two-player base game of Agent Avenue, then
use it as the foundation for a reproducible reinforcement-learning self-play lab. The lightweight
web UI exists to QA rules and opponents; it is not the primary product. Read `README.md` for the
research goals and `RULES.md` before changing game behavior.

All gameplay content belongs to Nerdlab Games. Do not turn this repository into a hosted public
copy of the game or add copyrighted assets that are not already authorized for this research repo.

## Development environment

- Workspace: `~/workspace/agent-avenue-ai`
- Runtime: Python 3.12
- Package/environment manager: `uv`
- Default local/proxied web port: `8000`; servers must bind `0.0.0.0`
- The Git remote uses the VM's `jamesrp-agent-avenue-ai` integration. Keep the
  `https://github.int.exe.xyz/...` remote URL so clone, fetch, and push remain authenticated.
- Do not commit secrets, virtual environments, generated datasets, replay corpora, model
  checkpoints, or experiment artifacts.

## Bootstrap, build, run, and test

```bash
make setup                       # create .venv, lock, and install development dependencies
make run                         # run the current package entry point
make format                      # format Python files
make lint                        # Ruff checks
make typecheck                   # strict mypy
make test                        # pytest suite
make check                       # lint + typecheck + tests; required before pushing
uv sync --extra web              # add web dependencies when working on the QA UI
uv sync --extra rl               # add NumPy/PyTorch when working on learned agents
uv sync --extra web --extra rl   # full local environment
```

Use `uv add`, `uv add --dev`, or `uv add --optional <extra>` to change dependencies; never hand-edit
`uv.lock`. Commit `pyproject.toml` and `uv.lock` together.

## Intended package boundaries

Keep boundaries explicit even while the tree is evolving:

- `agent_avenue.engine`: rules, immutable/explicit state transitions, semantic decisions/actions,
  terminal outcomes, and replay support. It must not import web, agents, encoders, or PyTorch.
- `agent_avenue.observation`: player-visible observations and public decision context. Hidden deck
  order, facedown cards, and opposing hands must never leak.
- `agent_avenue.agents`: random, scripted, heuristic, and learned-agent adapters. Agents choose
  actions; they do not mutate engine state.
- `agent_avenue.runners`: trusted single-game, batch, self-play, and arena orchestration.
- `agent_avenue.encoding`: versioned, viewpoint-relative conversion of safe observations to model
  inputs.
- `agent_avenue.learning`: model definitions, training, checkpointing, and inference.
- `agent_avenue.storage`: compact replay/run formats, schema validation, fingerprints, and metadata.
- `agent_avenue.web`: thin QA interface over public observations and semantic actions.

Avoid a generic `utils.py`. Put behavior in the narrowest owning module. Keep optional web and RL
imports out of the core import path so the engine works after a plain `uv sync`.

## Engine invariants

- Model the game as an explicit state machine: a `Decision` describes the current semantic choice;
  applying one legal `Action` produces the next state, next decision, or terminal result.
- The engine never calls an agent and never handles model tensors.
- Transitions are deterministic. Setup randomness and agent randomness enter through explicit,
  seeded RNG/configuration and are recorded for replay.
- Prefer frozen dataclasses, enums, tuples, and pure functions for authoritative state. If mutable
  structures are needed for performance, contain them behind a tested deterministic interface.
- Actions are stable structured values, not UI strings or ephemeral legal-action indices.
- Validate actions at the engine boundary and fail clearly on illegal or stale actions.
- Define tie-breaking, deck exhaustion, simultaneous win/loss conditions, and unusual duplicate-hand
  cases exactly as specified in `RULES.md`.
- Observation construction is a security boundary. Learned or heuristic candidate evaluation may
  consume only the acting player's observation plus public decision context.

## Reproducibility and data compatibility

Every replay, dataset, checkpoint, run, and arena report should carry, as applicable:

- schema/encoder/model version;
- code or rules fingerprint;
- seed and RNG algorithm/state needed to reproduce behavior;
- complete normalized configuration;
- checkpoint and dataset fingerprints;
- creation timestamp and parent run/checkpoint lineage.

Reject incompatible inputs loudly rather than guessing or silently migrating. Migrations must be
explicit, versioned, and tested. Keep compact machine-readable artifacts separate from derived
reports. Large artifacts belong outside Git (or in explicitly configured LFS/object storage), while
small fixtures may live under `tests/fixtures/`.

## Testing conventions

- Mirror package paths under `tests/`; name tests `test_<behavior>` rather than implementation
  details.
- Add focused unit tests for every rule and boundary condition.
- Add property tests for conservation/card-count invariants, legal-action completeness, hidden-info
  safety, determinism under a seed, and replay round trips.
- Add end-to-end scripted games for regressions and terminal/tie scenarios.
- When fixing a bug, first add a regression test that fails for the reported behavior.
- Avoid flaky timing assertions and unseeded randomness. Tests must run usefully on a CPU-only VM.
- Compare agents with enough seeded games, alternating seats and reporting uncertainty—not a single
  anecdotal win rate.

## Python conventions

- Use type hints on public and internal APIs; strict mypy is the baseline.
- Prefer small pure functions and explicit domain types over dictionaries with magic keys.
- Use `pathlib.Path`, structured logging, and context managers.
- Keep public APIs documented. Comments should explain rule intent, information boundaries, or
  non-obvious tradeoffs rather than restating code.
- Ruff formatting is canonical. Do not introduce another formatter or overlapping linter without a
  clear reason.
- Keep CPU performance visible, but profile before optimizing. Preserve a clear path to batched and
  parallel execution without coupling the engine to multiprocessing or accelerators.

## Web UI conventions

The UI is a QA tool. Keep it server-rendered or otherwise lightweight until richer interaction is
justified. It should show the acting player's legal public view, decision, action history, and useful
debug/replay identifiers, but never render hidden authoritative state to the browser. Do not make
network access or a browser a requirement for core tests.

## Research-cycle coordination

- Shelley remains the single user-facing research lead. Use temporary workers only in isolated Git
  worktrees with bounded assignments and evidence-bearing completion reports; one lead integrates.
- A real cycle requires a committed, fingerprinted, user-approved agreement under
  `research/cycles/`. The local coordinator in `agent_avenue.research` orders trusted commands but
  does not replace experiment runners or make promotion/scientific decisions.
- Generated coordinator state belongs under ignored `runs/research-cycles/`. Every task declares
  outputs, timeout, dependencies, and at most one retry. Claim-generating tasks freeze a clean source
  and still use the owning runner's normalized plan/provenance checks.
- Do not use LLM polling to watch jobs. Detached managed processes report actual exit status; one
  terminal completion message may resume the lead for analysis/review/briefing.
- Stop and budget requests halt new dispatch and preserve evidence. Never launch an unapproved next
  cycle automatically. See `docs/RESEARCH_WORKFLOW.md`.

## Git workflow

- Keep commits small and descriptive; use imperative subjects.
- Run `make check` before pushing.
- Do not rewrite shared history or force-push unless explicitly asked.
- Update this file, `README.md`, and command help when workflows or architecture change.

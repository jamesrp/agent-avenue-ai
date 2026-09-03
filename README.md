# agent-avenue-ai

[Agent Avenue](https://boardgamegeek.com/boardgame/422732/agent-avenue) AI experiments.

All gameplay content is copyright Nerdlab Games. This is an ML research project; it is not intended
as a public hosted copy of the game.

## Project goal

Build a strong, reproducible, and inspectable AI for the two-player base game of **Agent Avenue**—not
just a rules simulator. The project combines a faithful deterministic implementation, self-play,
learned evaluation, controlled experiments, and artifacts that let results be reproduced and
improved over time. It is inspired by the end-to-end ambition of
[Keldon Jones's Race for the Galaxy AI project](https://www.keldon.net/rftg/) without committing to
its exact methods.

The long-term measure of success is an AI that becomes meaningfully stronger through iterative
self-play while remaining fair, testable, auditable, and practical to run on a CPU-only development
machine.

## Current status — September 3, 2026

| Milestone | Status | Result |
| --- | --- | --- |
| [1: deterministic rules engine](docs/MILESTONE1.md) | Complete | Typed, immutable, seeded, replayable two-player engine |
| [2: lightweight web QA](docs/MILESTONE2.md) | Complete | Human hot-seat UI and controller seam |
| [3: baseline agents and arena](docs/MILESTONE3.md) | Complete | Random, scripted, and `greedy-public-v1`; paired arena |
| 4: neural data/training foundation | Complete | Safe 87-feature encoder, datasets, model, and checkpoints |
| 5: first learned checkpoint | Complete | q0 trained and evaluated against random and heuristic |
| [6: frozen self-play and promotion](docs/MILESTONE6.md) | Complete | Four production generations, locked final evaluation, and verified archive |
| [7: controlled RL experiments](docs/MILESTONE7.md) | Planned | Recipe abstraction, environment adapters, and measured ablations |

The current learned champion is **q0**. Milestone 6 completed all four frozen self-play proposals;
each candidate beat q0 head-to-head but failed the predeclared heuristic non-regression guardrail, so
q0 was correctly retained. Its locked-final evaluation used fresh 500-pair blocks:

| Matchup | Games | q0 win rate | Paired-bootstrap 95% interval |
| --- | ---: | ---: | ---: |
| q0 vs random | 1,000 | **75.5%** | 72.9%–78.1% |
| q0 vs `greedy-public-v1` | 1,000 | **57.3%** | 54.2%–60.4% |

Full q1–q4 decisions, independent reproduction evidence, lineage, and archive checksums are in
[`docs/MILESTONE6_RESULTS.md`](docs/MILESTONE6_RESULTS.md). The earlier q0 evaluation and training
history remain in [`docs/MILESTONE5_RESULTS.md`](docs/MILESTONE5_RESULTS.md). The living
milestone/result index is [`docs/STATUS.md`](docs/STATUS.md).

**Milestone 6 is a project milestone, not a sixth neural generation.** It covered generations q1–q4:
freeze the incumbent, collect one generation of self-play, warm-start a candidate, evaluate it
against the incumbent and guardrails, and promote it only when the predeclared statistical gate
passes. The four-generation budget ended without a promotion or practical-equivalence plateau;
this is reported as `budget_exhausted_inconclusive`, and q0 remains the incumbent.

## Research principles

The engine is an explicit state machine. Every player choice is a semantic `Decision`; applying one
semantic `Action` produces the next state, decision, or terminal result. Setup randomness and agent
randomness use explicit, independently derived seeds. Transitions are deterministic and replayable.

Authoritative state is strictly separated from player-visible observations. Learned components
consume only versioned `PlayerObservation + Action` inputs and public decision context. Candidate
evaluation must never use hidden deck order, an opposing hand, an unknown face-down card, or private
future draws.

Rules execution, observation construction, encoding, model definition, training, inference,
self-play orchestration, storage, evaluation, and web presentation remain separate modules. The core
engine does not import web or PyTorch dependencies.

Experiments follow the declared workflow in
[`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md): state a hypothesis, freeze a recipe and
seed domains, retain reproducible artifacts, evaluate with matched seat-balanced games, and record
negative results without changing the gate after seeing them.

## Development

The repository targets Python 3.12 and uses [uv](https://docs.astral.sh/uv/) for environments and
dependency locking.

```bash
make setup
make check
make run
```

Optional dependencies can be installed independently:

```bash
uv sync --extra web
uv sync --extra rl
uv sync --extra web --extra rl
```

Useful checks:

```bash
make test-web
make check-rl
make arena-smoke
make neural-smoke
```

See `AGENTS.md` for package boundaries, engine invariants, testing conventions, and contribution
rules.

## Neural training and self-play

The first learned baseline is deliberately small and inspectable:

- `candidate-public-v1`: a viewpoint-relative 87-feature observation-and-action encoder;
- `candidate-mlp-v1`: an 87→128→1 PyTorch value network with 11,393 parameters;
- terminal-outcome Monte Carlo regression from verified semantic game records;
- one batched score for every legal action, followed by information-safe greedy selection;
- frozen-checkpoint self-play with explicit epsilon exploration; and
- paired, seat-balanced arena evaluation with deterministic block bootstrap intervals.

The research rationale and complete implementation plan are in
[`docs/NEURAL_AI_RESEARCH.md`](docs/NEURAL_AI_RESEARCH.md) and
[`docs/NEURAL_AI_PLAN.md`](docs/NEURAL_AI_PLAN.md).

### Historical q0 recipe

The recorded q0 corpus, dataset, and checkpoint were produced at source revision
`7fe2e5af6339efc86db285349043bb16b298eed1`. Exact reproduction requires that revision and its
locked environment; running the same commands at a later revision creates a new, separately
fingerprinted artifact set.

```bash
git switch --detach 7fe2e5af6339efc86db285349043bb16b298eed1
uv sync --extra rl --locked
uv run python -m agent_avenue corpus-generate runs/q0-corpus \
  --games 4000 --seed 20260829 --run-id q0-bootstrap
uv run python -m agent_avenue dataset-build runs/q0-corpus runs/q0-dataset.npz \
  --split-seed 20260830
uv run python -m agent_avenue train runs/q0-dataset.npz checkpoints/q0 \
  --seed 20260831
uv run python -m agent_avenue checkpoint-inspect checkpoints/q0
```

The recorded q0 arenas used source revision
`d46350356a7bf461468ba33c6bf1f28b3231cd14` and its locked environment. See
[`docs/MILESTONE5_RESULTS.md`](docs/MILESTONE5_RESULTS.md) for the exact evaluation commands and
artifact identities. Return to the current development branch before using the current self-play
CLI shown below:

```bash
git switch main
uv sync --extra rl --locked
```

Corpora, datasets, checkpoints, full reports, and game records are generated research artifacts and
remain outside normal Git history. Their manifests carry fingerprints, source revisions, seeds,
normalized configuration, and lineage. The q0 corpus bundle is about 1.5 MB for 4,000 games,
including about 1.2 MB of compressed semantic records, so the project retains complete corpora and
checkpoints rather than reducing them to aggregate metrics.

Checkpoint-backed play and evaluation are available through the optional RL extra:

```bash
uv run python -m agent_avenue game --player-one learned \
  --player-one-checkpoint checkpoints/q0 --player-two heuristic --seed 17
uv run python -m agent_avenue arena --agent-a learned \
  --agent-a-checkpoint checkpoints/q0 --agent-b random --pairs 400 --seed 2026083001
```

### Completed frozen generations

Milestone 6 ran q1 through q4 at source revision
`139318bad909438e8a3e1cb9dd962c80653875d0`. Each attempt retained q0 solely because the candidate
failed the aligned heuristic non-regression guardrail. All attempts include compressed training and
arena records, exact Git/lock identity, compatibility evidence, practical-effect assessment, and an
immutable decision. A separate full q1 regeneration reproduced every semantic artifact identity.

The exact commands, tables, and checksums are in
[`docs/MILESTONE6_RESULTS.md`](docs/MILESTONE6_RESULTS.md). Inspect a plan without writing a run:

```bash
uv run python -m agent_avenue iterate runs/milestone6/q1-a1 \
  --incumbent checkpoints/q0 --generation 1 --attempt-id q1-a1 \
  --seed 2026090100 --dry-run
```

Execute or validate the same immutable plan:

```bash
uv run python -m agent_avenue iterate runs/milestone6/q1-a1 \
  --incumbent checkpoints/q0 --generation 1 --attempt-id q1-a1 \
  --seed 2026090100
```

The orchestrator freezes paths, seeds, corpus size, epsilon, training configuration, and promotion
policy in `plan.json`. It validates and reuses completed artifacts, generates only missing games,
trains a warm-started candidate, retains compressed semantic records for every arena, runs primary
It evaluates candidate strength against the frozen incumbent, random, and aligned heuristic
and guardrail comparisons, performs the one allowed confirmation block if required, and emits an
immutable `promotion-decision.json`. It never silently mutates a global champion pointer.

The terminal-safety experiment adds a resumable generation-zero bootstrap runner, replay-derived
per-decision safety diagnostics, per-side CLI shield composition, and a held-out diagnostic that
scores each `qn` on records from every unordered prior-policy pair. Those diagnostic records never
enter training or promotion. Run the frozen q0–q4 declaration with
`scripts/run_terminal_safety_v1.py`.

## Lightweight web QA interface

The private, server-rendered web UI exists to inspect rules, observations, and opponent behavior—not
to be a public game service.

```bash
uv sync --extra web
make web
# serves http://0.0.0.0:8000
```

It currently supports:

- explicit or generated seeds;
- human-versus-human hot-seat play;
- human-versus-random and human-versus-heuristic play;
- either human seat;
- semantic form actions and public turn history; and
- terminal/reproduction metadata without hidden authoritative state.

Learned-checkpoint selection is planned but not yet implemented. When added, checkpoints will come
from a validated server-side allowlist rather than arbitrary browser-supplied paths. See
[`docs/WEB_QA.md`](docs/WEB_QA.md) for the manual checklist.

## Deterministic rules engine and baseline arena

```python
from agent_avenue.engine import apply_action, legal_actions, new_game

state = new_game(seed=17)
state = apply_action(state, legal_actions(state)[0])
```

Run a seeded game, save and verify its semantic record, or execute a paired arena:

```bash
uv run python -m agent_avenue game --seed 17 \
  --player-one heuristic --player-two random --output game.json
uv run python -m agent_avenue replay game.json
uv run python -m agent_avenue arena \
  --agent-a heuristic --agent-b random --pairs 10 --seed 17
```

Completed-game records contain normalized engine and agent configurations, independently derived
setup/agent seeds, semantic actions, terminal metadata, schema versions, and code/rules
fingerprints. Arena reports alternate seats for each paired setup seed and include paired bootstrap
and Wilson intervals, seat splits, score margins, terminal reasons, and throughput.

See [`docs/MILESTONE3_RESULTS.md`](docs/MILESTONE3_RESULTS.md) for the checked random/heuristic
baseline and `benchmarks/milestone3-smoke.json` for its fast smoke configuration.

The canonical 38-card deck cannot naturally reach deck-exhaustion adjudication: once all six
Codebreakers have been recruited, one player necessarily has the three copies required for an
earlier instant win. The exhaustion rule is nevertheless implemented and tested in isolation to
match the published base rules.

## Documentation map

- [`RULES.md`](RULES.md): normalized two-player base-game rules.
- [`docs/STATUS.md`](docs/STATUS.md): current milestone and result index.
- [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md): experiment, metrics, and retention
  policy.
- [`docs/NEURAL_AI_RESEARCH.md`](docs/NEURAL_AI_RESEARCH.md): research review and algorithm rationale.
- [`docs/NEURAL_AI_PLAN.md`](docs/NEURAL_AI_PLAN.md): detailed neural implementation plan and gates.
- [`docs/MILESTONE1.md`](docs/MILESTONE1.md), [`docs/MILESTONE2.md`](docs/MILESTONE2.md), and
  [`docs/MILESTONE3.md`](docs/MILESTONE3.md): completed foundational milestone references.
- [`docs/NEURAL_AI_PLAN.md`](docs/NEURAL_AI_PLAN.md#milestone-4-safe-encoder-dataset-model-and-checkpoint),
  [`docs/MILESTONE5_RESULTS.md`](docs/MILESTONE5_RESULTS.md), and
  [`docs/MILESTONE6_RESULTS.md`](docs/MILESTONE6_RESULTS.md): neural foundation and measured results.
- [`docs/MILESTONE6.md`](docs/MILESTONE6.md) and [`docs/MILESTONE7.md`](docs/MILESTONE7.md): current
  and planned self-play/RL milestone references.
- [`docs/MILESTONE3_RESULTS.md`](docs/MILESTONE3_RESULTS.md),
  [`docs/MILESTONE5_RESULTS.md`](docs/MILESTONE5_RESULTS.md), and
  [`docs/MILESTONE6_RESULTS.md`](docs/MILESTONE6_RESULTS.md): completed benchmark reports.
- [`docs/TERMINAL_SAFETY_EXPERIMENT.md`](docs/TERMINAL_SAFETY_EXPERIMENT.md): implemented zero-ply
  terminal-safety shield, controlled q0–q4 rerun, held-out all-pairs diagnostic plan, and Milestone 6
  failure analysis. Controlled results are pending.
- [`docs/WEB_QA.md`](docs/WEB_QA.md): manual web security and behavior checks.

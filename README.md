# agent-avenue-ai
[Agent Avenue](https://boardgamegeek.com/boardgame/422732/agent-avenue) AI experiments.

All gameplay content is copyright Nerdlab Games. This is just an ML research project and
I don't intend to host the game as a playable app.

## Project goal
Build a strong, reproducible, and inspectable AI for the two-player base game of **Agent Avenue**—not
just a rules simulator. The broader aim is a complete playable opponent and research platform in
the spirit of [Keldon Jones's 2009 Race for the Galaxy AI project](https://www.keldon.net/rftg/):
combine a faithful game implementation with self-play, learned evaluation, rigorous comparison,
and artifacts that let results be reproduced and improved over time. The project is inspired by
that end-to-end ambition rather than committed to reproducing Keldon's exact methods.

The engine should be deterministic for all the base two-player game for now.
Setup randomness is explicit; every player action is a first-class decision;
transitions are deterministic and replayable; and authoritative state
is strictly separated from player-visible observations. Random, scripted, and simple heuristic
agents plus single-game and pull-based multi-game runners establish non-ML baselines and the batch
orchestration seam.

There should be a lightweight web UI to play the game (to verify the rules engine is correct
and kick the tires of an AI opponent), but this can be barebones as it's not the main goal.
Just enough for to QA the system.

The first learned baseline: a viewpoint-relative flat observation-and-action encoder, a small
PyTorch candidate-value network, terminal-outcome training from compact replay data,
information-safe one-ply candidate selection, frozen-checkpoint iterative self-play, and
statistically grounded arena evaluation. It must run usefully on a CPU-only exe.dev development box
while retaining clean paths to parallel actors and GPU-backed training or inference. The research
review and implementation breakdown are in
[`docs/NEURAL_AI_RESEARCH.md`](docs/NEURAL_AI_RESEARCH.md) and
[`docs/NEURAL_AI_PLAN.md`](docs/NEURAL_AI_PLAN.md).

Preserve the engine as an explicit state machine. Every player choice remains a semantic
`Decision`, and applying one semantic `Action` advances to the next decision or terminal result.
The engine never calls agents or exposes model tensors. Learned
components consume only versioned player-safe observations and public decision context; trusted
orchestration may operate the engine but must not allow candidate evaluation to exploit hidden
deck order, facedown cards, or hands.

Keep rules execution, observation and public-context construction, encoding, model definition,
training, inference, self-play orchestration, replay storage, evaluation, and profiling loosely
coupled. Actions remain stable structured data rather than display strings or legal-action
indices. Every dataset, checkpoint, run, and arena report records enough schema versions,
fingerprints, seeds, and configuration to reject incompatible inputs and reproduce its result.

The long-term measure of success is an AI that becomes meaningfully stronger through
iterative self-play while remaining fair, testable, auditable, and practical to run—not merely a
neural-network demo attached to the rules engine.

## Development

This repository targets Python 3.12 and uses [uv](https://docs.astral.sh/uv/) for environments and
dependency locking.

```bash
make setup
make check
make run
```

Optional dependencies can be installed with `uv sync --extra web`, `uv sync --extra rl`, or both.
See `AGENTS.md` for architecture, reproducibility, testing, and contribution conventions.

## Lightweight web QA interface

Milestone 2 provides a private, server-rendered hot-seat interface for checking game rules and
hidden-information behavior. It is a QA tool for this research project, not a public hosted copy of
the game. Install the optional dependencies and start the single-process in-memory server:

```bash
uv sync --extra web
make web
# serves http://0.0.0.0:8000
```

The interface supports explicit or generated seeds, human-versus-human play, human-versus-random
and human-versus-heuristic games from either seat, semantic form actions, public turn summaries,
terminal results, and reproduction metadata. Automated controllers advance only until the next
human decision and receive the same player-safe observations as every other agent. Active games are
intentionally lost when the process restarts. Run its focused tests with `make test-web`; see
[`docs/WEB_QA.md`](docs/WEB_QA.md) for the manual smoke checklist.

## Deterministic rules engine

Milestone 1 provides a typed, immutable two-player base-game engine in
`agent_avenue.engine` and a separate player-safe observation boundary in
`agent_avenue.observation`.

```python
from agent_avenue.engine import apply_action, legal_actions, new_game

state = new_game(seed=17)
state = apply_action(state, legal_actions(state)[0])
```

Run a seeded automated game, save its completed-game record, verify it, or execute a paired arena:

```bash
uv run python -m agent_avenue game --seed 17 \
  --player-one heuristic --player-two random --output game.json
uv run python -m agent_avenue replay game.json
uv run python -m agent_avenue arena \
  --agent-a heuristic --agent-b random --pairs 10 --seed 17
```

Completed-game records contain normalized engine and agent configurations, independently derived
setup/agent seeds, semantic actions, terminal metadata, versions, and code/rules fingerprints.
Arena reports alternate seats for every paired setup seed and include Wilson 95% confidence
intervals, seat splits, score margins, terminal reasons, and throughput. See
[`docs/MILESTONE3_RESULTS.md`](docs/MILESTONE3_RESULTS.md) for the checked baseline and
`benchmarks/milestone3-smoke.json` for the fast smoke configuration.

The Milestone 1 engine replay format remains supported by the `replay` command. The committed
`tests/fixtures/scripted_seed17.replay.json` is a small compatibility fixture.

The canonical 38-card deck cannot naturally reach the deck-exhaustion adjudication: once all six
Codebreakers have been recruited, at least one of two players necessarily has the three copies
required for an earlier instant win. The exhaustion rule is nevertheless implemented and tested as
an isolated adjudication rule, matching the published base rules.

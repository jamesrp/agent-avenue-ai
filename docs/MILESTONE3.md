# Milestone 3: Initial AI Opponent and Baseline Arena

## Goal

Add fair, reproducible automated opponents on top of the Milestone 1 engine and make them playable
through the Milestone 2 web UI. This milestone establishes the agent/runner/evaluation seams needed
for later self-play without introducing a learned model yet.

The principal playable opponent is a simple information-safe heuristic agent. Random and scripted
agents provide control baselines and deterministic test tools.

## Prerequisites

- Milestone 1 provides stable semantic decisions/actions, safe observations, deterministic replay,
  and structured terminal outcomes.
- Milestone 2 provides a controller seam that can assign a human or automated controller to either
  seat.

An agent must not receive authoritative state merely because the runner is trusted. If the desired
heuristic cannot be implemented from the acting player's information, simplify the heuristic rather
than weakening this boundary.

## Scope

### Agent interface

Define a narrow synchronous protocol equivalent to:

```python
class Agent(Protocol):
    def choose_action(
        self,
        observation: PlayerObservation,
        decision: PublicDecision,
        legal_actions: tuple[Action, ...],
        rng: RandomSource,
    ) -> Action: ...
```

The exact signature may differ, but it must guarantee:

- observations and decision context are viewer-safe;
- legal actions are stable semantic values, not UI indexes;
- agent randomness comes from an explicit seeded stream;
- agents return a choice but never mutate or advance engine state;
- an illegal returned action fails loudly at the runner boundary instead of silently falling back.

Keep setup randomness and each agent's randomness in separate, reproducibly derived streams. Record
the seed derivation scheme so adding a random call inside one agent cannot silently reshuffle setup.

### Baseline agents

Implement three agents:

1. **Random agent**: uniformly selects from legal actions using only its supplied RNG.
2. **Scripted agent**: consumes semantic actions or a decision-aware callback and is intended for
   tests, demos, and replay regressions.
3. **Greedy heuristic agent**: scores legal actions from the current observation and public card
   definitions, with deterministic seeded tie-breaking.

A useful initial heuristic can use:

- immediate score effects based on public recruited-card counts;
- distance to the seven-point win threshold;
- Codebreaker and Daredevil instant win/loss risk;
- worst-case or expected immediate score swing when offering two known cards;
- an expected value for an unknown face-down recruit based on the information-visible remaining
  card multiset, never the actual hidden card;
- small configurable weights expressed in a normalized, serializable agent configuration.

Document the policy's assumptions. In particular, distinguish “unknown to the agent” from “known by
the engine.” Do not use authoritative candidate afterstates, deck order, the opponent's hand, or the
true face-down card to score a choice.

### Trusted game runners

Add orchestration outside the engine:

- a single-game runner that alternates controllers until terminal;
- a step/resume runner suitable for human-versus-agent web games;
- a pull-based batch iterator that yields completed game records without retaining an entire run;
- an arena runner that schedules paired seeds and alternates seats.

Each completed game record should include:

- rules/replay schema versions and code/rules fingerprint;
- normalized engine and agent configurations;
- setup and per-agent seeds;
- seat assignment;
- semantic action replay;
- terminal winner/reason, final score, and decision/turn counts;
- creation timestamp and run identifier.

Keep parallelism out of the first implementation, but make one game an independent unit so later
workers can consume scheduled game specifications without changing semantics.

### Arena evaluation

Compare agents over a declared, reproducible seed set with balanced seats. Report at least:

- total games and paired seed count;
- wins and win rate overall and by seat;
- a 95% binomial confidence interval for win rate;
- terminal reasons and average score margin;
- average turns/decisions and games per second;
- full normalized configurations and fingerprints.

The report must not label one agent stronger based on a single game or an unsupported point
estimate. Commit a small, fast smoke benchmark configuration; keep large reports and replay corpora
out of Git.

### Web and CLI integration

Extend new-game setup to support:

- human versus human;
- human versus random;
- human versus heuristic;
- human seat selection and explicit game seed.

After a human submits an action, the controller should advance automated decisions until the next
human decision or terminal outcome. Render only the human viewer's safe observation and public
transition summaries; agent diagnostics must not leak hidden state.

Add CLI commands or an equivalent entry point for:

- one seeded agent-versus-agent game with optional replay output;
- a deterministic arena run;
- replaying a recorded game and verifying its fingerprint/outcome.

## Suggested package layout

```text
src/agent_avenue/
  agents/
    base.py
    random.py
    scripted.py
    heuristic.py
  runners/
    game.py
    batch.py
    arena.py
    records.py
  storage/
    game_record.py
  cli.py
```

Agent modules may import observation/public card definitions, but not authoritative engine state or
web code. Runners may import both engine and agents because they are the trusted orchestration layer.

## Implementation sequence

1. Define the agent protocol, random-source abstraction, and versioned agent configuration.
2. Implement the scripted and random agents and validate the runner boundary with them.
3. Implement deterministic single-game and step/resume runners.
4. Add compact completed-game records and replay verification.
5. Implement and unit-test heuristic feature calculations from safe observations.
6. Implement the greedy policy and deterministic tie-breaking.
7. Add paired-seed, alternating-seat arena scheduling and statistical reports.
8. Add CLI single-game, replay, and arena commands.
9. Connect human-versus-agent modes to the web controller seam.
10. Run a documented random-versus-random fairness check and heuristic-versus-random baseline.
11. Profile throughput and record the CPU baseline before considering optimization.

## Test plan

Cover at least:

- every agent always returns a member of the supplied legal-action tuple;
- the agent protocol provides no path to authoritative state;
- the random agent selects each legal action when driven by a controlled RNG that returns the
  corresponding choice, without relying on a probabilistic test;
- fixed setup/agent seeds reproduce the same action sequence and outcome;
- setup randomness is unchanged when an agent consumes additional random values;
- scripted agents produce expected complete-game regressions and fail clearly when scripts diverge;
- heuristic card-count, score-effect, threshold, and instant-condition features;
- unknown face-down evaluation uses only the viewer's information set;
- deliberately different hidden states with the same observation produce the same heuristic scores
  and seeded choice;
- single-game records replay to the same outcome and fingerprints;
- batch results match individually executed game specifications;
- arena schedules balance seats and reuse paired setup seeds correctly;
- confidence intervals and summary statistics match known fixtures;
- invalid agent output is rejected without mutating game state;
- human-versus-agent web flows stop at each human decision and do not leak agent-only information;
- terminal results work when an AI action ends the game;
- deterministic CPU throughput smoke tests report performance without flaky time assertions.

## Deliverables

- Random, scripted, and versioned greedy heuristic agents.
- Single-game, resumable, batch, and arena runners.
- Compact versioned game records with replay verification.
- CLI commands for games, arenas, and replay checks.
- Human-versus-random and human-versus-heuristic web modes.
- A checked-in small benchmark specification and a documented local baseline result.

## Definition of done

- A human can complete a game against either random or heuristic AI from either seat.
- Agents consume only player-safe observations, public context, legal actions, and explicit RNG.
- Repeating an arena with the same code, configurations, and seeds reproduces every game.
- Arena output separates seat effects and includes uncertainty rather than anecdotal claims.
- The heuristic-versus-random result is recorded honestly; any strength claim is supported by its
  confidence interval.
- The implementation runs usefully on the CPU-only development VM without loading RL dependencies.
- `make check` and documented CLI/web smoke commands pass.

## Out of scope

- Neural networks, tensor observation encoders, PyTorch, or checkpoint loading.
- Self-play training, target generation, model promotion, or distributed actors.
- Information-set search, hidden-state sampling beyond the documented simple expectation, or tree
  search.
- Declaring the heuristic a strong final AI; it is a reproducible baseline for the learned stage.
- Large replay datasets, experiment artifacts, or model files in Git.

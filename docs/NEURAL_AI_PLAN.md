# Neural AI implementation plan

**Status:** proposed plan after Milestones 1–3
**Date:** August 29, 2026
**Companion research:** [`NEURAL_AI_RESEARCH.md`](NEURAL_AI_RESEARCH.md)

## Goal

Build the first reproducible learned Agent Avenue opponent on a CPU budget of low tens of thousands
of games. The initial system should be strong enough to compare honestly with the random and
`greedy-public-v1` baselines, but simple enough that information safety, target construction, and
artifact lineage remain inspectable.

The baseline is deliberately not AlphaZero. It is a small Monte Carlo-trained candidate-action
value network with frozen-checkpoint self-play:

```text
Q_theta(current acting-player observation, legal semantic action)
    = probability that the acting player eventually wins
```

At inference, encode all legal actions, evaluate them in one batch, and choose the maximum. The
network never receives `GameState`, a true face-down card unknown to the actor, the opposing hand,
or deck order.

## Existing baseline and budget

The committed Milestone 3 benchmark establishes:

- random versus random: 51.0% over 400 games;
- heuristic versus random: 86.25% over 400 games;
- about 13.3–13.7 decisions and 6.6–6.8 turns per game;
- maximum legal-action count 12 by the engine's legal-action construction; and
- about 44–45 games/second without neural inference on the current VM.

At those observed game lengths, a 20,000-game training program yields roughly 265,000 chosen
state/action/outcome examples across all generations. An 87-to-128-to-1 network has only 11,393
trainable parameters, so model arithmetic should not be the bottleneck.

## Non-negotiable information boundary

### Safe inputs

The learned policy may consume only:

- the current actor's typed `PlayerObservation`;
- its public decision context;
- the supplied tuple of semantic legal actions; and
- its explicit seeded agent RNG.

The encoder must accept `PlayerObservation` plus one legal `Action`; it must not accept
`GameState`.

This is an architectural, regression-tested boundary for trusted repository code, not a Python
sandbox. An in-process hostile agent could inspect process memory or stack frames. Supporting
untrusted third-party agents would require a separate process/RPC boundary with allowlisted
serialized messages.

### Why this is action-conditioned rather than a raw successor value

Applying candidate actions to the authoritative state is unsafe for comparison:

- after a play action, the real successor contains the opponent's actual hand and the offerer's real
  replacement draws; and
- after a recruit action, the real successor reveals the face-down card through recruitment,
  scoring, and possibly terminal adjudication.

Even if a successor observation is safe for its own viewer, it may contain private or future
information unavailable to the player who is currently selecting the action. Therefore the v1
“afterstate” is an **information-set afterstate** represented by the actor's current observation plus
the contemplated action—not an engine successor.

The authoritative hidden state remains valid for:

- executing the action after the policy chooses it;
- producing the eventual terminal outcome label; and
- reconstructing safe observations inside trusted replay extraction.

It is not valid as candidate-selection input.

### Required invariance test

If two authoritative states produce the same acting-player observation and legal semantic actions,
then the following must be identical under the same checkpoint and RNG:

- every encoded candidate vector;
- every model score; and
- the selected action.

This is the central learned-agent information-safety regression.

## Baseline learning design

### Value semantics

For actor `i`, observation `o_i`, and legal action `a`:

```text
Q_theta(o_i, a) = P(final winner is i | o_i, choose a, generating continuation policy)
```

The target is binary:

- `1.0` if the acting player at that sample eventually wins;
- `0.0` otherwise.

There are no draw labels because the engine always declares one winner. This is policy-conditional
Monte Carlo action value, not a claim to game-theoretic equilibrium value.

### Encoder v1: `candidate-public-v1`

Use a fixed `tuple(CardName)` ordering and canonicalize players to `self` and `opponent`. The encoder
accepts only a nonterminal observation owned by its decision actor and one member of
`observation.legal_actions`.

Proposed width: **87 features**.

| Feature group | Width | Encoding |
| --- | ---: | --- |
| Decision phase | 2 | one-hot play/recruit |
| Turn and remaining deck | 2 | clipped fractions using the 19-turn mechanical ceiling and 30 post-deal deck cards |
| Own score, opposing score, gap | 3 | own/opponent clipped to `[-14, 14] / 14`; gap clipped to `[-7, 7] / 7` |
| Own/opposing hand sizes | 2 | fraction of four |
| Own hand counts | 8 | count divided by canonical copies for each card name |
| Own recruited effect thresholds | 18 | three thresholds for repeatable score cards, two for Codebreaker/Daredevil, one for Sidekick/Mole |
| Opposing recruited effect thresholds | 18 | same, viewpoint-relative |
| Information-visible unseen counts | 8 | remaining unknown count divided by canonical copies |
| Current recruit face-up card | 8 | one-hot; all zero during play |
| Candidate play face-up card | 8 | one-hot; all zero during recruit |
| Candidate play face-down card | 8 | one-hot; all zero during recruit |
| Candidate recruit slot | 2 | one-hot face-up/face-down; all zero during play |

The 19-turn scale follows the canonical card flow: 30 post-deal deck cards refill two offered cards
for 15 turns, leaving two four-card hands; four more no-refill turns consume those eight cards, and
deck exhaustion adjudicates after turn 19. Normal games terminate earlier, but a boundary fixture
must verify this mechanical scale against the engine.

The recruited thresholds deliberately omit impossible nonterminal features: Sidekick/Mole counts
above one and Codebreaker/Daredevil counts at three. The latter terminate the game immediately, so
no later decision observation can contain them.

The recruit actor must always have `known_face_down is None`; reject any impossible actor payload
rather than encode a dead “known hidden card” feature.

`unseen counts` are derived only from the safe observation: canonical deck counts minus both public
recruited sets, own hand, and the currently visible face-up offer when present. They intentionally
combine opponent hand, deck, and the unknown face-down card.

The v1 encoder omits chronological history. That loses policy-induced belief information but keeps
the first experiment small. Source corpora retain complete semantic game records, so a future
`candidate-public-history-v2` can add padded public history and re-encode the same games without
regenerating them.

The implementation must emit stable feature names alongside the vector. Encoder fingerprinting
covers version, feature names/order, card order, scaling constants, and width.

### Model v1: `candidate-mlp-v1`

Conceptual architecture:

```python
Linear(87, 128)
Tanh()
Linear(128, 1)
Sigmoid()
```

Implementation details:

- return a logit from `forward()`;
- train with `BCEWithLogitsLoss`;
- apply sigmoid only for reported probabilities;
- explicitly seeded Xavier initialization;
- no dropout, batch normalization, residual layers, embeddings, recurrent state, or policy head;
- batch all 1–12 legal candidates in one forward pass; and
- run inference under `torch.inference_mode()` on CPU.

Parameter count: `87 * 128 + 128 + 128 + 1 = 11,393`.

### Action policy

Deployment policy:

1. validate observation/decision/legal-action consistency;
2. sort candidates by an explicit semantic key based on fixed card/slot order, independent of the
   supplied tuple or hand order;
3. encode and score the sorted candidate batch;
4. find the maximal logit;
5. use the supplied agent RNG only to break exact ties in the same semantic order; and
6. return the matching original supplied semantic action value.

Self-play policy wraps greedy selection with seeded epsilon exploration. Serialize epsilon as an
exact rational numerator/denominator and decide exploration with
`rng.randbelow(denominator) < numerator`; exploratory selection is uniform over the semantically
sorted candidates. Values are `1/5`, `1/10`, `3/40`, `1/20`, and `1/40` for the planned schedule.
The config also fixes whether epsilon 0 consumes no RNG call, epsilon 1 skips greedy inference, and
how exact greedy ties consume RNG.

Do not special-case a candidate using the authoritative engine's actual terminal result. At a
recruit decision, terminality may depend on the unknown face-down card. The model must rank the
choice from pre-action information.

## Data and artifact design

### Raw corpus

The source of truth remains verified `GameRecord` data. Store generated games as an append-only,
compressed JSON Lines corpus plus a small manifest. **Raw game records are trusted, omniscient
research artifacts**: setup seeds plus semantic actions can reconstruct deck order, both hands, and
face-down cards. Never expose a corpus handle, replay seed, or raw record to a live agent or browser.

The safety guarantee applies to extracted model features and agent-visible diagnostics, not to the
trusted raw replay corpus.

A corpus manifest records at least:

- schema version plus a semantic identity fingerprint;
- run/generation ID and creation timestamp;
- ordered game-record fingerprints;
- rules, replay, game-record, RNG, and code fingerprints;
- normalized engine and both agent configurations;
- root seed and seed-derivation domains;
- behavior checkpoint fingerprint or baseline policy version;
- exploration configuration; and
- game count, decision count, outcomes, seats, and terminal reasons.

Corpus identity hashes an `identity_payload` that excludes its self-hash, timestamps, output paths,
elapsed times, and gzip metadata, plus the ordered semantic game-record fingerprints. Reproduction
means equal semantic identities and contents after canonical decompression, not byte-identical
timestamped files.

Corpora, datasets, checkpoints, and reports belong under ignored `runs/` or `artifacts/`; only tiny
test fixtures and small benchmark specifications belong in Git.

### Replay-to-sample extraction

For every verified completed game:

1. reconstruct from setup seed and semantic actions;
2. before each action, determine the actor;
3. build `observe(state, actor)`;
4. pair that observation with the actual chosen action;
5. assign the terminal winner label from that actor's viewpoint;
6. encode the pair using the declared encoder; and
7. advance with the recorded semantic action.

The extractor reads a trusted record, replays only the prefix available before each decision, and
emits no current deck, opponent hand, true unknown face-down card, setup seed, or authoritative state
into the materialized model input. The terminal label is intentionally future information used only
for offline training.

Historical corpora use an explicit semantic-verification policy. Because the current code
fingerprint covers all package Python files, adding learning modules will make old records fail the
default exact-code check. A recorded compatibility decision may allow a code mismatch only after
replay/rules schemas, stored/current fingerprints, semantic replay, and terminal/final-state checks
pass. Never silently set `verify_code=False`.

### Materialized dataset

Use a compact NumPy artifact for training speed:

- `features`: float32 `[N, 87]`;
- `targets`: uint8 or float32 `[N]`;
- `game_index`: integer `[N]` or game-offset table;
- audit provenance outside model input: game-record fingerprint, decision index, actor, winner, and
  behavior policy ID;
- optional phase metadata for diagnostics, not model input; and
- a JSON manifest with source corpus fingerprints and exact split assignment.

Split by complete games—and by pair ID when paired records exist—never by individual decisions.
Default split: deterministic 90% train / 10% validation. Hash the split seed plus the game/pair
identity with SHA-256 and assign by a documented integer threshold, so input order cannot change the
split. Promotion and final arena seeds are not present in either split.

### Checkpoint format

Use two distinct artifact types.

Immutable inference bundle:

```text
checkpoint-id/
  manifest.json
  weights.pt
  metrics.json
```

Optional resumable trainer snapshot:

```text
training-state-id/
  manifest.json
  current-weights.pt
  optimizer.pt
  trainer-state.json
```

The trainer snapshot additionally records epoch, current/best checkpoint identities, best metric,
early-stop counter, shuffle-generator state, and every state needed to continue deterministically.
Resume support may be deferred from v1; do not imply that optimizer state alone is sufficient.

Validate manifest, named file digests, encoder/model/rules compatibility, and weights before creating
a game session. Load model weights with the safe weights-only PyTorch path. A process-level cache may
hold a validated immutable inference model and let per-game factories return lightweight agents that
share it read-only.

Checkpoint identity hashes a canonical `identity_payload` that excludes its self-hash, timestamps,
paths, elapsed times, and volatile environment diagnostics, plus named weight/metrics file digests.
It is not a mutable filesystem path or a hash that recursively includes itself.

The manifest contains:

- checkpoint schema and fingerprint;
- model version, dimensions, parameter count, and output semantics;
- encoder version/fingerprint/feature names;
- rules/replay/code fingerprints;
- Torch, NumPy, Python, platform, device, and thread settings;
- deterministic algorithm settings and all training seeds;
- optimizer and normalized training configuration;
- source corpus/dataset fingerprints and sample counts;
- train/validation metrics;
- parent checkpoint and generation lineage; and
- inference-versus-training-state artifact kind.

Loading fails loudly on schema, encoder, width, model, rules, or digest mismatch.

## Initial training defaults

Treat these as versioned baseline defaults, not universally optimal hyperparameters:

- optimizer: AdamW;
- learning rate: `1e-3`;
- weight decay: `1e-4`;
- batch size: `1024`;
- maximum epochs: `50`;
- early-stopping patience: `8` validation epochs;
- loss: binary cross entropy with logits;
- shuffle: one explicitly seeded Torch generator;
- data-loader workers: `0` initially;
- CPU threads: explicitly configured and recorded; use one thread for the strict reproducibility
  baseline, then profile a faster recorded setting;
- gradient clipping: disabled unless a measured instability justifies it; and
- model selection: lowest equal-game-weighted validation log loss: compute mean BCE within each held-
  out game, then average those game means; report pooled decision loss separately.

Train/validation reports include equal-game-weighted and pooled loss, Brier score, accuracy at 0.5,
outcome balance, phase split, calibration bins, examples/second, and wall-clock duration. Arena
strength, not validation accuracy, is the promotion criterion.

## Self-play program and game budget

### Generation 0: heuristic bootstrap

Generate **4,000 games** with both seats using `greedy-public-v1` wrapped in epsilon exploration at
`epsilon = 1/5`. This provides a sensible continuation policy plus broad candidate coverage without
inventing synthetic labels.

Extract chosen candidates and terminal outcomes and train checkpoint `q0` only on this generation's
games. Its declared target is the outcome under the epsilon-heuristic continuation policy. Evaluate
q0 on 400 paired seeds each versus random and the heuristic. Require the pair-bootstrap 95% lower
bound above 0.50 versus random before calling q0 a reasonable opponent or proceeding to the
iterative strength milestone.

An optional fallback experiment may pretrain candidate rankings from safe heuristic scores before
terminal fine-tuning. It is not part of the required baseline and must have a separate configuration
and ablation report.

### Generations 1–4: frozen checkpoint self-play

For each generation `k`, let `c{k-1}` be the current incumbent:

1. freeze incumbent checkpoint `c{k-1}` for the entire generation;
2. generate **4,000 games** with that checkpoint in both seats;
3. use seeded epsilon values `1/10`, `3/40`, `1/20`, and `1/40` for generations 1–4;
4. build this generation's train/validation dataset and retain older corpora for audit/experiments,
   not baseline fitting;
5. initialize candidate `q{k}` from the incumbent weights;
6. train offline only on the current generation—never while a game is in progress;
7. declare the target as outcome under that generation's epsilon-incumbent continuation policy;
8. evaluate candidate versus incumbent and guardrail baselines; and
9. set `c{k} = q{k}` only if the gate passes; otherwise set `c{k} = c{k-1}`.

Training generation total: `4,000 + 4 * 4,000 = 20,000 games`, approximately 265,000 decisions
across retained corpora at committed baseline lengths. The fixed base evaluation program adds 1,600
q0 games, 8,800 promotion/guardrail games, and up to 4,000 unique final-matchup games: **34,400
games total**. One optional earlier-champion diagnostic adds 1,000; confirmation blocks can add at
most 8,000 more, for a predeclared worst case of 43,400. The training budget itself remains 20,000.

Current-generation-only fitting is the v1 rule so the sigmoid has a named behavior-policy target.
Cumulative replay would instead learn a heterogeneous behavior-mixture outcome score; keep that as a
separately named experiment rather than silently changing `Q^pi` semantics. Every sample retains its
generating policy metadata.

### Promotion gate

Use fresh, predeclared promotion blocks per generation, derived from a promotion master seed and the
generation number. Alternate Player One/Player Two seats for every setup seed. Reserve a separate
locked final block.

Primary comparison:

- 500 paired setup seeds = 1,000 candidate-versus-incumbent games;
- pair score is candidate wins in that two-seat pair divided by two;
- 20,000 deterministic bootstrap resamples of the 500 pair blocks, sampled with replacement using
  the project `DeterministicRandom` and a recorded bootstrap seed;
- sort the 20,000 bootstrap means ascending and take zero-based indices 499 and 19,499 (nearest-rank
  2.5% and 97.5% endpoints); and
- promote only if the lower endpoint is above 0.50.

Guardrails:

- candidate point-estimate win rate is at least 45% as Player One and at least 45% as Player Two;
- 200 paired seeds candidate versus random, with the same pair-bootstrap lower endpoint above 0.50;
- 200 shared seed blocks for candidate-versus-heuristic and incumbent-versus-heuristic, each block
  containing both seat-swapped matchups for both learned agents; bootstrap the block-level
  candidate-minus-incumbent score difference and require its lower endpoint above `-0.05`; and
- all compatibility, replay, hidden-information, and deterministic reproduction checks pass.

The q0 gate and every 200-pair guardrail use the same 20,000-resample algorithm, order-statistic
indices, and domain-separated bootstrap-seed derivation.

Wilson intervals remain in reports for continuity with Milestone 3, but only the block-aware
bootstrap supports promotion claims.

A primary lower endpoint from 0.49 through 0.50 may trigger exactly one predeclared fresh
1,000-pair confirmation block; promotion then requires its lower endpoint above 0.50. No other
borderline reruns are allowed. Zero promotions is a valid research result. Keep the incumbent and
report failure rather than weakening the gate after seeing results.

### Final report

Use 500 paired setup seeds per unique declared matchup from a locked seed block unused for training
or promotion. Compare the final champion with the unique applicable members of:

- random;
- `greedy-public-v1`;
- initial `q0`;
- its immediate parent, if one exists and differs from q0; and
- one earlier champion only when needed to diagnose cycling.

Deduplicate identical checkpoint identities. If no promotion occurs, report q0 as the retained
champion and omit champion-versus-q0 and parent comparisons rather than fabricating them.

Report paired and seat-specific win rates, paired confidence intervals, ordinary Wilson intervals
for continuity, score margin, terminal reasons, game lengths, candidate evaluations/second,
games/second, and full artifact fingerprints.

## Package and dependency boundaries

Suggested additions:

```text
src/agent_avenue/
  encoding/
    __init__.py
    candidate_v1.py
    schema.py
  learning/
    __init__.py
    model.py
    checkpoint.py
    dataset.py
    train.py
  agents/
    exploration.py
    learned.py
  runners/
    self_play.py
    iteration.py
  storage/
    corpus.py
    dataset_manifest.py
    checkpoint_manifest.py
```

Ownership rules:

- `encoding` is pure Python and imports observation/engine domain values, never Torch or
  authoritative `GameState`.
- `learning` owns NumPy/Torch, tensor conversion, model, training, and checkpoint I/O.
- `agents.learned` adapts a loaded inference model to the existing safe `Agent` protocol.
- `runners` schedules independent games/generations and never exposes hidden state to agents.
- `storage` owns normalized schemas/fingerprints, not training behavior.
- `engine` and `observation` remain unchanged unless a missing safe public field is proven necessary.

Do not eagerly import the learned agent or Torch from `agent_avenue.agents.__init__`; a plain
`uv sync` must still import and run the engine, existing agents, and replay tools. RL commands use
lazy imports and require `uv sync --extra rl`.

## CLI and workflow target

Exact command names can be refined during implementation, but the workflow must expose independent,
scriptable stages equivalent to:

```bash
# Generate verified baseline or checkpoint games.
uv run python -m agent_avenue corpus generate ...

# Reconstruct safe samples and materialize a versioned dataset.
uv run python -m agent_avenue dataset build ...

# Train or resume one declared checkpoint.
uv run python -m agent_avenue train ...

# Run a normal paired arena with a checkpoint-backed AgentSpec.
uv run python -m agent_avenue arena --agent-a checkpoint:... ...

# Orchestrate one frozen generation and emit, but do not silently apply, a promotion decision.
uv run python -m agent_avenue iterate ...

# Inspect artifact compatibility and lineage without loading a game.
uv run python -m agent_avenue checkpoint inspect ...
```

Every command supports explicit output paths, root seeds, normalized config output, and `--dry-run`
or equivalent schedule inspection where useful. A high-level iteration command composes the lower
level stages; it does not hide their manifests.

## Milestone 4: Safe encoder, dataset, model, and checkpoint

### Objective

Freeze the learned-data contracts and prove deterministic CPU training before adding a neural agent
to gameplay.

### Implementation steps

1. Add `agent_avenue.encoding` with card order, feature schema, normalized config, version, and
   fingerprint types.
2. Implement `candidate-public-v1` from `PlayerObservation` plus one legal action.
3. Add golden feature-name/vector fixtures for play and recruit decisions, duplicate hands, threshold
   counts, negative scores, all card types, and the 19-turn/deck-exhaustion scaling boundary.
4. Add hidden-equivalent-state tests proving identical candidate vectors.
5. Define corpus and dataset manifest schemas with canonical JSON serialization and digests.
6. Implement verified `GameRecord` corpus reading/writing and replay-to-sample extraction.
7. Materialize deterministic game-level train/validation splits and NumPy arrays.
8. Add `candidate-mlp-v1`, explicit initialization, batch inference, and parameter reporting.
9. Add deterministic CPU trainer, metrics, early stopping, and tiny overfit fixture.
10. Add checkpoint save/load/inspect with compatibility and tamper rejection.
11. Add RL-specific Make targets and tests while retaining a plain-core import smoke test without
    Torch.
12. Profile encode, dataset extraction, train step, and inference batch throughput.

### Acceptance criteria

- Encoder width is exactly 87 and names/order match the committed golden fixture.
- Equal acting-player observations/actions always encode equally regardless of hidden state.
- Replay extraction reproduces the exact ordered samples and labels from a fixture.
- Train/validation splits are deterministic and game-level.
- No model-input feature or agent-visible diagnostic contains an authoritative/hidden field; audit
  metadata and terminal targets are stored separately from features.
- A tiny declared corpus can be intentionally overfit, proving target perspective and model wiring.
- Repeated strict-CPU training with the same environment/config/data/seed produces bitwise-equal
  model state tensors and the same canonical tensor digest; raw `.pt` file bytes need not match.
- Checkpoint schema/encoder/model/rules/digest mismatches fail before inference.
- Plain engine and existing-agent imports work without NumPy/Torch installed.
- Existing `make check` remains green; a documented RL check passes after `uv sync --extra rl`.

### Out of scope

- Gameplay integration, self-play generation, web UI, TD targets, search, or strength claims.

## Milestone 5: Learned agent and first usable checkpoint

### Objective

Integrate the action-conditioned model through the existing safe agent protocol, generate the
heuristic bootstrap corpus, train `q0`, and establish honest random/heuristic arena results.

### Implementation steps

1. Implement a generic serializable epsilon-exploration wrapper using the supplied `RandomSource`.
2. Implement `LearnedValueAgent` using an injected, prevalidated process-level inference object;
   do not defer compatibility or weight validation until the first decision.
3. Validate each candidate action, encode the full legal tuple, and batch one forward pass.
4. Serialize checkpoint identity, encoder/model versions, epsilon, device, and tie policy in agent
   configuration.
5. Add learned-agent factories to CLI/game/arena paths without eager Torch imports.
6. Add corpus generation with deterministic game specifications, streaming record sink, resumable
   manifest, and frozen agent configuration.
7. Generate 4,000 epsilon-heuristic bootstrap games under a committed run specification.
8. Build the dataset, train `q0`, and record train/validation/calibration results.
9. Run paired q0-versus-random and q0-versus-heuristic arenas on disjoint declared seeds.
10. Add a small neural smoke benchmark specification; keep checkpoints and large reports ignored.
11. Profile neural games/second and candidate evaluations/second.
12. Document failures and ablations if q0 does not become usable; do not proceed directly to MCTS.

### Acceptance criteria

- The learned agent uses the unchanged `Agent` protocol and always returns a supplied semantic
  action.
- Different hidden states with equal actor observations/actions produce equal logits and choices.
- Fixed checkpoint, setup seeds, agent seeds, and epsilon reproduce every semantic action and
  semantic game-record fingerprint; timestamps and compressed file bytes need not match.
- Neural `GameRecord`s replay and verify through existing storage APIs.
- Loading a checkpoint with incompatible metadata fails before game creation.
- q0 is evaluated on exactly 400 paired seeds each versus random and heuristic; the q0-versus-random
  pair-bootstrap lower endpoint must exceed 0.50 before Milestone 6 strength work begins. Failure
  does not invalidate the infrastructure milestone, but it blocks claims of a reasonable opponent
  and triggers encoder/data debugging.
- The result against `greedy-public-v1` is reported without requiring q0 to win.

### Out of scope

- Online learning, automatic promotion, opponent pools, history encoders, or web play.

## Milestone 6: Frozen iterative self-play and promotion

### Objective

Run reproducible offline policy-improvement generations and produce a champion selected by a
predeclared statistical gate.

### Implementation steps

1. Define normalized generation and iteration configs with parent checkpoint lineage.
2. Implement deterministic schedules for 4,000 independent games per generation.
3. Freeze the incumbent for generation; prohibit checkpoint mutation while any game uses it.
4. Add resumable corpus writing that verifies existing records before skipping completed work.
5. Build each candidate dataset from its current generation and retain all older corpora for audit
   and separately named replay experiments.
6. Initialize each candidate from the incumbent and train with the fixed v1 defaults.
7. Implement pair-aware promotion statistics and deterministic bootstrap confidence intervals.
8. Run candidate-versus-incumbent, random guardrail, and heuristic non-regression arenas.
9. Emit an explicit promotion decision artifact; update a champion pointer only through a separate,
   auditable operation.
10. Execute up to four planned generations with the declared epsilon schedule.
11. Run the locked final benchmark and write a results document analogous to
    `MILESTONE3_RESULTS.md`.
12. Add optional human-versus-champion web mode only after checkpoint safety and arena behavior are
    stable; render checkpoint identity but no candidate scores or RNG state.

### Acceptance criteria

- Re-running an iteration from its root config reproduces game schedules, semantic game-record
  fingerprints, dataset identity, model tensor digest, checkpoint lineage, and promotion report in
  the declared environment; timestamped file bytes are not the reproducibility contract.
- Generated games identify the exact frozen behavior checkpoint and exploration config.
- No training/promotion seed is present in the locked final set.
- Promotion uses the predeclared pair-aware gate; an incumbent is retained on failure.
- Final reports include random, heuristic, q0, parent, seat, uncertainty, throughput, and
  compatibility metadata.
- At least one complete 4,000-game generation is reproducible end-to-end; the performance target is
  under 60 minutes on the current CPU VM under otherwise low load, reported as a benchmark rather
  than enforced by a flaky timing test.
- A human can optionally play the promoted checkpoint from either seat without changing the core
  observation or agent boundary.

### Out of scope

- Requiring that a promotion occur, declaring equilibrium strength, distributed actors, GPU
  training, or public hosting.

## Milestone 7 experiments: only after the MC baseline

These are ordered experiments, not part of the initial implementation commitment.

1. **Public-history encoder v2:** add fixed padded history or compact recent-turn features; compare on
   exactly the same corpus and arenas.
2. **Blended MC/TD(lambda):** sequence-level lambda returns with a frozen target checkpoint, gamma 1,
   and a retained terminal-loss component.
3. **Opponent/champion pool:** sample frozen older checkpoints if self-play cycling or forgetting is
   measured.
4. **Heuristic ranking warm start:** safe auxiliary candidate-ranking pretraining if q0 coverage is
   inadequate.
5. **Search distillation:** shallow information-set-safe rollouts or a learned belief model whose
   action rankings are distilled into the candidate network.
6. **Larger/two-layer MLP:** only after profiling and an encoder/data ablation show underfitting.
7. **Policy head:** only if repeated candidate inference or planned search makes it useful.
8. **Belief-aware tree search/ReBeL-style work:** a separate research milestone after direct value
   learning plateaus.

For each experiment, change one major factor, reuse fixed corpora/seeds where valid, and require an
arena win over the simpler parent. Do not replace the permanent MC baseline.

## Test matrix

### Encoding and safety

- every card and legal action type has a golden encoding;
- own/opponent seat swaps canonicalize correctly;
- tuple/card order does not alter count features;
- duplicate hands preserve legal-action distinctions;
- unknown face-down identities cannot affect recruit candidate vectors;
- opponent hands and deck permutations cannot affect any candidate vector;
- no seed, revision history artifact, UI index, or authoritative field enters features;
- terminal and non-actor observations fail clearly.

### Data and storage

- corpus append/resume/round-trip and tamper rejection;
- game-record verification before extraction;
- labels use each sample actor, not a fixed seat;
- split isolation at game/pair level;
- stable manifests and fingerprints;
- incompatible historical records require an explicit policy, never a silent bypass;
- corpus-to-dataset regeneration is deterministic.

### Model and checkpoint

- exact tensor shapes and parameter count;
- logits/probabilities finite for every legal fixture;
- batched scores equal per-candidate scores;
- deterministic initialization/training under strict settings;
- overfit and constant-label controls;
- safe weights-only loading;
- metadata/digest/width/rules mismatch rejection;
- inference never enables gradients.

### Agent, runner, and self-play

- returned action is always from the supplied tuple;
- exact ties use only supplied agent RNG;
- epsilon rationals 0/1 and 1/1 follow the declared RNG-consumption contract under a controlled RNG;
- setup RNG is unaffected by model/exploration calls;
- fixed checkpoint/seeds reproduce action sequences;
- frozen weights do not change during games/generation;
- resumed generation matches uninterrupted generation;
- existing random/heuristic/web flows remain unchanged.

### Evaluation

- paired schedules preserve setup and logical-agent RNG across seat swaps;
- pair-bootstrap confidence intervals match known fixtures;
- promotion gates pass/fail boundary fixtures;
- training, promotion, and final seed manifests are disjoint;
- reports identify all agent/checkpoint/config fingerprints;
- no claim-generating statistic depends on one anecdotal game.

## Performance plan

Measure before optimizing:

- safe observations encoded/second;
- candidate vectors encoded/second;
- batched MLP candidates/second for batch sizes 2, 6, and 12;
- complete neural games/second;
- corpus MB and decisions per 1,000 games;
- replay extraction examples/second;
- training examples/second and epoch duration; and
- percentage of decision time in engine validation, encoding, Torch, and storage.

Likely first optimizations:

1. batch all legal candidates;
2. load and validate one immutable model per checkpoint per process/run, then share it read-only
   through lightweight per-game agents;
3. stream records and datasets;
4. cache observation-base features while adding action suffixes; and
5. only if profiling proves necessary, add a trusted internal transition/validation fast path with
   equivalence tests—never expose it to agents.

## Principal risks and responses

| Risk | Response |
| --- | --- |
| Hidden-state leakage | `PlayerObservation + Action` only; hidden-equivalence tests |
| Chosen-action coverage | epsilon exploration, varied setup seeds, iterative on-policy data |
| Optimistic unseen actions | monitor action frequencies/logits; increase exploration before adding Bellman max targets |
| Self-play cycles | frozen generations and parent/heuristic arenas; opponent pool only if measured |
| History omission harms beliefs | retain raw replays; add a separately versioned history encoder ablation |
| Validation metric does not imply strength | arena result is the promotion criterion |
| Torch breaks core install | lazy optional imports and plain-core smoke test |
| Artifact incompatibility | strict manifests, digests, and explicit migrations/policies |
| Paired games treated as independent | add pair-aware bootstrap while retaining Wilson for continuity |
| CPU slowdown from engine validation | profile; optimize only behind tested trusted interfaces |

## Definition of done for the planned neural stage

The initial neural stage is complete when:

- the 87-feature action-conditioned encoder and model/checkpoint contracts are frozen and tested;
- q0 can be regenerated from declared bootstrap games and training config;
- at least one frozen self-play generation runs end-to-end reproducibly;
- every learned decision is invariant to hidden states outside the actor's observation;
- the best checkpoint has paired, seat-balanced results versus random, heuristic, q0, and parent;
- promotion and non-promotion decisions follow a predeclared gate;
- all artifacts carry enough lineage and fingerprints to reject incompatible reuse;
- core engine use remains independent of RL dependencies;
- `make check` and the documented RL checks pass; and
- results are recorded honestly even if iterative self-play does not beat the heuristic yet.

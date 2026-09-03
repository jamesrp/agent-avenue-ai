# Terminal-Safety Hybrid Policy Experiment

**Status:** `terminal-safety-v1` and held-out cross-policy diagnostics implemented; controlled runs pending
**Declaration date:** September 2, 2026
**Implementation date:** September 3, 2026
**Parent result:** [Milestone 6 frozen self-play](MILESTONE6_RESULTS.md)

## Motivation and observed failure

Milestone 6 showed that pure greedy neural action selection is not yet a sufficient MVP policy.
Every q1–q4 candidate beat q0 directly, but every candidate failed the aligned
`greedy-public-v1` non-regression guardrail. The candidates became specialized responses to q0
rather than robust policies.

The most important concrete defect was avoidable immediate loss:

- across the four candidate-versus-heuristic arenas, candidates selected a visible third Daredevil
  102 times in 113 opportunities (90.3%);
- recorded examples include taking the fatal card while leading 11–8, 18–17, 9–3, and 12–8;
- candidate games contained 200 losses involving their own third Daredevil, versus 156 for q0 on
  the aligned blocks; and
- this was not caused by a missing representation: `candidate-public-v1` already encodes whether
  each player has at least one or two Daredevils.

The broader failure was opponent-specific action ranking:

- candidates selected the face-up recruit slot on 94–98% of decisions against q0 and 78.0% against
  the heuristic, while q0 selected it 52.2% of the time against the heuristic;
- the characteristic `Codebreaker face-up / Saboteur face-down` offer exploited q0 because q0 chose
  the hidden Saboteur 328/357 times, but the heuristic chose the visible Codebreaker 170/170 times;
- candidates gave the heuristic Codebreakers 21.14% of the time, versus 11.61% from q0 offers, and
  retained Saboteurs themselves 24.91% of the time, versus 15.26% for q0; and
- candidate first-offer agreement with the public heuristic evaluator was only 22.5–35.0%, compared
  with 50.5–61.5% for q0, contributing to a combined 37.9% candidate win rate from seat one.

The training mechanism explains the brittleness. The Monte Carlo dataset labels only the action
actually selected with the eventual winner. Alternative legal actions receive no counterfactual or
ranking target, but inference greedily maximizes over all legal actions. Current-generation-only
q0 self-play therefore rewards correlations specific to q0's responses and can assign unsupported
high values to alternatives. Validation loss improved from 0.5345 for q1 to 0.4913 for q4 while
heuristic win rate varied from 37.5% to 52.25%, confirming that same-distribution outcome prediction
was not a reliable robustness metric.

## Hypothesis

A narrow, deterministic, information-safe terminal-safety layer will eliminate avoidable immediate
losses and provide a better minimum viable learned policy without embedding a broad hand-authored
strategy. Keeping every other encoder, model, target, budget, and gate fixed will measure whether
this small hybrid structure improves robustness and prevents catastrophic neural extrapolation.

## Frozen change: `terminal-safety-v1`

Implement a composable policy shield between legal-action generation and any base policy. The
shield consumes only `PlayerObservation`, public decision context, semantic legal actions, and card
rules. It must never accept authoritative state, deck order, the actual unknown face-down card, or an
opposing hand.

For each legal action, perform only the current offer/recruit resolution and terminal adjudication;
do not search future turns or assign a general strategic score.

### Recruit decisions

For each slot, enumerate every face-down card consistent with the viewer-visible remaining multiset.
Classify an action as a **provable immediate loss** only if every information-consistent assignment
ends the current turn with the acting player losing. This rejects taking a visible third Daredevil
when the loss cannot be avoided by the unknown assignment, while refusing to use the true hidden
card.

### Play decisions

The offerer knows both cards in the proposed action. Resolve both possible opponent recruit choices.
Classify an offer as unsafe if an adversarial opponent has at least one choice that makes the offerer
lose at the end of the current turn. This catches offers that allow the opponent to leave the
offerer a fatal third Daredevil. Terminal ties must use the engine's exact active-player rule rather
than an approximate score test.

### Selection and fallback

- If at least one legal action is not classified as a provable immediate loss, expose only those
  actions to the wrapped policy.
- If every legal action is losing, preserve the complete legal set and let the wrapped policy choose;
  the shield must distinguish forced losses from avoidable mistakes.
- Apply the shield before both base-policy selection and epsilon exploration so exploration cannot
  reintroduce a vetoed action.
- Do not add bonuses for immediate wins, card values, score changes, Codebreaker progress, or any
  other heuristic in this experiment. The sole changed factor is vetoing provable immediate losses.

The normalized policy identity should name the wrapper, version, base policy, fallback rule, public
uncertainty enumeration, and terminal evaluator version. Game records must identify both the shield
and wrapped policy.

## Implementation

The shield is implemented in `agent_avenue.agents.terminal_safety` as
`TerminalSafetyAgent`. It wraps the complete behavior policy, so the required composition for
exploratory policies is `TerminalSafetyAgent(EpsilonGreedyAgent(base, epsilon))`. The nested
normalized configuration records the shield, epsilon wrapper, and underlying policy identities.

The corpus and iteration CLIs expose this composition through `--terminal-safety`; shield identity is
included in generation fingerprints, plans, corpus manifests, arena configurations, and per-seat
game-record policy configurations.

`filter_terminal_actions` exposes deterministic per-turn diagnostics: all publicly provable loss
actions, actions actually vetoed, the actions supplied to the base policy, and whether an all-losing
forced fallback occurred. Terminal resolution reuses the engine's public-material
`engine-terminal-v1` evaluator rather than duplicating tie logic in the agent package.

The implementation and regression suite do not start or alter the controlled experiment chain
below. Historical Milestone 6 artifacts remain unchanged. The committed experiment driver is
`scripts/run_terminal_safety_v1.py`; it refuses to begin from a tracked-dirty source tree and writes
an immutable top-level declaration before collecting games.

## Required tests

1. A visible third Daredevil is excluded when the other recruit action is not a provable loss.
2. A fatal third-Daredevil offer is excluded when another safe offer exists.
3. Simultaneous win/loss and both-player condition ties match engine active-player adjudication.
4. A hidden recruit card is never inspected; changing only authoritative hidden identity cannot
   change the filtered set for the same public observation.
5. Information-consistent hidden-card enumeration handles exhausted card counts exactly.
6. If all actions lose, fallback preserves the original legal-action tuple.
7. Epsilon exploration samples only from the filtered set.
8. Seeded shielded games and semantic records reproduce exactly.
9. The engine and observation packages remain independent of agents and PyTorch.

## Controlled rerun plan

Treat this as a new experiment chain rather than rewriting q0 or the immutable Milestone 6 result.
Use names such as `q0-terminal-safety-v1` and `q1-terminal-safety-v1`; retain the original q0 and
q1–q4 artifacts permanently.

### Phase 1: retrain and snapshot q0

1. Freeze a clean implementation revision and `uv.lock` digest.
2. Regenerate 4,000 epsilon-heuristic bootstrap games with `terminal-safety-v1` applied before the
   fixed 1/5 epsilon choice.
3. Build the same deterministic game-level 90/10 dataset.
4. Train the unchanged `candidate-public-v1` / `candidate-mlp-v1` Monte Carlo recipe from the same
   initialization and optimizer settings.
5. Snapshot the resulting checkpoint with explicit parent experiment lineage but no claim that it
   is the historical q0.
6. Evaluate it on matched development blocks against random, `greedy-public-v1`, and historical q0.
7. Record avoidable/provable immediate-loss opportunity and veto counts for all agents.

Intentionally reuse the historical setup-seed blocks where useful for paired diagnosis, but place
all artifacts in a distinct experiment namespace and derive agent/training/bootstrap seeds from the
new experiment ID. Previously observed blocks are development data, not locked final data.

### Phase 2: rerun q1–q4

Run the frozen Milestone 6 loop with only these changes:

- every learned incumbent and candidate is wrapped in `terminal-safety-v1`;
- safety filtering occurs before epsilon exploration; and
- diagnostics count filtered opportunities, vetoes, forced-loss fallbacks, and any executed
  provable immediate losses.

Keep unchanged:

- 4,000 current-generation-only games;
- epsilon schedule 1/10, 3/40, 1/20, and 1/40;
- encoder, model, Monte Carlo target, warm start, optimizer, epochs, and split;
- 500-pair primary comparison;
- 200-pair random and aligned heuristic guardrails;
- confirmation and promotion thresholds; and
- plateau and four-generation stopping rules.

If a candidate is promoted, use it as the next behavior checkpoint exactly as the original protocol
requires. If no candidate is promoted, continue from the retained shielded incumbent. Do not add
replay-mixture training, heuristic distillation, counterfactual targets, or search during this chain.
Those are separate experiments.

### Held-out all-pairs self-play diagnostic

After training each `qn`, including the new `q0`, evaluate its chosen-action outcome predictions on
fresh retained games generated by all unordered policy pairs with replacement from

`P_n = (greedy-public-v1, q0, ..., q(n-1))`.

Here `qk` identifies the generated terminal-safety experiment proposal even when it failed promotion;
the diagnostic pool is therefore different from the incumbent pool used for training. Every learned
behavior policy is its greedy checkpoint wrapped by `terminal-safety-v1`; the heuristic member is the
unchanged unshielded `greedy-public-v1` baseline. Use 200 paired setup blocks (400 seat-swapped games)
per cell. The resulting matrix contains 1, 3, 6, 10, and 15 cells for q0 through q4, respectively,
or 14,000 diagnostic games across the complete chain.

Score only the behavior-selected action at each recorded decision against the actor-relative terminal
outcome. Report paired-block equal-game log loss and Brier score by cell, pooled calibration/phase
metrics, an equal-cell macro result, and the worst cell. Verify that no normalized game-config/setup-
seed pair overlaps any q0-through-qn training corpus. These data are never fitted, never enter early
stopping, and never alter promotion. They diagnose generalization across prior-policy interactions;
they are not counterfactual action labels or playing-strength evidence.

This interpretation makes “all pairs of opponents” concrete without changing the declared training
opponent or introducing opponent-pool training as a second experimental factor.

### Phase 3: locked final evaluation

After champion selection, use a fresh, predeclared locked setup block disjoint from all prior
training, Milestone 6, and development/promotion blocks. Evaluate the selected shielded champion
against the unique applicable set of:

- random;
- `greedy-public-v1`;
- historical q0;
- the shielded q0 snapshot;
- its immediate parent; and
- earlier promoted shielded champions only when diagnostically necessary.

Retain compressed semantic arena records and report paired/seat uncertainty, margins, terminal
reasons, game lengths, throughput, artifact identities, and safety diagnostics.

## Success and interpretation

The primary implementation invariant is:

> When at least one non-losing action is publicly provable, the executed action is never a provable
> immediate loss.

The experiment succeeds structurally only if this invariant holds over tests and all retained games.
Playing-strength conclusions continue to use the unchanged paired promotion gate. In particular:

- eliminating third-Daredevil blunders without improving heuristic results means other offer and
  distribution-shift failures dominate;
- improved heuristic robustness with weaker direct parent performance is a tradeoff, not an
  automatic promotion;
- a stronger chain supports retaining the hybrid safety layer in the MVP; and
- no result from this experiment establishes that selected-action Monte Carlo training is generally
  sufficient. Counterfactual/ranking supervision and mixed-opponent replay remain likely follow-ups.

## Frozen execution declaration

The controlled run uses experiment root seed `2026090301` and derives all q0 bootstrap, q1–q4
iteration, all-pairs diagnostic, development, and locked-final seeds with the project
`sha256-domain-separation-v1` derivation. Names are fixed as `q0-terminal-safety-v1-a1` through
`q4-terminal-safety-v1-a1`. q0 uses 400 paired development blocks per matchup; each diagnostic cell
uses 200 paired blocks; final unique matchups use 500 paired blocks. The driver writes every derived
seed and exact source/lock identity to `runs/terminal-safety-v1/experiment-plan.json` before work.

The q0 gate remains the historical requirement that the paired-bootstrap 95% lower endpoint against
random exceed 0.50. The q1–q4 gate remains unchanged. A q0 gate failure stops the chain rather than
silently weakening the prerequisite.

## Retention and reporting

Retain every bootstrap corpus, dataset, rejected candidate, promoted checkpoint, arena aggregate,
compressed arena record corpus, plan, safety diagnostic, and immutable decision. Record exact Git
revision, lock digest, normalized configuration, root seeds, and a verified checksum archive.
Publish a result document alongside, without modifying the historical Milestone 5 or Milestone 6
reports except to add a link to the new result.

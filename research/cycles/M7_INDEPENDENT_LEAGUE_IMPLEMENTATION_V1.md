# Step 5 implementation record: independent league and promotion v1

**Agreement:** [`M7_INDEPENDENT_LEAGUE_PROMOTION_V1.md`](M7_INDEPENDENT_LEAGUE_PROMOTION_V1.md)
**Input registry:** [`m7-independent-league-inputs.json`](m7-independent-league-inputs.json), fingerprint `df238b7d9c948563ee15e4e2cfc59fe1525a9721b10e24362a7d251c1dfb7d48`
**State:** runner, independent validator, and mutation tests implemented; learned-checkpoint smoke on
the retained inputs, runtime preflight, claim source freeze, and the locked-final run are pending.
No locked-final game has been played. Nothing here changes the selected champion or the web default.

## Components

| Component | Path | Role |
| --- | --- | --- |
| Production core | `src/agent_avenue/runners/league.py` | Registry authentication, schedule reconstruction, policy construction with declaration checks, holdout scan, cells, tactical replay, aligned prefix audit, statistics, decision, execution state, checksums, preflight |
| Runner CLI | `scripts/run_independent_league_promotion_v1.py` | `run` (claim or `--smoke-pairs N`) and `preflight` |
| Independent validator | `scripts/validate_independent_league_promotion_v1.py` | Imports nothing from `runners.league`; reads roles from the registry; replays every record with the independent public-win oracle; recomputes every artifact and the decision |
| Tests | `tests/runners/test_league.py`, `tests/runners/test_league_validator.py` | Schedule, declaration, holdout, prefix, tactical, bootstrap, threshold, decision, checksum, and independence mutations plus one end-to-end toy league |

`DeterministicRandom.randbelow_batch` is a throughput path for bootstrap resampling that returns the
identical stream and counter as sequential `randbelow` calls; the arena paired bootstrap now draws
in chunks through it. Both equivalences are regression-tested against the sequential definition, so
historical arena intervals are unchanged.

## Commands

```bash
# 1. Learned-checkpoint smoke on separate nonclaim seed domains (retained inputs under runs/).
uv run --extra rl python scripts/run_independent_league_promotion_v1.py run \
  --output runs/m7-independent-league-smoke --smoke-pairs 2
uv run --extra rl python scripts/validate_independent_league_promotion_v1.py \
  runs/m7-independent-league-smoke

# 2. Measured runtime gate (x1.20 linear projection must be <= 420 minutes).
uv run --extra rl python scripts/run_independent_league_promotion_v1.py preflight \
  runs/m7-independent-league-smoke --output runs/m7-independent-league-preflight.json

# 3. After committing a tracked-clean claim source with the same package code fingerprint:
uv run --extra rl python scripts/run_independent_league_promotion_v1.py run \
  --preflight runs/m7-independent-league-preflight.json
uv run --extra rl python scripts/validate_independent_league_promotion_v1.py \
  runs/m7-independent-league-promotion-v1
```

A claim run refuses to start unless the registry is the committed one, every checkpoint/evidence/
archive identity authenticates, the registry schedule reconstructs exactly, the tracked tree is
clean, output is `runs/m7-independent-league-promotion-v1`, holdout scope is exactly `runs`, and a
sealed passing preflight matches the current package code fingerprint and registry. A smoke never
emits claim evidence: its namespace is `m7-independent-league-promotion-v1:nonclaim-smoke` and its
root seed is derived from the registry root under `...:nonclaim-smoke-root:v1`.

## Implementation interpretations requiring sign-off before the claim

The agreement fixes the estimands, gates, entrants, and seeds. These implementation-level choices
were not spelled out and are recorded here so they can be reviewed before claim dispatch:

1. **Declaration equality.** Registry declarations omit four terminal-safety version fields that
   `TerminalSafetyAgent` serializes (`fallback`, `public_uncertainty`, `resolution_scope`,
   `terminal_evaluator`). The runtime config must contain every declared key with an equal value;
   `checkpoint_ref` nodes must match the registry's kind, checkpoint fingerprint, tensor digest,
   and encoder version; and the only permitted undeclared keys are those four, pinned to their
   current constants. The full runtime configs are retained in `plan.json` and every record.
2. **Bootstrap seeds.** Every league bootstrap uses `derive_seed(root, "<namespace>:<domain>")`
   with the same string as its RNG domain: `matchup-family-stratified-bootstrap:v1:<left>--vs--<right>`,
   `anchor-joint-family-stratified-bootstrap:v1`, and `m-family-nested-bootstrap:v1`. Family A
   draws precede family B in every resample. Per-cell arena intervals keep the existing arena
   definition.
3. **M nested bootstrap.** Each resample draws three replicate indexes with replacement, then one
   aligned block-index set per family shared by all sampled replicates and all nine non-M
   opponents. The worst-opponent interval is the distribution of the resampled minimum.
4. **No-worse condition.** An aligned regression is a game in which the incumbent wins and the
   challenger, in the same setup, seat, and opponent stream, loses. Condition 2 requires zero such
   games and therefore zero such blocks; both counts are reported.
5. **Prefix endpoint.** The endpoint is the first q0-variant decision with a public guaranteed
   current-turn win. The challenger must choose a guaranteed action and its record must end there
   (after the opponent's recruit for a play action) in its own win. Games with no endpoint must be
   identical in actions and outcome.
6. **Integrity versus retention.** Failures of conditions 1–7 yield
   `retain_q0_terminal_safety_v1`, including an avoidable loss by the incumbent or challenger
   (conditions 4–5). Condition-8 failures and envelope violations by the descriptive policies (an
   avoidable loss by q1–q4 or M1–M3, or a missed/false forced win by M1–M3) indicate an evidence
   defect and yield `blocked_no_decision`.
7. **Input identity.** Checkpoint directories must contain exactly the declared files; source
   evidence must match its SHA-256, its declared `plan_fingerprint`, and its declared seal, which
   the Step-1/3/4 result writers store as `result_fingerprint` and every other retained artifact
   (including the Step-0 result) as `artifact_fingerprint`; claim runs require the four archives at
   the paths named in the prior result documents.
8. **Deadline accounting.** The 465-minute claim cutoff applies to accumulated active runner time
   (planning/input hashing, holdout, cells, audits, statistics, and finalization) across the
   initial attempt and the single permitted resume; only the final `runtime.json`/`checksums.json`
   writes are untimed. Time inside a phase killed by SIGKILL is not recovered. The validator fails
   the step if runner plus validator time exceeds 480 minutes.
9. **Runtime projection (needs explicit sign-off).** §10 multiplies "the measured combined
   projection" by 1.20 but does not say how a small smoke is scaled. The preflight counts phases
   whose cost does not depend on block count once (runner planning/input hashing and holdout scan;
   validator authentication, input hashing, holdout scan, and final checks) and scales every other
   measured second by 200 / smoke pairs. The all-phases-linear projection is also recorded; with
   a large `runs/` the linear rule would multiply the fixed holdout scans by 100 and could fail
   the gate for a reason unrelated to the claim's cost. If the phase-aware rule is not accepted,
   use the recorded linear figure or a larger smoke.
10. **Holdout at validation.** The validator requires the retained holdout to have passed, the
    proposed setup identities to reproduce, a fresh scan of `runs` to show zero overlap, and every
    retained prior corpus that is still present to be byte-identical. Exact equality of the whole
    prior inventory at validation time is reported but not required, because `runs/` may change
    after the claim starts.
11. **Checksum scope.** Mutable and wrapper files are excluded only at the output root; corpus
    `.lock` files are excluded anywhere. Dot-temporary files left by an interrupted atomic write
    are swept (and listed in `runtime.json`) before checksums are written.

## Known limitations

- A source repair changes `plan.json`'s source identity, so the immutable plan refuses to resume
  after any code change; a repair would need an explicit repair ledger and, if partial claim
  corpora stay under `runs/`, a fresh holdout decision. No such path is implemented.
- As in earlier cycles, both runner and validator classify provable losses with the production
  terminal-safety filter; only guaranteed wins use the independent public oracle.
- Learned configs' `model_version`, `encoder_fingerprint`, `device`, `inference`, and
  `tie_breaking` are not declared in the registry; they are checked for consistency between the
  plan and every record, not against the registry.

## Verification in the development container

- `make check`-equivalent with RL and web extras: Ruff, strict mypy, and 357 tests pass.
- Toy-random smoke (`--smoke-pairs 2`, 528 games): runner and independent validator agree on every
  artifact, zero prefix/alignment failures, decision `retain_q0_terminal_safety_v1` (expected for
  random stand-ins).
- Learned smoke with synthetic checkpoints of the real architectures bound by a resealed fixture
  registry (`--smoke-pairs 2`): validator passed with zero problems. Runner 101 s, validator 75 s;
  the gate projection was 351 minutes. This is indicative only: retained checkpoints, real game
  lengths, and the target VM determine the binding preflight.

- A fresh-context review found that the source-evidence check used the wrong seal key for three
  result files (which would have refused the claim at input authentication), that the
  all-linear preflight would scale fixed scans by 100x, that incumbent/challenger avoidable losses
  were classified as blocking rather than retaining, and several resume/holdout/checksum-scope
  edge cases. All were fixed with regression tests except the documented limitations above.

## Remaining steps

1. Run the learned smoke and validator on the VM against retained inputs; fix any input-layout
   mismatch before freezing (no claim evidence exists yet, so these are preclaim fixes).
2. Generate the preflight; if the projection exceeds 420 minutes the claim is blocked and may not
   be reduced.
3. Fresh-context review, commit a tracked-clean claim source, run the locked-final league once,
   and run the independent validator at that exact source.
4. Only if the validated decision is `promote_q0_terminal_offense_v1`: a separate post-validation
   commit adds and tests the `TerminalOffense(TerminalSafety(q0))` web composition bound to the
   decision, validation, and checksum fingerprints, then changes the champion/default.

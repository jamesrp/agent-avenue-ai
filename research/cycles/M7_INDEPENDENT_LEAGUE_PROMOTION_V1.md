# Approved Step 5: independent league and promotion v1

**Program:** `m7-stronger-policy-program-v1`
**Step:** 5 of 5
**Cycle:** `m7-independent-league-promotion-v1`
**Status:** Approved and design-frozen; implementation pending
**Approved:** September 17, 2026 under the September 11 five-step program authorization
**Input registry:** [`m7-independent-league-inputs.json`](m7-independent-league-inputs.json), fingerprint `df238b7d9c948563ee15e4e2cfc59fe1525a9721b10e24362a7d251c1dfb7d48`
**Budget:** one CPU process at a time; claim cutoff 7 h 45 min; hard whole-step stop 8 h

## 1. Question and decision boundary

Run one untouched locked-final league after all recipe-development decisions are closed. Determine
whether the already validated exact immediate-win wrapper should replace `q0-terminal-safety-v1` as
the selected champion, while measuring the retained learned families without reopening recipe
selection.

Only `q0-terminal-offense-v1` is promotion eligible. Step-3 mixed-data structured-v2 M is descriptive
because Steps 3 and 4 did not pass their frozen advancement gates. No q1-q4, M replicate, rollout
checkpoint, baseline, or post-result combination may be promoted.

Step 5 performs no training, target generation, architecture choice, hyperparameter choice, replicate
selection, or search. It may change the champion/web default only through a separate post-validation
commit after the immutable decision passes independent validation.

## 2. Exact entrants

Freeze these twelve policies and the full file identities in the input registry. The registry contains
one explicitly nested normalized config and one stable RNG identity for every policy; runtime config
serialization must equal those declarations exactly.

| ID | Live policy | Role |
| --- | --- | --- |
| `q0-terminal-safety-v1` | terminal safety around retained q0 | incumbent/reference |
| `q0-terminal-offense-v1` | terminal offense around the same safety-wrapped q0 | sole promotion challenger |
| `q1-terminal-safety-v1` through `q4-terminal-safety-v1` | historical q1-q4 safety-only policies | descriptive descendants |
| `M1` through `M3` | terminal offense + terminal safety around original Step-3 structured-v2 M checkpoints | descriptive training-replicate family |
| `historical-q0` | original raw q0 checkpoint | anchor |
| `greedy-public-v1` | canonical fixed public heuristic | anchor |
| `random-agent-v1` | canonical random policy | anchor |

q1-q4 remain their historical safety-only policies; do not add the offense wrapper. M1-M3 use the
original Step-3 checkpoint paths, not Step-4 reproduced controls and not rollout treatments. The
three M replicates remain exchangeable evidence; never select or label a best replicate.

## 3. Locked-final schedule

Use root seed `2026091705`, family order `family-a`, `family-b`, and the exact SHA-256-domain
schedule committed in the registry:

| Family | Seed domain | Master seed | 200-setup fingerprint |
| --- | --- | ---: | --- |
| `family-a` | `m7-independent-league-promotion-v1:seed-family:family-a` | `3091215482018554961` | `56960400d09509bc59bd20fc5c7dd67f97512c31d5ce2ea331ca9a66d30508d6` |
| `family-b` | `m7-independent-league-promotion-v1:seed-family:family-b` | `1709486255448338158` | `1a5e6dacebe22b39a7773c2a9c0d5b03a1f6cf268bf59502e55ca84228c14c93` |

For family `F`, derive `master = derive_seed(2026091705,
"m7-independent-league-promotion-v1:seed-family:F") & ((1 << 63) - 1)`. For block index `i` in
`0..199`, derive setup seed `derive_seed(master, "arena:pair:i:setup") & ((1 << 64) - 1)`.
The registry fixes policy order, all 132 ordered cell keys/run IDs, pair counts, and the expected
52,800-game manifest. Runner and validator must independently reconstruct that exact schedule.

The predeclared setup families are exactly `family-a` and `family-b` as serialized in the registry.
For each family, derive exactly 200 setup blocks. Use every block for every unordered policy pair,
with both physical-seat assignments. The complete twelve-policy round robin therefore contains:

- 66 unordered matchups;
- 132 family-specific arena cells;
- 200 paired blocks / 400 games per family-specific cell;
- 400 paired blocks / 800 games per unordered matchup; and
- **52,800 physical games total**.

Open both families together. No result from family A may affect family B or any configuration.
No confirmation or extra block is allowed.

Before any game, recursively scan all retained completed corpora, arenas, diagnostics, development,
promotion, and locked-final schedules under repository `runs/`. Require zero setup overlap with both
families and zero A/B overlap. Persist the complete prior inventory, proposed identities, overlap
counts, and fingerprints. Any overlap blocks execution; seeds or counts may not be changed after
inspection.

## 4. RNG and alignment contract

Use the existing SHA-256 domain-separated seed derivation and deterministic engine setup. Within each
family, every matchup shares the same setup-block identities.

For all `q0-terminal-safety-v1` versus opponent X and `q0-terminal-offense-v1` versus the same X:

- align family, block, setup, physical seat, opponent RNG identity, and pair index;
- give both q0 variants logical core RNG identity `q0-terminal-core-v1`;
- keep X's logical RNG identity identical across the aligned cells; and
- compare semantic prefixes before the first guaranteed-win intervention.

The direct q0-offense versus q0-safety cell is descriptive because the deliberate common core RNG
identity makes it an intervention diagnostic rather than an independent promotion estimand.

Every other policy has one stable logical RNG identity across opponents and seats. Record full
setup, agent, family, block, pair, and policy identities in every game record and report.

## 5. League reporting

For all 66 unordered matchups report:

- paired win rate and deterministic 20,000-resample 95% paired-block bootstrap interval;
- physical-seat win rates and Wilson intervals;
- score margins, turns, decisions, terminal reasons, and throughput;
- tactical counts and policy/config identities; and
- both family-specific estimates plus the combined estimate.

Also report the complete matrix, equal-opponent macro, and worst opponent for every policy. Scalar
rankings such as Elo or Copeland may be descriptive only and may not replace the matrix or promotion
rule.

For M1-M3, report each replicate separately and one nested family summary. The nested procedure
resamples M replicate identity outside and aligned family/block indexes inside. Report M versus the
incumbent, M versus the tactical challenger, an equal-weight macro across the nine non-M policies,
and worst-opponent behavior. Exclude M-M cells from the family macro. No M statistic can promote or
select a replicate.

## 6. Promotion estimand

For each independent anchor `O` in:

- `historical-q0`;
- `greedy-public-v1`;
- `random-agent-v1`;

define the aligned paired contrast:

```text
Delta_O = score(q0-terminal-offense-v1, O)
          - score(q0-terminal-safety-v1, O)
```

where win = 1, loss = 0, and the active-player tie rule is already reflected in game outcomes. Form
the equal-weight three-anchor macro after computing each anchor contrast.

Use a deterministic 20,000-resample joint family-stratified common-block bootstrap. Preserve family
A/B as fixed strata, resample paired block indexes within each family, use identical resampled indexes
for incumbent/challenger and all three anchors, then average anchors equally. The nearest-rank 95%
interval uses sorted indices 499 and 19,499. Do not pool individual games or count correlated q1-q4
or M replicates as independent promotion anchors.

## 7. Tactical and integrity audits

For every aligned incumbent/challenger common-opponent game, independently establish:

- semantic histories are identical until the first guaranteed-win intervention;
- setup, seat, family, opponent stream, RNG, and metadata align exactly;
- no game has incumbent win while challenger loses;
- the challenger has zero missed publicly guaranteed current-turn wins;
- the challenger has zero false forced-win classifications;
- both policies have zero executed avoidable provable immediate losses; and
- after a challenger intervention, the game terminates in the guaranteed win.

Across every safety-enveloped league policy, require zero executed avoidable provable losses. Across
M1-M3, which include terminal offense, additionally require zero missed guaranteed wins and zero
false forced wins. q1-q4 are safety-only; report their missed-win counts without treating them as an
integrity failure.

Require exact input, checkpoint, encoder, model, rules, source, schedule, seat, record, replay,
information-boundary, checksum, deadline, and holdout validation.

## 8. Immutable promotion rule

Promote `q0-terminal-offense-v1` only if every condition is true:

1. the three-anchor macro `Delta` 95% lower endpoint is **strictly above +0.25 percentage points**;
2. there are zero aligned common-opponent blocks in which the incumbent wins and challenger loses;
3. there are zero pre-intervention semantic-prefix or alignment failures;
4. the challenger has zero missed guaranteed wins, false forced wins, and avoidable provable losses;
5. the incumbent has zero avoidable provable losses;
6. challenger-versus-random paired win-rate lower endpoint is strictly above 50%;
7. the minimum challenger physical-seat point estimate across challenger-versus-incumbent,
   challenger-versus-heuristic, and challenger-versus-random is at least 45%; and
8. every provenance, holdout, source, schedule, replay, checksum, deadline, and independent-validator
   condition passes.

Decision values are:

- `promote_q0_terminal_offense_v1` if all conditions pass;
- `retain_q0_terminal_safety_v1` if the run is valid but any promotion condition fails; or
- `blocked_no_decision` if evidence integrity or independent validation fails.

A retention decision means only that this locked-final protocol did not justify changing the
champion. It does not establish optimality or authorize a sixth recipe.

## 9. Independent validation

A separate validator must avoid importing production league aggregation, bootstrap, tactical-prefix,
holdout, or promotion-decision functions. It must independently:

- authenticate every committed input file and exact source identity;
- reconstruct both setup families, all 132 cells, seats, RNG identities, and holdout inventory;
- replay all 52,800 semantic game records;
- reconstruct local arena reports, seat summaries, tactical/prefix audits, and the full matrix;
- recompute the joint anchor bootstrap and nested M-family summaries;
- reproduce every promotion condition and the immutable decision;
- verify result cross-references and exact checksum scope; and
- reject any extra, missing, malformed, stale, or incompatible artifact.

Mutation tests must cover schedule/family overlap, seat inversion, policy wrapping, q0 core RNG
alignment, setup or opponent-stream mismatch, prefix divergence, false or missed forced wins,
avoidable loss, checkpoint substitution, replicate selection, bootstrap-index misalignment,
threshold boundaries, decision mutation, and checksum scope.

## 10. Runtime and operational bounds

Before claim source freeze, run one isolated end-to-end smoke on separate nonclaim domains. Measure
production arena throughput, record I/O, tactical-prefix analysis, independent replay, statistics,
and checksum work. The measured combined projection, multiplied by 1.20, must be at most 420 minutes.
Failure blocks the claim and does not permit reducing policies, families, blocks, games, replay, or
validation.

Claim execution cutoff is 465 minutes and the hard whole-step limit is 480 minutes. Use one
substantial process at a time. The approved trivial-repair policy permits at most three mechanically
proven, estimand-preserving repairs; no repair may change entrants, wrappers, seeds, families,
blocks, games, statistics, thresholds, or promotion logic.

## 11. Required artifacts

Retain:

- `plan.json`, source identity, input audit, and full input registry copy;
- holdout inventory and both family schedules;
- all compressed semantic records and per-cell reports;
- complete league matrix, per-policy summaries, nested M-family statistics, and anchor contrasts;
- tactical, prefix, seat, worst-opponent, and runtime reports;
- immutable `promotion-decision.json`, result, validation, and checksums;
- runner/validator logs, execution state, repair ledger if any, fresh-context review, and verified
  archive.

A promotion decision does not itself prove that the current web loader can construct the challenger.
If promotion passes, a separate post-validation commit must first add and test the exact
`TerminalOffense(TerminalSafety(q0))` server-side composition, bind it to the validated decision,
validation, and checksum fingerprints, and only then change the champion/default. The research
runner must never mutate deployment settings.

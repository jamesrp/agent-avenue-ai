# Approved operational repair: population replay validator bootstrap order

**Program:** `m7-stronger-policy-program-v1`
**Step:** 2 of 5
**Cycle:** `m7-population-replay-v1`
**Repair:** `repair-1`
**Status:** Approved under the step's one-repair allowance
**Repair date:** September 12, 2026
**Original claim source:** `fde19b5d3c327e539c29913a973b5e4d76ffff5b`

## Failure

The claim runner completed all six corpora, six datasets, six checkpoints, 54 development arenas,
statistics, decision, and immutable result. The independent validator then failed on deterministic
nested-bootstrap percentile endpoints.

The validator consumed its RNG stream in a different order from the frozen production statistic:

- production first materialized all three outer replicate draws, then sampled inner blocks; and
- validator used a generator that interleaved each outer replicate draw with that replicate's inner
  block draws.

Point estimates, retained games, models, arena aggregates, and the scientific decision were
unchanged. Several bootstrap endpoints differed slightly. Both versions failed the same frozen
heuristic non-regression and minimum-seat conditions.

## Authorized repair

Change only:

- `scripts/validate_population_replay_v1.py`;
- `tests/runners/test_population_experiment.py`; and
- this repair agreement.

The validator must materialize the three outer replicate indexes before consuming any inner block
randomness and must match `nested_population_bootstrap` byte-for-byte on a fixed regression fixture.

The repaired validator may run at a later clean Git revision only when:

- the original plan identifies claim source `fde19b5d3c327e539c29913a973b5e4d76ffff5b`;
- `uv.lock`, package code fingerprint, and rules fingerprint equal the original claim;
- `git diff --name-only fde19b5..HEAD` contains only the three allowed paths above; and
- repair provenance is embedded in the validation result.

## Prohibited changes

Do not rerun or alter claim games, corpora, datasets, checkpoints, arenas, seeds, bootstrap method,
resample count, order-statistic indexes, thresholds, decision criteria, or result artifact. No other
source or documentation path may change before repaired validation completes.

## Completion

Run the repaired independent validator once with `--allow-validator-repair`. If it reproduces the
retained statistics and decision, continue with fresh-context review, archive, Step-2 briefing, and
Step-3 design. A second operational repair is not authorized.

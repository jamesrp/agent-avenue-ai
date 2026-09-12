# Completed operational repair: structured-model checksum scope

**Program:** `m7-stronger-policy-program-v1`
**Step:** 3 of 5
**Cycle:** `m7-structured-model-v2`
**Repair:** `repair-1`
**Status:** Complete; exact-source validation passed
**Repair date:** September 12, 2026
**Claim source:** `a63f081`

## Failure

The claim runner completed all six datasets, six checkpoints, 60 arenas, statistics, safety,
selection, checksums, and result. Independent validation stopped before scientific recomputation
because `driver.stdout` no longer matched its checksum.

The runner created `checksums.json` before the CLI printed the final result. Because wrapper-managed
stdout was redirected inside the run root, the checksum captured the empty file while the CLI later
appended the result JSON. No scientific payload changed.

## Authorized repair

The one operational repair restored `driver.stdout` to the empty SHA-256 state already declared by
`checksums.json`, then reran the independent validator at the exact original source. The original full
stdout was preserved outside the retained claim root during validation. No corpus, dataset,
checkpoint, arena, statistic, safety artifact, selection, result, source file, seed, or threshold was
changed.

The exact-source validator then reconstructed all six v2 datasets through validator-local 519-feature
logic, verified checkpoints and q0 lineage, locally recomputed all 60 complete arena reports and
record schedules/configs/RNG metadata, reproduced nested statistics and selection, and passed.

## Post-validation prevention

After validation, source code excludes wrapper-owned `driver.stdout`, `driver.stderr`, and
`wrapper-exit.json` from future immutable checksum scopes, alongside validation timing/output files.
A regression test freezes that boundary. This prevention change is not part of the claim source and
does not trigger a second validation or alter the retained Step-3 result.

No second operational repair is authorized.

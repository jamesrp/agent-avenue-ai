# Milestone 5: q0 Learned-Agent Evaluation

**Status:** Complete
**Date:** August 30, 2026
**Current role:** q0 is the learned incumbent for Milestone 6

## Result

The checkpoint-backed `LearnedValueAgent` ranks every legal semantic action in one CPU batch using
only `PlayerObservation + Action` encodings. Checkpoint loading remains lazy and optional, and
baseline engine/agent/CLI imports do not import PyTorch.

q0 was trained from 4,000 epsilon-greedy heuristic self-play games containing 62,184 decisions. The
planned seat-balanced q0 arenas completed successfully:

| Matchup | Paired seeds | Games | q0 wins | q0 win rate | Paired-bootstrap 95% CI | Player One | Player Two |
|---|---:|---:|---:|---:|---:|---:|---:|
| q0 vs random | 400 | 800 | 603 | 75.375% | [72.25%, 78.375%] | 76.75% | 74.00% |
| q0 vs `greedy-public-v1` | 400 | 800 | 481 | 60.125% | [56.75%, 63.50%] | 61.50% | 58.75% |

q0 therefore passes the predeclared Generation 0 gate against random: the paired-bootstrap lower
endpoint is above 50%. On this fixed evaluation it also beat the bootstrap heuristic with a lower
endpoint above 50%, though that stronger result was not required by the q0 gate.

## Training summary

| Item | Value |
| --- | ---: |
| Corpus games | 4,000 |
| Corpus decisions | 62,184 |
| Training games/samples | 3,627 / 56,544 |
| Validation games/samples | 373 / 5,640 |
| Best epoch | 46 of 50 |
| Training equal-game loss | 0.540577 |
| Validation equal-game loss | 0.550372 |
| Validation pooled loss | 0.574345 |
| Validation Brier score | 0.197040 |
| Validation accuracy | 68.475% |
| Training wall-clock time | 11.74 seconds |
| Training examples/second | 240,848 |

The accuracy and loss values describe held-out outcome prediction under the bootstrap behavior
policy. They are diagnostics, not gameplay-strength estimates; the paired arenas above support the
strength claim.

## Source and artifact lineage

| Artifact/stage | Identity |
| --- | --- |
| Corpus source revision | `7fe2e5af6339efc86db285349043bb16b298eed1` |
| Corpus/training code fingerprint | `b92b07727d89eb2f9471316fa44992a2eb337da2186a25f3bc4637404bb09d8d` |
| Corpus fingerprint | `55224e5454c250563715d87c6adca1b84f034a79d2c386c650f99535f44a6479` |
| Dataset fingerprint | `263bfcefeae992baa6cbbb2c0936682fdfe1e8a5618d4b87cab8bf705afd9aae` |
| Encoder fingerprint | `e122abbf302e44d1ab62411c63b76d58a02dbd4c87a4d8c5ca5501625960d052` |
| Checkpoint fingerprint | `bc6f070aa30e2155ed6953a065c9f67e10357b54f222e3e944273493ab4e6e8e` |
| Model tensor digest | `c8239721005aedcdee9185475db80018dbfc524e833b069ca1c540d9032041b6` |
| Arena source revision | `d46350356a7bf461468ba33c6bf1f28b3231cd14` |
| Arena code fingerprint | `6909e9316510f48cad515b99a8f9b19288ee5c43d5925f84b88ad853479031bc` |
| Rules fingerprint | `877731565922c8e65245f78a8bb71995cfd26edd81dab13fac2520f970695d09` |

The corpus/training revision is the exact package source used to generate q0. The later
`3dd8ffd50fba79530529514c750659c29b7d73d2` commit changed only ignore configuration and therefore
has the same package code fingerprint.

## Reproduction commands

Training at the recorded source and lockfile revision:

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

Evaluation at the recorded learned-agent/arena revision:

```bash
git switch --detach d46350356a7bf461468ba33c6bf1f28b3231cd14
uv sync --extra rl --locked
uv run python -m agent_avenue arena \
  --agent-a learned --agent-a-checkpoint checkpoints/q0 \
  --agent-b random --pairs 400 --seed 2026083001 \
  --run-id q0-vs-random-400-pairs \
  --output logs/q0-vs-random-400-pairs.json

uv run python -m agent_avenue arena \
  --agent-a learned --agent-a-checkpoint checkpoints/q0 \
  --agent-b heuristic --pairs 400 --seed 2026083002 \
  --run-id q0-vs-greedy-public-v1-400-pairs \
  --output logs/q0-vs-greedy-public-v1-400-pairs.json
```

The corpus, split, training, and two arena master seeds are distinct. Each arena master seed
deterministically derives 400 setup seeds, logical-agent RNG streams, and a separate bootstrap RNG
stream.

## Statistical method

Each observation unit is the two-game seat-swapped block for one setup seed. q0 wins per block are
resampled with replacement 20,000 times using `DeterministicRandom`. Sorted bootstrap win-rate means
use zero-based order-statistic indices 499 and 19,499. Wilson intervals remain in the machine report
for continuity but are not used for the gate.

Pair outcome counts were:

- versus random: 27 zero-win pairs, 143 split pairs, and 230 two-win pairs;
- versus heuristic: 62 zero-win pairs, 195 split pairs, and 143 two-win pairs.

The measured neural arena throughput was 9.84 games/second against random and 8.82 games/second
against the heuristic on the development VM under the recorded run conditions.

## Artifact archive

The retained q0 artifact set contains:

- the complete 4,000-game compressed corpus and manifest;
- materialized dataset and manifest;
- immutable checkpoint bundle;
- training log and checkpoint inspection output; and
- both aggregate arena reports and command outputs.

The original arena commands did not request `--records-dir`, so individual per-game q0 evaluation
records are not present. This is a documented exception to the retention protocol adopted on
September 2, 2026. The exact evaluation games can still be regenerated using the retained
checkpoint, recorded arena source revision and lockfile, normalized agent/game configuration, and
declared seeds. Future claim-generating arenas retain compressed individual game records by
default.

Measured sizes:

| Artifact | Bytes |
| --- | ---: |
| Compressed corpus records | 1,229,542 |
| Corpus manifest | 268,640 |
| Dataset arrays | 1,525,316 |
| Dataset manifest | 539,749 |
| Checkpoint bundle | 399,336 |
| Training/evaluation logs and aggregate reports | 766,302 |
| Complete archived input set | 4,728,885 |
| Compressed archive | 3,539,016 |

Local archive:

```text
artifacts/archive/q0-artifacts-2026-09-02.tar.gz
```

Archive SHA-256:

```text
2434d3a2700d1668f773143b9f87a91b0824d8e3af97f4969fbcdaa2a4576b95
```

The archive contains a per-file checksum manifest and was copied off the VM to the repository
owner on September 2, 2026. Generated artifacts remain ignored by Git in accordance with the project
retention policy.

## Interpretation

These arenas provide the first direct q0 gameplay measurement. The result supports calling q0 a
reasonable opponent and proceeding to the frozen-generation self-play milestone. It does not
establish optimal play, robustness to every opposing policy, or generalization beyond this rules,
encoder, checkpoint, and declared seed program.

q0 is the current learned incumbent until a later candidate passes the complete Milestone 6
promotion gate.

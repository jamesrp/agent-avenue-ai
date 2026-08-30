# Milestone 5: q0 learned-agent evaluation

Date: August 30, 2026

## Result

The checkpoint-backed `LearnedValueAgent` now ranks all legal semantic actions in one CPU batch
using only `PlayerObservation + Action` encodings. Checkpoint loading remains lazy and optional, and
baseline engine/agent/CLI imports do not import PyTorch.

The planned seat-balanced q0 arenas completed successfully:

| Matchup | Paired seeds | Games | q0 wins | q0 win rate | Paired-bootstrap 95% CI | Player One | Player Two |
|---|---:|---:|---:|---:|---:|---:|---:|
| q0 vs random | 400 | 800 | 603 | 75.375% | [72.25%, 78.375%] | 76.75% | 74.00% |
| q0 vs `greedy-public-v1` | 400 | 800 | 481 | 60.125% | [56.75%, 63.50%] | 61.50% | 58.75% |

q0 therefore passes the predeclared Generation 0 gate against random: the paired-bootstrap lower
endpoint is above 50%. On this fixed evaluation it also beat the bootstrap heuristic with a lower
endpoint above 50%, though that stronger result was not required by the q0 gate.

## Reproduction contract

Checkpoint fingerprint:

```text
bc6f070aa30e2155ed6953a065c9f67e10357b54f222e3e944273493ab4e6e8e
```

Rules fingerprint:

```text
877731565922c8e65245f78a8bb71995cfd26edd81dab13fac2520f970695d09
```

Evaluation code fingerprint:

```text
6909e9316510f48cad515b99a8f9b19288ee5c43d5925f84b88ad853479031bc
```

Commands:

```bash
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

The two master seeds are distinct from each other and from corpus, split, and training seeds. Each
master seed deterministically derives 400 setup seeds, logical-agent RNG streams, and a separate
bootstrap RNG stream.

## Statistical method

Each observation unit is the two-game seat-swapped block for one setup seed. q0 wins per block are
resampled with replacement 20,000 times using `DeterministicRandom`. Sorted bootstrap win-rate
means use zero-based order-statistic indices 499 and 19,499. Wilson intervals remain in the machine
report for continuity but are not used for the gate.

Pair outcome counts were:

- versus random: 27 zero-win pairs, 143 split pairs, and 230 two-win pairs;
- versus heuristic: 62 zero-win pairs, 195 split pairs, and 143 two-win pairs.

## Interpretation

The earlier 68.5% validation accuracy and 0.5504 equal-game loss described outcome prediction, not
playing strength. These arenas provide the first direct q0 gameplay measurement. The result supports
calling q0 a reasonable opponent and proceeding to the frozen-generation self-play milestone. It
does not establish optimal play, robustness to other policies, or generalization beyond this rules,
encoder, checkpoint, and declared seed program.

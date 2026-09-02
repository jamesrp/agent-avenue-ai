# Milestone 6: Frozen Self-Play Results

**Status:** Complete — four-generation budget exhausted, evidence inconclusive; q0 retained
**Run date:** September 2, 2026
**Implementation revision:** `139318bad909438e8a3e1cb9dd962c80653875d0`
**`uv.lock` SHA-256:** `2c56e1ba299c5a9b0f9363d6080e1bc4ed8f49162f7cccdacc2dc3734c9131c8`

## Result

The frozen q1–q4 baseline completed without weakening the declared gate. Every candidate beat q0
convincingly in its direct 500-pair primary arena, but every candidate failed the aligned
`greedy-public-v1` non-regression guardrail. No candidate was promoted, so **q0 remains the learned
champion**.

This is a negative selection result, not evidence that the model family is plateaued. All four
familywise practical-effect intervals were well above the equivalence region because the proposals
were much stronger than q0 head-to-head. The run therefore stopped as
`budget_exhausted_inconclusive`, not `plateaued`.

## Frozen declaration

Each generation used 4,000 fresh current-generation-only self-play games, a deterministic 90/10
game split, warm-started `candidate-mlp-v1` training, 500 paired candidate/incumbent seeds, 200
paired random seeds, and two aligned 200-pair heuristic arenas. The one allowed confirmation block
was never triggered because every primary lower endpoint was above 50%.

| Generation | Root seed | Epsilon | Incumbent |
| --- | ---: | ---: | --- |
| q1 | 2026090100 | 1/10 | q0 |
| q2 | 2026090200 | 3/40 | q0 |
| q3 | 2026090300 | 1/20 | q0 |
| q4 | 2026090400 | 1/40 | q0 |

All plans declared the exact clean Git revision and lockfile digest above and were marked
claim-eligible.

## Promotion results

Intervals below are deterministic 95% paired-block bootstrap intervals. “Heuristic difference” is
candidate win rate minus q0 win rate on aligned paired setup blocks; promotion required its lower
endpoint to be greater than -5 percentage points.

| Candidate | vs q0 | Seat 1 | Seat 2 | vs random | vs heuristic | q0 vs heuristic | Heuristic difference | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| q1 | 77.9% [75.5, 80.3] | 77.0% | 78.8% | 77.0% [72.75, 81.0] | 45.75% | 54.5% | -8.75% [-14.25, -3.25] | Retain q0 |
| q2 | 81.9% [79.5, 84.2] | 80.8% | 83.0% | 77.75% [73.5, 81.75] | 52.25% | 58.0% | -5.75% [-12.25, 0.75] | Retain q0 |
| q3 | 82.4% [80.1, 84.6] | 79.2% | 85.6% | 72.0% [67.5, 76.25] | 37.5% | 59.25% | -21.75% [-28.0, -15.5] | Retain q0 |
| q4 | 83.5% [81.2, 85.7] | 80.0% | 87.0% | 75.25% [71.25, 79.25] | 42.5% | 59.75% | -17.25% [-23.75, -10.75] | Retain q0 |

Every random lower endpoint exceeded 50%, and every candidate exceeded the 45% primary seat
minimum. The heuristic non-regression guardrail was the sole rejection reason in all four immutable
promotion decisions.

The result suggests severe policy-specific forgetting or specialization: current-generation
self-play produced candidates that dominated their parent while becoming less robust against the
permanent public heuristic. Cumulative replay, opponent pools, or altered targets remain Milestone 7
experiments; they were not introduced after seeing this result.

## Training and throughput

| Candidate | Best epoch | Best validation loss | End-to-end elapsed | Primary games/s |
| --- | ---: | ---: | ---: | ---: |
| q1 | 22 | 0.53455 | 11.38 min | 17.46 |
| q2 | 39 | 0.50691 | 11.72 min | 17.78 |
| q3 | 50 | 0.50269 | 11.57 min | 17.66 |
| q4 | 47 | 0.49131 | 11.46 min | 18.19 |

All generations were far below the declared 60-minute CPU target. Elapsed values use the attempt
plan and immutable decision file publication times. Candidate-evaluations/second was not
instrumented in the v1 runner; games/second, average turns/decisions, margins, terminal reasons, and
seat splits remain in each aggregate arena artifact.

## Locked final evaluation

After selection ended, q0 was evaluated on an untouched 500-pair block with master seed
`2026091000`. The final schedule was checked for setup-seed overlap with every training and
promotion arena corpus. Because q0 was both the initial and selected champion, the unique applicable
opponents were random and `greedy-public-v1`; duplicate q0/parent matchups were omitted.

| Matchup | Games | q0 win rate | Paired-bootstrap 95% interval | Seat 1 | Seat 2 | Avg margin | Avg turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q0 vs random | 1,000 | 75.5% | 72.9%–78.1% | 72.0% | 79.0% | +4.071 | 6.312 |
| q0 vs `greedy-public-v1` | 1,000 | 57.3% | 54.2%–60.4% | 56.8% | 57.8% | +3.271 | 7.190 |

The final result fingerprint is
`85f42426cef0d5cc9b96ecbffa774688823d043de5a74bcc71945645eb9c3826`.

## Independent reproduction

q1 was regenerated from scratch in a separate directory with the same declaration. It reproduced
all of the following exactly:

- plan fingerprint;
- ordered semantic training-corpus fingerprint;
- dataset fingerprint;
- candidate checkpoint and tensor digest;
- all four aggregate arena fingerprints;
- all four compressed arena-record corpus fingerprints; and
- immutable promotion-decision fingerprint.

The independent run completed in 11.41 minutes. Its machine-readable comparison is retained as
`runs/milestone6/reproduction-result.json`.

## Artifact identities

| Generation | Plan | Corpus | Dataset | Candidate checkpoint | Tensor digest | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| q1 | `74629199eb54…` | `0868118a18fd…` | `cc38f6490ae5…` | `2e863544a12b…` | `f22d9a2877fd…` | `edbb8f437564…` |
| q2 | `2578ea8e6537…` | `831e9217c6e1…` | `62067fa786a8…` | `58ea8856e74b…` | `b8e8f50e983a…` | `cbeb5cfc809b…` |
| q3 | `501411dcd400…` | `f7cba7e6f87d…` | `b47b73da6f4a…` | `7f8ce7b5e47c…` | `e08c2dbe1b3b…` | `b60b93063c1f…` |
| q4 | `b20d5d03b86c…` | `6139b3f89ea4…` | `feedecc0ea98…` | `5a07a5c2a7f3…` | `d03eeac8a89c…` | `3aec8d30810c…` |

The retained archive is `artifacts/milestone6/milestone6-baseline.tar.gz` (21,097,384 bytes,
137 payload members). Its SHA-256 is
`9264aadbfdabb3045d805f96d898a76c8b180ea39250c9dbe69ce4370cab8b3a`; archive creation verified
every member against the embedded checksum manifest. A byte-identical secondary retained copy is at
`~/.local/share/agent-avenue-ai-archives/milestone6-baseline.tar.gz`. Generated artifacts remain
ignored by Git.

## Reproduction commands

Use the recorded implementation revision and lockfile:

```bash
git switch --detach 139318bad909438e8a3e1cb9dd962c80653875d0
uv sync --extra rl --locked

uv run python -m agent_avenue iterate runs/milestone6/q1-a1 \
  --incumbent checkpoints/q0 --generation 1 --attempt-id q1-a1 --seed 2026090100
uv run python -m agent_avenue iterate runs/milestone6/q2-a1 \
  --incumbent checkpoints/q0 --generation 2 --attempt-id q2-a1 --seed 2026090200
uv run python -m agent_avenue iterate runs/milestone6/q3-a1 \
  --incumbent checkpoints/q0 --generation 3 --attempt-id q3-a1 --seed 2026090300
uv run python -m agent_avenue iterate runs/milestone6/q4-a1 \
  --incumbent checkpoints/q0 --generation 4 --attempt-id q4-a1 --seed 2026090400
```

The selected incumbent remains q0 for every command because each preceding candidate was retained.
The final and archive plans, reports, compressed records, checksums, and complete commands are
retained inside the verified archive.

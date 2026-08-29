# Milestone 3 baseline results

Recorded on August 29, 2026 on the CPU-only exe.dev development VM. These are reproducibility
checks and baseline measurements, not permanent performance claims. Throughput varies with the
machine and concurrent load.

Both runs used 200 paired setup seeds (400 games), alternated seats within every pair, preserved
each logical agent's RNG seed across the seat swap, and used master seed `20260829`.

| Matchup (agent A vs B) | A wins | A win rate | Wilson 95% CI | A as P1 | A as P2 | Games/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random vs random | 204/400 | 51.0% | 46.11%–55.87% | 50.5% | 51.5% | 44.1 |
| Greedy heuristic vs random | 345/400 | 86.25% | 82.53%–89.28% | 88.5% | 84.0% | 45.2 |

The random-versus-random interval includes 50%, so this run does not show evidence that the two
identical policies differ. The heuristic baseline's interval is wholly above 50% against this
versioned random policy for this declared seed set. This supports the narrow claim that
`greedy-public-v1` outperformed the random baseline in this run; it is not a claim that the policy
is strong in an absolute sense.

Additional aggregate results:

- Random vs random: average A score margin `+0.2475`, average `6.8275` turns and `13.655`
  decisions; all 400 games ended by a card/score condition.
- Greedy heuristic vs random: average A score margin `+3.99`, average `6.635` turns and
  `13.27` decisions; all 400 games ended by a card/score condition.

## Reproduce

```bash
uv run python -m agent_avenue arena \
  --agent-a random --agent-b random \
  --pairs 200 --seed 20260829 --run-id random-fairness

uv run python -m agent_avenue arena \
  --agent-a heuristic --agent-b random \
  --pairs 200 --seed 20260829 --run-id heuristic-baseline
```

For a faster local check, use the two 10-pair runs declared in
`benchmarks/milestone3-smoke.json`. Arena JSON includes normalized agent/engine configurations,
seed derivation and RNG algorithm versions, rules/code fingerprints, seat-specific results, terminal
reasons, score margin, decision/turn averages, confidence interval, and measured throughput. Large reports and
individual game records belong under ignored `runs/` or `replays/`, not in Git.

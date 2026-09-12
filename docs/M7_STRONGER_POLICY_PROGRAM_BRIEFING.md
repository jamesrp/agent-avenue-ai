# Stronger-policy program v1: stopped briefing

**Program:** `m7-stronger-policy-program-v1`
**Status:** Blocked at Step 4; Step 5 not executed
**Stopped:** September 12, 2026
**Agreement:** [`research/cycles/M7_STRONGER_POLICY_PROGRAM_V1.md`](../research/cycles/M7_STRONGER_POLICY_PROGRAM_V1.md)

## Executive summary

The program found two real improvements but did not complete its final promotion path:

1. The exact immediate-win envelope is structurally correct and measurably improves q0.
2. Mixed replay and structured v2 each produce large direct gains over their matched controls.
3. Neither learned recipe passed the frozen q0-parent heuristic non-regression gate.
4. Counterfactual rollout supervision was not scientifically evaluated because infrastructure failed
   before canonical targets, training, or arenas.
5. The selected champion and web default remain `q0-terminal-safety-v1`; Step 5 was not run.

## Step outcomes

| Step | Outcome | Main result |
| --- | --- | --- |
| 1. Terminal offense | Adopted for research | +0.75 pp anchor macro [0.60, 0.908]; 3,385/3,385 guaranteed wins converted |
| 2. Population replay v1 | Does not advance | Mixed beat q0-only control 62.10% [60.13%, 64.00%], but guardrails failed |
| 3. Structured model v2 | Does not advance | V2 beat v1 in both data arms; mixed-v2 selected only as Step-4 development input |
| 4. Counterfactual rollout | Infrastructure blocked | Panels and partial replicate-1 targets only; repair exhausted before training |
| 5. Independent league | Not executed | Sequential prerequisite absent |

## Scientific takeaways

The evidence strongly supports the original diagnosis: the selected-action model and narrow
self-play distribution are limiting action ranking. Population diversity substantially changes
policy strength, and public history/consequence structure gives another consistent improvement.
However, direct gains coexist with weaker uncertainty against the q0-parent heuristic reference.
This is genuine non-transitivity/robustness tension, not a simple monotonic Elo ladder.

Step 3's mixed structured recipe is the strongest development direction produced by the program, but
it is not promoted. Its architecture effect combines representation, phase routing, and added
capacity, and its heuristic non-regression lower bound still failed.

Step 4's partial targets support only infrastructure claims. The one-position repair reproduced all
replicate-1 rows exactly; the final refusal was one-ULP metadata strictness. No outcome about rollout
supervision itself can be inferred.

## Current policy status

- Selected hybrid champion: `q0-terminal-safety-v1`.
- Permanent pure-neural baseline: historical q0.
- Terminal offense: validated fixed tactical component for future research, not deployed as web
  default by this program.
- Mixed structured v2: development-only evidence, not a champion.
- Rollout treatment: nonexistent; no valid checkpoint was trained.

## Decision needed

A new user authorization is required to perform a narrow second Step-4 repair. The scientifically
clean option is to preserve every frozen panel, seed, world, target row, loss, arena, and gate while
changing only shard publication/accounting to compare integer terminal/leaf counts or accept the
bounded one-ULP complement difference. If that completes and validates, Step 5 can then be separately
resumed under the already approved high-level direction.

Without that authorization, the program remains stopped and no further experiment should launch.

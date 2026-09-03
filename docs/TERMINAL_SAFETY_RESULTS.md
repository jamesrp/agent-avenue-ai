# Terminal-Safety Hybrid Policy Results

**Status:** Complete — q0 retrained, q1–q4 executed, all-pairs diagnostics complete, shielded q0 retained
**Run date:** September 3, 2026
**Source revision:** `19c2871080503c62a53522415e0645913d7674b0`
**Parent experiment:** [Milestone 6 frozen self-play](MILESTONE6_RESULTS.md)
**Declaration:** [Terminal-safety hybrid policy experiment](TERMINAL_SAFETY_EXPERIMENT.md)

## Executive summary

The combined terminal-safety hybrid package materially improved the minimum viable policy, but it did
not fix the selected-action Monte Carlo self-play specialization problem. This experiment retrained
the checkpoint on shielded behavior data and deployed it behind the shield; it was not a shield-only
causal ablation.

A separately named `q0-terminal-safety-v1` was trained from 4,000 shielded epsilon-heuristic games
containing 65,746 decisions. It passed its q0 gate, produced strong random/heuristic development
results, and beat historical q0 directly:

| Development matchup | Games | Shielded q0 win rate | Paired-bootstrap 95% interval |
| --- | ---: | ---: | ---: |
| vs random | 800 | **83.0%** | 80.375%–85.5% |
| vs `greedy-public-v1` | 800 | **68.5%** | 65.25%–71.75% |
| vs historical q0 | 800 | **55.5%** | 52.5%–58.5% |

The q1–q4 candidates again became strong direct opponents to their shielded q0 parent: they won
75.5%–80.5% head-to-head. All four nevertheless failed the unchanged aligned heuristic
non-regression guardrail, so every immutable decision retained shielded q0. The four-generation
stopping result is again `budget_exhausted_inconclusive`, not a plateau.

On the untouched locked-final block, the retained shielded q0 scored:

| Locked-final matchup | Games | Shielded q0 win rate | Paired-bootstrap 95% interval | Seat 1 | Seat 2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| vs random | 1,000 | **77.9%** | 75.3%–80.5% | 76.4% | 79.4% |
| vs `greedy-public-v1` | 1,000 | **67.4%** | 64.4%–70.4% | 67.4% | 67.4% |
| vs historical q0 | 1,000 | **56.7%** | 54.1%–59.3% | 51.8% | 61.6% |

The declared safety invariant held over **591,219 audited shielded decisions** spanning training,
promotion, diagnostic, and final records: zero executed a publicly provable avoidable immediate
loss. The shield preserved 16,756 all-actions-losing forced fallbacks. This is empirical coverage of
the retained records, not a universal proof over every reachable game state.

## q0 training and development

The new q0 kept the historical encoder, model architecture, Monte Carlo target, split method and
90/10 ratio, optimizer, and training budget. It used newly domain-separated corpus, split, model, and
training seeds, so both the realized data split and initialized weights differ from historical q0.
The intended recipe change was `terminal-safety-v1`, applied outside the fixed 1/5 epsilon wrapper so
exploration could not restore a vetoed action.

| Corpus | Games | Decisions | Best epoch | Validation equal-game log loss |
| --- | ---: | ---: | ---: | ---: |
| `q0-terminal-safety-v1-a1` | 4,000 | 65,746 | 50 | 0.55740 |

The random development lower endpoint was 80.375%, comfortably above the predeclared 50% q0 gate.
The checkpoint and tensor identities prove this was a separately trained model: hybrid q0 is
`b511ae16…` / `b1562237…`, versus historical q0 at `bc6f070a…` / `c8239721…`. Its 55.5% direct
result against historical q0 measures the strength of the combined retraining-plus-shield package.

## q1–q4 promotion results

Every candidate and incumbent was greedily deployed behind the same terminal-safety shield. The
heuristic difference is the candidate win rate minus the shielded q0 win rate on aligned setup
blocks. Promotion required its lower endpoint to be above -5 percentage points, in addition to the
direct-parent, random, seat, and integrity gates. The bracketed direct and random intervals below
are the promotion-gate recomputations with their declared bootstrap seeds; they can differ by one
tenth of a point from the descriptive arena-report bootstrap using a different seed.

| Candidate | vs shielded q0 | Seat 1 | Seat 2 | vs random | vs heuristic | q0 vs heuristic | Heuristic difference | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| q1 | 80.5% [78.1, 82.9] | 78.0% | 83.0% | 87.25% [84.0, 90.25] | 58.75% | 64.75% | -6.0% [-13.25, 1.0] | Retain q0 |
| q2 | 78.8% [76.3, 81.3] | 77.6% | 80.0% | 84.75% [81.0, 88.25] | 55.0% | 66.25% | -11.25% [-17.0, -5.5] | Retain q0 |
| q3 | 79.5% [77.0, 82.0] | 76.2% | 82.8% | 84.5% [81.0, 88.0] | 58.0% | 64.25% | -6.25% [-12.25, -0.25] | Retain q0 |
| q4 | 75.5% [73.0, 78.0] | 75.4% | 75.6% | 77.25% [73.25, 81.25] | 50.75% | 62.75% | -12.0% [-18.0, -5.75] | Retain q0 |

q1 and q3 had point regressions near the allowed five-point tolerance and were rejected
conservatively because their lower confidence endpoints crossed the frozen guardrail. q2 and q4
showed clear heuristic regressions. No candidate was practically equivalent to q0 on the direct
comparison; each plateau assessment was `inconclusive` because the candidates were much stronger
head-to-head while failing robustness.

## Training and safety diagnostics

| Generation | Training decisions | Best epoch | Validation loss | Vetoed actions | Forced-loss fallbacks | Executed avoidable losses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q0 | 65,746 | 50 | 0.55740 | 39,208 | 1,955 | **0** |
| q1 | 61,712 | 33 | 0.53130 | 30,334 | 1,852 | **0** |
| q2 | 60,550 | 48 | 0.53866 | 30,049 | 1,802 | **0** |
| q3 | 60,134 | 44 | 0.48617 | 29,261 | 1,851 | **0** |
| q4 | 60,696 | 25 | 0.53603 | 29,859 | 1,856 | **0** |

The shield therefore did exactly what it was designed to do in retained-record replay. In the locked
final against historical q0, the shielded champion executed zero avoidable provable losses, while the
unshielded historical q0 executed 274. Against random, the random policy executed 375 such losses
while shielded q0 executed zero. The audit deliberately reuses the production public filter, so it is
a deterministic replay-consistency check rather than an independent oracle. These are policy
diagnostics, not independent causal estimates of the win-rate gain.

## Held-out all-pairs diagnostic

For each `qn`, fresh seat-swapped games were generated for every unordered pair with replacement
from `(heuristic, q0, ..., q(n-1))`. Rejected proposals remained in later diagnostic pools. Each cell
used 200 paired setup blocks; all 14,000 games were excluded from training and promotion.

| Model | Policy-pair cells | Games | Decisions | Equal-cell macro log loss | 95% interval | Worst cell |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| q0 | 1 | 400 | 6,570 | 0.5413 | 0.5079–0.5760 | heuristic–heuristic: 0.5413 |
| q1 | 3 | 1,200 | 18,802 | 0.5447 | 0.5248–0.5652 | heuristic–heuristic: 0.5649 |
| q2 | 6 | 2,400 | 37,796 | 0.5511 | 0.5367–0.5659 | q1–q1: 0.6248 |
| q3 | 10 | 4,000 | 61,384 | 0.5582 | 0.5484–0.5682 | q1–q1: 0.6210 |
| q4 | 15 | 6,000 | 92,768 | 0.5628 | 0.5547–0.5710 | q2–q3: 0.6303 |

The rows below are the complete paired-block log-loss matrix. They are **not** a common-test-set
longitudinal model comparison: each `qn` used fresh independently seeded cells, and the macro pool
expands with n. Comparisons among cells within one model are aligned with that model's declared
diagnostic; repeated cell names across models are separate samples.

| Scored model | Generating policy pair | Log loss |
| --- | --- | ---: |
| q0 | heuristic–heuristic | 0.5413 |
| q1 | heuristic–heuristic | 0.5649 |
| q1 | heuristic–q0 | 0.5353 |
| q1 | q0–q0 | 0.5339 |
| q2 | heuristic–heuristic | 0.6099 |
| q2 | heuristic–q0 | 0.5272 |
| q2 | heuristic–q1 | 0.5969 |
| q2 | q0–q0 | 0.4998 |
| q2 | q0–q1 | 0.4479 |
| q2 | q1–q1 | 0.6248 |
| q3 | heuristic–heuristic | 0.5725 |
| q3 | heuristic–q0 | 0.5447 |
| q3 | heuristic–q1 | 0.5825 |
| q3 | heuristic–q2 | 0.5976 |
| q3 | q0–q0 | 0.5284 |
| q3 | q0–q1 | 0.4487 |
| q3 | q0–q2 | 0.4623 |
| q3 | q1–q1 | 0.6210 |
| q3 | q1–q2 | 0.6113 |
| q3 | q2–q2 | 0.6132 |
| q4 | heuristic–heuristic | 0.6044 |
| q4 | heuristic–q0 | 0.5581 |
| q4 | heuristic–q1 | 0.5863 |
| q4 | heuristic–q2 | 0.5650 |
| q4 | heuristic–q3 | 0.5751 |
| q4 | q0–q0 | 0.4992 |
| q4 | q0–q1 | 0.4560 |
| q4 | q0–q2 | 0.4387 |
| q4 | q0–q3 | 0.4376 |
| q4 | q1–q1 | 0.6269 |
| q4 | q1–q2 | 0.5915 |
| q4 | q1–q3 | 0.6218 |
| q4 | q2–q2 | 0.6216 |
| q4 | q2–q3 | 0.6303 |
| q4 | q3–q3 | 0.6293 |

The diagnostic sharpens the original failure analysis. Within each model's fresh matrix, later
models predicted q0-containing games well, especially q0-versus-candidate cells, while prediction
was worse on interactions among rejected self-play candidates and on heuristic-only play. Because
these targets follow different continuation policies and label only selected actions, the matrix
does not measure counterfactual action quality, and it does not causally identify the training target
as the failure source. It does show that lower same-generation validation loss was insufficient as a
robustness indicator: q3 had the best training validation loss (0.4862) but still failed the heuristic
gate and had roughly 0.61–0.62 loss on its candidate-only cells.

## Interpretation

The experiment supports retaining `terminal-safety-v1` in the MVP policy under the declared gate:

- it removed publicly provable avoidable immediate losses from all 591,219 audited shielded
  decisions, while making no claim about longer-horizon or non-provable mistakes;
- the combined retraining-plus-shield package beat historical q0 directly; and
- its final seat results against the heuristic were exactly balanced at 67.4% from both seats.

Historical q0's prior heuristic aggregate was 57.3%, while the hybrid scored 67.4% on a different
fresh final block. That descriptive comparison and the direct 56.7% hybrid-versus-historical-q0
result support selecting the package, but they do not isolate the deployment veto from changed
training data, split, initialization, or seeds. A common-block old/new-checkpoint × shield/no-shield
factorial evaluation would be required for that causal attribution.

The experiment does **not** validate incumbent-only selected-action Monte Carlo self-play as a robust
improvement method. q1–q4 continued to dominate q0 while regressing against the heuristic, and the
all-pairs matrix exposed distribution-sensitive prediction loss on interactions among prior rejected
proposals. The selected-action target is a plausible mechanism, not a causally isolated finding. The
next strength experiment should change the data/target recipe rather than merely extending the same
chain—most plausibly predeclared mixed-opponent replay and/or counterfactual candidate-ranking
supervision. Historical q0 remains the permanent pure-neural baseline; `q0-terminal-safety-v1` is the
selected hybrid champion under this protocol for subsequent controlled work.

All four retained per-generation attempt assessments are `inconclusive`. The top-level
`budget_exhausted_inconclusive` label is a reproducible application of the frozen stopping rule to
those decisions, but the original driver did not emit a separate immutable stopping-decision file.

## Holdout, provenance, and retention

The locked-final validator checked 87 retained corpora and 55,900 unique prior setup blocks,
including the historical Milestone 6 artifacts and reserved q0 arena schedules. The final 500-pair
block had zero normalized game-config/setup-seed overlap.

| Item | Identity |
| --- | --- |
| Git revision | `19c2871080503c62a53522415e0645913d7674b0` |
| `uv.lock` SHA-256 | `2c56e1ba299c5a9b0f9363d6080e1bc4ed8f49162f7cccdacc2dc3734c9131c8` |
| Package code fingerprint | `9c8b39bb4e81bb29e0902c56198557f63b7e9992bdaff845967c562fc064f429` |
| Experiment plan | `3e6bbc29041f3d2b9c1b55dcac52cdb89cfe4e7bff8b858d3b8151fa2c2b6fe3` |
| Result artifact | `ed0a4f7da5767d7b8e146624d0c61d417df4c32f989030c69780e4f90c300d23` |
| Shielded q0 checkpoint | `b511ae162b794da6450fd3151762d98b4e5c8475c1af23e5b9289d0140a0a06d` |
| Historical q0 checkpoint | `bc6f070aa30e2155ed6953a065c9f67e10357b54f222e3e944273493ab4e6e8e` |
| Locked-final holdout validation | `c3c7a27266c609de75941e1c027490b1d11bdb9da9c0d9f98aed0cf34a0f293f` |

The ignored verified archive is
`artifacts/archive/terminal-safety-v1-artifacts-2026-09-03.tar.gz` (30,036,123 bytes). Its embedded
manifest checksums 237 non-lock payload files; the tar also contains directories, lock files, and the
manifest itself. Its SHA-256 is
`d33c3e4bc3fea107a4d7ffe93dc841d9af8d12687269b7da783f540167fb27b9`.
The archive embeds the frozen historical q0 input and a member-level checksum manifest. It is the
sealed source of truth. The live `driver.stdout` grew after sealing when the driver printed the final
JSON result, so the mutable live run tree is not claimed to be byte-identical to the archive.

## Reproduction and validation

To validate an already restored completed archive at the recorded source revision:

```bash
git switch --detach 19c2871080503c62a53522415e0645913d7674b0
uv sync --extra rl --locked
uv run python scripts/run_terminal_safety_v1.py
```

When the fixed archive and sidecar are present, this validates their checksums and returns the
retained result without rewriting them. A fresh end-to-end regeneration additionally requires the
historical q0 checkpoint, historical `runs/q0-corpus`, and restored Milestone 6 retained corpora at
the paths referenced by the driver; those inputs are used for direct comparison and locked-final
setup-overlap validation.

The complete terminal-safety plan, q0 gate, q1–q4 promotion decisions, cross-policy reports,
semantic records, checkpoints, final reports, and checksums are retained in the verified archive.

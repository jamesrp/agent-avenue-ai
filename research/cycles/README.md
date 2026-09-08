# Research cycle records

Each substantive cycle starts with one user-approved agreement and ends with one briefing. Keep the
small, human-readable agreement and compact result note in Git. Keep generated state, logs, corpora,
checkpoints, and evaluation records under `runs/`/`artifacts/` according to
[`docs/EXPERIMENT_PROTOCOL.md`](../../docs/EXPERIMENT_PROTOCOL.md).

A cycle record should contain:

1. question, hypotheses, and why the answer matters;
2. work packages, dependencies, deliverables, and completion criteria;
3. frozen evaluation criteria and scientific constraints;
4. runtime/compute, concurrency, worker, and retry limits;
5. preapproved operational repairs or confirmation analyses;
6. decisions reserved for the user;
7. exact source/configuration identities once implementation is frozen; and
8. final state plus links to the briefing and retained evidence.

Executable JSON plans use `bounded-research-workflow-v1` and are fingerprinted. A proposed agreement
may remain Markdown until the user approves it. Approval is a scientific decision, not merely a
change from `"proposed"` to `"approved"`: the lead records what was agreed, then commits the exact
version before claim-generating execution.

The setup-only executable plan is `setup-smoke-v1.json`. It is explicitly smoke evidence and must
not be cited as a gameplay-strength result.

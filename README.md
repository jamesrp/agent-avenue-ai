# agent-avenue-ai
[Agent Avenue](https://boardgamegeek.com/boardgame/422732/agent-avenue) AI experiments.

All gameplay content is copyright Nerdlab Games. This is just an ML research project and
I don't intend to host the game as a playable app.

## Project goal
Build a strong, reproducible, and inspectable AI for the two-player base game of **Agent Avenue**—not
just a rules simulator. The broader aim is a complete playable opponent and research platform in
the spirit of [Keldon Jones's 2009 Race for the Galaxy AI project](https://www.keldon.net/rftg/):
combine a faithful game implementation with self-play, learned evaluation, rigorous comparison,
and artifacts that let results be reproduced and improved over time. The project is inspired by
that end-to-end ambition rather than committed to reproducing Keldon's exact methods.

The engine should be deterministic for all the base two-player game for now.
Setup randomness is explicit; every player action is a first-class decision;
transitions are deterministic and replayable; and authoritative state
is strictly separated from player-visible observations. Random, scripted, and simple heuristic
agents plus single-game and pull-based multi-game runners establish non-ML baselines and the batch
orchestration seam.

There should be a lightweight web UI to play the game (to verify the rules engine is correct
and kick the tires of an AI opponent), but this can be barebones as it's not the main goal.
Just enough for to QA the system.

The first learned baseline: a viewpoint-relative flat observation encoder, a small
PyTorch value network, terminal-outcome training from compact replay data, information-safe
one-ply afterstate selection, frozen-checkpoint iterative self-play, and statistically grounded
arena evaluation. It must run usefully on a CPU-only exe.dev development box while retaining clean
paths to parallel actors and GPU-backed training or inference.

Preserve the engine as an explicit state machine. Every player choice remains a semantic
`Decision`, and applying one semantic `Action` advances to the next decision or terminal result.
The engine never calls agents or exposes model tensors. Learned
components consume only versioned player-safe observations and public decision context; trusted
orchestration may operate the engine but must not allow candidate evaluation to exploit hidden
deck order, facedown cards, or hands.

Keep rules execution, observation and public-context construction, encoding, model definition,
training, inference, self-play orchestration, replay storage, evaluation, and profiling loosely
coupled. Actions remain stable structured data rather than display strings or legal-action
indices. Every dataset, checkpoint, run, and arena report records enough schema versions,
fingerprints, seeds, and configuration to reject incompatible inputs and reproduce its result.

The long-term measure of success is an AI that becomes meaningfully stronger through
iterative self-play while remaining fair, testable, auditable, and practical to run—not merely a
neural-network demo attached to the rules engine.

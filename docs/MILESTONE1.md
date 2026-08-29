# Milestone 1: Deterministic Rules Engine

## Goal

Implement the complete two-player base game in `RULES.md` as a deterministic, replayable state
machine. At the end of this milestone, trusted Python code can set up a seeded game, inspect the
current semantic decision, apply one legal semantic action, and continue until a fully explained
terminal result.

This milestone is the foundation for the web and AI milestones. Correctness, information hiding,
and stable domain types take priority over UI or optimization.

## Scope

### Domain model

Create explicit, typed representations for:

- the eight card names and their copy-count effects;
- the 38-card base deck and a validated deck/configuration definition;
- player identifiers, scores, hands, recruited cards, active player, and turn number;
- the face-up/face-down offer and the play, recruit, and end-of-turn phases;
- semantic decisions and actions, including a deterministic decision revision used to reject stale
  submissions;
- terminal outcomes, winner, reason, and the facts used to resolve a tie.

Use frozen dataclasses, enums, and tuples for authoritative state wherever practical. Identical card
copies do not need artificial identities unless an identity is required to make replay or validation
unambiguous.

### State-machine API

Expose a small engine API along these lines:

1. `new_game(config, seed)` creates the shuffled deck, deals both hands, and selects the configured
   starting player.
2. `current_decision(state)` returns the one decision that must be answered, or a terminal outcome.
3. `legal_actions(state)` returns stable structured actions rather than display labels or indexes.
4. `apply_action(state, action)` validates the decision revision and action, then returns a new state.
5. Replay support reconstructs a game from normalized configuration, seed, and semantic actions and
   verifies the final fingerprint/outcome.

The engine must never invoke an agent, render UI, or depend on FastAPI, NumPy, or PyTorch.

### Rule execution

Implement the turn in the exact order specified by `RULES.md`:

1. The active player selects an ordered face-up/face-down pair from hand.
2. The names must differ unless every card in that hand has the same name.
3. The active player draws up to four cards after both cards are played.
4. The opponent chooses the face-up or face-down offer slot to recruit.
5. The active player receives the remaining card.
6. Each recruit effect is calculated from that player's resulting count of that card name.
7. Both score changes are applied before terminal conditions are adjudicated.
8. Win, loss, simultaneous-condition, deck-exhaustion, and active-player tie rules are resolved.
9. If the game continues, active player and turn advance.

Centralize terminal adjudication so precedence and tie behavior are explicit and independently
testable. Preserve the active player through end-of-turn resolution because that player wins every
tie described by the rules.

### Player-safe observations

Add a separate `agent_avenue.observation` boundary before either the web UI or agents are built.
Given authoritative state and a viewer, it should return only information that viewer may know,
including:

- their own hand;
- public scores, recruited cards, active player, turn, and remaining deck count;
- the public face-up card during recruitment;
- the viewer's own remembered face-down card only when the rules permit that viewer to know it;
- public action history and current decision context.

It must not expose deck order, the opponent's hand, or an opponent's unknown face-down card through
fields, legal actions, debug text, serialization, or fingerprints.

### Errors and validation

Fail clearly for:

- actions that are illegal for the current decision;
- actions submitted for an old decision revision;
- malformed or impossible states/configurations;
- replay inputs with unsupported schema/rules versions or mismatched fingerprints.

Errors should carry enough structured context for a CLI or web adapter to explain the problem
without parsing an exception string.

## Suggested package layout

```text
src/agent_avenue/
  engine/
    cards.py          # card definitions and deck construction
    model.py          # authoritative state, decisions, actions, outcomes
    setup.py          # seeded setup and normalized configuration
    transitions.py    # legal actions and pure state transitions
    terminal.py       # end-of-turn adjudication
    replay.py         # replay records and deterministic verification
  observation/
    model.py          # player-visible immutable types
    build.py          # authoritative-state to safe-observation conversion
```

Exact filenames may change, but the engine/observation dependency boundary should not.

## Implementation sequence

1. Define card, player, action, decision, outcome, configuration, and state types.
2. Implement and validate the canonical 38-card deck.
3. Implement seeded setup and normalized configuration serialization.
4. Implement offer-generation rules and play-action validation.
5. Implement draw, recruit, copy-count effects, and score updates.
6. Implement all terminal checks and tie resolution.
7. Add observation construction and explicit safe serialization.
8. Add replay recording, schema/rules versioning, and round-trip verification.
9. Add scripted end-to-end game fixtures for regressions.
10. Profile representative games and remove only demonstrated bottlenecks.

## Test plan

Mirror the package layout under `tests/` and cover at least:

- canonical deck composition and 38-card conservation across every zone;
- four-card initial hands, explicit starting player, and reproducibility for a fixed seed;
- all legal ordered offers, including duplicate cards and the all-one-name exception;
- rejection of same-name offers when another name is available;
- drawing to four, partial draws, and empty-deck behavior;
- first, second, and third-or-later effects for all six repeated card types;
- Sidekick and Mole scoring;
- Codebreaker instant wins and Daredevil instant losses;
- every simultaneous win/loss combination and active-player tie rule;
- score-gap wins and deck-exhaustion wins/ties;
- illegal, malformed, and stale actions leaving the original state unchanged;
- deterministic state fingerprints and replay round trips;
- property tests for card conservation, legal-action completeness, and deterministic transitions;
- hidden-information tests that recursively inspect serialized observations for leaks;
- scripted complete games that exercise ordinary, instant, and exhaustion endings.

## Deliverables

- Typed engine and observation packages with documented public APIs.
- A versioned replay schema and small replay fixtures committed under `tests/fixtures/`.
- A minimal command that can run or replay a scripted game for developer verification.
- Updated `README.md` development instructions if commands or package layout change.

## Definition of done

- Every rule in `RULES.md` maps to an implementation and focused test.
- The same configuration, seed, and action sequence always produce the same states and outcome.
- A game can be replayed without storing hidden-state snapshots.
- Player observations pass explicit hidden-information tests.
- Core imports and tests work after plain `uv sync`; no web or RL dependency is required.
- `make check` passes.

## Out of scope

- Browser UI or network APIs.
- Agent strategy and automated action selection.
- Learned models, tensor encoders, self-play training, or large replay datasets.
- More than the two-player base game described in `RULES.md`.

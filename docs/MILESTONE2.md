# Milestone 2: Lightweight Web QA UI

**Status:** Complete
**Completed:** August 29, 2026

This document describes the original hot-seat milestone. Milestone 3 subsequently reused the
controller seam to add human-versus-random and human-versus-heuristic play from either seat; see
[`WEB_QA.md`](WEB_QA.md) for the current checklist.

## Goal

Provide a deliberately small, server-rendered web interface for playing and inspecting a game built
on Milestone 1. Its purpose is to find rules, observation, and interaction bugs—not to become a
public hosted copy of Agent Avenue.

The first version should support seeded two-player hot-seat play. It should also leave a clean
controller seam so the initial AI from Milestone 3 can occupy either seat without changing game
rules or templates.

## Prerequisites

- Milestone 1's state-machine and observation APIs are stable enough to consume.
- Replay records can identify and reconstruct a game.
- The engine can explain illegal/stale actions with structured errors.

Do not work around missing engine behavior in route handlers. Fix it in the owning engine or
observation module first.

## Scope

### Application shape

Build `agent_avenue.web` as an optional FastAPI application. Keep the core package importable after a
plain `uv sync`; importing `agent_avenue.engine` must never import FastAPI or template code.

Use:

- server-rendered HTML and ordinary form submissions;
- small, local CSS with no frontend build chain;
- an application factory for tests and deployment;
- in-memory server-side game/session storage for this milestone;
- opaque game/session identifiers rather than serialized state in cookies or URLs.

A process restart may discard active games. Durable multiplayer accounts, matchmaking, and database
operations are not part of this milestone.

### User flow

1. A landing page explains the QA purpose and starts a new game.
2. The user may choose a seed or let the server generate and display one.
3. A pass-device screen identifies which player should take control without revealing that player's
   private view early.
4. The acting player sees their own hand, legal play controls, both public tableaux and scores, turn
   information, and remaining deck count.
5. During recruitment, the opponent sees the face-up option and an unrevealed face-down option, then
   selects one offer slot.
6. After recruitment, the result and score changes are shown before the next pass-device step.
7. Terminal pages show the winner, terminal reason, final public state, seed, replay identifier, and
   a way to start or replay another game.

Face-down cards and opposing hands must never be present in HTML, JSON, form values, comments,
`data-*` attributes, error pages, or browser logs before they are public.

### QA information

Display only public or viewer-safe information useful for verification:

- current phase, decision actor, active player, and turn number;
- each score and recruited-card counts/cards;
- the viewer's hand when they are entitled to see it;
- face-up and properly hidden face-down offer slots;
- legal semantic controls for the current decision;
- public action/recruit/score history;
- normalized seed/configuration, rules version, and replay/fingerprint identifiers.

Do not expose an authoritative-state dump or add a client-side “debug” endpoint. If maintainers need
one, keep it as a trusted Python/test helper outside the browser surface.

### HTTP boundary

Use routes equivalent to:

```text
GET  /                         landing/new-game form
POST /games                    create a seeded game
GET  /games/{game_id}/pass     pass-device interstitial
GET  /games/{game_id}          render the current viewer-safe decision
POST /games/{game_id}/actions  submit one semantic action
GET  /games/{game_id}/result   render a terminal result
GET  /healthz                  process health only
```

The exact URLs may differ. Regardless of route design:

- derive the current decision and legal actions on the server;
- submit semantic action fields plus the decision revision, never a transient legal-action index;
- validate every submission again at the engine boundary;
- use POST/redirect/GET so refresh does not repeat an action;
- return a clear 4xx response or safe page for malformed, illegal, stale, or missing-game requests;
- prevent one browser session from guessing another session's active game identifier.

### Presentation

Keep the interface accessible and functional at phone and desktop widths:

- semantic HTML, labels, buttons, headings, and visible keyboard focus;
- sufficient contrast without relying on card artwork;
- textual card names and effects as the authoritative presentation;
- clear face-up/face-down distinction and selected-card state;
- no copyrighted assets beyond material already authorized in the repository;
- no JavaScript requirement for completing a game.

Small progressive enhancements are acceptable only if the form-based path remains complete.

## Suggested package layout

```text
src/agent_avenue/web/
  app.py             # application factory and health route
  routes.py          # HTTP-to-application translation
  sessions.py        # in-memory game/session repository
  presenters.py      # observation/outcome to template view models
  templates/
  static/
tests/web/
```

Presenters should consume safe observations and public outcomes, not authoritative state.

## Implementation sequence

1. Add and lock only the required packages under the existing `web` optional dependency.
2. Add an application factory, configuration type, and health check.
3. Implement opaque session/game storage with deterministic engine state kept only server-side.
4. Implement new-game creation with seed/configuration display.
5. Add pass-device handling and viewer selection.
6. Render play decisions from safe observations and submit semantic offer actions.
7. Render recruit decisions without exposing the face-down card.
8. Render transition summaries, public history, and terminal outcomes.
9. Add replay identifiers and a developer-friendly replay/start-over flow.
10. Add responsive styling and complete accessibility/keyboard checks.
11. Add a production run command binding to `0.0.0.0:8000` and document it.

## Test plan

Use application-level tests as well as a short real-browser smoke pass:

- create games with explicit and generated seeds;
- confirm identical seeds/configurations create identical initial games;
- exercise a complete hot-seat game through HTTP requests;
- verify POST/redirect/GET and refresh safety;
- reject stale decision revisions, illegal card pairs, invalid offer slots, and unknown games;
- confirm that only actions returned for the current decision are accepted;
- assert opposing hands, deck order, and hidden offer cards never appear in response bodies;
- assert hidden values do not appear in HTML attributes, scripts, comments, or error responses;
- verify the pass-device page contains no private player information;
- test ordinary, instant win/loss, simultaneous-condition, and deck-exhaustion result pages;
- test session isolation with two clients;
- test useful behavior when optional web dependencies are not installed;
- browser-check play and recruitment at narrow and desktop viewport sizes;
- browser-check keyboard navigation and visible focus for all actions.

Prefer small deterministic fixtures from Milestone 1 over long random games in route tests.

## Deliverables

- Optional `agent_avenue.web` package and server-rendered templates/styles.
- A documented command that serves the app on `0.0.0.0:8000`.
- HTTP and hidden-information regression tests.
- A short manual QA checklist for playing one complete hot-seat game.
- README links to the web run command and the project's non-public research purpose.

## Definition of done

- Two people can complete a legal game by passing one browser between them.
- Every browser view is constructed from a player-safe observation or public terminal result.
- Viewing source and inspecting network responses reveals no hidden authoritative state.
- Invalid or repeated submissions cannot corrupt or advance a game incorrectly.
- The UI clearly exposes enough public history, seed, and replay metadata to reproduce a bug.
- Core engine usage still works without the `web` extra.
- `make check` and the documented web test command pass.

## Out of scope

- Public hosting, user accounts, matchmaking, chat, or remote multiplayer.
- Persistent game storage or cross-process session sharing.
- Rich client frameworks, WebSockets, animations, or a frontend build toolchain.
- An AI controller; Milestone 3 plugs one into the controller seam.
- Training dashboards, model management, or research experiment visualization.

# Web QA checklist

The web interface is a private rules and observation QA tool. Do not deploy it as a public playable
copy. It stores games only in one process and intentionally has no accounts or durable storage.

## Start the server

```bash
uv sync --extra web
make web
```

Open `http://localhost:8000`. The server binds to `0.0.0.0:8000` and must run with one worker because
active sessions are in memory.

## Complete-game smoke pass

- [ ] Start one game with a known seed and another with a blank seed; confirm both displayed seeds.
- [ ] On every handoff, confirm the pass-device page names only the next player and shows no hand or
      offer card.
- [ ] As the active player, confirm only your hand is visible and every legal ordered offer is a
      keyboard-focusable button.
- [ ] Submit an offer, refresh, and confirm the action is not repeated.
- [ ] As the recruiter, confirm the face-up card is named and the face-down card remains “Hidden
      agent,” including in page source and form values.
- [ ] Recruit each slot at least once during the game.
- [ ] After recruitment, verify both revealed cards, assignments, score changes, public tableaux,
      and history before continuing.
- [ ] Finish the game and verify winner, reason, resolution, seed, replay ID, rules/shuffle versions,
      and public fingerprint.
- [ ] Use **Replay this seed** and confirm the initial hand matches.
- [ ] Start human-versus-random and human-versus-heuristic games with the human in each seat;
      confirm an AI opening advances directly to the first human decision.
- [ ] After each human action in an AI game, confirm automated decisions advance only until the
      next human decision or terminal result and resolved actions appear in public history.
- [ ] Inspect AI-game responses and confirm they contain no agent RNG seed, heuristic score,
      opposing hand, deck order, or unrevealed face-down identity.

## Security and responsive checks

- [ ] Use a private/incognito second browser session and confirm it cannot open the first session’s
      copied game URL.
- [ ] Use browser Back after a handoff and confirm a previous private view is not restored from cache.
- [ ] Inspect source and network responses at play, recruit, summary, error, and terminal screens;
      confirm no opposing hand, deck order, or unrevealed face-down value appears.
- [ ] Check the play and recruit screens around 390 px and 1280 px widths.
- [ ] Navigate every control using Tab/Shift-Tab and confirm the gold focus outline is visible.
- [ ] Disable JavaScript and complete at least one full turn.

Automated coverage is available with `make test-web` and the full required check is `make check`.

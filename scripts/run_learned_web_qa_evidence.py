#!/usr/bin/env python3
"""Exercise the two real allowlisted checkpoints without exposing server paths."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse

from fastapi.testclient import TestClient

from agent_avenue.web import create_app
from agent_avenue.web.sessions import BrowserSession


def _csrf(body: str) -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', body)
    if match is None:
        raise RuntimeError("landing page has no CSRF token")
    return match.group(1)


def run(output: Path) -> None:
    app = create_app()
    cases: list[dict[str, object]] = []
    for key in ("historical-q0", "terminal-safety-q0"):
        for seat in ("player_one", "player_two"):
            client = TestClient(app)
            landing = client.get("/")
            token = _csrf(landing.text)
            response = client.post(
                "/games",
                data={
                    "csrf_token": token,
                    "seed": "23",
                    "mode": f"human-learned:{key}",
                    "human_seat": seat,
                },
                follow_redirects=False,
            )
            if response.status_code != 303:
                raise RuntimeError(f"allowlisted game creation failed for {key}/{seat}")
            game_id = urlparse(response.headers["location"]).path.split("/")[2]
            page = client.get(f"/games/{game_id}")
            repository = app.state.session_repository
            session = repository.session(client.cookies["agent_avenue_session"])
            if not isinstance(session, BrowserSession):
                raise RuntimeError("browser session was not retained")
            game = repository.game(session, game_id)
            if game is None or game.revealed_actor is None:
                raise RuntimeError("learned game did not reach a human decision")
            automated = next(
                controller for controller in game.controllers if controller.kind == "agent"
            )
            fingerprint = automated.config.get("checkpoint_fingerprint")
            base = automated.config.get("base")
            if fingerprint is None and isinstance(base, dict):
                fingerprint = base.get("checkpoint_fingerprint")
            if not isinstance(fingerprint, str):
                raise RuntimeError("learned controller lacks checkpoint identity")
            if (
                page.status_code != 200
                or game.revealed_actor.value != seat
                or fingerprint not in page.text
                or "weights.pt" in page.text
                or "tensor_digest" in page.text
            ):
                raise RuntimeError("learned web evidence failed identity or safety checks")
            cases.append(
                {
                    "opponent_key": key,
                    "human_seat": seat,
                    "checkpoint_fingerprint": fingerprint,
                    "status": "ok",
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "version": "learned-web-qa-evidence-v1",
                "status": "completed",
                "cases": cases,
                "information_boundary": (
                    "Responses expose an opaque allowlist key, public label, and checkpoint "
                    "fingerprint; no checkpoint path, tensor digest, logits, or hidden game state."
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()

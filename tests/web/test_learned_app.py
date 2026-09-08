from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("torch")

from agent_avenue.engine import Phase, PlayOfferAction, RecruitAction, legal_actions
from agent_avenue.learning import create_model, save_checkpoint
from agent_avenue.web import LearnedOpponentConfig, WebConfig, create_app
from agent_avenue.web.sessions import BrowserSession, SessionRepository, WebGame


def _csrf(body: str) -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', body)
    assert match
    return match.group(1)


def _checkpoint(path: Path, seed: int) -> tuple[Path, str]:
    saved = save_checkpoint(path, create_model(seed=seed), metrics={"web_fixture": seed})
    return path, saved.checkpoint_fingerprint


def _app(tmp_path: Path) -> tuple[object, dict[str, str], dict[str, Path]]:
    historical, historical_fingerprint = _checkpoint(tmp_path / "historical", 11)
    champion, champion_fingerprint = _checkpoint(tmp_path / "champion", 12)
    config = WebConfig(
        learned_opponents=(
            LearnedOpponentConfig("historical-q0", "Historical q0", historical),
            LearnedOpponentConfig(
                "terminal-safety-q0",
                "Terminal-safety q0",
                champion,
                terminal_safety=True,
            ),
        )
    )
    return (
        create_app(config),
        {
            "historical-q0": historical_fingerprint,
            "terminal-safety-q0": champion_fingerprint,
        },
        {"historical-q0": historical, "terminal-safety-q0": champion},
    )


def _new_game(
    client: TestClient, *, mode: str, human_seat: str, seed: str = "23"
) -> tuple[str, str]:
    landing = client.get("/")
    token = _csrf(landing.text)
    response = client.post(
        "/games",
        data={
            "csrf_token": token,
            "seed": seed,
            "mode": mode,
            "human_seat": human_seat,
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    game_id = urlparse(response.headers["location"]).path.split("/")[2]
    return game_id, token


def _stored_game(client: TestClient, app: object, game_id: str) -> WebGame:
    repository: SessionRepository = app.state.session_repository  # type: ignore[attr-defined]
    session = repository.session(client.cookies["agent_avenue_session"])
    assert isinstance(session, BrowserSession)
    game = repository.game(session, game_id)
    assert game is not None
    return game


def _action_data(action: PlayOfferAction | RecruitAction, token: str) -> dict[str, str]:
    data = {"csrf_token": token, "revision": str(action.revision)}
    if isinstance(action, PlayOfferAction):
        data.update({"face_up": action.face_up.value, "face_down": action.face_down.value})
    else:
        data["slot"] = action.slot.value
    return data


def test_landing_exposes_only_server_labels_not_checkpoint_paths(tmp_path: Path) -> None:
    app, _, paths = _app(tmp_path)
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert "human-learned:historical-q0" in response.text
    assert "human-learned:terminal-safety-q0" in response.text
    assert "Historical q0" in response.text
    assert "Terminal-safety q0" in response.text
    assert all(str(path) not in response.text for path in paths.values())


def test_allowlisted_learned_agents_play_from_either_seat_and_show_identity(
    tmp_path: Path,
) -> None:
    app, fingerprints, paths = _app(tmp_path)
    for key, human_seat in (
        ("historical-q0", "player_one"),
        ("terminal-safety-q0", "player_two"),
    ):
        client = TestClient(app)
        game_id, _ = _new_game(client, mode=f"human-learned:{key}", human_seat=human_seat)
        game = _stored_game(client, app, game_id)
        page = client.get(f"/games/{game_id}")
        assert page.status_code == 200
        assert game.revealed_actor is not None
        assert game.revealed_actor.value == human_seat
        assert fingerprints[key] in page.text
        assert str(paths[key]) not in page.text
        assert "tensor_digest" not in page.text
        automated = next(
            controller for controller in game.controllers if controller.kind == "agent"
        )
        assert automated.config


def test_unknown_or_path_shaped_learned_key_is_rejected_without_echo(tmp_path: Path) -> None:
    app, _, _ = _app(tmp_path)
    client = TestClient(app)
    token = _csrf(client.get("/").text)
    submitted = "/tmp/not-allowlisted/weights.pt"
    response = client.post(
        "/games",
        data={
            "csrf_token": token,
            "seed": "17",
            "mode": f"human-learned:{submitted}",
            "human_seat": "player_one",
        },
    )
    assert response.status_code == 400
    assert submitted not in response.text
    assert "weights.pt" not in response.text


def test_validated_model_is_reused_and_replay_preserves_opaque_key(tmp_path: Path) -> None:
    app, fingerprints, paths = _app(tmp_path)
    client = TestClient(app)
    first_id, token = _new_game(
        client, mode="human-learned:historical-q0", human_seat="player_one", seed="4"
    )
    second_id, _ = _new_game(
        client, mode="human-learned:historical-q0", human_seat="player_one", seed="5"
    )
    first = _stored_game(client, app, first_id)
    second = _stored_game(client, app, second_id)
    first_agent = next(
        controller.agent for controller in first.controllers if controller.kind == "agent"
    )
    second_agent = next(
        controller.agent for controller in second.controllers if controller.kind == "agent"
    )
    assert first_agent is second_agent

    for _ in range(100):
        if first.state.phase is Phase.TERMINAL:
            break
        action = legal_actions(first.state)[0]
        response = client.post(
            f"/games/{first_id}/actions",
            data=_action_data(action, token),
            follow_redirects=False,
        )
        assert response.status_code == 303
    assert first.state.phase is Phase.TERMINAL
    result = client.get(f"/games/{first_id}/result")
    assert 'name="mode" value="human-learned:historical-q0"' in result.text
    assert fingerprints["historical-q0"] in result.text
    assert str(paths["historical-q0"]) not in result.text

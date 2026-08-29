import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from fastapi.testclient import TestClient

from agent_avenue.engine import Phase, PlayerId, PlayOfferAction, RecruitAction, legal_actions
from agent_avenue.web import create_app
from agent_avenue.web.sessions import BrowserSession, SessionRepository, WebGame


def _csrf(body: str) -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', body)
    assert match
    return match.group(1)


def _new_game(
    client: TestClient,
    seed: str = "17",
    mode: str = "human-human",
    human_seat: str = "player_one",
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
    path = urlparse(response.headers["location"]).path
    game_id = path.split("/")[2]
    return game_id, token


def _stored_game(client: TestClient, app: object, game_id: str) -> WebGame:
    repository: SessionRepository = app.state.session_repository  # type: ignore[attr-defined]
    session_id = client.cookies["agent_avenue_session"]
    session = repository.session(session_id)
    assert isinstance(session, BrowserSession)
    game = repository.game(session, game_id)
    assert game is not None
    return game


def _reveal(client: TestClient, game_id: str, token: str) -> None:
    response = client.post(
        f"/games/{game_id}/pass/reveal",
        data={"csrf_token": token},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == f"/games/{game_id}"


def _action_data(action: PlayOfferAction | RecruitAction, token: str) -> dict[str, str]:
    data = {"csrf_token": token, "revision": str(action.revision)}
    if isinstance(action, PlayOfferAction):
        data.update({"face_up": action.face_up.value, "face_down": action.face_down.value})
    else:
        data["slot"] = action.slot.value
    return data


def test_health_and_seeded_creation_are_reproducible() -> None:
    app = create_app()
    first = TestClient(app)
    second = TestClient(app)
    assert first.get("/healthz").json() == {"status": "ok"}
    first_id, _ = _new_game(first, "42")
    second_id, _ = _new_game(second, "42")
    assert _stored_game(first, app, first_id).state == _stored_game(second, app, second_id).state


def test_generated_seed_is_displayed_and_in_range() -> None:
    app = create_app()
    client = TestClient(app)
    game_id, _ = _new_game(client, "")
    response = client.get(f"/games/{game_id}/pass")
    seed = _stored_game(client, app, game_id).state.seed
    assert 0 <= seed < 2**64
    assert f"<code>{seed}</code>" in response.text


def test_pass_page_and_direct_decision_get_do_not_reveal_private_cards() -> None:
    app = create_app()
    client = TestClient(app)
    game_id, _ = _new_game(client, "31")
    game = _stored_game(client, app, game_id)
    response = client.get(f"/games/{game_id}/pass")
    assert response.headers["cache-control"] == "no-store, private"
    assert "form-action 'self'" in response.headers["content-security-policy"]
    assert "Your hand" not in response.text
    assert "Face up:" not in response.text
    direct = client.get(f"/games/{game_id}", follow_redirects=False)
    assert direct.status_code == 303
    assert direct.headers["location"].endswith("/pass")
    assert all(card.value not in response.text for card in game.state.hands[0])


def test_recruiter_never_receives_hidden_offer_value() -> None:
    app = create_app()
    client = TestClient(app)
    game_id, token = _new_game(client, "31")
    game = _stored_game(client, app, game_id)
    action = next(
        item
        for item in legal_actions(game.state)
        if isinstance(item, PlayOfferAction)
        and item.face_down not in game.state.hands[1]
        and item.face_down is not item.face_up
    )
    _reveal(client, game_id, token)
    played = client.post(
        f"/games/{game_id}/actions",
        data=_action_data(action, token),
        follow_redirects=False,
    )
    assert played.status_code == 303
    pass_page = client.get(played.headers["location"])
    assert action.face_down.value.replace("_", " ").title() not in pass_page.text
    _reveal(client, game_id, token)
    recruiter = client.get(f"/games/{game_id}")
    assert recruiter.status_code == 200
    assert action.face_up.value.replace("_", " ").title() in recruiter.text
    assert action.face_down.value.replace("_", " ").title() not in recruiter.text
    assert "Hidden agent" in recruiter.text
    assert "known_face_down" not in recruiter.text


def test_complete_hot_seat_game_uses_prg_and_public_summary() -> None:
    app = create_app()
    client = TestClient(app)
    game_id, token = _new_game(client, "1")
    game = _stored_game(client, app, game_id)
    for _ in range(200):
        if game.state.phase is Phase.TERMINAL:
            break
        _reveal(client, game_id, token)
        page = client.get(f"/games/{game_id}")
        assert page.status_code == 200
        action = legal_actions(game.state)[0]
        response = client.post(
            f"/games/{game_id}/actions",
            data=_action_data(action, token),
            follow_redirects=False,
        )
        assert response.status_code == 303
        if isinstance(action, PlayOfferAction):
            assert response.headers["location"].endswith("/pass")
        else:
            assert response.headers["location"].endswith("/turn-result")
            summary = client.get(response.headers["location"])
            assert "Recruitment resolved" in summary.text
            response = client.post(
                f"/games/{game_id}/turn-result/continue",
                data={"csrf_token": token},
                follow_redirects=False,
            )
            assert response.status_code == 303
    assert game.state.phase is Phase.TERMINAL
    result = client.get(response.headers["location"])
    assert result.status_code == 200
    assert "Game complete" in result.text
    assert game.state.outcome is not None
    assert game.state.outcome.winner.value.replace("_", " ").title() in result.text
    assert "Your hand" not in result.text
    assert "Replay this seed" in result.text


def test_human_can_play_random_or_heuristic_from_either_seat() -> None:
    for mode, human_seat in (
        ("human-random", "player_one"),
        ("human-heuristic", "player_two"),
    ):
        app = create_app()
        client = TestClient(app)
        game_id, token = _new_game(client, "23", mode, human_seat)
        game = _stored_game(client, app, game_id)
        page = client.get(f"/games/{game_id}")
        assert page.status_code == 200
        assert game.revealed_actor is not None
        assert game.revealed_actor.value == human_seat
        assert "decision</h1>" in page.text
        assert f"/games/{game_id}/actions" in page.text
        action = legal_actions(game.state)[0]
        response = client.post(
            f"/games/{game_id}/actions",
            data=_action_data(action, token),
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"] in {
            f"/games/{game_id}",
            f"/games/{game_id}/result",
        }
        assert game.state.revision >= action.revision + 1
        if game.state.phase is not Phase.TERMINAL:
            assert game.revealed_actor is not None
            assert game.revealed_actor.value == human_seat
        assert all(controller.kind in {"human", "agent"} for controller in game.controllers)


def test_agent_web_flow_never_renders_hidden_face_down_or_diagnostics() -> None:
    app = create_app()
    client = TestClient(app)
    game_id, token = _new_game(client, "31", "human-random", "player_one")
    game = _stored_game(client, app, game_id)
    play = legal_actions(game.state)[0]
    assert isinstance(play, PlayOfferAction)
    response = client.post(
        f"/games/{game_id}/actions",
        data=_action_data(play, token),
        follow_redirects=False,
    )
    assert response.headers["location"] == f"/games/{game_id}"
    page = client.get(response.headers["location"])
    assert page.status_code == 200
    assert "Hidden agent" in page.text
    assert "heuristic" not in page.text.lower()
    assert "agent_seed" not in page.text
    assert "known_face_down" not in page.text


def test_ai_result_replay_preserves_mode_and_human_seat() -> None:
    app = create_app()
    client = TestClient(app)
    game_id, token = _new_game(client, "4", "human-random", "player_two")
    game = _stored_game(client, app, game_id)
    for _ in range(100):
        if game.state.phase is Phase.TERMINAL:
            break
        action = legal_actions(game.state)[0]
        response = client.post(
            f"/games/{game_id}/actions",
            data=_action_data(action, token),
            follow_redirects=False,
        )
        assert response.status_code == 303
    assert game.state.phase is Phase.TERMINAL
    result = client.get(f"/games/{game_id}/result")
    assert 'name="mode" value="human-random"' in result.text
    assert 'name="human_seat" value="player_two"' in result.text
    assert "Human (Player Two) versus Random" in result.text


def test_ai_action_can_end_web_game() -> None:
    app = create_app()
    client = TestClient(app)
    game_id, token = _new_game(client, "0", "human-random", "player_one")
    game = _stored_game(client, app, game_id)
    last_response = None
    for _ in range(100):
        if game.state.phase is Phase.TERMINAL:
            break
        action = legal_actions(game.state)[0]
        last_response = client.post(
            f"/games/{game_id}/actions",
            data=_action_data(action, token),
            follow_redirects=False,
        )
    assert game.state.phase is Phase.TERMINAL
    assert game.state.actions[-1].actor is PlayerId.PLAYER_TWO
    assert last_response is not None
    assert last_response.headers["location"] == f"/games/{game_id}/result"
    assert client.get(last_response.headers["location"]).status_code == 200


def test_repeated_stale_illegal_and_malformed_actions_are_safe() -> None:
    app = create_app()
    client = TestClient(app)
    game_id, token = _new_game(client, "9")
    game = _stored_game(client, app, game_id)
    _reveal(client, game_id, token)
    action = legal_actions(game.state)[0]
    assert isinstance(action, PlayOfferAction)
    data = _action_data(action, token)
    first = client.post(f"/games/{game_id}/actions", data=data, follow_redirects=False)
    assert first.status_code == 303
    revision = game.state.revision
    repeated = client.post(f"/games/{game_id}/actions", data=data)
    assert repeated.status_code == 409
    assert game.state.revision == revision
    assert action.face_down.value not in repeated.text

    _reveal(client, game_id, token)
    illegal = client.post(
        f"/games/{game_id}/actions",
        data={"csrf_token": token, "revision": str(revision), "slot": "not-a-slot"},
    )
    assert illegal.status_code == 409
    assert "not-a-slot" not in illegal.text
    assert game.state.revision == revision


def test_concurrent_duplicate_submission_advances_once() -> None:
    app = create_app()
    client = TestClient(app)
    game_id, token = _new_game(client, "12")
    game = _stored_game(client, app, game_id)
    _reveal(client, game_id, token)
    action = legal_actions(game.state)[0]
    assert isinstance(action, PlayOfferAction)
    data = _action_data(action, token)

    def submit() -> int:
        return client.post(
            f"/games/{game_id}/actions", data=data, follow_redirects=False
        ).status_code

    with ThreadPoolExecutor(max_workers=2) as executor:
        statuses = sorted(executor.map(lambda _index: submit(), range(2)))
    assert statuses == [303, 409]
    assert game.state.revision == 1


def test_unknown_games_and_sessions_are_isolated() -> None:
    app = create_app()
    owner = TestClient(app)
    stranger = TestClient(app)
    game_id, _ = _new_game(owner, "5")
    stranger.get("/")
    response = stranger.get(f"/games/{game_id}/pass")
    missing = stranger.get("/games/not-a-real-game/pass")
    assert response.status_code == missing.status_code == 404
    assert response.text == missing.text


def test_invalid_seed_and_csrf_fail_without_echoing_input() -> None:
    app = create_app()
    client = TestClient(app)
    landing = client.get("/")
    token = _csrf(landing.text)
    bad_seed = "18446744073709551616"
    response = client.post("/games", data={"csrf_token": token, "seed": bad_seed})
    assert response.status_code == 400
    assert bad_seed not in response.text
    csrf = client.post("/games", data={"csrf_token": "wrong", "seed": "1"})
    assert csrf.status_code == 400
    assert "wrong" not in csrf.text


def test_core_engine_import_does_not_require_web_dependencies() -> None:
    script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'fastapi', 'jinja2', 'starlette', 'uvicorn'}:
            raise ImportError(fullname)
        return None
sys.meta_path.insert(0, Block())
import agent_avenue.engine
print('ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"

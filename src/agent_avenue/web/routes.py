"""HTTP translation for the server-rendered hot-seat workflow."""

import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from agent_avenue.engine import (
    Action,
    CardName,
    EngineError,
    GameConfig,
    OfferSlot,
    Phase,
    PlayerId,
    PlayOfferAction,
    RecruitAction,
    apply_action,
    new_game,
)
from agent_avenue.engine.setup import RULES_VERSION, SHUFFLE_VERSION
from agent_avenue.observation import observe
from agent_avenue.observation.model import PlayContext, RecruitContext

from .presenters import (
    PLAYER_LABELS,
    completed_turn_view,
    observation_view,
    outcome_view,
    public_board_view,
    public_fingerprint,
)
from .sessions import BrowserSession, SessionRepository, WebGame, WebStage

TEMPLATES = Jinja2Templates(directory=Path(__file__).with_name("templates"))


def _session(request: Request) -> BrowserSession:
    return request.state.browser_session  # type: ignore[no-any-return]


def _game(request: Request, repository: SessionRepository, game_id: str) -> WebGame:
    game = repository.game(_session(request), game_id)
    if game is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Game not found")
    return game


def _csrf(request: Request, submitted: str) -> None:
    if not secrets.compare_digest(_session(request).csrf_token, submitted):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid form token")


def _metadata(game: WebGame) -> dict[str, object]:
    return {
        "seed": game.state.seed,
        "starting_player": PLAYER_LABELS[game.state.config.starting_player],
        "rules_version": RULES_VERSION,
        "shuffle_version": SHUFFLE_VERSION,
        "replay_id": game.replay_id,
        "public_fingerprint": public_fingerprint(game.state.seed, game.state.history),
    }


def _redirect(path: str) -> RedirectResponse:
    return RedirectResponse(path, status_code=status.HTTP_303_SEE_OTHER)


def build_router(repository: SessionRepository) -> APIRouter:
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse)
    def landing(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(
            request=request,
            name="landing.html",
            context={"csrf_token": _session(request).csrf_token},
        )

    @router.post("/games")
    def create_game_route(
        request: Request,
        csrf_token: Annotated[str, Form()],
        seed: Annotated[str, Form()] = "",
    ) -> RedirectResponse:
        _csrf(request, csrf_token)
        try:
            actual_seed = secrets.randbits(64) if not seed.strip() else int(seed, 10)
            state = new_game(GameConfig(), actual_seed)
        except (ValueError, EngineError) as exc:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "Seed must be an unsigned 64-bit integer",
            ) from exc
        game = repository.create_game(_session(request), state)
        return _redirect(f"/games/{game.game_id}/pass")

    @router.get("/games/{game_id}/pass", response_class=HTMLResponse)
    def pass_device(request: Request, game_id: str) -> Response:
        game = _game(request, repository, game_id)
        with game.lock:
            if game.stage is WebStage.DECISION:
                return _redirect(f"/games/{game_id}")
            if game.stage is WebStage.SUMMARY:
                return _redirect(f"/games/{game_id}/turn-result")
            if game.stage is WebStage.TERMINAL:
                return _redirect(f"/games/{game_id}/result")
            decision = game.state.active_player
            if game.state.phase is Phase.RECRUIT:
                decision = game.state.active_player.other()
            return TEMPLATES.TemplateResponse(
                request=request,
                name="pass.html",
                context={
                    "game_id": game_id,
                    "player": PLAYER_LABELS[decision],
                    "csrf_token": _session(request).csrf_token,
                    "metadata": _metadata(game),
                },
            )

    @router.post("/games/{game_id}/pass/reveal")
    def reveal(
        request: Request,
        game_id: str,
        csrf_token: Annotated[str, Form()],
    ) -> RedirectResponse:
        _csrf(request, csrf_token)
        game = _game(request, repository, game_id)
        with game.lock:
            if game.stage is not WebStage.PASS:
                raise HTTPException(status.HTTP_409_CONFLICT, "This handoff is no longer current")
            actor = (
                game.state.active_player
                if game.state.phase is Phase.PLAY
                else game.state.active_player.other()
            )
            game.revealed_actor = actor
            game.stage = WebStage.DECISION
        return _redirect(f"/games/{game_id}")

    @router.get("/games/{game_id}", response_class=HTMLResponse)
    def decision_page(request: Request, game_id: str) -> Response:
        game = _game(request, repository, game_id)
        with game.lock:
            if game.stage is WebStage.PASS:
                return _redirect(f"/games/{game_id}/pass")
            if game.stage is WebStage.SUMMARY:
                return _redirect(f"/games/{game_id}/turn-result")
            if game.stage is WebStage.TERMINAL:
                return _redirect(f"/games/{game_id}/result")
            if game.revealed_actor is None:
                raise HTTPException(status.HTTP_409_CONFLICT, "No player has taken control")
            observation = observe(game.state, game.revealed_actor)
            return TEMPLATES.TemplateResponse(
                request=request,
                name="game.html",
                context={
                    "game_id": game_id,
                    "view": observation_view(observation),
                    "actions": observation.legal_actions,
                    "csrf_token": _session(request).csrf_token,
                    "metadata": _metadata(game),
                },
            )

    @router.post("/games/{game_id}/actions")
    async def submit_action(request: Request, game_id: str) -> RedirectResponse:
        form = await request.form()
        values: Mapping[str, Any] = form
        submitted_csrf = values.get("csrf_token")
        if not isinstance(submitted_csrf, str):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Malformed action")
        _csrf(request, submitted_csrf)
        game = _game(request, repository, game_id)
        with game.lock:
            if game.stage is not WebStage.DECISION or game.revealed_actor is None:
                raise HTTPException(status.HTTP_409_CONFLICT, "This decision is no longer current")
            try:
                revision_value = values.get("revision")
                if not isinstance(revision_value, str):
                    raise ValueError
                revision = int(revision_value, 10)
                decision = observe(game.state, game.revealed_actor).decision
                action: Action
                if isinstance(decision, PlayContext):
                    face_up = values.get("face_up")
                    face_down = values.get("face_down")
                    if not isinstance(face_up, str) or not isinstance(face_down, str):
                        raise ValueError
                    action = PlayOfferAction(
                        revision, game.revealed_actor, CardName(face_up), CardName(face_down)
                    )
                elif isinstance(decision, RecruitContext):
                    slot = values.get("slot")
                    if not isinstance(slot, str):
                        raise ValueError
                    action = RecruitAction(revision, game.revealed_actor, OfferSlot(slot))
                else:
                    raise ValueError
                next_state = apply_action(game.state, action)
            except (ValueError, EngineError) as exc:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "The action is malformed, illegal, or stale",
                ) from exc
            was_recruit = game.state.phase is Phase.RECRUIT
            game.state = next_state
            game.revealed_actor = None
            if was_recruit:
                game.stage = WebStage.SUMMARY
                return _redirect(f"/games/{game_id}/turn-result")
            game.stage = WebStage.PASS
            return _redirect(f"/games/{game_id}/pass")

    @router.get("/games/{game_id}/turn-result", response_class=HTMLResponse)
    def turn_result(request: Request, game_id: str) -> Response:
        game = _game(request, repository, game_id)
        with game.lock:
            if game.stage is WebStage.TERMINAL:
                return _redirect(f"/games/{game_id}/result")
            if game.stage is not WebStage.SUMMARY or not game.state.history:
                return _redirect(f"/games/{game_id}/pass")
            observation = observe(game.state, game.state.active_player)
            return TEMPLATES.TemplateResponse(
                request=request,
                name="summary.html",
                context={
                    "game_id": game_id,
                    "summary": completed_turn_view(game.state.history[-1]),
                    "board": public_board_view(observation),
                    "terminal": game.state.phase is Phase.TERMINAL,
                    "csrf_token": _session(request).csrf_token,
                    "metadata": _metadata(game),
                },
            )

    @router.post("/games/{game_id}/turn-result/continue")
    def continue_after_summary(
        request: Request,
        game_id: str,
        csrf_token: Annotated[str, Form()],
    ) -> RedirectResponse:
        _csrf(request, csrf_token)
        game = _game(request, repository, game_id)
        with game.lock:
            if game.stage is not WebStage.SUMMARY:
                raise HTTPException(status.HTTP_409_CONFLICT, "This summary is no longer current")
            if game.state.phase is Phase.TERMINAL:
                game.stage = WebStage.TERMINAL
                return _redirect(f"/games/{game_id}/result")
            game.stage = WebStage.PASS
            return _redirect(f"/games/{game_id}/pass")

    @router.get("/games/{game_id}/result", response_class=HTMLResponse)
    def result(request: Request, game_id: str) -> Response:
        game = _game(request, repository, game_id)
        with game.lock:
            if game.stage is not WebStage.TERMINAL or game.state.outcome is None:
                if game.stage is WebStage.SUMMARY:
                    return _redirect(f"/games/{game_id}/turn-result")
                return _redirect(f"/games/{game_id}/pass")
            observation = observe(game.state, PlayerId.PLAYER_ONE)
            return TEMPLATES.TemplateResponse(
                request=request,
                name="result.html",
                context={
                    "outcome": outcome_view(game.state.outcome),
                    "board": public_board_view(observation),
                    "metadata": _metadata(game),
                    "csrf_token": _session(request).csrf_token,
                },
            )

    return router

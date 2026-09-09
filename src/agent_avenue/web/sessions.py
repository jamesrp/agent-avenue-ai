"""In-memory, session-owned game storage for the QA application."""

import secrets
import threading
from dataclasses import dataclass, field
from enum import StrEnum

from agent_avenue.engine import GameState, PlayerId

from .controllers import HumanController, SeatController


def _human_controllers() -> tuple[SeatController, SeatController]:
    return (
        HumanController(PlayerId.PLAYER_ONE),
        HumanController(PlayerId.PLAYER_TWO),
    )


class WebStage(StrEnum):
    PASS = "pass"
    DECISION = "decision"
    SUMMARY = "summary"
    TERMINAL = "terminal"


@dataclass(slots=True)
class WebGame:
    game_id: str
    replay_id: str
    state: GameState
    stage: WebStage = WebStage.PASS
    revealed_actor: PlayerId | None = None
    skip_take_control: bool = False
    controllers: tuple[SeatController, SeatController] = field(default_factory=_human_controllers)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self) -> None:
        if tuple(controller.player for controller in self.controllers) != tuple(PlayerId):
            raise ValueError("controllers must be ordered player one, player two")


@dataclass(slots=True)
class BrowserSession:
    session_id: str
    csrf_token: str
    games: dict[str, WebGame] = field(default_factory=dict)


class SessionRepository:
    """Thread-safe process-local repository with ownership checks."""

    def __init__(self) -> None:
        self._sessions: dict[str, BrowserSession] = {}
        self._lock = threading.RLock()

    def create_session(self) -> BrowserSession:
        with self._lock:
            session = BrowserSession(secrets.token_urlsafe(32), secrets.token_urlsafe(32))
            self._sessions[session.session_id] = session
            return session

    def session(self, session_id: str | None) -> BrowserSession | None:
        if session_id is None:
            return None
        with self._lock:
            return self._sessions.get(session_id)

    def create_game(
        self,
        session: BrowserSession,
        state: GameState,
        controllers: tuple[SeatController, SeatController] | None = None,
        *,
        skip_take_control: bool = False,
    ) -> WebGame:
        with self._lock:
            game = WebGame(
                secrets.token_urlsafe(18),
                secrets.token_urlsafe(18),
                state,
                skip_take_control=skip_take_control,
                controllers=controllers or _human_controllers(),
            )
            session.games[game.game_id] = game
            return game

    def game(self, session: BrowserSession, game_id: str) -> WebGame | None:
        with self._lock:
            return session.games.get(game_id)

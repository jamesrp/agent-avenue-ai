"""FastAPI application factory for the local QA interface."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .routes import TEMPLATES, build_router
from .sessions import BrowserSession, SessionRepository


@dataclass(frozen=True, slots=True)
class WebConfig:
    """Deployment settings; storage remains deliberately process-local."""

    cookie_name: str = "agent_avenue_session"
    cookie_secure: bool = False


def create_app(
    config: WebConfig | None = None,
    repository: SessionRepository | None = None,
) -> FastAPI:
    """Create an isolated QA application suitable for tests or one-worker Uvicorn."""
    actual_config = config or WebConfig()
    sessions = repository or SessionRepository()
    app = FastAPI(title="Agent Avenue QA", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.session_repository = sessions
    app.mount(
        "/static",
        StaticFiles(directory=Path(__file__).with_name("static")),
        name="static",
    )

    @app.middleware("http")
    async def browser_session(request: Request, call_next: object) -> Response:
        if request.url.path == "/healthz" or request.url.path.startswith("/static/"):
            response = await call_next(request)  # type: ignore[operator]
            return cast(Response, response)
        session_id = request.cookies.get(actual_config.cookie_name)
        session = sessions.session(session_id)
        created = session is None
        if session is None:
            session = sessions.create_session()
        request.state.browser_session = session
        response = cast(Response, await call_next(request))  # type: ignore[operator]
        response.headers["Cache-Control"] = "no-store, private"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self'; form-action 'self'; "
            "base-uri 'none'; frame-ancestors 'none'"
        )
        if created:
            response.set_cookie(
                actual_config.cookie_name,
                session.session_id,
                httponly=True,
                secure=actual_config.cookie_secure,
                samesite="strict",
                path="/",
            )
        return response

    @app.exception_handler(HTTPException)
    async def safe_http_error(request: Request, exc: HTTPException) -> Response:
        if request.url.path == "/healthz":
            return JSONResponse({"detail": "unavailable"}, status_code=exc.status_code)
        session = cast(BrowserSession | None, getattr(request.state, "browser_session", None))
        response = TEMPLATES.TemplateResponse(
            request=request,
            name="error.html",
            context={
                "status_code": exc.status_code,
                "message": str(exc.detail),
                "csrf_token": session.csrf_token if session else "",
            },
            status_code=exc.status_code,
        )
        response.headers["Cache-Control"] = "no-store, private"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self'; form-action 'self'; "
            "base-uri 'none'; frame-ancestors 'none'"
        )
        return response

    @app.get("/healthz", include_in_schema=False)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(build_router(sessions))
    return app


app = create_app()

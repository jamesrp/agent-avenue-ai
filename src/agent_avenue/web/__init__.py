"""Optional server-rendered QA web application."""

from .app import WebConfig, create_app
from .opponents import LearnedOpponentConfig

__all__ = ["LearnedOpponentConfig", "WebConfig", "create_app"]

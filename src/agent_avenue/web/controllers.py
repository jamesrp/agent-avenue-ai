"""Web-facing re-exports of the shared controller seam."""

from agent_avenue.runners import AgentController, HumanController
from agent_avenue.runners.game import Controller as SeatController

__all__ = ["AgentController", "HumanController", "SeatController"]

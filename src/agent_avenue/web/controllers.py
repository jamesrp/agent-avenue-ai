"""Seat-controller seam for human and future automated players."""

from dataclasses import dataclass
from typing import Protocol

from agent_avenue.engine import PlayerId


class SeatController(Protocol):
    """Identifies who owns a seat without coupling controllers to game rules."""

    @property
    def kind(self) -> str: ...


@dataclass(frozen=True, slots=True)
class HumanController:
    """A local human using the pass-device flow."""

    player: PlayerId

    @property
    def kind(self) -> str:
        return "human"

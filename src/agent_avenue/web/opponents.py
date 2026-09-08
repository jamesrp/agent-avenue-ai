"""Server-side allowlist for checkpoint-backed web QA opponents.

The module stays free of optional RL imports until an allowlisted opponent is selected.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_avenue.agents import Agent


@dataclass(frozen=True, slots=True)
class LearnedOpponentConfig:
    """One trusted server configuration; keys are safe browser-facing identifiers."""

    key: str
    label: str
    checkpoint: Path
    terminal_safety: bool = False

    def __post_init__(self) -> None:
        if not self.key or not self.label:
            raise ValueError("learned opponent key and label must be non-empty")
        if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in self.key):
            raise ValueError("learned opponent key must use lowercase letters, digits, and hyphens")


@dataclass(frozen=True, slots=True)
class LoadedLearnedOpponent:
    key: str
    label: str
    checkpoint_fingerprint: str
    agent_id: str
    agent_config: dict[str, object]
    agent: Agent


class LearnedOpponentRegistry:
    """Validate each allowlisted checkpoint once and reuse its immutable inference agent."""

    def __init__(self, configured: tuple[LearnedOpponentConfig, ...]) -> None:
        by_key = {item.key: item for item in configured}
        if len(by_key) != len(configured):
            raise ValueError("learned opponent keys must be unique")
        self._configured = by_key
        self._loaded: dict[str, LoadedLearnedOpponent] = {}
        self._lock = threading.RLock()

    def configured(self) -> tuple[LearnedOpponentConfig, ...]:
        return tuple(self._configured[key] for key in sorted(self._configured))

    def resolve(self, key: str) -> LoadedLearnedOpponent:
        """Resolve only an exact configured key; browser values are never treated as paths."""
        with self._lock:
            existing = self._loaded.get(key)
            if existing is not None:
                return existing
            configured = self._configured.get(key)
            if configured is None:
                raise ValueError("learned opponent is not allowlisted")

            from agent_avenue.agents.learned import LearnedValueAgent
            from agent_avenue.agents.terminal_safety import TerminalSafetyAgent
            from agent_avenue.learning import load_checkpoint

            checkpoint = load_checkpoint(configured.checkpoint)
            base = LearnedValueAgent.from_checkpoint(checkpoint)
            agent: Agent = TerminalSafetyAgent(base) if configured.terminal_safety else base
            config_to_data = getattr(agent, "config_to_data", None)
            agent_config = config_to_data() if callable(config_to_data) else base.config.to_data()
            loaded = LoadedLearnedOpponent(
                key=configured.key,
                label=configured.label,
                checkpoint_fingerprint=checkpoint.checkpoint_fingerprint,
                agent_id=f"learned-{configured.key}",
                agent_config=agent_config,
                agent=agent,
            )
            self._loaded[key] = loaded
            return loaded

    def public_label(self, agent_id: str) -> str | None:
        prefix = "learned-"
        if not agent_id.startswith(prefix):
            return None
        configured = self._configured.get(agent_id.removeprefix(prefix))
        return configured.label if configured else None

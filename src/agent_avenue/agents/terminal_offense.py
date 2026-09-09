"""Information-safe current-turn forced-win policy wrapper."""

from dataclasses import dataclass, replace
from typing import Final

from agent_avenue.engine.model import Action
from agent_avenue.observation.model import PlayerObservation

from .base import Agent, PublicDecision
from .random_source import RandomSource
from .terminal_safety import terminal_outcomes_for_action

TERMINAL_OFFENSE_VERSION: Final = "terminal-offense-v1"
WIN_DEFINITION_VERSION: Final = "public-guaranteed-current-turn-win-v1"
SELECTION_VERSION: Final = "base-policy-on-forced-win-set-v1"


@dataclass(frozen=True, slots=True)
class TerminalOffenseFilter:
    """One immediate-win restriction decision."""

    allowed_actions: tuple[Action, ...]
    forced_win_actions: tuple[Action, ...]


def filter_immediate_win_actions(
    observation: PlayerObservation, legal_actions: tuple[Action, ...]
) -> TerminalOffenseFilter:
    """Restrict to actions that guarantee a current-turn win over the public information set."""
    if not legal_actions:
        raise ValueError("terminal offense requires at least one legal action")
    forced_wins = tuple(
        action
        for action in legal_actions
        if all(
            outcome is not None and outcome.winner is observation.viewer
            for outcome in terminal_outcomes_for_action(observation, action)
        )
    )
    return TerminalOffenseFilter(forced_wins or legal_actions, forced_wins)


def _base_config(agent: Agent) -> dict[str, object]:
    config = getattr(agent, "config", None)
    to_data = getattr(config, "to_data", None)
    if callable(to_data):
        data = to_data()
    else:
        config_to_data = getattr(agent, "config_to_data", None)
        if not callable(config_to_data):
            raise ValueError("wrapped agent must expose normalized configuration")
        data = config_to_data()
    if not isinstance(data, dict):
        raise ValueError("wrapped agent configuration must be a JSON object")
    return data


@dataclass(frozen=True, slots=True)
class TerminalOffenseAgent:
    """Let the base policy choose only among publicly guaranteed current-turn wins when present."""

    base: Agent

    def config_to_data(self) -> dict[str, object]:
        return {
            "type": "terminal_offense",
            "version": TERMINAL_OFFENSE_VERSION,
            "base": _base_config(self.base),
            "win_definition": WIN_DEFINITION_VERSION,
            "selection": SELECTION_VERSION,
        }

    def choose_action(
        self,
        observation: PlayerObservation,
        decision: PublicDecision,
        legal_actions: tuple[Action, ...],
        rng: RandomSource,
    ) -> Action:
        if decision != observation.decision or legal_actions != observation.legal_actions:
            raise ValueError("agent inputs must describe one consistent public turn")
        result = filter_immediate_win_actions(observation, legal_actions)
        filtered_observation = replace(observation, legal_actions=result.allowed_actions)
        action = self.base.choose_action(
            filtered_observation,
            filtered_observation.decision,
            result.allowed_actions,
            rng,
        )
        if action not in result.allowed_actions:
            raise ValueError("wrapped agent returned an action outside the forced-win action set")
        return action

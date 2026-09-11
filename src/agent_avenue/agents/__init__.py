"""Information-safe automated action choosers."""

from .base import Agent, AgentTurn, PublicDecision, choose_agent_action
from .exploration import EpsilonConfig, EpsilonGreedyAgent
from .heuristic import (
    HEURISTIC_VERSION,
    GreedyHeuristicAgent,
    GreedyHeuristicConfig,
    score_actions,
    visible_unknown_multiset,
)
from .random import RandomAgent, RandomAgentConfig
from .random_source import (
    RNG_ALGORITHM,
    SEED_DERIVATION,
    DeterministicRandom,
    RandomSource,
    derive_seed,
)
from .scoring import LearnedCandidateScores
from .scripted import ScriptCallback, ScriptedAgent
from .terminal_offense import (
    SELECTION_VERSION,
    TERMINAL_OFFENSE_VERSION,
    WIN_DEFINITION_VERSION,
    TerminalOffenseAgent,
    TerminalOffenseFilter,
    filter_immediate_win_actions,
)
from .terminal_safety import (
    FALLBACK_VERSION,
    RESOLUTION_SCOPE,
    TERMINAL_SAFETY_VERSION,
    UNCERTAINTY_VERSION,
    TerminalSafetyAgent,
    TerminalSafetyFilter,
    filter_terminal_actions,
    information_consistent_face_down_cards,
    terminal_outcomes_for_action,
)

__all__ = [
    "FALLBACK_VERSION",
    "HEURISTIC_VERSION",
    "RESOLUTION_SCOPE",
    "RNG_ALGORITHM",
    "SEED_DERIVATION",
    "SELECTION_VERSION",
    "TERMINAL_OFFENSE_VERSION",
    "TERMINAL_SAFETY_VERSION",
    "UNCERTAINTY_VERSION",
    "WIN_DEFINITION_VERSION",
    "Agent",
    "AgentTurn",
    "DeterministicRandom",
    "EpsilonConfig",
    "EpsilonGreedyAgent",
    "GreedyHeuristicAgent",
    "GreedyHeuristicConfig",
    "LearnedCandidateScores",
    "PublicDecision",
    "RandomAgent",
    "RandomAgentConfig",
    "RandomSource",
    "ScriptCallback",
    "ScriptedAgent",
    "TerminalOffenseAgent",
    "TerminalOffenseFilter",
    "TerminalSafetyAgent",
    "TerminalSafetyFilter",
    "choose_agent_action",
    "derive_seed",
    "filter_immediate_win_actions",
    "filter_terminal_actions",
    "information_consistent_face_down_cards",
    "score_actions",
    "terminal_outcomes_for_action",
    "visible_unknown_multiset",
]

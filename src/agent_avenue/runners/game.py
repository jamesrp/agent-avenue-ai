"""Trusted orchestration for single and resumable games."""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from agent_avenue.agents import (
    RNG_ALGORITHM,
    Agent,
    AgentTurn,
    DeterministicRandom,
    choose_agent_action,
)
from agent_avenue.engine import (
    Action,
    GameConfig,
    GameState,
    Phase,
    PlayerId,
    apply_action,
    new_game,
)
from agent_avenue.observation import observe
from agent_avenue.storage import AgentSeatRecord, GameRecord, create_game_record


class InvalidAgentActionError(ValueError):
    """An automated controller returned an illegal or malformed action."""


@dataclass(frozen=True, slots=True)
class HumanController:
    player: PlayerId

    @property
    def kind(self) -> str:
        return "human"


@dataclass(slots=True)
class AgentController:
    player: PlayerId
    agent_id: str
    config: dict[str, object]
    seed: int
    agent: Agent
    rng: DeterministicRandom

    @property
    def kind(self) -> str:
        return "agent"


type Controller = HumanController | AgentController


@dataclass(slots=True)
class GameSession:
    state: GameState
    controllers: tuple[Controller, Controller]

    def __post_init__(self) -> None:
        if tuple(controller.player for controller in self.controllers) != tuple(PlayerId):
            raise ValueError("controllers must be ordered player one, player two")


@dataclass(frozen=True, slots=True)
class AdvanceResult:
    actions: tuple[Action, ...]
    completed_turns: int
    terminal: bool


AgentFactory = Callable[[], Agent]


def _copy_config(config: Mapping[str, object]) -> dict[str, object]:
    """Detach a normalized JSON configuration from caller-owned mutable objects."""
    value = json.loads(json.dumps(dict(config), sort_keys=True))
    if not isinstance(value, dict):  # pragma: no cover - guarded by the annotation
        raise TypeError("agent configuration must be a JSON object")
    return value


def _agent_config(agent: Agent) -> dict[str, object]:
    config = getattr(agent, "config", None)
    to_data = getattr(config, "to_data", None)
    if callable(to_data):
        return _copy_config(to_data())
    config_to_data = getattr(agent, "config_to_data", None)
    if callable(config_to_data):
        return _copy_config(config_to_data())
    raise ValueError("runnable agents must expose normalized configuration metadata")


@dataclass(frozen=True, slots=True)
class AgentSpec:
    agent_id: str
    config: Mapping[str, object]
    factory: AgentFactory

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise ValueError("agent_id must be non-empty")
        object.__setattr__(self, "config", MappingProxyType(_copy_config(self.config)))


@dataclass(frozen=True, slots=True)
class GameSpec:
    run_id: str
    game_id: str
    pair_id: str | None
    config: GameConfig
    setup_seed: int
    seats: tuple[AgentSpec, AgentSpec]
    agent_seeds: tuple[int, int]
    agent_seed_derivations: tuple[str, str] = ("supplied", "supplied")


def decision_actor(state: GameState) -> PlayerId | None:
    if state.phase is Phase.TERMINAL:
        return None
    return state.active_player if state.phase is Phase.PLAY else state.active_player.other()


def controller_for(session: GameSession, player: PlayerId) -> Controller:
    return session.controllers[0 if player is PlayerId.PLAYER_ONE else 1]


def create_agent_session(spec: GameSpec) -> GameSession:
    controllers: list[AgentController] = []
    for index, player in enumerate(PlayerId):
        agent = spec.seats[index].factory()
        actual_config = _agent_config(agent)
        if actual_config != spec.seats[index].config:
            raise ValueError("agent factory configuration does not match recorded configuration")
        controllers.append(
            AgentController(
                player=player,
                agent_id=spec.seats[index].agent_id,
                config=_copy_config(spec.seats[index].config),
                seed=spec.agent_seeds[index],
                agent=agent,
                rng=DeterministicRandom(
                    spec.agent_seeds[index], f"agent:{spec.seats[index].agent_id}"
                ),
            )
        )
    return GameSession(new_game(spec.config, spec.setup_seed), (controllers[0], controllers[1]))


def step_agent(session: GameSession) -> Action:
    """Request and apply exactly one safe automated action."""
    actor = decision_actor(session.state)
    if actor is None:
        raise InvalidAgentActionError("cannot step a terminal game")
    controller = controller_for(session, actor)
    if not isinstance(controller, AgentController):
        raise InvalidAgentActionError("current decision belongs to a human controller")
    observation = observe(session.state, actor)
    turn = AgentTurn.from_observation(observation)
    before = session.state
    try:
        action = choose_agent_action(controller.agent, turn, controller.rng)
    except (TypeError, ValueError) as exc:
        if session.state is not before:
            raise RuntimeError("agent mutated runner state") from exc
        raise InvalidAgentActionError("agent returned an invalid action") from exc
    if (
        type(action) not in {type(item) for item in turn.legal_actions}
        or action not in turn.legal_actions
    ):
        raise InvalidAgentActionError("agent returned an invalid action")
    next_state = apply_action(before, action)
    session.state = next_state
    return action


def submit_human_action(session: GameSession, action: Action) -> None:
    actor = decision_actor(session.state)
    if actor is None or not isinstance(controller_for(session, actor), HumanController):
        raise ValueError("current decision does not belong to a human")
    session.state = apply_action(session.state, action)


def advance_until_human_or_terminal(session: GameSession) -> AdvanceResult:
    """Advance consecutive automated decisions without retaining hidden diagnostics."""
    actions: list[Action] = []
    starting_turns = len(session.state.history)
    while (actor := decision_actor(session.state)) is not None:
        if isinstance(controller_for(session, actor), HumanController):
            break
        actions.append(step_agent(session))
    return AdvanceResult(
        tuple(actions),
        len(session.state.history) - starting_turns,
        session.state.phase is Phase.TERMINAL,
    )


def run_game(spec: GameSpec) -> GameRecord:
    """Run one independent agent-versus-agent game to completion."""
    session = create_agent_session(spec)
    advance_until_human_or_terminal(session)
    if session.state.phase is not Phase.TERMINAL:
        raise RuntimeError("agent-only game stopped before terminal")
    for controller in session.controllers:
        if not isinstance(controller, AgentController):
            raise RuntimeError("agent-only game contains a human controller")
        assert_exhausted = getattr(controller.agent, "assert_exhausted", None)
        if callable(assert_exhausted):
            assert_exhausted()
    seat_records = tuple(
        AgentSeatRecord(
            player,
            spec.seats[index].agent_id,
            _copy_config(spec.seats[index].config),
            spec.agent_seeds[index],
            RNG_ALGORITHM,
            spec.agent_seed_derivations[index],
            f"agent:{spec.seats[index].agent_id}",
        )
        for index, player in enumerate(PlayerId)
    )
    return create_game_record(
        run_id=spec.run_id,
        game_id=spec.game_id,
        pair_id=spec.pair_id,
        seats=(seat_records[0], seat_records[1]),
        final_state=session.state,
    )

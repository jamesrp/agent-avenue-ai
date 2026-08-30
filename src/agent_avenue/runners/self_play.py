"""Frozen learned-checkpoint self-play generation scheduling."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from agent_avenue.agents import SEED_DERIVATION, EpsilonConfig, EpsilonGreedyAgent, derive_seed
from agent_avenue.engine import GameConfig

from .game import AgentSpec, GameSpec

GENERATION_CONFIG_VERSION = "frozen-self-play-generation-v1"
GENERATION_EPSILONS: tuple[EpsilonConfig, ...] = (
    EpsilonConfig(1, 10),
    EpsilonConfig(3, 40),
    EpsilonConfig(1, 20),
    EpsilonConfig(1, 40),
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Identity and deterministic schedule for one immutable behavior-policy generation."""

    generation: int
    run_id: str
    game_count: int
    root_seed: int
    parent_checkpoint_fingerprint: str
    parent_tensor_digest: str
    epsilon: EpsilonConfig
    game_config: GameConfig = field(default_factory=GameConfig)

    def __post_init__(self) -> None:
        if self.generation < 1:
            raise ValueError("learned self-play generation must be at least one")
        if not self.run_id or self.game_count < 2:
            raise ValueError("generation requires a run id and at least two games")
        for label, digest in (
            ("parent checkpoint", self.parent_checkpoint_fingerprint),
            ("parent tensor", self.parent_tensor_digest),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{label} identity must be a lowercase SHA-256 digest")

    def normalized(self) -> dict[str, object]:
        from agent_avenue.engine.setup import normalize_config

        return {
            "version": GENERATION_CONFIG_VERSION,
            "generation": self.generation,
            "run_id": self.run_id,
            "game_count": self.game_count,
            "root_seed": self.root_seed,
            "parent_checkpoint_fingerprint": self.parent_checkpoint_fingerprint,
            "parent_tensor_digest": self.parent_tensor_digest,
            "epsilon": self.epsilon.to_data(),
            "game_config": normalize_config(self.game_config),
        }

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical_json(self.normalized())).hexdigest()


def planned_epsilon(generation: int) -> EpsilonConfig:
    """Return the predeclared baseline epsilon for generations one through four."""
    if not 1 <= generation <= len(GENERATION_EPSILONS):
        raise ValueError("baseline epsilon is declared only for generations 1 through 4")
    return GENERATION_EPSILONS[generation - 1]


def learned_self_play_agent(checkpoint: Path, epsilon: EpsilonConfig) -> AgentSpec:
    """Load one checkpoint once and expose an epsilon-wrapped immutable behavior policy."""
    from agent_avenue.agents.learned import LearnedValueAgent
    from agent_avenue.learning import load_checkpoint

    loaded = load_checkpoint(checkpoint)
    base = LearnedValueAgent.from_checkpoint(loaded)
    wrapped = EpsilonGreedyAgent(base, epsilon)
    config = wrapped.config_to_data()
    return AgentSpec("frozen-incumbent", config, lambda: wrapped)


def generation_config_from_agent(
    *,
    generation: int,
    run_id: str,
    game_count: int,
    root_seed: int,
    agent: AgentSpec,
    epsilon: EpsilonConfig,
    game_config: GameConfig | None = None,
) -> GenerationConfig:
    base = agent.config.get("base")
    if not isinstance(base, Mapping):
        raise ValueError("self-play agent must be an epsilon-wrapped learned checkpoint")
    checkpoint = base.get("checkpoint_fingerprint")
    tensor = base.get("tensor_digest")
    if not isinstance(checkpoint, str) or not isinstance(tensor, str):
        raise ValueError("self-play agent config is missing checkpoint identity")
    return GenerationConfig(
        generation=generation,
        run_id=run_id,
        game_count=game_count,
        root_seed=root_seed,
        parent_checkpoint_fingerprint=checkpoint,
        parent_tensor_digest=tensor,
        epsilon=epsilon,
        game_config=game_config or GameConfig(),
    )


def schedule_generation(config: GenerationConfig, agent: AgentSpec) -> Iterator[GameSpec]:
    """Yield independent games whose random domains are fixed by generation and index."""
    base = agent.config.get("base")
    if not isinstance(base, Mapping) or (
        base.get("checkpoint_fingerprint") != config.parent_checkpoint_fingerprint
        or base.get("tensor_digest") != config.parent_tensor_digest
    ):
        raise ValueError("behavior agent does not match the generation parent checkpoint")
    if agent.config.get("epsilon") != config.epsilon.to_data():
        raise ValueError("behavior agent does not match the generation epsilon")
    for index in range(config.game_count):
        domain = f"self-play:g{config.generation}:game:{index}"
        yield GameSpec(
            config.run_id,
            f"game-{index:06d}",
            None,
            config.game_config,
            derive_seed(config.root_seed, f"{domain}:setup") & ((1 << 64) - 1),
            (agent, agent),
            (
                derive_seed(config.root_seed, f"{domain}:player-one"),
                derive_seed(config.root_seed, f"{domain}:player-two"),
            ),
            (SEED_DERIVATION, SEED_DERIVATION),
        )

"""Public-only Step-4 panel identities and fixed-stratum selection.

This module intentionally retains a source record fingerprint only as an audit locator.  It never
participates in a safe identity, rank, seed, or panel order.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Final

from agent_avenue.agents.ordering import semantic_action_key
from agent_avenue.engine import Phase, apply_action, new_game
from agent_avenue.engine.cards import CANONICAL_DECK
from agent_avenue.engine.model import Action
from agent_avenue.observation import observation_to_data, observe
from agent_avenue.observation.model import PlayerObservation, RecruitContext
from agent_avenue.storage.game_record import GameRecord, game_record_fingerprint, verify_game_record

PANEL_SELECTION_VERSION: Final[str] = "step4-safe-panel-v1"
PANEL_SIZE_PER_STRATUM: Final[int] = 20
PANEL_STRATA: Final[tuple[str, ...]] = (
    "play:turn_1_3:legal_1_2",
    "play:turn_1_3:legal_6",
    "play:turn_1_3:legal_12",
    "play:turn_4_6:legal_1_2",
    "play:turn_4_6:legal_6",
    "play:turn_4_6:legal_12",
    "play:turn_7_plus:legal_1_2",
    "play:turn_7_plus:legal_6",
    "play:turn_7_plus:legal_12",
    "recruit:turn_1_3:unseen_ge_21",
    "recruit:turn_4_6:unseen_ge_21",
    "recruit:turn_7_plus:unseen_le_12",
    "recruit:turn_7_plus:unseen_13_20",
    "recruit:turn_7_plus:unseen_ge_21",
)


class PanelSelectionError(ValueError):
    """Raised when a public panel candidate cannot satisfy the fixed Step-4 contract."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def canonical_legal_actions(actions: Iterable[Action]) -> tuple[Action, ...]:
    """Return the complete legal set in repository semantic order."""
    result = tuple(sorted(actions, key=semantic_action_key))
    if not result or len(set(result)) != len(result):
        raise PanelSelectionError("legal actions must be a non-empty unique semantic set")
    return result


def canonical_observation(observation: PlayerObservation) -> PlayerObservation:
    """Normalize only semantic action ordering for identity serialization.

    The observation representation itself remains the repository's allowlisted public data format.
    Hand order is deliberately not rewritten here: the safe identity is explicitly based on the
    complete observed public value.  The latent-policy boundary performs its own hand
    canonicalization before any policy invocation.
    """
    return replace(observation, legal_actions=canonical_legal_actions(observation.legal_actions))


def safe_position_identity(
    observation: PlayerObservation,
    *,
    replicate_id: str,
    stratum: str,
) -> str:
    """Hash canonical public material only; audit/source information is intentionally absent."""
    if not replicate_id:
        raise PanelSelectionError("replicate_id cannot be empty")
    if stratum not in PANEL_STRATA:
        raise PanelSelectionError("stratum is not one of the frozen Step-4 strata")
    canonical = canonical_observation(observation)
    decision = canonical.decision
    actor = getattr(decision, "actor", None)
    revision = getattr(decision, "revision", None)
    if canonical.viewer is not actor or type(revision) is not int:
        raise PanelSelectionError("a panel observation must be from the decision actor viewpoint")
    payload = {
        "observation": observation_to_data(canonical),
        "decision_actor": actor.value,
        "decision_revision": revision,
        "replicate_id": replicate_id,
        "stratum": stratum,
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _turn_band(turn: int) -> str | None:
    if 1 <= turn <= 3:
        return "turn_1_3"
    if 4 <= turn <= 6:
        return "turn_4_6"
    if turn >= 7:
        return "turn_7_plus"
    return None


def public_unseen_count(observation: PlayerObservation) -> int:
    """Count copy-weighted names not identified by the decision actor at recruit.

    This is the current face-down card plus offerer hand plus hidden deck in a normal recruit
    position.  It is calculated from public card counts rather than a source state's hidden zones.
    """
    decision = observation.decision
    if not isinstance(decision, RecruitContext):
        raise PanelSelectionError("public unseen count is defined for recruit observations only")
    if decision.known_face_down is not None:
        raise PanelSelectionError("recruit panel observations may not expose the face-down card")
    remaining = Counter(CANONICAL_DECK)
    for player in observation.players:
        remaining.subtract(player.recruited)
    remaining.subtract(observation.own_hand)
    remaining[decision.face_up] -= 1
    if any(count < 0 for count in remaining.values()):
        raise PanelSelectionError("observation has impossible public card multiplicities")
    return sum((+remaining).values())


def stratum_for_observation(observation: PlayerObservation) -> str | None:
    """Return the exact frozen stratum, or ``None`` when the public position is ineligible."""
    canonical = canonical_observation(observation)
    if canonical.viewer is not getattr(canonical.decision, "actor", None):
        return None
    band = _turn_band(canonical.turn)
    if band is None:
        return None
    decision = canonical.decision
    if decision.kind == "play":
        bucket = {1: "legal_1_2", 2: "legal_1_2", 6: "legal_6", 12: "legal_12"}.get(
            len(canonical.legal_actions)
        )
        return None if bucket is None else f"play:{band}:{bucket}"
    if not isinstance(decision, RecruitContext) or decision.known_face_down is not None:
        return None
    unseen = public_unseen_count(canonical)
    if band in {"turn_1_3", "turn_4_6"}:
        return f"recruit:{band}:unseen_ge_21" if unseen >= 21 else None
    if unseen <= 12:
        suffix = "unseen_le_12"
    elif unseen <= 20:
        suffix = "unseen_13_20"
    else:
        suffix = "unseen_ge_21"
    return f"recruit:{band}:{suffix}"


def _valid_fingerprint(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True, slots=True)
class PanelCandidate:
    """One train-only public decision with an audit-only source-game locator."""

    replicate_id: str
    observation: PlayerObservation
    source_record_fingerprint: str
    stratum: str | None = None

    def __post_init__(self) -> None:
        if not _valid_fingerprint(self.source_record_fingerprint):
            raise PanelSelectionError("source_record_fingerprint must be a SHA-256 digest")
        inferred = stratum_for_observation(self.observation)
        declared = inferred if self.stratum is None else self.stratum
        if declared is None or declared != inferred or declared not in PANEL_STRATA:
            raise PanelSelectionError("candidate does not belong to its declared frozen stratum")
        object.__setattr__(self, "stratum", declared)

    @property
    def safe_identity(self) -> str:
        assert self.stratum is not None
        return safe_position_identity(
            self.observation, replicate_id=self.replicate_id, stratum=self.stratum
        )

    @property
    def selection_hash(self) -> str:
        assert self.stratum is not None
        payload = f"step4:panel:{self.replicate_id}:{self.stratum}:{self.safe_identity}".encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class PanelPosition:
    """A selected, deduplicated safe position plus its non-semantic audit locator."""

    replicate_id: str
    stratum: str
    safe_identity: str
    selection_hash: str
    observation: PlayerObservation
    audit_record_fingerprint: str

    def __post_init__(self) -> None:
        if self.stratum not in PANEL_STRATA or not _valid_fingerprint(self.safe_identity):
            raise PanelSelectionError("selected position has invalid frozen identity metadata")
        if not _valid_fingerprint(self.audit_record_fingerprint):
            raise PanelSelectionError("selected position audit locator must be a SHA-256 digest")
        expected = safe_position_identity(
            self.observation, replicate_id=self.replicate_id, stratum=self.stratum
        )
        if self.safe_identity != expected:
            raise PanelSelectionError(
                "selected position identity does not match its public observation"
            )
        expected_rank = hashlib.sha256(
            f"step4:panel:{self.replicate_id}:{self.stratum}:{self.safe_identity}".encode()
        ).hexdigest()
        if self.selection_hash != expected_rank:
            raise PanelSelectionError("selected position rank does not match the frozen contract")


@dataclass(frozen=True, slots=True)
class PanelSelectionResult:
    """Selected panel plus the complete safe/audit-only selection accounting."""

    positions: tuple[PanelPosition, ...]
    eligible_identity_counts: dict[str, int]

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            _canonical_json(
                {
                    "version": PANEL_SELECTION_VERSION,
                    "eligible_identity_counts": self.eligible_identity_counts,
                    "positions": [
                        {
                            "replicate_id": position.replicate_id,
                            "stratum": position.stratum,
                            "safe_identity": position.safe_identity,
                            "selection_hash": position.selection_hash,
                            "audit_record_fingerprint": position.audit_record_fingerprint,
                        }
                        for position in self.positions
                    ],
                }
            )
        ).hexdigest()


def extract_train_panel_candidates(
    records: Sequence[GameRecord],
    *,
    replicate_id: str,
    train_record_fingerprints: Iterable[str],
    verify_code: bool = True,
) -> tuple[PanelCandidate, ...]:
    """Replay only declared train games into detached decision-actor public candidates.

    Authoritative states are used only transiently to verify/replay the frozen record and construct
    a ``PlayerObservation``.  The returned candidates retain none of their setup seed, actions,
    hidden zones, outcome, or provenance beyond their audit-only record fingerprint.
    """
    train = set(train_record_fingerprints)
    if not train or any(not _valid_fingerprint(value) for value in train):
        raise PanelSelectionError("train_record_fingerprints must be non-empty SHA-256 identities")
    result: list[PanelCandidate] = []
    seen_records: set[str] = set()
    for record in records:
        fingerprint = game_record_fingerprint(record)
        if fingerprint not in train:
            continue
        if fingerprint in seen_records:
            raise PanelSelectionError("train records contain duplicate record fingerprints")
        seen_records.add(fingerprint)
        verify_game_record(record, verify_code=verify_code)
        state = new_game(record.replay.config, record.replay.seed)
        for action in record.replay.actions:
            if state.phase is Phase.TERMINAL:
                raise PanelSelectionError("verified record contains an action after termination")
            actor = (
                state.active_player if state.phase is Phase.PLAY else state.active_player.other()
            )
            observation = observe(state, actor)
            stratum = stratum_for_observation(observation)
            if stratum is not None:
                result.append(
                    PanelCandidate(replicate_id, canonical_observation(observation), fingerprint)
                )
            state = apply_action(state, action)
    if seen_records != train:
        raise PanelSelectionError("declared train records were not supplied exactly once")
    return tuple(result)


def select_panel_result(
    candidates: Sequence[PanelCandidate], *, replicate_id: str
) -> PanelSelectionResult:
    """Select all frozen quotas and retain per-stratum deduplicated eligibility counts."""
    if not replicate_id:
        raise PanelSelectionError("replicate_id cannot be empty")
    if any(candidate.replicate_id != replicate_id for candidate in candidates):
        raise PanelSelectionError("all candidates must belong to the requested replicate")
    by_identity: dict[str, PanelCandidate] = {}
    for candidate in candidates:
        identity = candidate.safe_identity
        prior = by_identity.get(identity)
        if prior is None or candidate.source_record_fingerprint < prior.source_record_fingerprint:
            by_identity[identity] = candidate
    counts = {
        stratum: sum(candidate.stratum == stratum for candidate in by_identity.values())
        for stratum in PANEL_STRATA
    }
    selected: list[PanelPosition] = []
    used_source_games: set[str] = set()
    for stratum in PANEL_STRATA:
        eligible = sorted(
            (candidate for candidate in by_identity.values() if candidate.stratum == stratum),
            key=lambda candidate: (candidate.selection_hash, candidate.safe_identity),
        )
        kept: list[PanelCandidate] = []
        for candidate in eligible:
            if candidate.source_record_fingerprint in used_source_games:
                continue
            kept.append(candidate)
            used_source_games.add(candidate.source_record_fingerprint)
            if len(kept) == PANEL_SIZE_PER_STRATUM:
                break
        if len(kept) != PANEL_SIZE_PER_STRATUM:
            raise PanelSelectionError(
                f"stratum {stratum!r} has only {len(kept)} eligible identities "
                "after source-game cap"
            )
        selected.extend(
            PanelPosition(
                candidate.replicate_id,
                stratum,
                candidate.safe_identity,
                candidate.selection_hash,
                canonical_observation(candidate.observation),
                candidate.source_record_fingerprint,
            )
            for candidate in kept
        )
    return PanelSelectionResult(tuple(selected), counts)


def select_panel_positions(
    candidates: Sequence[PanelCandidate], *, replicate_id: str
) -> tuple[PanelPosition, ...]:
    """Select exactly 14x20 train-only positions with global source-game uniqueness."""
    return select_panel_result(candidates, replicate_id=replicate_id).positions


def canonical_panel_order(positions: Sequence[PanelPosition]) -> tuple[PanelPosition, ...]:
    """Return the immutable training order: stratum index, selection hash, safe identity."""
    if len(positions) != len(set(position.safe_identity for position in positions)):
        raise PanelSelectionError("panel contains duplicate safe identities")
    index = {stratum: ordinal for ordinal, stratum in enumerate(PANEL_STRATA)}
    return tuple(
        sorted(
            positions,
            key=lambda position: (
                index[position.stratum],
                position.selection_hash,
                position.safe_identity,
            ),
        )
    )


__all__ = [
    "PANEL_SELECTION_VERSION",
    "PANEL_SIZE_PER_STRATUM",
    "PANEL_STRATA",
    "PanelCandidate",
    "PanelPosition",
    "PanelSelectionError",
    "PanelSelectionResult",
    "canonical_legal_actions",
    "canonical_observation",
    "canonical_panel_order",
    "extract_train_panel_candidates",
    "public_unseen_count",
    "safe_position_identity",
    "select_panel_positions",
    "select_panel_result",
    "stratum_for_observation",
]

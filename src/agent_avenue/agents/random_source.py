"""Versioned, domain-separated deterministic randomness for agents."""

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

RNG_ALGORITHM = "sha256-counter-rejection-v1"
SEED_DERIVATION = "sha256-domain-v1"


def _exact_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    return value


def derive_seed(root_seed: int, domain: str, *, version: str = SEED_DERIVATION) -> int:
    """Derive an independent seed from a root seed and stable, explicit domain."""
    _exact_int(root_seed, "root_seed")
    if not domain or not isinstance(domain, str):
        raise ValueError("domain must be a non-empty string")
    if version != SEED_DERIVATION:
        raise ValueError(f"unsupported seed derivation version: {version!r}")
    payload = json.dumps(
        {"domain": domain, "root_seed": root_seed, "version": version},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


@runtime_checkable
class RandomSource(Protocol):
    """The deliberately tiny random interface available to an agent."""

    def randbelow(self, upper_bound: int) -> int:
        """Return an integer in ``range(upper_bound)``."""


@dataclass(slots=True)
class DeterministicRandom:
    """Cross-platform deterministic SHA-256 counter stream.

    Unlike ``random.Random``, the byte stream and bounded-integer mapping are part of this
    package's explicitly named compatibility contract.
    """

    seed: int
    domain: str
    algorithm: str = RNG_ALGORITHM
    _counter: int = 0

    def __post_init__(self) -> None:
        _exact_int(self.seed, "seed")
        if not self.domain or not isinstance(self.domain, str):
            raise ValueError("domain must be a non-empty string")
        if self.algorithm != RNG_ALGORITHM:
            raise ValueError(f"unsupported RNG algorithm: {self.algorithm!r}")
        _exact_int(self._counter, "counter")
        if self._counter < 0:
            raise ValueError("counter cannot be negative")

    @classmethod
    def from_root_seed(cls, root_seed: int, domain: str) -> "DeterministicRandom":
        """Construct an independent stream using the versioned derivation scheme."""
        return cls(derive_seed(root_seed, domain), domain)

    def _word(self) -> int:
        payload = json.dumps(
            {
                "algorithm": self.algorithm,
                "counter": self._counter,
                "domain": self.domain,
                "seed": self.seed,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        self._counter += 1
        return int.from_bytes(hashlib.sha256(payload).digest(), "big")

    def randbelow(self, upper_bound: int) -> int:
        """Return an unbiased bounded integer using rejection sampling."""
        _exact_int(upper_bound, "upper_bound")
        if upper_bound <= 0:
            raise ValueError("upper_bound must be positive")
        modulus = 1 << 256
        limit = modulus - (modulus % upper_bound)
        while True:
            value = self._word()
            if value < limit:
                return value % upper_bound

    def randbelow_batch(self, upper_bound: int, count: int) -> list[int]:
        """Return ``count`` consecutive ``randbelow`` draws from the identical stream.

        This is a throughput path for bootstrap resampling. It hashes the exact canonical word
        payload that ``_word`` serializes, but builds the constant prefix and suffix once, so the
        values and the final counter equal ``count`` sequential ``randbelow`` calls.
        """
        _exact_int(upper_bound, "upper_bound")
        _exact_int(count, "count")
        if upper_bound <= 0:
            raise ValueError("upper_bound must be positive")
        if count < 0:
            raise ValueError("count cannot be negative")
        prefix = ('{"algorithm":' + json.dumps(self.algorithm) + ',"counter":').encode()
        suffix = (
            ',"domain":' + json.dumps(self.domain) + ',"seed":' + str(self.seed) + "}"
        ).encode()
        modulus = 1 << 256
        limit = modulus - (modulus % upper_bound)
        sha256 = hashlib.sha256
        counter = self._counter
        values: list[int] = []
        append = values.append
        while len(values) < count:
            value = int.from_bytes(sha256(prefix + str(counter).encode() + suffix).digest(), "big")
            counter += 1
            if value < limit:
                append(value % upper_bound)
        self._counter = counter
        return values

    @classmethod
    def from_data(cls, data: object) -> "DeterministicRandom":
        """Restore a stream from an exact normalized state representation."""
        if not isinstance(data, dict) or set(data) != {
            "algorithm",
            "counter",
            "domain",
            "seed",
            "seed_derivation",
        }:
            raise ValueError("malformed deterministic RNG state")
        if data["seed_derivation"] != SEED_DERIVATION:
            raise ValueError("unsupported RNG seed derivation")
        return cls(
            seed=data["seed"],
            domain=data["domain"],
            algorithm=data["algorithm"],
            _counter=data["counter"],
        )

    def to_data(self) -> dict[str, object]:
        """Return sufficient normalized state to reproduce subsequent draws."""
        return {
            "algorithm": self.algorithm,
            "counter": self._counter,
            "domain": self.domain,
            "seed": self.seed,
            "seed_derivation": SEED_DERIVATION,
        }

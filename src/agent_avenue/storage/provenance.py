"""Exact source and dependency-lock identity for reproducible experiments."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path


class SourceIdentityError(ValueError):
    """Raised when the repository source identity cannot be established."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    git_revision: str
    uv_lock_sha256: str
    tracked_tree_clean: bool
    tracked_diff_sha256: str

    def to_data(self) -> dict[str, object]:
        return {
            "git_revision": self.git_revision,
            "uv_lock_sha256": self.uv_lock_sha256,
            "tracked_tree_clean": self.tracked_tree_clean,
            "tracked_diff_sha256": self.tracked_diff_sha256,
        }


def inspect_source_identity(root: Path | None = None) -> SourceIdentity:
    """Inspect HEAD, tracked modifications, and the exact dependency lock bytes."""
    root = (root or repository_root()).resolve()
    lock = root / "uv.lock"
    if lock.is_symlink() or not lock.is_file():
        raise SourceIdentityError("uv.lock must be a regular file")
    try:
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ("git", "status", "--porcelain=v1", "--untracked-files=no"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        diff = subprocess.run(
            ("git", "diff", "--binary", "HEAD", "--", "."),
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SourceIdentityError("unable to inspect Git source identity") from exc
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise SourceIdentityError("Git revision is not a full lowercase object id")
    return SourceIdentity(
        revision,
        _sha256_file(lock),
        not bool(status.strip()),
        hashlib.sha256(diff).hexdigest(),
    )

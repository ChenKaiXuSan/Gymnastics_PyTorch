"""Immutable data contracts for the FreeMan benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArchiveEntry:
    """One file published by the gated Hugging Face dataset repository."""

    path: str
    size: int
    sha256: str | None

    def __post_init__(self) -> None:
        path = Path(self.path)
        if (
            not self.path
            or path.is_absolute()
            or ".." in path.parts
            or self.size <= 0
        ):
            raise ValueError(f"invalid archive entry: {self.path!r}")
        if self.sha256 is not None:
            digest = self.sha256.lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"invalid SHA256 for {self.path!r}")
            object.__setattr__(self, "sha256", digest)


@dataclass(frozen=True)
class PreflightReport:
    """Read-only result of authentication, access, and storage checks."""

    repo_id: str
    revision: str
    hf_executable: Path
    authenticated_user: str
    access_granted: bool
    archive_root: Path
    required_bytes: int
    free_bytes: int
    reserve_bytes: int
    entries: tuple[ArchiveEntry, ...]

    def __post_init__(self) -> None:
        if not self.repo_id or not self.revision or not self.authenticated_user:
            raise ValueError("repository, revision, and authenticated user are required")
        if min(self.required_bytes, self.free_bytes, self.reserve_bytes) < 0:
            raise ValueError("preflight byte counts must be non-negative")
        object.__setattr__(self, "hf_executable", Path(self.hf_executable).resolve())
        object.__setattr__(self, "archive_root", Path(self.archive_root).resolve())
        object.__setattr__(self, "entries", tuple(self.entries))

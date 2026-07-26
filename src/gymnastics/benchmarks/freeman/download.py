"""Hugging Face inventory, preflight, and resumable FreeMan download."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import yaml

from gymnastics.common.paths import PROJECT_ROOT

from .schema import ArchiveEntry, PreflightReport


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _resolved_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the committed FreeMan YAML configuration."""
    config_path = _resolved_path(path)
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("FreeMan config must contain a mapping")
    config = dict(loaded)
    paths = config.get("paths")
    dataset = config.get("dataset")
    if not isinstance(paths, Mapping) or not isinstance(dataset, Mapping):
        raise ValueError("FreeMan config requires paths and dataset mappings")
    config["paths"] = {
        str(name): _resolved_path(value)
        for name, value in paths.items()
    }
    fps_values = tuple(int(value) for value in dataset.get("fps_subsets", ()))
    if not fps_values or any(value not in {30, 60} for value in fps_values):
        raise ValueError("dataset.fps_subsets must contain only 30 and/or 60")
    subjects = tuple(int(value) for value in dataset.get("subjects", ()))
    if not subjects or len(set(subjects)) != len(subjects):
        raise ValueError("dataset.subjects must contain unique subject IDs")
    if any(value < 1 or value > 40 for value in subjects):
        raise ValueError("dataset.subjects must be within 1..40")
    frame_stride = int(dataset.get("frame_stride", 0))
    if frame_stride < 1:
        raise ValueError("dataset.frame_stride must be positive")
    config["dataset"] = {
        **dict(dataset),
        "fps_subsets": list(fps_values),
        "subjects": list(subjects),
        "frame_stride": frame_stride,
    }
    return config


def _sibling_sha256(sibling: Any) -> str | None:
    lfs = getattr(sibling, "lfs", None)
    if lfs is None:
        return None
    if isinstance(lfs, Mapping):
        value = lfs.get("sha256")
    else:
        value = getattr(lfs, "sha256", None)
    return str(value) if value else None


def fetch_hub_inventory(
    api: Any,
    repo_id: str,
    revision: str,
) -> tuple[ArchiveEntry, ...]:
    """Fetch a complete, size-aware dataset file inventory."""
    info = api.dataset_info(
        repo_id,
        revision=revision,
        files_metadata=True,
    )
    entries = tuple(
        sorted(
            (
                ArchiveEntry(
                    path=str(sibling.rfilename),
                    size=int(sibling.size),
                    sha256=_sibling_sha256(sibling),
                )
                for sibling in info.siblings
            ),
            key=lambda entry: entry.path,
        )
    )
    paths = {entry.path for entry in entries}
    missing = [
        f"subj{subject:02d}.zip"
        for subject in range(1, 41)
        if f"subj{subject:02d}.zip" not in paths
    ]
    if missing:
        raise RuntimeError(
            "FreeMan release inventory is missing required subject archives: "
            + ", ".join(missing)
        )
    return entries


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entry_is_valid(
    entry: ArchiveEntry,
    archive_root: Path,
    *,
    verify_sha256: bool,
) -> bool:
    path = archive_root / entry.path
    if not path.is_file() or path.stat().st_size != entry.size:
        return False
    return not (
        verify_sha256
        and entry.sha256 is not None
        and _sha256(path) != entry.sha256
    )


def _existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise RuntimeError(f"no existing parent for storage path {path}")
        candidate = candidate.parent
    return candidate


def run_preflight(
    config: Mapping[str, Any],
    *,
    runner: Runner = subprocess.run,
    api: Any | None = None,
) -> PreflightReport:
    """Check CLI authentication, gated access, local files, and free space."""
    hf = shutil.which("hf")
    if hf is None:
        raise RuntimeError(
            "hf executable not found; install the current Hugging Face Hub CLI"
        )
    auth = runner(
        [hf, "auth", "whoami"],
        capture_output=True,
        text=True,
        check=False,
    )
    if auth.returncode != 0 or not auth.stdout.strip():
        raise RuntimeError(
            "local Hugging Face authentication is unavailable; run `hf auth login`"
        )
    repository = config["repository"]
    repo_id = str(repository["repo_id"])
    revision = str(repository["revision"])
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    try:
        entries = fetch_hub_inventory(api, repo_id, revision)
    except Exception as error:
        raise RuntimeError(
            f"FreeMan gated access is unavailable for {repo_id}: {error}"
        ) from error
    archive_root = Path(config["paths"]["archive_root"]).resolve()
    download = config.get("download", {})
    verify_sha256 = bool(download.get("verify_sha256", True))
    reserve_bytes = int(download.get("reserve_bytes", 0))
    if reserve_bytes < 0:
        raise ValueError("download.reserve_bytes must be non-negative")
    required_bytes = sum(
        entry.size
        for entry in entries
        if not _entry_is_valid(
            entry,
            archive_root,
            verify_sha256=verify_sha256,
        )
    )
    free_bytes = int(
        shutil.disk_usage(_existing_parent(archive_root.parent)).free
    )
    if free_bytes < required_bytes + reserve_bytes:
        raise RuntimeError(
            "insufficient free space for FreeMan download: "
            f"need {required_bytes + reserve_bytes} bytes including reserve, "
            f"have {free_bytes}"
        )
    return PreflightReport(
        repo_id=repo_id,
        revision=revision,
        hf_executable=Path(hf),
        authenticated_user=auth.stdout.strip(),
        access_granted=True,
        archive_root=archive_root,
        required_bytes=required_bytes,
        free_bytes=free_bytes,
        reserve_bytes=reserve_bytes,
        entries=entries,
    )


def validate_downloads(
    entries: Sequence[ArchiveEntry],
    archive_root: Path,
) -> tuple[Path, ...]:
    """Require every inventory entry to match its published size and digest."""
    root = Path(archive_root).resolve()
    validated: list[Path] = []
    for entry in entries:
        path = root / entry.path
        if not _entry_is_valid(entry, root, verify_sha256=True):
            raise RuntimeError(f"downloaded FreeMan file is missing or invalid: {entry.path}")
        validated.append(path)
    return tuple(validated)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def download_release(
    config: Mapping[str, Any],
    report: PreflightReport,
    *,
    runner: Runner = subprocess.run,
) -> Path:
    """Download the complete release into one local directory and verify it."""
    report.archive_root.mkdir(parents=True, exist_ok=True)
    command = [
        str(report.hf_executable),
        "download",
        report.repo_id,
        "--repo-type",
        "dataset",
        "--revision",
        report.revision,
        "--local-dir",
        str(report.archive_root),
    ]
    completed = runner(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "FreeMan download failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    paths = validate_downloads(report.entries, report.archive_root)
    manifest_root = Path(config["paths"]["manifest_root"]).resolve()
    inventory_payload = {
        "repo_id": report.repo_id,
        "revision": report.revision,
        "entries": [asdict(entry) for entry in report.entries],
    }
    state_payload = {
        **inventory_payload,
        "authenticated_user": report.authenticated_user,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "required_bytes_before_download": report.required_bytes,
        "free_bytes_before_download": report.free_bytes,
        "reserve_bytes": report.reserve_bytes,
        "files": [
            {
                "path": str(path.relative_to(report.archive_root)),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in paths
        ],
    }
    _write_json_atomic(manifest_root / "remote_inventory.json", inventory_payload)
    _write_json_atomic(manifest_root / "download_state.json", state_payload)
    return report.archive_root

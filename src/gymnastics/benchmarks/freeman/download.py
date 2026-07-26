"""Hugging Face inventory, preflight, and resumable FreeMan download."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
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


def subject_archive_set(
    subject_id: int,
    entries: Sequence[ArchiveEntry],
) -> tuple[ArchiveEntry, ...]:
    """Return one subject's contiguous numeric pieces followed by its ZIP."""
    if subject_id < 1 or subject_id > 40:
        raise ValueError("FreeMan subject_id must be within 1..40")
    stem = f"subj{subject_id:02d}"
    terminal = next(
        (entry for entry in entries if entry.path == f"{stem}.zip"),
        None,
    )
    if terminal is None:
        raise RuntimeError(f"missing required subject archive {stem}.zip")
    numbered: list[tuple[int, ArchiveEntry]] = []
    pattern = re.compile(rf"^{re.escape(stem)}\.(\d+)$")
    for entry in entries:
        match = pattern.fullmatch(entry.path)
        if match:
            numbered.append((int(match.group(1)), entry))
    numbered.sort(key=lambda item: item[0])
    if numbered:
        observed = [index for index, _ in numbered]
        expected = list(range(1, observed[-1] + 1))
        if observed != expected:
            raise RuntimeError(
                f"non-contiguous numeric volumes for {stem}: {observed}"
            )
    return tuple(entry for _, entry in numbered) + (terminal,)


def _completed(
    runner: Runner,
    command: list[str],
) -> subprocess.CompletedProcess[str]:
    return runner(
        command,
        capture_output=True,
        text=True,
        check=False,
    )


def _seven_zip() -> str:
    executable = shutil.which("7z")
    if executable is None:
        raise RuntimeError(
            "7z executable not found; install p7zip to extract FreeMan archives"
        )
    return executable


def _archive_test(
    archive: Path,
    *,
    seven_zip: str,
    runner: Runner,
) -> bool:
    return _completed(runner, [seven_zip, "t", str(archive)]).returncode == 0


def _archive_members(
    archive: Path,
    *,
    seven_zip: str,
    runner: Runner,
) -> tuple[str, ...]:
    completed = _completed(
        runner,
        [seven_zip, "l", "-slt", str(archive)],
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"cannot list FreeMan archive {archive.name}: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    members = tuple(
        line.split("=", 1)[1].strip()
        for line in completed.stdout.splitlines()
        if line.startswith("Path =") and line.split("=", 1)[1].strip()
    )
    if not members:
        raise RuntimeError(f"FreeMan archive has no members: {archive.name}")
    return members


def _validate_archive_members(members: Sequence[str], output_root: Path) -> None:
    root = output_root.resolve()
    for member in members:
        normalized = member.replace("\\", "/")
        pure = PurePosixPath(normalized)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or re.match(r"^[A-Za-z]:", normalized)
        ):
            raise RuntimeError(f"unsafe archive member: {member}")
        destination = (root / Path(*pure.parts)).resolve()
        if destination != root and root not in destination.parents:
            raise RuntimeError(f"unsafe archive member: {member}")


def _extract_archive(
    archive: Path,
    output_root: Path,
    *,
    seven_zip: str,
    runner: Runner,
) -> None:
    members = _archive_members(archive, seven_zip=seven_zip, runner=runner)
    _validate_archive_members(members, output_root)
    completed = _completed(
        runner,
        [
            seven_zip,
            "x",
            str(archive),
            f"-o{output_root}",
            "-y",
        ],
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"cannot extract FreeMan archive {archive.name}: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )


def _contains_signature(path: Path, signature: bytes, *, tail: bool) -> bool:
    with path.open("rb") as stream:
        if tail:
            stream.seek(max(0, path.stat().st_size - 65_557))
        data = stream.read(65_557 if tail else 4)
    return signature in data


def _reconstruct_numeric_archive(
    subject_id: int,
    pieces: Sequence[Path],
    work_root: Path,
    *,
    seven_zip: str,
    runner: Runner,
) -> Path:
    starts = [
        path
        for path in pieces
        if _contains_signature(path, b"PK\x03\x04", tail=False)
    ]
    finals = [
        path
        for path in pieces
        if _contains_signature(path, b"PK\x05\x06", tail=True)
    ]
    if len(starts) != 1 or len(finals) != 1 or starts[0] == finals[0]:
        raise RuntimeError(
            f"cannot determine numeric ZIP volume order for subject {subject_id:02d}"
        )
    numeric = sorted(
        (path for path in pieces if path.suffix[1:].isdigit()),
        key=lambda path: int(path.suffix[1:]),
    )
    terminal = next(path for path in pieces if path.suffix.lower() == ".zip")
    if starts[0] in numeric and finals[0] == terminal:
        ordered = numeric + [terminal]
    elif starts[0] == terminal and finals[0] in numeric:
        ordered = [terminal] + numeric
    else:
        raise RuntimeError(
            f"ambiguous numeric ZIP volume order for subject {subject_id:02d}"
        )
    work_root.mkdir(parents=True, exist_ok=True)
    reconstructed = work_root / f"subject_{subject_id:02d}.reconstructed.zip"
    temporary = reconstructed.with_suffix(reconstructed.suffix + ".partial")
    with temporary.open("wb") as target:
        for part in ordered:
            with part.open("rb") as source:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    temporary.replace(reconstructed)
    if not _archive_test(reconstructed, seven_zip=seven_zip, runner=runner):
        reconstructed.unlink(missing_ok=True)
        raise RuntimeError(
            f"reconstructed ZIP validation failed for subject {subject_id:02d}"
        )
    return reconstructed


def _subject_archive_input(
    subject_id: int,
    archive_root: Path,
    work_root: Path,
    *,
    seven_zip: str,
    runner: Runner,
) -> tuple[Path, bool]:
    terminal = archive_root / f"subj{subject_id:02d}.zip"
    if not terminal.is_file():
        raise FileNotFoundError(terminal)
    if _archive_test(terminal, seven_zip=seven_zip, runner=runner):
        return terminal, False
    numeric = sorted(
        archive_root.glob(f"subj{subject_id:02d}.[0-9]*"),
        key=lambda path: int(path.suffix[1:]),
    )
    if not numeric:
        raise RuntimeError(
            f"invalid FreeMan subject archive without numeric volumes: {terminal.name}"
        )
    observed = [int(path.suffix[1:]) for path in numeric]
    if observed != list(range(1, observed[-1] + 1)):
        raise RuntimeError(
            f"non-contiguous numeric volumes for subject {subject_id:02d}: {observed}"
        )
    reconstructed = _reconstruct_numeric_archive(
        subject_id,
        [*numeric, terminal],
        work_root,
        seven_zip=seven_zip,
        runner=runner,
    )
    return reconstructed, True


def _has_subject_video(subject_root: Path) -> bool:
    for fps in ("30FPS", "60FPS"):
        if any((subject_root / fps / "videos").glob("*/vframes/c01.mp4")):
            return True
    return False


def extract_subject(
    subject_id: int,
    archive_root: Path,
    work_root: Path,
    *,
    runner: Runner = subprocess.run,
) -> Path:
    """Safely extract one subject, reconstructing numeric ZIP pieces if needed."""
    if subject_id < 1 or subject_id > 40:
        raise ValueError("FreeMan subject_id must be within 1..40")
    archives = Path(archive_root).resolve()
    work = Path(work_root).resolve()
    subject_root = work / f"subject_{subject_id:02d}"
    if subject_root.exists():
        if _has_subject_video(subject_root):
            return subject_root
        raise RuntimeError(f"existing subject workspace is invalid: {subject_root}")
    partial = work / f"subject_{subject_id:02d}.partial"
    if partial.exists():
        raise RuntimeError(f"partial subject workspace requires inspection: {partial}")
    seven_zip = _seven_zip()
    archive, disposable = _subject_archive_input(
        subject_id,
        archives,
        work,
        seven_zip=seven_zip,
        runner=runner,
    )
    _extract_archive(
        archive,
        partial,
        seven_zip=seven_zip,
        runner=runner,
    )
    if not _has_subject_video(partial):
        raise RuntimeError(
            f"subject {subject_id:02d} extraction contains no FreeMan c01 videos"
        )
    partial.replace(subject_root)
    if disposable:
        archive.unlink()
    return subject_root


def cleanup_subject_workspace(
    subject_id: int,
    subject_root: Path,
    work_root: Path,
) -> None:
    """Remove only the exact non-symlink subject workspace requested."""
    work = Path(work_root).resolve()
    target = Path(subject_root)
    expected = work / f"subject_{subject_id:02d}"
    if (
        target.is_symlink()
        or target.resolve(strict=False) != expected
        or expected.parent != work
    ):
        raise ValueError(f"refusing to remove unsafe subject workspace: {target}")
    if expected.exists():
        shutil.rmtree(expected)


_SHARED_ARCHIVES = ("cameras.zip", "keypoints2d.zip", "keypoints3d.zip")
_SHARED_TEXT_FILES = {
    "session_list.txt",
    "session_list_mono.txt",
    "ignore_list.txt",
    "train.txt",
    "valid.txt",
    "validation.txt",
    "test.txt",
}


def _shared_tree_valid(root: Path) -> bool:
    return (
        any(root.rglob("session_list.txt"))
        and any(path for path in root.rglob("*.json") if "camera" in str(path.parent).lower())
        and any(path for path in root.rglob("*.npy") if "keypoints2d" in str(path.parent))
        and any(path for path in root.rglob("*.npy") if "keypoints3d" in str(path.parent))
    )


def extract_shared_annotations(
    entries: Sequence[ArchiveEntry],
    archive_root: Path,
    work_root: Path,
    *,
    runner: Runner = subprocess.run,
) -> Path:
    """Publish only the shared annotations consumed by the benchmark."""
    archives = Path(archive_root).resolve()
    work = Path(work_root).resolve()
    shared = work / "shared"
    manifest_entries = [
        asdict(entry)
        for entry in entries
        if Path(entry.path).name in {*_SHARED_ARCHIVES, *_SHARED_TEXT_FILES}
    ]
    identity = {
        "consumed_entries": sorted(manifest_entries, key=lambda item: item["path"])
    }
    if shared.exists():
        manifest = shared / "extraction_manifest.json"
        if (
            manifest.is_file()
            and json.loads(manifest.read_text(encoding="utf-8")) == identity
            and _shared_tree_valid(shared)
        ):
            return shared
        raise RuntimeError(f"existing shared FreeMan workspace is invalid: {shared}")
    partial = work / "shared.partial"
    if partial.exists():
        raise RuntimeError(f"partial shared workspace requires inspection: {partial}")
    partial.mkdir(parents=True)
    seven_zip = _seven_zip()
    by_name = {Path(entry.path).name: entry for entry in entries}
    for name in _SHARED_ARCHIVES:
        entry = by_name.get(name)
        if entry is None:
            raise RuntimeError(f"missing required shared archive: {name}")
        archive = archives / entry.path
        if not _archive_test(archive, seven_zip=seven_zip, runner=runner):
            raise RuntimeError(f"invalid required shared archive: {name}")
        _extract_archive(
            archive,
            partial,
            seven_zip=seven_zip,
            runner=runner,
        )
    for name in sorted(_SHARED_TEXT_FILES):
        source = archives / name
        if source.is_file():
            shutil.copy2(source, partial / name)
    if not _shared_tree_valid(partial):
        raise RuntimeError("shared FreeMan extraction is missing required annotations")
    _write_json_atomic(partial / "extraction_manifest.json", identity)
    partial.replace(shared)
    return shared

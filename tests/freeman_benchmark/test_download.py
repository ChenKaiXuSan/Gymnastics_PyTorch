from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from gymnastics.benchmarks.freeman.download import (
    cleanup_subject_workspace,
    download_release,
    extract_shared_annotations,
    extract_subject,
    fetch_hub_inventory,
    run_preflight,
    subject_archive_set,
    validate_downloads,
)
from gymnastics.benchmarks.freeman.schema import ArchiveEntry


@dataclass(frozen=True)
class FakeSibling:
    rfilename: str
    size: int
    sha256: str | None = None

    @property
    def lfs(self):
        if self.sha256 is None:
            return None
        return SimpleNamespace(sha256=self.sha256)


class FakeApi:
    def __init__(self, siblings, error: Exception | None = None):
        self.siblings = siblings
        self.error = error

    def dataset_info(self, repo_id, *, revision, files_metadata):
        assert repo_id == "wjwow/FreeMan"
        assert revision == "main"
        assert files_metadata is True
        if self.error is not None:
            raise self.error
        return SimpleNamespace(siblings=self.siblings)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _release_siblings() -> list[FakeSibling]:
    siblings = [
        FakeSibling(f"subj{subject:02d}.zip", subject)
        for subject in range(1, 41)
    ]
    siblings.extend(
        [
            FakeSibling("subj02.01", 101),
            FakeSibling("subj02.02", 102),
            FakeSibling("subj02.03", 103),
            FakeSibling("cameras.zip", 11),
            FakeSibling("keypoints2d.zip", 12),
            FakeSibling("keypoints3d.zip", 13),
            FakeSibling("motions.zip", 14),
            FakeSibling("README.md", 15),
        ]
    )
    return siblings


def _config(tmp_path: Path) -> dict:
    return {
        "repository": {"repo_id": "wjwow/FreeMan", "revision": "main"},
        "paths": {
            "archive_root": tmp_path / "archives",
            "manifest_root": tmp_path / "manifests",
        },
        "download": {"reserve_bytes": 100, "verify_sha256": True},
        "dataset": {
            "fps_subsets": [30, 60],
            "subjects": list(range(1, 41)),
            "frame_stride": 1,
        },
    }


def _authenticated_runner(command, **kwargs):
    assert command == ["/opt/hf", "auth", "whoami"]
    return subprocess.CompletedProcess(command, 0, stdout="freeman-user\n", stderr="")


def test_inventory_preserves_every_subject_and_numeric_volume() -> None:
    entries = fetch_hub_inventory(
        FakeApi(reversed(_release_siblings())),
        "wjwow/FreeMan",
        "main",
    )

    assert tuple(item.path for item in entries) == tuple(
        sorted(item.path for item in entries)
    )
    assert len([item for item in entries if item.path.endswith(".zip")]) == 44
    assert {item.path for item in entries} >= {
        "subj01.zip",
        "subj40.zip",
        "subj02.01",
        "subj02.02",
        "subj02.03",
        "cameras.zip",
        "keypoints2d.zip",
        "keypoints3d.zip",
        "motions.zip",
    }
    assert all(item.size > 0 for item in entries)


def test_inventory_rejects_release_without_all_forty_subjects() -> None:
    incomplete = [
        sibling
        for sibling in _release_siblings()
        if sibling.rfilename != "subj17.zip"
    ]

    with pytest.raises(RuntimeError, match="subj17.zip"):
        fetch_hub_inventory(FakeApi(incomplete), "wjwow/FreeMan", "main")


def test_preflight_rejects_missing_hf(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: None)

    with pytest.raises(RuntimeError, match="hf executable"):
        run_preflight(_config(tmp_path), runner=_authenticated_runner, api=FakeApi([]))


def test_preflight_rejects_unapproved_gated_access(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: "/opt/hf")

    with pytest.raises(RuntimeError, match="gated access"):
        run_preflight(
            _config(tmp_path),
            runner=_authenticated_runner,
            api=FakeApi([], error=PermissionError("pending approval")),
        )


def test_preflight_enforces_remaining_bytes_plus_reserve(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: "/opt/hf")
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _: shutil._ntuple_diskusage(total=1000, used=849, free=151),
    )

    with pytest.raises(RuntimeError, match="free space"):
        run_preflight(
            _config(tmp_path),
            runner=_authenticated_runner,
            api=FakeApi(_release_siblings()),
        )


def test_preflight_subtracts_only_checksum_valid_local_files(
    monkeypatch, tmp_path: Path
) -> None:
    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    good = b"good"
    bad = b"bad!"
    (archive_root / "subj01.zip").write_bytes(good)
    (archive_root / "subj02.zip").write_bytes(bad)
    siblings = _release_siblings()
    siblings[0] = FakeSibling("subj01.zip", len(good), _digest(good))
    siblings[1] = FakeSibling("subj02.zip", len(bad), _digest(b"nope"))
    expected_remaining = sum(item.size for item in siblings) - len(good)

    monkeypatch.setattr(shutil, "which", lambda _: "/opt/hf")
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _: shutil._ntuple_diskusage(
            total=expected_remaining + 1000,
            used=0,
            free=expected_remaining + 1000,
        ),
    )

    report = run_preflight(
        _config(tmp_path),
        runner=_authenticated_runner,
        api=FakeApi(siblings),
    )

    assert report.authenticated_user == "freeman-user"
    assert report.required_bytes == expected_remaining
    assert report.reserve_bytes == 100
    assert report.access_granted is True


def test_download_uses_hf_local_dir_and_writes_validated_state(
    monkeypatch, tmp_path: Path
) -> None:
    payloads = {
        f"subj{subject:02d}.zip": bytes([subject])
        for subject in range(1, 41)
    }
    payloads.update(
        {
            "cameras.zip": b"cameras",
            "keypoints2d.zip": b"k2",
            "keypoints3d.zip": b"k3",
            "motions.zip": b"motion",
        }
    )
    siblings = [
        FakeSibling(name, len(payload), _digest(payload))
        for name, payload in payloads.items()
    ]
    monkeypatch.setattr(shutil, "which", lambda _: "/opt/hf")
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _: shutil._ntuple_diskusage(total=10_000, used=0, free=10_000),
    )
    config = _config(tmp_path)
    report = run_preflight(
        config,
        runner=_authenticated_runner,
        api=FakeApi(siblings),
    )
    commands: list[list[str]] = []

    def download_runner(command, **kwargs):
        commands.append(command)
        archive_root = Path(command[command.index("--local-dir") + 1])
        archive_root.mkdir(parents=True, exist_ok=True)
        for name, payload in payloads.items():
            (archive_root / name).write_bytes(payload)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    target = download_release(config, report, runner=download_runner)

    assert commands == [
        [
            "/opt/hf",
            "download",
            "wjwow/FreeMan",
            "--repo-type",
            "dataset",
            "--revision",
            "main",
            "--local-dir",
            str(tmp_path / "archives"),
        ]
    ]
    assert target == tmp_path / "archives"
    assert validate_downloads(report.entries, target) == tuple(
        target / item.path for item in report.entries
    )
    assert (tmp_path / "manifests" / "remote_inventory.json").is_file()
    assert (tmp_path / "manifests" / "download_state.json").is_file()
    assert not list((tmp_path / "manifests").glob("*.tmp"))


def test_subject_archive_set_orders_numeric_parts_before_zip() -> None:
    entries = (
        ArchiveEntry("subj02.zip", 4, None),
        ArchiveEntry("subj02.03", 3, None),
        ArchiveEntry("subj02.01", 1, None),
        ArchiveEntry("subj01.zip", 5, None),
        ArchiveEntry("subj02.02", 2, None),
    )

    selected = subject_archive_set(2, entries)

    assert [item.path for item in selected] == [
        "subj02.01",
        "subj02.02",
        "subj02.03",
        "subj02.zip",
    ]


class FakeSevenZip:
    def __init__(self, *, listing: tuple[str, ...], expected_reconstructed: bytes | None = None):
        self.listing = listing
        self.expected_reconstructed = expected_reconstructed

    def __call__(self, command, **kwargs):
        operation = command[1]
        archive = Path(command[-1]) if operation in {"t", "l"} else Path(command[2])
        if operation == "t":
            if archive.name.endswith(".reconstructed.zip"):
                success = (
                    self.expected_reconstructed is not None
                    and archive.read_bytes() == self.expected_reconstructed
                )
                return subprocess.CompletedProcess(
                    command, 0 if success else 2, stdout="", stderr=""
                )
            return subprocess.CompletedProcess(command, 2, stdout="", stderr="split")
        if operation == "l":
            output = "\n".join(f"Path = {member}" for member in self.listing)
            return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")
        if operation == "x":
            output_arg = next(item for item in command if item.startswith("-o"))
            output_root = Path(output_arg[2:])
            for member in self.listing:
                target = output_root / member
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"fixture")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {command}")


def test_extract_subject_reconstructs_numeric_parts_and_removes_temporary_zip(
    monkeypatch, tmp_path: Path
) -> None:
    archive_root = tmp_path / "archives"
    work_root = tmp_path / "work"
    archive_root.mkdir()
    first = b"PK\x03\x04first"
    middle = b"middle"
    final = b"lastPK\x05\x06"
    (archive_root / "subj02.01").write_bytes(first)
    (archive_root / "subj02.02").write_bytes(middle)
    (archive_root / "subj02.zip").write_bytes(final)
    monkeypatch.setattr(shutil, "which", lambda _: "/opt/7z")
    runner = FakeSevenZip(
        listing=("30FPS/videos/session_subj02/vframes/c01.mp4",),
        expected_reconstructed=first + middle + final,
    )

    subject_root = extract_subject(
        2,
        archive_root,
        work_root,
        runner=runner,
    )

    assert subject_root == work_root / "subject_02"
    assert (
        subject_root / "30FPS/videos/session_subj02/vframes/c01.mp4"
    ).is_file()
    assert not (work_root / "subject_02.reconstructed.zip").exists()
    assert (archive_root / "subj02.01").is_file()
    assert (archive_root / "subj02.zip").is_file()


def test_extract_rejects_archive_member_path_traversal(
    monkeypatch, tmp_path: Path
) -> None:
    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    (archive_root / "subj01.zip").write_bytes(b"PK\x03\x04safePK\x05\x06")
    monkeypatch.setattr(shutil, "which", lambda _: "/opt/7z")

    class TraversalRunner(FakeSevenZip):
        def __call__(self, command, **kwargs):
            if command[1] == "t":
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            return super().__call__(command, **kwargs)

    with pytest.raises(RuntimeError, match="unsafe archive member"):
        extract_subject(
            1,
            archive_root,
            tmp_path / "work",
            runner=TraversalRunner(listing=("../../escape.mp4",)),
        )

    assert not (tmp_path / "escape.mp4").exists()


def test_extract_ignores_7z_archive_header_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    archive = archive_root / "subj01.zip"
    archive.write_bytes(b"PK\x03\x04safePK\x05\x06")
    monkeypatch.setattr(shutil, "which", lambda _: "/opt/7z")
    member = "30FPS/videos/session_subj01/vframes/c01.mp4"

    class HeaderRunner:
        def __call__(self, command, **kwargs):
            operation = command[1]
            if operation == "t":
                return subprocess.CompletedProcess(
                    command, 0, stdout="", stderr=""
                )
            if operation == "l":
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=(
                        f"Path = {archive.resolve()}\n"
                        "Type = zip\n"
                        "----------\n"
                        f"Path = {member}\n"
                    ),
                    stderr="",
                )
            output_arg = next(
                item for item in command if item.startswith("-o")
            )
            target = Path(output_arg[2:]) / member
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"video")
            return subprocess.CompletedProcess(
                command, 0, stdout="", stderr=""
            )

    subject_root = extract_subject(
        1,
        archive_root,
        tmp_path / "work",
        runner=HeaderRunner(),
    )

    assert (subject_root / member).is_file()


@pytest.mark.parametrize(
    "target_factory",
    [
        lambda work: Path("/"),
        lambda work: work,
        lambda work: work / "subject_02",
    ],
)
def test_cleanup_rejects_any_target_other_than_exact_subject(
    tmp_path: Path, target_factory
) -> None:
    work_root = tmp_path / "work"
    work_root.mkdir()
    target = target_factory(work_root)
    if target != Path("/") and target != work_root:
        target.mkdir()

    with pytest.raises(ValueError, match="refusing"):
        cleanup_subject_workspace(1, target, work_root)


def test_cleanup_rejects_subject_symlink_escaping_work_root(tmp_path: Path) -> None:
    work_root = tmp_path / "work"
    outside = tmp_path / "outside"
    work_root.mkdir()
    outside.mkdir()
    link = work_root / "subject_01"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="refusing"):
        cleanup_subject_workspace(1, link, work_root)

    assert outside.is_dir()


def test_cleanup_removes_only_exact_subject_workspace(tmp_path: Path) -> None:
    work_root = tmp_path / "work"
    subject = work_root / "subject_01"
    other = work_root / "subject_02"
    subject.mkdir(parents=True)
    other.mkdir()
    (subject / "artifact.txt").write_text("done", encoding="utf-8")

    cleanup_subject_workspace(1, subject, work_root)

    assert not subject.exists()
    assert other.is_dir()


def test_extract_shared_annotations_publishes_only_consumed_archives(
    monkeypatch, tmp_path: Path
) -> None:
    archive_root = tmp_path / "archives"
    work_root = tmp_path / "work"
    archive_root.mkdir()
    members = {
        "cameras.zip": ("30FPS/cameras/session_subj01.json",),
        "keypoints2d.zip": ("30FPS/keypoints2d/session_subj01.npy",),
        "keypoints3d.zip": ("30FPS/keypoints3d/session_subj01.npy",),
    }
    entries = []
    for name in (*members, "motions.zip", "bbox2d.zip"):
        payload = b"PK\x03\x04" + name.encode() + b"PK\x05\x06"
        (archive_root / name).write_bytes(payload)
        entries.append(ArchiveEntry(name, len(payload), _digest(payload)))
    (archive_root / "session_list.txt").write_text(
        "session_subj01\n", encoding="utf-8"
    )
    (archive_root / "train.txt").write_text("session_subj01\n", encoding="utf-8")
    (archive_root / "valid.txt").write_text("", encoding="utf-8")
    (archive_root / "test.txt").write_text("", encoding="utf-8")
    for name in ("session_list.txt", "train.txt", "valid.txt", "test.txt"):
        path = archive_root / name
        entries.append(ArchiveEntry(name, path.stat().st_size or 1, None))
    monkeypatch.setattr(shutil, "which", lambda _: "/opt/7z")

    class SharedRunner:
        def __call__(self, command, **kwargs):
            operation = command[1]
            archive = Path(command[-1]) if operation in {"t", "l"} else Path(command[2])
            if operation == "t":
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            if operation == "l":
                output = "\n".join(
                    f"Path = {member}" for member in members[archive.name]
                )
                return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")
            output_arg = next(item for item in command if item.startswith("-o"))
            output_root = Path(output_arg[2:])
            for member in members[archive.name]:
                target = output_root / member
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"annotation")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    shared_root = extract_shared_annotations(
        entries,
        archive_root,
        work_root,
        runner=SharedRunner(),
    )

    assert shared_root == work_root / "shared"
    assert (shared_root / "30FPS/cameras/session_subj01.json").is_file()
    assert (shared_root / "30FPS/keypoints2d/session_subj01.npy").is_file()
    assert (shared_root / "30FPS/keypoints3d/session_subj01.npy").is_file()
    assert (shared_root / "session_list.txt").is_file()
    assert not (shared_root / "motions").exists()
    assert not (shared_root / "bbox2d").exists()


def test_extract_shared_annotations_keeps_same_named_fps_archives(
    monkeypatch,
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archives"
    work_root = tmp_path / "work"
    entries = []
    archive_members = {}
    for fps in (30, 60):
        for kind, suffix in (
            ("cameras", "json"),
            ("keypoints2d", "npy"),
            ("keypoints3d", "npy"),
        ):
            relative = f"{fps}FPS/{kind}.zip"
            archive = archive_root / relative
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_bytes(b"PK\x03\x04fixturePK\x05\x06")
            entries.append(
                ArchiveEntry(relative, archive.stat().st_size, None)
            )
            archive_members[relative] = (
                f"{fps}FPS/{kind}/session_{fps}_subj01.{suffix}",
            )
        for name in ("session_list.txt", "train.txt", "valid.txt", "test.txt"):
            relative = f"{fps}FPS/{name}"
            path = archive_root / relative
            path.write_text(
                f"session_{fps}_subj01\n" if name != "valid.txt" else "",
                encoding="utf-8",
            )
            entries.append(
                ArchiveEntry(relative, max(path.stat().st_size, 1), None)
            )
    monkeypatch.setattr(shutil, "which", lambda _: "/opt/7z")

    class NestedRunner:
        def __call__(self, command, **kwargs):
            operation = command[1]
            archive = (
                Path(command[-1])
                if operation in {"t", "l"}
                else Path(command[2])
            )
            relative = archive.relative_to(archive_root).as_posix()
            if operation == "t":
                return subprocess.CompletedProcess(
                    command, 0, stdout="", stderr=""
                )
            if operation == "l":
                output = "\n".join(
                    f"Path = {member}"
                    for member in archive_members[relative]
                )
                return subprocess.CompletedProcess(
                    command, 0, stdout=output, stderr=""
                )
            output_arg = next(
                item for item in command if item.startswith("-o")
            )
            output_root = Path(output_arg[2:])
            for member in archive_members[relative]:
                target = output_root / member
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"annotation")
            return subprocess.CompletedProcess(
                command, 0, stdout="", stderr=""
            )

    shared = extract_shared_annotations(
        entries,
        archive_root,
        work_root,
        runner=NestedRunner(),
    )

    assert (shared / "30FPS/cameras/session_30_subj01.json").is_file()
    assert (shared / "60FPS/cameras/session_60_subj01.json").is_file()
    assert (shared / "30FPS/session_list.txt").is_file()
    assert (shared / "60FPS/session_list.txt").is_file()

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from gymnastics.benchmarks.freeman.download import (
    download_release,
    fetch_hub_inventory,
    run_preflight,
    validate_downloads,
)


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

"""Out-of-fold run audit and publication contracts."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Literal, Mapping

import numpy as np
import torch

from gymnastics.fusion.rotation_aware.dataset import build_split_manifest

from .cohorts import sha256_file


@dataclass(frozen=True)
class OOFRun:
    """One trained model and its test-only inference source."""

    outer_fold: int
    run_id: str
    seed: int
    checkpoint: Path
    split_manifest: Path
    inference_root: Path


@dataclass(frozen=True)
class OOFCycle:
    """One validated cycle eligible for the immutable OOF publication."""

    person_id: str
    cohort: Literal["elderly", "student"]
    outer_fold: int
    run_id: str
    seed: int
    cycle_id: str
    sequence_path: Path
    metadata_path: Path
    checkpoint_sha256: str
    split_hash: str
    cache_manifest_hash: str
    sequence_sha256: str
    face_map_sha256: str
    side_map_sha256: str


def semantic_split_hash(path: str | Path) -> str:
    """Hash split membership using the rotation-aware provenance contract."""
    manifest = build_split_manifest(path)
    encoded = json.dumps(
        asdict(manifest),
        sort_keys=True,
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def collect_oof_cycles(
    runs: list[OOFRun],
    cohort_by_person: Mapping[str, str],
    *,
    expected_people: set[str],
    expected_cycles: int,
) -> tuple[list[OOFCycle], dict[str, Any]]:
    """Validate runs and return publishable test cycles."""
    cycles: list[OOFCycle] = []
    assigned_people: dict[str, int] = {}
    ignored_source_people: dict[str, list[str]] = {}
    run_summaries: list[dict[str, Any]] = []

    for run in sorted(runs, key=lambda item: item.outer_fold):
        split = build_split_manifest(run.split_manifest)
        split_hash = semantic_split_hash(run.split_manifest)
        checkpoint_hash = sha256_file(run.checkpoint)
        checkpoint = torch.load(
            run.checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"checkpoint is not a mapping: {run.checkpoint}")
        training = checkpoint.get("training_config")
        provenance = checkpoint.get("provenance")
        if not isinstance(training, Mapping) or not isinstance(
            provenance, Mapping
        ):
            raise ValueError(f"checkpoint metadata is incomplete: {run.run_id}")
        _reject_forbidden_training_dependencies(
            {"training_config": training, "provenance": provenance},
            run_id=run.run_id,
        )
        if provenance.get("split_hash") != split_hash:
            raise ValueError(
                f"checkpoint split hash mismatch for {run.run_id}"
            )
        if str(training.get("ablation")) != "A6":
            raise ValueError(f"OOF checkpoint is not A6: {run.run_id}")
        if int(training.get("seed", -1)) != run.seed:
            raise ValueError(f"checkpoint seed mismatch for {run.run_id}")

        run_metadata_path = run.checkpoint.parents[1] / "run_metadata.json"
        run_metadata = _load_json(run_metadata_path)
        if run_metadata.get("no_pseudo_gt_training") is not True:
            raise ValueError(
                f"run lacks no_pseudo_gt_training attestation: {run.run_id}"
            )

        cache_manifests = provenance.get("cache_manifests")
        if not isinstance(cache_manifests, Mapping):
            raise ValueError(
                f"checkpoint cache manifests are missing: {run.run_id}"
            )

        source_people = {
            path.name.removeprefix("person_")
            for path in run.inference_root.glob("person_*")
            if path.is_dir()
        }
        ignored_source_people[run.run_id] = sorted(
            source_people - set(split.test),
            key=_person_sort_key,
        )

        for person_id in split.test:
            if person_id in assigned_people:
                raise ValueError(
                    "duplicate OOF person "
                    f"{person_id} in folds {assigned_people[person_id]} "
                    f"and {run.outer_fold}"
                )
            assigned_people[person_id] = run.outer_fold
            cohort = cohort_by_person.get(person_id)
            if cohort not in {"elderly", "student"}:
                raise ValueError(f"missing cohort for OOF person {person_id}")
            cache_identity = cache_manifests.get(person_id)
            if not isinstance(cache_identity, Mapping):
                raise ValueError(
                    f"checkpoint cache identity missing for person {person_id}"
                )
            expected_trials = cache_identity.get("trials")
            if not isinstance(expected_trials, list) or not all(
                isinstance(item, str) for item in expected_trials
            ):
                raise ValueError(
                    f"invalid cache trial list for person {person_id}"
                )
            person_root = run.inference_root / f"person_{person_id}"
            actual_trials = {
                path.name
                for path in person_root.glob("cycle_*")
                if path.is_dir()
            }
            if actual_trials != set(expected_trials):
                missing = sorted(set(expected_trials) - actual_trials)
                extra = sorted(actual_trials - set(expected_trials))
                raise ValueError(
                    f"inference cycle mismatch for person {person_id}: "
                    f"missing={missing}, extra={extra}"
                )
            for cycle_id in sorted(expected_trials):
                cycle_root = person_root / cycle_id
                sequence_path = cycle_root / "fused_sequence.npz"
                metadata_path = cycle_root / "metadata.json"
                if not sequence_path.is_file() or not metadata_path.is_file():
                    raise ValueError(
                        f"incomplete inference publication: {cycle_root}"
                    )
                metadata = _load_json(metadata_path)
                _validate_cycle_metadata(
                    metadata,
                    run=run,
                    person_id=person_id,
                    cycle_id=cycle_id,
                    checkpoint_hash=checkpoint_hash,
                    split_hash=split_hash,
                    cache_identity=cache_identity,
                )
                with np.load(sequence_path, allow_pickle=False) as archive:
                    if "face_map" not in archive or "side_map" not in archive:
                        raise ValueError(
                            f"frame maps missing from {sequence_path}"
                        )
                    face_map_hash = _array_hash(archive["face_map"])
                    side_map_hash = _array_hash(archive["side_map"])
                cycles.append(
                    OOFCycle(
                        person_id=person_id,
                        cohort=cohort,
                        outer_fold=run.outer_fold,
                        run_id=run.run_id,
                        seed=run.seed,
                        cycle_id=cycle_id,
                        sequence_path=sequence_path,
                        metadata_path=metadata_path,
                        checkpoint_sha256=checkpoint_hash,
                        split_hash=split_hash,
                        cache_manifest_hash=str(
                            cache_identity.get("manifest_hash", "")
                        ),
                        sequence_sha256=sha256_file(sequence_path),
                        face_map_sha256=face_map_hash,
                        side_map_sha256=side_map_hash,
                    )
                )
        run_summaries.append(
            {
                "outer_fold": run.outer_fold,
                "run_id": run.run_id,
                "seed": run.seed,
                "test_people": len(split.test),
                "checkpoint_sha256": checkpoint_hash,
                "split_hash": split_hash,
            }
        )

    actual_people = set(assigned_people)
    missing_people = sorted(
        expected_people - actual_people,
        key=_person_sort_key,
    )
    unexpected_people = sorted(
        actual_people - expected_people,
        key=_person_sort_key,
    )
    if missing_people:
        raise ValueError(f"missing OOF people: {missing_people}")
    if unexpected_people:
        raise ValueError(f"unexpected OOF people: {unexpected_people}")
    if len(cycles) != expected_cycles:
        raise ValueError(
            f"OOF cycle count mismatch: {len(cycles)} != {expected_cycles}"
        )

    cycles.sort(
        key=lambda item: (
            _person_sort_key(item.person_id),
            item.cycle_id,
        )
    )
    audit = {
        "schema_version": 1,
        "valid": True,
        "people": len(actual_people),
        "cycles": len(cycles),
        "folds": len(runs),
        "runs": run_summaries,
        "ignored_source_people": ignored_source_people,
        "duplicate_people": [],
        "missing_people": [],
        "unexpected_people": [],
    }
    return cycles, audit


def publish_oof_cycles(
    cycles: list[OOFCycle],
    audit: Mapping[str, Any],
    output_root: str | Path,
) -> None:
    """Publish validated cycles and their provenance."""
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"OOF publication already exists: {output}")
    staging = output.with_name(output.name + ".tmp")
    if staging.exists():
        raise FileExistsError(f"OOF staging path already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        rows: list[dict[str, object]] = []
        for cycle in cycles:
            target = (
                staging
                / f"person_{cycle.person_id}"
                / cycle.cycle_id
            )
            target.mkdir(parents=True)
            sequence_target = target / "prediction.npz"
            metadata_target = target / "source_metadata.json"
            _link_or_copy(cycle.sequence_path, sequence_target)
            _link_or_copy(cycle.metadata_path, metadata_target)
            row = {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(cycle).items()
                if key not in {"sequence_path", "metadata_path"}
            }
            row["prediction_path"] = str(
                sequence_target.relative_to(staging)
            )
            row["source_metadata_path"] = str(
                metadata_target.relative_to(staging)
            )
            rows.append(row)

        fieldnames = list(rows[0]) if rows else []
        with (staging / "oof_provenance.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        (staging / "oof_audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        publication_manifest = {
            "schema_version": 1,
            "people": int(audit["people"]),
            "cycles": int(audit["cycles"]),
            "provenance_sha256": sha256_file(
                staging / "oof_provenance.csv"
            ),
            "audit_sha256": sha256_file(staging / "oof_audit.json"),
            "predictions": {
                str(row["prediction_path"]): str(row["sequence_sha256"])
                for row in rows
            },
        }
        (staging / "oof_manifest.json").write_text(
            json.dumps(
                publication_manifest,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _validate_cycle_metadata(
    metadata: Mapping[str, Any],
    *,
    run: OOFRun,
    person_id: str,
    cycle_id: str,
    checkpoint_hash: str,
    split_hash: str,
    cache_identity: Mapping[str, Any],
) -> None:
    expected = {
        "ablation": "A6",
        "checkpoint_sha256": checkpoint_hash,
        "no_pseudo_gt_training": True,
        "person_id": person_id,
        "run_id": run.run_id,
        "split_hash": split_hash,
        "trial_id": cycle_id,
    }
    mismatched = [
        key for key, value in expected.items() if metadata.get(key) != value
    ]
    if mismatched:
        raise ValueError(
            f"inference provenance mismatch for {person_id}/{cycle_id}: "
            + ", ".join(mismatched)
        )
    if metadata.get("consumed_cache_manifest") != dict(cache_identity):
        raise ValueError(
            f"cache manifest mismatch for {person_id}/{cycle_id}"
        )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required provenance file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON provenance must be a mapping: {path}")
    return value


def _array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(json.dumps(list(contiguous.shape)).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copy2(source, target)


def _person_sort_key(person_id: str) -> tuple[int, str]:
    return (
        int(person_id) if person_id.isdigit() else 2**31 - 1,
        person_id,
    )


def _reject_forbidden_training_dependencies(
    value: object,
    *,
    run_id: str,
    path: tuple[str, ...] = (),
) -> None:
    forbidden = ("triangulated", "pseudo_reference", "pseudo_target", "pseudo_gt")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            current_path = (*path, str(key))
            if any(marker in key_text for marker in forbidden):
                raise ValueError(
                    "forbidden training dependency in "
                    f"{run_id}: {'.'.join(current_path)}"
                )
            _reject_forbidden_training_dependencies(
                nested,
                run_id=run_id,
                path=current_path,
            )
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_forbidden_training_dependencies(
                nested,
                run_id=run_id,
                path=(*path, str(index)),
            )

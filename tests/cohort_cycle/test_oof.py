from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gymnastics.analysis.cohort_cycle.oof import (
    OOFRun,
    collect_oof_cycles,
    publish_oof_cycles,
    semantic_split_hash,
)


def _cache_identity(person_id: str, trial: str) -> dict[str, object]:
    return {
        "config_hash": "config",
        "generation": f"generation_{person_id}",
        "layout": "immutable_generation",
        "manifest_hash": f"manifest_{person_id}",
        "person_id": person_id,
        "source_hash": f"source_{person_id}",
        "trials": [trial],
    }


def _make_run(
    root: Path,
    *,
    fold: int,
    person_id: str,
    run_id: str,
) -> OOFRun:
    run_root = root / "runs" / run_id
    inference_root = root / "inference" / run_id
    checkpoint = run_root / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    split = root / f"fold_{fold:02d}.json"
    split.write_text(
        json.dumps(
            {
                "train": [f"train_{fold}"],
                "val": [f"val_{fold}"],
                "test": [person_id],
            }
        ),
        encoding="utf-8",
    )
    split_hash = semantic_split_hash(split)
    cache_identity = _cache_identity(person_id, "cycle_000")
    torch.save(
        {
            "training_config": {
                "ablation": "A6",
                "seed": 0,
                "epochs": 100,
            },
            "provenance": {
                "split_hash": split_hash,
                "cache_manifests": {person_id: cache_identity},
            },
        },
        checkpoint,
    )
    (run_root / "run_metadata.json").write_text(
        json.dumps({"no_pseudo_gt_training": True}),
        encoding="utf-8",
    )
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    cycle_root = inference_root / f"person_{person_id}" / "cycle_000"
    cycle_root.mkdir(parents=True)
    np.savez_compressed(
        cycle_root / "fused_sequence.npz",
        kpts_body=np.zeros((64, 70, 3), dtype=np.float32),
        face_map=np.arange(64, dtype=np.int64),
        side_map=np.arange(64, dtype=np.int64) + 1,
    )
    (cycle_root / "metadata.json").write_text(
        json.dumps(
            {
                "ablation": "A6",
                "checkpoint_sha256": checkpoint_hash,
                "consumed_cache_manifest": cache_identity,
                "no_pseudo_gt_training": True,
                "person_id": person_id,
                "run_id": run_id,
                "split_hash": split_hash,
                "trial_id": "cycle_000",
            }
        ),
        encoding="utf-8",
    )
    return OOFRun(
        outer_fold=fold,
        run_id=run_id,
        seed=0,
        checkpoint=checkpoint,
        split_manifest=split,
        inference_root=inference_root,
    )


def test_collect_oof_cycles_accepts_only_test_people_and_preserves_provenance(
    tmp_path: Path,
):
    """Publishing a non-test person or losing frame maps would invalidate OOF."""
    runs = [
        _make_run(tmp_path, fold=0, person_id="1", run_id="run0"),
        _make_run(tmp_path, fold=1, person_id="81", run_id="run1"),
    ]

    cycles, audit = collect_oof_cycles(
        runs,
        {"1": "elderly", "81": "student"},
        expected_people={"1", "81"},
        expected_cycles=2,
    )

    assert [(cycle.person_id, cycle.outer_fold) for cycle in cycles] == [
        ("1", 0),
        ("81", 1),
    ]
    assert cycles[0].face_map_sha256 != cycles[0].side_map_sha256
    assert audit["people"] == 2
    assert audit["cycles"] == 2
    assert audit["valid"] is True


def test_collect_oof_cycles_rejects_checkpoint_split_mismatch(tmp_path: Path):
    """A checkpoint trained under another split must never publish this fold."""
    run = _make_run(tmp_path, fold=0, person_id="1", run_id="run0")
    payload = torch.load(run.checkpoint, map_location="cpu", weights_only=False)
    payload["provenance"]["split_hash"] = "wrong"
    torch.save(payload, run.checkpoint)

    with pytest.raises(ValueError, match="split hash"):
        collect_oof_cycles(
            [run],
            {"1": "elderly"},
            expected_people={"1"},
            expected_cycles=1,
        )


def test_collect_oof_cycles_rejects_triangulated_training_dependency(
    tmp_path: Path,
):
    """A checkpoint declaring pseudo-reference input is ineligible for OOF."""
    run = _make_run(tmp_path, fold=0, person_id="1", run_id="run0")
    payload = torch.load(run.checkpoint, map_location="cpu", weights_only=False)
    payload["provenance"]["triangulated_root"] = "/forbidden/reference"
    torch.save(payload, run.checkpoint)

    with pytest.raises(ValueError, match="forbidden training dependency"):
        collect_oof_cycles(
            [run],
            {"1": "elderly"},
            expected_people={"1"},
            expected_cycles=1,
        )


def test_collect_oof_cycles_rejects_duplicate_or_missing_people(tmp_path: Path):
    """Each participant must have exactly one outer-test publication."""
    run0 = _make_run(tmp_path, fold=0, person_id="1", run_id="run0")
    run1 = _make_run(tmp_path, fold=1, person_id="1", run_id="run1")

    with pytest.raises(ValueError, match="duplicate OOF person"):
        collect_oof_cycles(
            [run0, run1],
            {"1": "elderly"},
            expected_people={"1"},
            expected_cycles=2,
        )

    with pytest.raises(ValueError, match="missing OOF people"):
        collect_oof_cycles(
            [run0],
            {"1": "elderly", "81": "student"},
            expected_people={"1", "81"},
            expected_cycles=1,
        )


def test_publish_oof_cycles_writes_test_only_tree_and_provenance(tmp_path: Path):
    """The publication must be traceable without consulting source directories."""
    run = _make_run(tmp_path, fold=0, person_id="1", run_id="run0")
    cycles, audit = collect_oof_cycles(
        [run],
        {"1": "elderly"},
        expected_people={"1"},
        expected_cycles=1,
    )
    output = tmp_path / "oof_seed0"

    publish_oof_cycles(cycles, audit, output)

    assert (
        output / "person_1" / "cycle_000" / "prediction.npz"
    ).is_file()
    rows = list(
        csv.DictReader(
            (output / "oof_provenance.csv").open(encoding="utf-8")
        )
    )
    assert len(rows) == 1
    assert rows[0]["person_id"] == "1"
    assert rows[0]["cohort"] == "elderly"
    assert rows[0]["outer_fold"] == "0"
    saved_audit = json.loads(
        (output / "oof_audit.json").read_text(encoding="utf-8")
    )
    assert saved_audit["valid"] is True

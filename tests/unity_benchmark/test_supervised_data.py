from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gymnastics.benchmarks.unity.dataset import load_unity_benchmark
from gymnastics.benchmarks.unity.supervised_data import (
    UNITY_SUPERVISED_FOLDS,
    audit_fold_isolation,
    build_supervised_sequences,
    select_supervised_fold,
)


UNITY_ROOT = Path("/home/data/xchen/gymnastics/unity_benchmark")
SAM3D_ROOT = Path("local/runs/unity_benchmark/sam3d")
SKELETON = Path("configs/fusion/skeleton_mhr70.yaml")


def test_direction_folds_are_exact_and_static_is_evaluation_only() -> None:
    assert UNITY_SUPERVISED_FOLDS["left_to_right"].train_sequence == (
        "continuous_left_060_r00"
    )
    assert UNITY_SUPERVISED_FOLDS["left_to_right"].test_sequence == (
        "continuous_right_060_r00"
    )
    assert UNITY_SUPERVISED_FOLDS["right_to_left"].train_sequence == (
        "continuous_right_060_r00"
    )
    assert UNITY_SUPERVISED_FOLDS["right_to_left"].test_sequence == (
        "continuous_left_060_r00"
    )


def _real_sequences():
    benchmark = load_unity_benchmark(UNITY_ROOT)
    return build_supervised_sequences(
        benchmark,
        SAM3D_ROOT,
        skeleton_path=SKELETON,
        fps=60.0,
    )


def test_real_unity_fold_has_no_sample_or_sequence_leakage() -> None:
    sequences = _real_sequences()
    fold = UNITY_SUPERVISED_FOLDS["left_to_right"]
    train, test, static = select_supervised_fold(sequences, fold)

    audit_fold_isolation(fold, train, test, static)
    assert len(train.sample_ids) == 97
    assert len(test.sample_ids) == 97
    assert len(static.sample_ids) == 5
    assert not set(train.sample_ids) & set(test.sample_ids)
    assert not set(train.sample_ids) & set(static.sample_ids)


def test_fold_audit_rejects_sample_leakage() -> None:
    sequences = _real_sequences()
    fold = UNITY_SUPERVISED_FOLDS["left_to_right"]
    train, test, static = select_supervised_fold(sequences, fold)
    leaked_ids = np.array(test.sample_ids, copy=True)
    leaked_ids[0] = train.sample_ids[0]
    leaked_test = replace(test, sample_ids=leaked_ids)

    with pytest.raises(ValueError, match="sample leakage"):
        audit_fold_isolation(fold, train, leaked_test, static)


def test_sequence_filtered_loader_materializes_only_training_direction() -> None:
    benchmark = load_unity_benchmark(
        UNITY_ROOT,
        sequence_ids=("continuous_left_060_r00",),
    )

    assert len(benchmark.frames) == 97
    assert {
        frame.sequence_id for frame in benchmark.frames
    } == {"continuous_left_060_r00"}

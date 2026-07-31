from __future__ import annotations

import numpy as np

from gymnastics.analysis.cohort_cycle.phase_statistics import (
    cluster_permutation_test,
)


def test_cluster_permutation_finds_contiguous_person_level_effect():
    """Cycle-level shuffling or pointwise tests would misstate phase evidence."""
    rng = np.random.default_rng(3)
    people = np.array(
        [f"s{i}" for i in range(10)] + [f"e{i}" for i in range(10)]
    )
    cohorts = np.array(["student"] * 10 + ["elderly"] * 10)
    curves = rng.normal(0.0, 0.15, size=(20, 101))
    curves[10:, 35:56] += 1.2

    clusters = cluster_permutation_test(
        curves,
        people,
        cohorts,
        permutations=499,
        seed=11,
    )

    assert clusters
    strongest = min(clusters, key=lambda item: item["p_value"])
    assert strongest["start_phase"] <= 0.35
    assert strongest["end_phase"] >= 0.55
    assert strongest["p_value"] < 0.05

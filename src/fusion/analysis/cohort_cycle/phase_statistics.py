"""Phase-resolved cluster permutation contracts."""

from __future__ import annotations

import numpy as np
from scipy import stats


def cluster_permutation_test(
    curves: np.ndarray,
    person_ids: np.ndarray,
    cohorts: np.ndarray,
    *,
    permutations: int,
    seed: int,
    alpha: float = 0.05,
) -> list[dict[str, float | int]]:
    """Two-sided max-cluster permutation with person-level label exchange."""
    values = np.asarray(curves, dtype=np.float64)
    people = np.asarray(person_ids).astype(str)
    labels = np.asarray(cohorts).astype(str)
    if values.ndim != 2 or len(values) != len(people) or len(values) != len(
        labels
    ):
        raise ValueError("phase arrays have incompatible shapes")
    unique_people = np.unique(people)
    person_curves = []
    person_labels = []
    for person_id in unique_people:
        selected = people == person_id
        person_curves.append(np.median(values[selected], axis=0))
        unique_labels = np.unique(labels[selected])
        if len(unique_labels) != 1:
            raise ValueError(f"person {person_id} has conflicting cohorts")
        person_labels.append(unique_labels[0])
    person_values = np.stack(person_curves)
    person_labels_array = np.asarray(person_labels)
    if set(person_labels_array) != {"elderly", "student"}:
        raise ValueError("phase test requires elderly and student cohorts")

    elderly = person_values[person_labels_array == "elderly"]
    student = person_values[person_labels_array == "student"]
    degrees = len(elderly) + len(student) - 2
    threshold = float(stats.t.ppf(1.0 - alpha / 2.0, degrees))
    observed_t = _welch_t(elderly, student)
    observed_clusters = _clusters(observed_t, threshold)
    if not observed_clusters:
        return []

    rng = np.random.default_rng(seed)
    null_max = np.zeros(permutations, dtype=np.float64)
    for permutation in range(permutations):
        shuffled = rng.permutation(person_labels_array)
        permuted_t = _welch_t(
            person_values[shuffled == "elderly"],
            person_values[shuffled == "student"],
        )
        clusters = _clusters(permuted_t, threshold)
        null_max[permutation] = max(
            (cluster[2] for cluster in clusters),
            default=0.0,
        )

    denominator = float(values.shape[1] - 1)
    results = []
    for start, end, mass in observed_clusters:
        p_value = (1 + int(np.sum(null_max >= mass))) / float(
            permutations + 1
        )
        results.append(
            {
                "start_index": start,
                "end_index": end,
                "start_phase": start / denominator,
                "end_phase": end / denominator,
                "cluster_mass": mass,
                "p_value": p_value,
                "direction": int(np.sign(np.mean(observed_t[start : end + 1]))),
            }
        )
    return results


def _welch_t(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    statistic = stats.ttest_ind(
        first,
        second,
        axis=0,
        equal_var=False,
        nan_policy="omit",
    ).statistic
    return np.nan_to_num(statistic, nan=0.0, posinf=0.0, neginf=0.0)


def _clusters(
    statistic: np.ndarray,
    threshold: float,
) -> list[tuple[int, int, float]]:
    active = np.abs(statistic) > threshold
    starts = np.flatnonzero(active & np.r_[True, ~active[:-1]])
    ends = np.flatnonzero(active & np.r_[~active[1:], True])
    return [
        (
            int(start),
            int(end),
            float(np.sum(np.abs(statistic[start : end + 1]))),
        )
        for start, end in zip(starts, ends, strict=True)
    ]

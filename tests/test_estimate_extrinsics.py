import cv2
import numpy as np

from gymnastics.triangulation.estimate_extrinsics import (
    chordal_mean,
    cluster_rigs,
    consensus_per_cluster,
    estimate_relative_pose,
    geodesic_deg,
    kabsch,
    metric_scale,
    reprojection_error,
    rotation_angle_deg,
)


def _rz(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


def _calib(fx=1710.0, cx=540.0, cy=960.0):
    return {
        "K": np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]]),
        "dist": np.zeros((5,), dtype=np.float64),
    }


def _project(points_3d, calib, R, t):
    rvec, _ = cv2.Rodrigues(R)
    pts, _ = cv2.projectPoints(
        points_3d, rvec, t.reshape(3, 1), calib["K"], calib["dist"]
    )
    return pts.reshape(-1, 2)


def _synthetic_pair(num_frames=16, num_joints=40, angle_deg=87.0, baseline=3.4, seed=0):
    """Two views of a moving point cloud under a known relative pose."""
    rng = np.random.default_rng(seed)
    R = _rz(angle_deg)
    t = R @ (-np.array([baseline, 0.0, 0.0]))
    frames = []
    for _ in range(num_frames):
        centre = np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.3, 0.3), rng.uniform(3.0, 4.0)])
        pts = centre + rng.uniform(-0.5, 0.5, size=(num_joints, 3))
        frames.append(pts)
    return R, t, frames


def test_rotation_helpers_are_consistent():
    R = _rz(87.0)
    assert rotation_angle_deg(R) == 87.0
    assert geodesic_deg(R, R) == 0.0
    # A rotation and its transpose differ by twice the rotation angle.
    assert np.isclose(geodesic_deg(R, R.T), 174.0)
    assert np.allclose(chordal_mean([R, R]), R)


def test_kabsch_recovers_a_similarity_transform():
    rng = np.random.default_rng(3)
    P = rng.normal(size=(30, 3))
    R_true, s_true, t_true = _rz(40.0), 1.7, np.array([0.4, -1.2, 3.0])
    Q = s_true * (P @ R_true.T) + t_true

    R, s, t = kabsch(P, Q)
    assert np.allclose(R, R_true, atol=1e-8)
    assert np.isclose(s, s_true)
    assert np.allclose(t, t_true, atol=1e-8)


def test_estimate_relative_pose_recovers_a_known_rig():
    calib_a, calib_b = _calib(), _calib(fx=1709.0, cx=588.0, cy=865.0)
    R_true, t_true, frames = _synthetic_pair()

    k2a = np.stack([_project(f, calib_a, np.eye(3), np.zeros(3)) for f in frames])
    k2b = np.stack([_project(f, calib_b, R_true, t_true) for f in frames])

    R, unit_t, inlier_ratio = estimate_relative_pose(k2a, k2b, calib_a, calib_b, 1.5)
    assert geodesic_deg(R, R_true) < 1.0
    assert inlier_ratio > 0.9
    # The essential matrix fixes direction but not length.
    assert np.allclose(unit_t, t_true / np.linalg.norm(t_true), atol=1e-2)


def test_metric_scale_recovers_the_baseline_from_monocular_3d():
    R_true, t_true, frames = _synthetic_pair()
    x3a = np.stack(frames)
    x3b = np.stack([f @ R_true.T + t_true for f in frames])
    assert np.isclose(metric_scale(x3a, x3b), np.linalg.norm(t_true), rtol=1e-6)


def test_reprojection_error_detects_a_wrong_rotation():
    calib_a, calib_b = _calib(), _calib(fx=1709.0)
    R_true, t_true, frames = _synthetic_pair()
    k2a = np.stack([_project(f, calib_a, np.eye(3), np.zeros(3)) for f in frames])
    k2b = np.stack([_project(f, calib_b, R_true, t_true) for f in frames])

    assert reprojection_error(R_true, t_true, k2a, k2b, calib_a, calib_b) < 1e-3
    wrong_R = _rz(75.0)
    assert reprojection_error(wrong_R, t_true, k2a, k2b, calib_a, calib_b) > 10.0


def test_reprojection_error_is_blind_to_baseline_scale():
    """Two-view geometry is scale-free, so this metric cannot police metric scale.

    Scaling the baseline scales the whole reconstruction with it and leaves every
    projection untouched.  Metric scale therefore rests entirely on the monocular
    3D that :func:`metric_scale` reads, and has to be checked against something
    dimensional (bone lengths, subject height) instead.
    """
    calib_a, calib_b = _calib(), _calib(fx=1709.0)
    R_true, t_true, frames = _synthetic_pair()
    k2a = np.stack([_project(f, calib_a, np.eye(3), np.zeros(3)) for f in frames])
    k2b = np.stack([_project(f, calib_b, R_true, t_true) for f in frames])

    exact = reprojection_error(R_true, t_true, k2a, k2b, calib_a, calib_b)
    inflated = reprojection_error(R_true, t_true * 1.5, k2a, k2b, calib_a, calib_b)
    assert np.isclose(exact, inflated, atol=1e-9)


def test_cluster_rigs_separates_the_two_mirror_configurations():
    left = {str(i): _rz(88.0 + i) for i in range(6)}
    right = {str(10 + i): _rz(88.0 + i).T for i in range(3)}
    labels, reference = cluster_rigs({**left, **right})

    # Cluster 0 is the majority configuration by convention.
    assert {labels[p] for p in left} == {0}
    assert {labels[p] for p in right} == {1}
    assert abs(rotation_angle_deg(reference) - 90.5) < 3.0


def test_consensus_per_cluster_ignores_members_without_a_finite_score():
    def entry(angle, baseline, holdout):
        R = _rz(angle)
        return {
            "R": R,
            "t": np.array([baseline, 0.0, 0.0]),
            "baseline_m": baseline,
            "holdout_px": holdout,
        }

    results = {
        "1": entry(87.0, 3.4, 5.0),
        "2": entry(88.0, 3.5, 6.0),
        "3": entry(89.0, 3.6, 7.0),
        "4": entry(200.0, 40.0, float("nan")),
    }
    labels = {"1": 0, "2": 0, "3": 0, "4": 0}

    clusters = consensus_per_cluster(results, labels)
    assert set(clusters) == {0}
    assert abs(rotation_angle_deg(clusters[0]["R"]) - 88.0) < 1.0
    assert abs(np.linalg.norm(clusters[0]["t"]) - 3.5) < 0.1


def test_consensus_per_cluster_skips_clusters_that_are_too_small():
    results = {
        "1": {
            "R": _rz(87.0),
            "t": np.array([3.4, 0.0, 0.0]),
            "baseline_m": 3.4,
            "holdout_px": 5.0,
        }
    }
    assert consensus_per_cluster(results, {"1": 0}) == {}

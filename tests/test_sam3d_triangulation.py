import cv2
import numpy as np

from triangulation.sam3d_from_split_cycle import triangulate_keypoints


def _project(points_3d, K, R, t, dist=None):
    rvec, _ = cv2.Rodrigues(R)
    pts, _ = cv2.projectPoints(
        points_3d.astype(np.float32),
        rvec,
        t.reshape(3, 1).astype(np.float32),
        K.astype(np.float32),
        np.zeros((5,), dtype=np.float32) if dist is None else dist.astype(np.float32),
    )
    return pts.reshape(-1, 2)


def test_triangulate_keypoints_uses_per_camera_intrinsics():
    face_K = np.array([[1200.0, 0.0, 540.0], [0.0, 1210.0, 960.0], [0.0, 0.0, 1.0]])
    side_K = np.array([[900.0, 0.0, 520.0], [0.0, 920.0, 930.0], [0.0, 0.0, 1.0]])
    dist = np.zeros((5,), dtype=np.float32)

    face_rt = {"R": np.eye(3, dtype=np.float32), "t": np.zeros(3, dtype=np.float32)}
    side_rt = {
        "R": np.eye(3, dtype=np.float32),
        "t": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    }

    points_3d = np.array(
        [[0.0, 0.0, 4.0], [0.3, -0.2, 5.0], [-0.4, 0.1, 6.0]],
        dtype=np.float32,
    )
    face_2d = _project(points_3d, face_K, face_rt["R"], face_rt["t"], dist)
    side_2d = _project(points_3d, side_K, side_rt["R"], side_rt["t"], dist)

    reconstructed = triangulate_keypoints(
        face_2d,
        side_2d,
        face_calib={"K": face_K, "dist": dist},
        side_calib={"K": side_K, "dist": dist},
        face_rt=face_rt,
        side_rt=side_rt,
    )

    assert np.allclose(reconstructed, points_3d, atol=1e-4)

from __future__ import annotations

import numpy as np

from gymnastics.benchmarks.freeman.mapping import (
    FREEMAN_COCO17_NAMES,
    MAPPING_VERSION,
    map_mhr70_to_freeman,
)
from gymnastics.common.skeletons.mhr70 import MHR70_INDEX


def test_maps_exact_coco17_names_in_official_order() -> None:
    points = np.zeros((2, 70, 3), dtype=np.float32)
    for index in range(70):
        points[:, index] = index + 1

    mapped = map_mhr70_to_freeman(points)

    assert mapped.points.shape == (2, 17, 3)
    assert mapped.joint_names == FREEMAN_COCO17_NAMES
    assert MAPPING_VERSION == "mhr70_to_freeman_coco17_v1"
    for target_index, name in enumerate(FREEMAN_COCO17_NAMES):
        np.testing.assert_allclose(
            mapped.points[:, target_index],
            points[:, MHR70_INDEX[name]],
        )
    np.testing.assert_allclose(
        mapped.points[:, FREEMAN_COCO17_NAMES.index("left-wrist")],
        points[:, 62],
    )
    np.testing.assert_allclose(
        mapped.points[:, FREEMAN_COCO17_NAMES.index("right-wrist")],
        points[:, 41],
    )


def test_mapping_propagates_only_corresponding_joint_validity() -> None:
    points = np.ones((1, 70, 3), dtype=np.float32)
    valid = np.ones((1, 70), dtype=bool)
    valid[:, MHR70_INDEX["left-wrist"]] = False

    mapped = map_mhr70_to_freeman(points, valid)

    left_wrist = FREEMAN_COCO17_NAMES.index("left-wrist")
    assert not mapped.valid[0, left_wrist]
    np.testing.assert_array_equal(mapped.points[0, left_wrist], np.zeros(3))
    assert mapped.valid.sum() == 16


def test_mapping_treats_nonfinite_prediction_as_invalid() -> None:
    points = np.ones((1, 70, 3), dtype=np.float32)
    points[0, MHR70_INDEX["nose"], 0] = np.nan

    mapped = map_mhr70_to_freeman(points)

    assert not mapped.valid[0, 0]
    np.testing.assert_array_equal(mapped.points[0, 0], np.zeros(3))

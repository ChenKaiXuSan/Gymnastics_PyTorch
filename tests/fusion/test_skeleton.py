from __future__ import annotations

import pytest

from common.skeletons.mhr70 import MHR70_MAJOR_JOINT_INDICES, mhr_names
from fusion.skeleton import build_common_skeleton


def test_full_skeleton_matches_mhr70():
    skeleton = build_common_skeleton("mhr70")
    assert skeleton.num_joints == 70
    assert skeleton.joint_names == tuple(mhr_names)
    assert skeleton.source_indices == tuple(range(70))
    assert skeleton.joint_names[skeleton.left_hip_index] == "left-hip"
    assert skeleton.joint_names[skeleton.right_hip_index] == "right-hip"


def test_major_skeleton_selects_major_joints():
    skeleton = build_common_skeleton("mhr70_major")
    assert skeleton.source_indices == tuple(MHR70_MAJOR_JOINT_INDICES)
    assert all("thumb" not in name for name in skeleton.joint_names)
    # Every left joint has a right partner and vice versa.
    for left, right in skeleton.left_right_pairs:
        assert skeleton.joint_names[left].replace("left-", "right-") == skeleton.joint_names[right]
    assert len(skeleton.left_right_pairs) == 9


def test_bones_reference_valid_joints_and_have_mirrors():
    skeleton = build_common_skeleton("mhr70_major")
    for start, end in skeleton.bones:
        assert 0 <= start < skeleton.num_joints and 0 <= end < skeleton.num_joints
    pairs = skeleton.bilateral_bone_pairs()
    assert pairs
    for left_bone, right_bone in pairs:
        left_names = tuple(skeleton.joint_names[i] for i in skeleton.bones[left_bone])
        right_names = tuple(skeleton.joint_names[i] for i in skeleton.bones[right_bone])
        assert tuple(n.replace("left-", "right-") for n in left_names) == right_names


def test_distal_joints_are_hands_and_feet():
    skeleton = build_common_skeleton("mhr70")
    names = {skeleton.joint_names[i] for i in skeleton.distal_indices}
    assert "left-wrist" in names and "right-heel" in names and "left-index-tip" in names
    assert "neck" not in names and "left-hip" not in names


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        build_common_skeleton("nope")

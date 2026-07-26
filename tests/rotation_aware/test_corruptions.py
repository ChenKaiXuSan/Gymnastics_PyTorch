import json
from pathlib import Path

import pytest
import torch

from gymnastics.fusion.rotation_aware.corruptions import (
    CorruptionConfig,
    apply_corruptions,
    write_corruption_manifest,
)
from gymnastics.fusion.rotation_aware.config import RoleSpec, SkeletonSpec, load_skeleton_spec


def _inputs():
    values = torch.arange(12 * 4 * 3, dtype=torch.float32).reshape(12, 4, 3) + 1
    return values, values + 100, torch.ones(12, 4, dtype=torch.bool), torch.ones(12, 4, dtype=torch.bool)


def _skeleton() -> SkeletonSpec:
    return SkeletonSpec(
        name="test",
        joint_names=("left-thorax", "right-thorax", "left-fallback", "right-fallback"),
        bones=(),
        roles={
            "thorax": RoleSpec(
                kind="midpoint",
                joints=("left-thorax", "right-thorax"),
                fallback=("left-fallback", "right-fallback"),
            )
        },
        required_roles=("thorax",),
        joint_groups={
            "upper_body": (
                "left-thorax",
                "right-thorax",
                "left-fallback",
                "right-fallback",
            )
        },
    )


def _joint_fallback_skeleton() -> SkeletonSpec:
    return SkeletonSpec(
        name="test-joint-fallback",
        joint_names=("thorax", "other", "fallback-thorax", "other-fallback"),
        bones=(),
        roles={"thorax": RoleSpec(kind="joint", joints=("thorax",), fallback=("fallback-thorax",))},
        required_roles=("thorax",),
        joint_groups={
            "upper_body": (
                "thorax",
                "other",
                "fallback-thorax",
                "other-fallback",
            )
        },
    )


def test_corruption_is_reproducible_and_reference_is_unchanged():
    face, side, valid_face, valid_side = _inputs()
    before = face.clone()
    cfg = CorruptionConfig(enabled_families=("spike_noise",), spike_probability=1.0, spike_scale=3.0)

    a = apply_corruptions(face, side, valid_face, valid_side, seed=17, config=cfg)
    b = apply_corruptions(face, side, valid_face, valid_side, seed=17, config=cfg)

    torch.testing.assert_close(a.corrupted_face, b.corrupted_face)
    torch.testing.assert_close(a.reference_face, before)
    torch.testing.assert_close(face, before)
    assert a.face_corruption_mask.dtype is torch.bool
    assert a.face_corruption_mask.all()


@pytest.mark.parametrize(
    ("family", "config_kwargs"),
    [
        ("joint_dropout", {"joint_dropout_probability": 1.0}),
        ("temporal_block_dropout", {"temporal_block_probability": 1.0, "block_length": 12}),
        ("spike_noise", {"spike_probability": 1.0, "spike_scale": 2.0}),
        ("random_walk_drift", {"drift_probability": 1.0, "drift_scale": 0.5}),
        ("thorax_rotation_bias", {"rotation_probability": 1.0, "rotation_degrees": 30.0}),
        ("freeze_segment", {"freeze_probability": 1.0, "freeze_length": 12}),
        ("integer_time_shift", {"time_shift_probability": 1.0, "max_time_shift": 2}),
    ],
)
def test_each_corruption_family_marks_exactly_the_changed_valid_points(family, config_kwargs):
    face, side, valid_face, valid_side = _inputs()
    batch = apply_corruptions(
        face,
        side,
        valid_face,
        valid_side,
        seed=11,
        config=CorruptionConfig(enabled_families=(family,), **config_kwargs),
        skeleton=_skeleton() if family == "thorax_rotation_bias" else None,
    )

    face_changed = (batch.corrupted_face != batch.reference_face).any(dim=-1)
    side_changed = (batch.corrupted_side != batch.reference_side).any(dim=-1)
    assert torch.equal(batch.face_corruption_mask, face_changed)
    assert torch.equal(batch.side_corruption_mask, side_changed)
    assert batch.face_corruption_mask.any()
    assert batch.side_corruption_mask.any()
    assert not (batch.face_corruption_mask & ~batch.valid_face).any()
    assert not (batch.side_corruption_mask & ~batch.valid_side).any()


def test_fixed_evaluation_manifest_is_stable_and_json_serializable(tmp_path):
    path = tmp_path / "corruption_manifest.json"
    manifest = write_corruption_manifest(path, ["person_2/cycle_001/0", "person_1/cycle_000/32"], seed=9)

    assert manifest["seed"] == 9
    assert manifest["windows"] == {
        "person_1/cycle_000/32": manifest["windows"]["person_1/cycle_000/32"],
        "person_2/cycle_001/0": manifest["windows"]["person_2/cycle_001/0"],
    }
    assert json.loads(path.read_text(encoding="utf-8")) == manifest


def test_time_shift_leaves_reference_invalid_targets_and_masks_untouched():
    face, side, valid_face, valid_side = _inputs()
    valid_face[:, 0] = False
    valid_side[:, 0] = False
    batch = apply_corruptions(
        face,
        side,
        valid_face,
        valid_side,
        seed=5,
        config=CorruptionConfig(
            enabled_families=("integer_time_shift",), time_shift_probability=1.0, max_time_shift=2
        ),
    )

    torch.testing.assert_close(batch.corrupted_face[:, 0], face[:, 0])
    torch.testing.assert_close(batch.corrupted_side[:, 0], side[:, 0])
    assert not batch.face_corruption_mask[:, 0].any()
    assert not batch.side_corruption_mask[:, 0].any()


def test_default_corruptions_do_not_require_a_caller_supplied_thorax_index():
    face, side, valid_face, valid_side = _inputs()

    batch = apply_corruptions(face, side, valid_face, valid_side, seed=3)

    assert batch.corrupted_face.shape == face.shape


def test_thorax_rotation_resolves_virtual_midpoint_and_fallback_from_skeleton():
    face, side, valid_face, valid_side = _inputs()
    valid_face[1, :2] = False
    valid_side[1, :2] = False
    config = CorruptionConfig(
        enabled_families=("thorax_rotation_bias",), rotation_probability=1.0, rotation_degrees=30.0
    )
    batch = apply_corruptions(face, side, valid_face, valid_side, seed=2, config=config, skeleton=_skeleton())

    torch.testing.assert_close(
        batch.corrupted_face[0, :2].mean(dim=0), batch.reference_face[0, :2].mean(dim=0)
    )
    torch.testing.assert_close(
        batch.corrupted_face[1, 2:].mean(dim=0), batch.reference_face[1, 2:].mean(dim=0)
    )
    assert batch.face_corruption_mask[0, :2].any()
    assert batch.face_corruption_mask[1, 2:].any()


def test_thorax_rotation_uses_joint_role_fallback_from_skeleton():
    face, side, valid_face, valid_side = _inputs()
    valid_face[1, 0] = False
    valid_side[1, 0] = False
    config = CorruptionConfig(
        enabled_families=("thorax_rotation_bias",), rotation_probability=1.0, rotation_degrees=30.0
    )
    batch = apply_corruptions(
        face, side, valid_face, valid_side, seed=2, config=config, skeleton=_joint_fallback_skeleton()
    )

    torch.testing.assert_close(batch.corrupted_face[1, 2], batch.reference_face[1, 2])
    assert batch.face_corruption_mask[1, 3]


def test_thorax_rotation_moves_only_named_upper_body_about_local_y() -> None:
    skeleton = load_skeleton_spec(Path("configs/fusion/skeleton_mhr70.yaml"))
    joints = len(skeleton.joint_names)
    face = torch.arange(2 * joints * 3, dtype=torch.float32).reshape(2, joints, 3) + 1
    valid = torch.ones((2, joints), dtype=torch.bool)
    left_acromion = skeleton.joint_index("left-acromion")
    right_acromion = skeleton.joint_index("right-acromion")
    face[:, left_acromion] = torch.tensor([-1.0, 2.0, 0.0])
    face[:, right_acromion] = torch.tensor([1.0, 2.0, 0.0])
    upper_body = skeleton.joint_group("upper_body")
    lower_body = tuple(index for index in range(joints) if index not in upper_body)

    batch = apply_corruptions(
        face,
        face,
        valid,
        valid,
        seed=2,
        config=CorruptionConfig(
            enabled_families=("thorax_rotation_bias",),
            rotation_probability=1.0,
            rotation_degrees=30.0,
        ),
        skeleton=skeleton,
    )

    torch.testing.assert_close(batch.corrupted_face[:, lower_body], face[:, lower_body])
    torch.testing.assert_close(batch.corrupted_face[:, upper_body, 1], face[:, upper_body, 1])
    torch.testing.assert_close(
        batch.corrupted_face[:, [left_acromion, right_acromion]].mean(dim=1),
        face[:, [left_acromion, right_acromion]].mean(dim=1),
    )
    assert batch.face_corruption_mask[:, upper_body].any()
    assert not batch.face_corruption_mask[:, lower_body].any()

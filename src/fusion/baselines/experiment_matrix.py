"""Run the deterministic fusion experiment matrix and evaluate it against the
triangulated 3D keypoints.

The method implementations live in :mod:`fusion.baselines.methods`, the
loaders and writers in :mod:`fusion.baselines.data` and the metrics in
:mod:`fusion.baselines.evaluation`; this module wires them together per person
and exposes the ``python -m fusion deterministic`` command line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from fusion.baselines.data import (
    DEFAULT_EXTRINSICS_PATH,
    DEFAULT_OUT_DIR,
    DEFAULT_SAM3D_ROOT,
    DEFAULT_SKELETON_PATH,
    DEFAULT_SPLIT_ROOT,
    DEFAULT_TRIANGULATED_ROOT,
    build_aligned_timeline,
    iter_person_ids,
    load_aligned_cycle_cache,
    load_extrinsic_rotation,
    load_sam3d_world_by_frame,
    load_split_alignment_offset,
    save_compact_sequence,
    write_csv,
)
from fusion.baselines.evaluation import (
    ALIGNMENT_MODES,
    DEFAULT_ALIGNMENT,
    JointMetric,
    PersonMetric,
    estimate_weights_from_triangulated,
    evaluate_sequence,
)
from fusion.baselines.methods import (
    AVAILABLE_METHODS,
    BASELINE_METHODS,
    EXTRINSIC_METHODS,
    NO_EXTRINSIC_METHODS,
    STABLE_SIM3_JOINTS,
    align_side_with_extrinsic_rotation,
    bodypart_weights,
    current_body_average,
    fuse_extrinsic_rotation,
    fuse_quality_weighted,
    fuse_weighted,
    kpts_world_to_body,
    root_align_to_reference,
    rotation_aware_quality_scores,
    rotation_aware_quality_scores_by_trial,
    sim3_align_to_reference,
    smooth_sequence,
)
from fusion.baselines.save import save_fused_kpts


def process_person(
    person_id: str,
    methods: Sequence[str],
    sam3d_root: Path,
    triangulated_root: Path,
    split_root: Path,
    out_root: Path,
    save_frame_npz: bool,
    alignment: str = DEFAULT_ALIGNMENT,
    extrinsics_path: Path = DEFAULT_EXTRINSICS_PATH,
    quality_skeleton: Any | None = None,
    aligned_cache_root: Path | None = None,
) -> Tuple[List[PersonMetric], List[JointMetric]]:
    if aligned_cache_root is None:
        face_by_frame = load_sam3d_world_by_frame(sam3d_root, person_id, "face")
        side_by_frame = load_sam3d_world_by_frame(sam3d_root, person_id, "side")
        split_offset, split_metadata = load_split_alignment_offset(
            split_root, person_id
        )
        face, side, face_map, side_map, offset = build_aligned_timeline(
            face_by_frame,
            side_by_frame,
            offset_override=split_offset,
        )
        sequence_scope = "full_overlap_timeline"
        time_alignment = "split_alignment_record"
    else:
        face, side, face_map, side_map, split_metadata = (
            load_aligned_cycle_cache(aligned_cache_root, person_id)
        )
        offset = int(split_metadata["offset_side_to_face"])
        sequence_scope = str(split_metadata["sequence_scope"])
        time_alignment = "immutable_split_cycle_cache"
    triangulated_person_root = triangulated_root / f"person_{person_id}"
    has_triangulated = triangulated_person_root.exists() and any(triangulated_person_root.glob("cycle_*"))

    needs_extrinsics = any(method in EXTRINSIC_METHODS for method in methods)
    extrinsic_rotation = None
    extrinsic_metadata: Dict[str, Any] | None = None
    if needs_extrinsics:
        extrinsic_rotation, extrinsic_metadata = load_extrinsic_rotation(
            extrinsics_path, person_id
        )
    face_quality = None
    side_quality = None
    if "extrinsic_r_quality_average" in methods:
        if quality_skeleton is None:
            raise ValueError(
                "quality_skeleton is required for extrinsic_r_quality_average"
            )
        trial_lengths = split_metadata.get("trial_lengths")
        if isinstance(trial_lengths, list):
            face_quality = rotation_aware_quality_scores_by_trial(
                face, quality_skeleton, trial_lengths
            )
            side_quality = rotation_aware_quality_scores_by_trial(
                side, quality_skeleton, trial_lengths
            )
        else:
            face_quality = rotation_aware_quality_scores(face, quality_skeleton)
            side_quality = rotation_aware_quality_scores(side, quality_skeleton)

    sim3_all = None
    sim3_stable = None
    sim3_stable_scales = None
    sim3_stable_smooth = None
    metrics: List[PersonMetric] = []
    joint_metrics: List[JointMetric] = []

    for method in methods:
        extra: Dict[str, Any] = {
            "time_alignment": time_alignment,
            "sequence_scope": sequence_scope,
            "offset_side_to_face": int(offset),
            "split_alignment": split_metadata,
            "fusion_method": method,
            "uses_camera_extrinsics": method in EXTRINSIC_METHODS,
        }
        if method == "avg_body_current":
            fused_world = current_body_average(face, side)
        elif method == "avg_world_face_ref":
            fused_world = 0.5 * (face + side)
        elif method == "root_face_stable":
            side_aligned = root_align_to_reference(side, face)
            fused_world = 0.5 * (face + side_aligned)
        elif method == "sim3_face_all":
            if sim3_all is None:
                sim3_all, scales = sim3_align_to_reference(side, face, tuple(range(face.shape[1])))
                extra["scale_mean"] = float(np.mean(scales))
            fused_world = 0.5 * (face + sim3_all)
        elif method == "sim3_face_stable":
            if sim3_stable is None:
                sim3_stable, sim3_stable_scales = sim3_align_to_reference(side, face, STABLE_SIM3_JOINTS)
            extra["sim3_joints"] = list(STABLE_SIM3_JOINTS)
            extra["scale_mean"] = float(np.mean(sim3_stable_scales))
            fused_world = 0.5 * (face + sim3_stable)
        elif method == "sim3_face_stable_joint_weight":
            if sim3_stable is None:
                sim3_stable, sim3_stable_scales = sim3_align_to_reference(side, face, STABLE_SIM3_JOINTS)
            if has_triangulated:
                weights = estimate_weights_from_triangulated(
                    face,
                    sim3_stable,
                    face_map,
                    side_map,
                    triangulated_person_root,
                    alignment=alignment,
                )
                extra["joint_weight_source"] = "triangulated"
            else:
                weights = np.full((face.shape[1], 2), 0.5, dtype=np.float32)
                extra["joint_weight_source"] = "missing_triangulated_equal_fallback"
            extra["sim3_joints"] = list(STABLE_SIM3_JOINTS)
            extra["joint_weights"] = weights.tolist()
            extra["scale_mean"] = float(np.mean(sim3_stable_scales))
            fused_world = fuse_weighted(face, sim3_stable, weights)
        elif method == "sim3_face_stable_bodypart_weight":
            if sim3_stable is None:
                sim3_stable, sim3_stable_scales = sim3_align_to_reference(side, face, STABLE_SIM3_JOINTS)
            weights = bodypart_weights(face.shape[1])
            extra["sim3_joints"] = list(STABLE_SIM3_JOINTS)
            extra["joint_weights"] = weights.tolist()
            extra["scale_mean"] = float(np.mean(sim3_stable_scales))
            fused_world = fuse_weighted(face, sim3_stable, weights)
        elif method == "sim3_face_stable_smooth_transform":
            if sim3_stable is None:
                sim3_stable, sim3_stable_scales = sim3_align_to_reference(side, face, STABLE_SIM3_JOINTS)
            if sim3_stable_smooth is None:
                sim3_stable_smooth = smooth_sequence(sim3_stable, win=5)
            extra["sim3_joints"] = list(STABLE_SIM3_JOINTS)
            extra["smooth_target"] = "side_after_sim3"
            extra["smooth_window"] = 5
            extra["scale_mean"] = float(np.mean(sim3_stable_scales))
            fused_world = 0.5 * (face + sim3_stable_smooth)
        elif method == "sim3_face_stable_smooth_kpt":
            if sim3_stable is None:
                sim3_stable, sim3_stable_scales = sim3_align_to_reference(side, face, STABLE_SIM3_JOINTS)
            extra["sim3_joints"] = list(STABLE_SIM3_JOINTS)
            extra["smooth_target"] = "fused_world"
            extra["smooth_window"] = 5
            extra["scale_mean"] = float(np.mean(sim3_stable_scales))
            fused_world = smooth_sequence(0.5 * (face + sim3_stable), win=5)
        elif method == "extrinsic_r_average":
            assert extrinsic_rotation is not None
            assert extrinsic_metadata is not None
            extra["extrinsics"] = extrinsic_metadata
            extra["extrinsic_alignment"] = (
                "side_to_face_rotation_only_after_pelvis_centering"
            )
            extra["camera_translation_used"] = False
            fused_world = fuse_extrinsic_rotation(
                face, side, extrinsic_rotation
            )
        elif method == "extrinsic_r_quality_average":
            assert extrinsic_rotation is not None
            assert extrinsic_metadata is not None
            assert face_quality is not None
            assert side_quality is not None
            side_aligned = align_side_with_extrinsic_rotation(
                side, face, extrinsic_rotation
            )
            fused_world, frame_weights = fuse_quality_weighted(
                face, side_aligned, face_quality, side_quality
            )
            extra["extrinsics"] = extrinsic_metadata
            extra["extrinsic_alignment"] = (
                "side_to_face_rotation_only_after_pelvis_centering"
            )
            extra["camera_translation_used"] = False
            extra["quality_source"] = "rotation_aware_fixed_quality_features"
            extra["quality_temporal_scope"] = (
                "per_split_cycle"
                if isinstance(split_metadata.get("trial_lengths"), list)
                else "full_overlap_timeline"
            )
            extra["mean_face_weight"] = float(np.mean(frame_weights[:, 0]))
            extra["mean_side_weight"] = float(np.mean(frame_weights[:, 1]))
        elif method in BASELINE_METHODS:
            from fusion.baselines.classical_baselines import fuse_baseline

            fused_world, baseline_extra = fuse_baseline(method, face, side, fps=60.0)
            extra.update(baseline_extra)
        else:
            raise ValueError(f"Unsupported method: {method}")

        fused_world = fused_world.astype(np.float32)
        fused_body = kpts_world_to_body(fused_world)
        save_compact_sequence(out_root, person_id, method, fused_world, fused_body, face_map, side_map, extra)
        if save_frame_npz:
            save_fused_kpts(
                fused_world=fused_world,
                fused_body=fused_body,
                face_map=face_map,
                side_map=side_map,
                person_id=person_id,
                out_root=out_root / method / f"person_{person_id}" / "frames_format",
                fps=60.0,
            )

        if has_triangulated:
            person_metric, per_joint = evaluate_sequence(
                person_id=person_id,
                method=method,
                fused_world=fused_world,
                face_map=face_map,
                side_map=side_map,
                triangulated_person_root=triangulated_person_root,
                alignment=alignment,
            )
            metrics.append(person_metric)
            joint_metrics.extend(per_joint)
            print(f"[metric] person_{person_id} {method}: mpjpe={person_metric.mpjpe:.6g}")
        else:
            print(f"[metric] person_{person_id} {method}: skipped missing triangulated GT")

    return metrics, joint_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run face/side/fuse experiment matrix.")
    parser.add_argument("--sam3d-root", type=Path, default=DEFAULT_SAM3D_ROOT)
    parser.add_argument("--triangulated-root", type=Path, default=DEFAULT_TRIANGULATED_ROOT)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--extrinsics-path",
        type=Path,
        default=DEFAULT_EXTRINSICS_PATH,
        help="Estimated per-person camera extrinsics JSON.",
    )
    parser.add_argument(
        "--skeleton-path",
        type=Path,
        default=DEFAULT_SKELETON_PATH,
        help="Skeleton definition used by the fixed quality score.",
    )
    parser.add_argument(
        "--aligned-cache-root",
        type=Path,
        default=None,
        help=(
            "Optional immutable split-cycle cache. When supplied, fusion is "
            "restricted to the cached evaluation cycles and skips raw frame NPZ reads."
        ),
    )
    parser.add_argument("--person", nargs="*", default=None, help="Optional person ids, e.g. 27 29")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=list(NO_EXTRINSIC_METHODS),
        choices=AVAILABLE_METHODS,
    )
    parser.add_argument("--save-frame-npz", action="store_true", help="Also save old per-frame npz format.")
    parser.add_argument(
        "--alignment",
        default=DEFAULT_ALIGNMENT,
        choices=ALIGNMENT_MODES,
        help=(
            "How to bring fused keypoints into the triangulated frame before "
            "measuring error. 'similarity' (default) removes the static world "
            "frame and scale mismatch once per sequence. 'root' is the legacy "
            "pelvis-only behaviour and leaves that mismatch in the error. "
            "'procrustes' aligns every frame and is a diagnostic shape-only bound."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    quality_skeleton = None
    if "extrinsic_r_quality_average" in args.methods:
        from fusion.keypoints.config import load_skeleton_spec

        quality_skeleton = load_skeleton_spec(args.skeleton_path)
    all_person_metrics: List[PersonMetric] = []
    all_joint_metrics: List[JointMetric] = []
    config = {
        "sam3d_root": str(args.sam3d_root),
        "triangulated_root": str(args.triangulated_root),
        "split_root": str(args.split_root),
        "extrinsics_path": str(args.extrinsics_path),
        "skeleton_path": str(args.skeleton_path),
        "aligned_cache_root": (
            None
            if args.aligned_cache_root is None
            else str(args.aligned_cache_root)
        ),
        "methods": list(args.methods),
        "method_groups": {
            "without_camera_extrinsics": list(NO_EXTRINSIC_METHODS),
            "with_camera_extrinsics": list(EXTRINSIC_METHODS),
        },
        "stable_sim3_joints": list(STABLE_SIM3_JOINTS),
        "save_frame_npz": bool(args.save_frame_npz),
        "alignment": str(args.alignment),
    }
    (args.out_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for person_id in iter_person_ids(args.sam3d_root, args.person):
        print(f"[person] {person_id}")
        person_metrics, joint_metrics = process_person(
            person_id=person_id,
            methods=args.methods,
            sam3d_root=args.sam3d_root,
            triangulated_root=args.triangulated_root,
            split_root=args.split_root,
            out_root=args.out_dir,
            save_frame_npz=args.save_frame_npz,
            alignment=args.alignment,
            extrinsics_path=args.extrinsics_path,
            quality_skeleton=quality_skeleton,
            aligned_cache_root=args.aligned_cache_root,
        )
        all_person_metrics.extend(person_metrics)
        all_joint_metrics.extend(joint_metrics)

    write_csv(
        args.out_dir / "metrics_by_person.csv",
        all_person_metrics,
        ("person_id", "method", "eval_frames", "valid_points", "mpjpe", "median", "p95", "max_error"),
    )
    write_csv(
        args.out_dir / "metrics_by_joint.csv",
        all_joint_metrics,
        ("person_id", "method", "joint", "valid_points", "mpjpe", "median", "p95", "max_error"),
    )

    for method in args.methods:
        values = [row.mpjpe for row in all_person_metrics if row.method == method]
        if values:
            print(f"[summary] {method}: mean_person_mpjpe={np.nanmean(values):.6g}")
        else:
            print(f"[summary] {method}: no triangulated GT metrics")
    print(f"[save] {args.out_dir / 'metrics_by_person.csv'}")
    print(f"[save] {args.out_dir / 'metrics_by_joint.csv'}")


if __name__ == "__main__":
    main()


"""MetaPose stage 2 with the released checkpoint (runs in the ``metapose`` TF env).

Writes the stage-1 results as an ``opt``-format record set in the layout
``launch_iterative_solver`` produces (two reported iterations: the initial
estimate and the optimum), runs the official ``metapose.train_metapose``
inference path (``--epochs_per_stage=0 --load_weights_from ckpt/h36m/cam2``)
on it and exports the predicted 3D poses to ``s2.npz``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def write_opt_records(directory: Path, third_party: Path) -> Path:
    if str(third_party) not in sys.path:
        sys.path.insert(0, str(third_party))
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from metapose import data_utils  # type: ignore

    # Materialise the arrays: NpzFile members decompress on every access.
    inputs = {k: v for k, v in np.load(directory / "inputs.npz").items()}
    s1 = {k: v for k, v in np.load(directory / "s1.npz").items()}
    n, cams = inputs["pose2d"].shape[:2]
    usable = np.asarray(s1["usable"], dtype=bool)
    # Frames without a stage-1 estimate get a neutral record so the stage-2
    # batch order stays aligned with the inputs; their outputs are discarded.
    filled = {k: np.where(usable.reshape((-1,) + (1,) * (s1[k].ndim - 1)), s1[k], 0.0) for k in ("pose_init", "pose_opt", "rot_init", "rot_opt", "scale_init", "scale_opt", "shift_init", "shift_opt", "loss_init", "loss_opt")}
    for k in ("rot_init", "rot_opt"):
        filled[k][~usable] = np.eye(3)
    for k in ("scale_init", "scale_opt"):
        filled[k][~usable] = 1.0
    for k in ("pose_init", "pose_opt"):
        filled[k][~usable] = 0.5
    report_n = 2
    shapes = {
        "loss": ([report_n], tf.float32), "iters": ([report_n], tf.int32),
        "pose3d_opt_preds": ([report_n, 17, 3], tf.float32), "cam_rot_opt_preds": ([report_n, cams, 3, 3], tf.float32),
        "scale_opt_preds": ([report_n, cams], tf.float32), "shift_opt_preds": ([report_n, cams, 3], tf.float32),
        "pose2d_opt_preds": ([report_n, cams, 17, 2], tf.float32), "pose3d_gt_aligned_pred_3d_proj": ([report_n, cams, 17, 2], tf.float32),
        "pose3d_pred_pmpjpe": ([report_n], tf.float32), "pose2d_pred_err": ([report_n], tf.float32), "pose2d_pred_vs_posenet_err": ([report_n], tf.float32),
        "pose2d_gt_posenet_err_mean": ([], tf.float32), "pose3d_gt_backaligned_pose2d_gt_err": ([report_n], tf.float32),
        "pose3d": ([17, 3], tf.float64), "cam_pose3d": ([cams, 3], tf.float64), "cam_rot": ([cams, 3, 3], tf.float64),
        "cam_intr": ([cams, 4], tf.float64), "cam_kd": ([cams, 5], tf.float64), "pose2d_gt": ([cams, 17, 2], tf.float64),
        "pose2d_repr": ([cams, 17, 2], tf.float64), "heatmaps": ([cams, 17, 4, 4], tf.float64), "pose2d_pred": ([cams, 17, 2], tf.float64),
        "keys": ([cams], tf.string), "bboxes": ([cams, 4], tf.int32), "pose3d_epi_pred": ([cams, 17, 3], tf.float32), "cam_subset": ([cams], tf.int32),
    }
    spec = tfds.features.FeaturesDict({k: tfds.features.Tensor(shape=s, dtype=d) for k, (s, d) in shapes.items()})
    bboxes = inputs["bboxes"].astype(np.float64)
    sizes = np.maximum(bboxes[..., 1] - bboxes[..., 0], bboxes[..., 3] - bboxes[..., 2])
    origins = np.stack([bboxes[..., 2], bboxes[..., 0]], axis=-1)
    pose2d_bbox = ((inputs["pose2d"] - origins[:, :, None, :]) / sizes[:, :, None, None]).astype(np.float64)
    heat_bbox = s1["heatmaps_bbox"].astype(np.float64)
    epi = inputs["epi"].astype(np.float32)
    boxes_i32 = inputs["bboxes"].astype(np.int32)

    def records():
        for i in range(n):
            yield {
                "loss": np.array([filled["loss_init"][i], filled["loss_opt"][i]], np.float32), "iters": np.array([0, 100], np.int32),
                "pose3d_opt_preds": np.stack([filled["pose_init"][i], filled["pose_opt"][i]]).astype(np.float32),
                "cam_rot_opt_preds": np.stack([filled["rot_init"][i], filled["rot_opt"][i]]).astype(np.float32),
                "scale_opt_preds": np.stack([filled["scale_init"][i], filled["scale_opt"][i]]).astype(np.float32),
                "shift_opt_preds": np.stack([filled["shift_init"][i], filled["shift_opt"][i]]).astype(np.float32),
                "pose2d_opt_preds": np.zeros((report_n, cams, 17, 2), np.float32), "pose3d_gt_aligned_pred_3d_proj": np.zeros((report_n, cams, 17, 2), np.float32),
                "pose3d_pred_pmpjpe": np.zeros(report_n, np.float32), "pose2d_pred_err": np.zeros(report_n, np.float32), "pose2d_pred_vs_posenet_err": np.zeros(report_n, np.float32),
                "pose2d_gt_posenet_err_mean": np.float32(0.0), "pose3d_gt_backaligned_pose2d_gt_err": np.zeros(report_n, np.float32),
                "pose3d": np.zeros((17, 3), np.float64), "cam_pose3d": np.zeros((cams, 3), np.float64), "cam_rot": np.tile(np.eye(3), (cams, 1, 1)),
                "cam_intr": np.zeros((cams, 4), np.float64), "cam_kd": np.zeros((cams, 5), np.float64),
                "pose2d_gt": pose2d_bbox[i], "pose2d_repr": pose2d_bbox[i], "heatmaps": heat_bbox[i], "pose2d_pred": pose2d_bbox[i],
                "keys": np.array([f"r{i:07d}_c{k:02d}".encode() for k in range(cams)], dtype=object), "bboxes": boxes_i32[i],
                "pose3d_epi_pred": epi[i], "cam_subset": np.arange(cams, dtype=np.int32),
            }

    out = directory / "opt" / "test"
    if out.exists():
        shutil.rmtree(out)
    data_utils.write_tfrec_feature_dict_ds(records(), spec, str(out))
    return out


def run_stage2(directory: Path, release_root: Path, third_party: Path) -> Path:
    log_dir = directory / "s2_tb"
    preds = directory / "s2_preds"
    if preds.exists():
        shutil.rmtree(preds)
    cmd = [
        sys.executable, "-m", "metapose.train_metapose", f"--data_root={directory}", "--experiment_name=strict_external", f"--tb_log_dir={log_dir}",
        "--dataset=opt", "--data_splits=test,test", "--dataset_warmup=false", "--n_cam=2", "--train_repeat_k=1", "--permute_cams_aug=false",
        "--use_equivariant_model=true", "--debug_show_single_frame_pmpjes=false", "--debug_enable_check_numerics=false", "--standardize_init_best=false",
        "--main_mlp_spec=512,512,ccat,512,512,ccat,512", f"--load_weights_from={release_root / 'ckpt' / 'h36m' / 'cam2' / 'model'}", "--load_stages_n=1",
        "--epochs_per_stage=0", "--max_stage_attempts=1", "--debug_take_n_train_batches=1", f"--save_preds_to={preds}",
    ]
    env = {**os.environ, "PYTHONPATH": str(third_party) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    subprocess.run(cmd, check=True, env=env, cwd=str(third_party))
    import tensorflow as tf

    poses = [d["pose3d"].numpy() for d in tf.data.experimental.load(str(preds))]
    pose = np.stack(poses).astype(np.float32)
    np.savez_compressed(directory / "s2.npz", pose=pose)
    print(f"[metapose-s2] {len(pose)} predictions -> {directory / 's2.npz'}")
    return directory / "s2.npz"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--third-party", type=Path, required=True)
    parser.add_argument("--rewrite", action="store_true", help="rewrite the opt records even if present")
    args = parser.parse_args(argv)
    directory, release_root, third_party = (Path(a).resolve() for a in (args.directory, args.release_root, args.third_party))
    if not (directory / "opt" / "test" / "dataset.tfrec").is_file() or args.rewrite:
        write_opt_records(directory, third_party)
    run_stage2(directory, release_root, third_party)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Batched port of MetaPose's stage-1 solver (runs in the ``metapose`` TF env).

The released ``launch_iterative_solver`` optimises one frame at a time and
re-traces its graph per record (about 10 s/frame on a CPU node), which is
unusable for the ~200k frames per dataset here. This module runs the *same*
algorithm on a batch of frames: the objective is the per-frame mean negative
GMM log-likelihood of the weak-perspective reprojection
(:func:`metapose.inference_time_optimization.total_frame_loss`), the
parameters are the 3D pose, the 6D camera rotations, inverse-softplus scales
and shifts of every camera but the first, and Adam(1e-2) takes 100 steps.
Frames are independent and Adam is elementwise, so optimising the sum of
per-frame objectives is the per-frame optimisation up to float rounding;
``validate`` checks that against the official code on released records.

Only the released ``metapose`` package functions are used for the
projection, the 6D rotation and the GMM likelihood (imported unmodified);
the initial estimate (Procrustes alignment of each view's monocular 3D onto
the first view) is re-implemented batched with the same formulas.

Input / output are NumPy bundles (see :mod:`fusion.external.published.metapose_pipeline`):

    inputs.npz    heatmaps [N, C, 17, 4, 4]  pose2d [N, C, 17, 2]  bboxes [N, C, 4]  epi [N, C, 17, 3]
    s1.npz        pose_init [N, 17, 3]  pose_opt [N, 17, 3]  rot_init/rot_opt [N, C, 3, 3]
                  scale_init/scale_opt [N, C]  shift_init/shift_opt [N, C, 3]  loss_init/loss_opt [N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

THIRD_PARTY = Path(__file__).resolve().parents[1] / "third_party"


def _import_metapose():
    if str(THIRD_PARTY) not in sys.path:
        sys.path.insert(0, str(THIRD_PARTY))
    from metapose import inference_time_optimization as inf_opt  # type: ignore

    return inf_opt


# ----------------------------------------------------------------------------- numpy pre-processing
def to_bbox_axis(pose2d: np.ndarray, heatmaps: np.ndarray, bboxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``convert_rec_pose2d_to_bbox_axis`` for a batch: pixels -> [0, 1] bbox units."""
    bboxes = np.asarray(bboxes, dtype=np.float64)
    sizes = np.maximum(bboxes[..., 1] - bboxes[..., 0], bboxes[..., 3] - bboxes[..., 2])  # [N, C]
    origins = np.stack([bboxes[..., 2], bboxes[..., 0]], axis=-1)  # [N, C, 2] (x0, y0)
    heat = np.asarray(heatmaps, dtype=np.float64).copy()
    heat[..., 1:3] = (heat[..., 1:3] - origins[:, :, None, None, :]) / sizes[:, :, None, None, None]
    heat[..., 3] = heat[..., 3] / sizes[:, :, None, None] ** 2
    pose = (np.asarray(pose2d, dtype=np.float64) - origins[:, :, None, :]) / sizes[:, :, None, None]
    return pose, heat


def _procrustes(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched ``inf_opt.procrustes``: rotation aligning centred ``a`` to centred ``b`` ([N, J, 3] each)."""
    a_m, b_m = a.mean(axis=1), b.mean(axis=1)
    a_c, b_c = a - a_m[:, None], b - b_m[:, None]
    cov = np.einsum("njd,nje->nde", a_c, b_c)
    u, _, vt = np.linalg.svd(cov)
    # tf.linalg.svd returns V (not V^T); tensordot(U, V, axes=(1, 1)) = U V^T.
    rotation = np.einsum("nij,nkj->nik", u, vt.transpose(0, 2, 1))
    return rotation, a_m, b_m


def initial_estimate(epi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batched ``initial_epi_estimate``: ``(pose, R [N, C, 3, 3], scale [N, C], shift [N, C, 3])``."""
    epi = np.asarray(epi, dtype=np.float64)
    n, cams = epi.shape[:2]
    aligned, rots, scales, shifts = [], [np.tile(np.eye(3), (n, 1, 1))], [np.ones(n)], []
    view0 = epi[:, 0]
    for k in range(1, cams):
        pred = epi[:, k]
        rotation, a_m, b_m = _procrustes(view0, pred)
        scale = (np.linalg.norm(pred - b_m[:, None], axis=(1, 2)) / np.linalg.norm(view0 - a_m[:, None], axis=(1, 2)))
        # b2a = (b - b_m) @ R^T / scale + a_m  (the pose of view k aligned onto view 0)
        b2a = np.einsum("njd,ned->nje", pred - b_m[:, None], rotation) / scale[:, None, None] + a_m[:, None]
        aligned.append(b2a)
        rots.append(rotation)
        scales.append(scale)
        shifts.append(b_m)
    if cams == 1:
        raise ValueError("at least two cameras are required")
    # first_view_params = [zeros(3), view0_mean (=a_m of the first alignment), eye, 1]
    a_m0 = _procrustes(view0, epi[:, 1])[1]
    pose = np.mean(aligned, axis=0) - a_m0[:, None]
    shift = np.stack([a_m0] + shifts, axis=1)
    return pose, np.stack(rots, axis=1), np.stack(scales, axis=1), shift


# ----------------------------------------------------------------------------- batched optimisation
def optimise(pose0: np.ndarray, rot0: np.ndarray, scale0: np.ndarray, shift0: np.ndarray, heatmaps: np.ndarray, *, steps: int = 100, learning_rate: float = 1e-2, batch: int = 4096) -> dict[str, np.ndarray]:
    """Adam over batches of frames with MetaPose's objective; returns init and final parameters."""
    import tensorflow as tf

    inf_opt = _import_metapose()
    n, cams, joints = heatmaps.shape[:3]
    out = {k: [] for k in ("pose_opt", "rot_opt", "scale_opt", "shift_opt", "loss_init", "loss_opt")}

    def objective(pose, rot_re, scale_re, shift, heat):
        # pose [B, J, 3]; rot_re [B, C, 2, 3]; scale_re [B, C]; shift [B, C, 3]; heat [B, C, J, 4, 4]
        b = tf.shape(pose)[0]
        rot = inf_opt.vec6d_to_rot_mat(tf.reshape(rot_re, (-1, 6)))
        rot = tf.reshape(rot, (b, cams, 3, 3))
        scale = tf.math.softplus(scale_re)
        views = tf.einsum("bjd,bkdo->bkjo", pose, rot) * scale[:, :, None, None] + shift[:, :, None, :]
        points = tf.reshape(views[..., :2], (-1, 2))
        params = tf.reshape(heat, (-1, 4, 4))
        logp = inf_opt.gaussian_mixture_log_prob(points, params, 1e-8)
        per_frame = -tf.reduce_mean(tf.reshape(logp, (b, cams * joints)), axis=1)
        return per_frame

    for start in range(0, n, batch):
        stop = min(n, start + batch)
        heat = tf.constant(heatmaps[start:stop], tf.float64)
        pose = tf.Variable(pose0[start:stop].astype(np.float32))
        rot_re = tf.Variable(rot0[start:stop, :, :2, :].astype(np.float32))
        scale_re = tf.Variable(np.log(np.exp(scale0[start:stop]) - 1.0).astype(np.float32))
        shift = tf.Variable(shift0[start:stop].astype(np.float32))
        opt = tf.keras.optimizers.Adam(learning_rate)
        variables = [pose, rot_re, scale_re, shift]

        @tf.function
        def step():
            with tf.GradientTape() as tape:
                per_frame = objective(pose, rot_re, scale_re, shift, heat)
                total = tf.reduce_sum(per_frame)
            grads = tape.gradient(total, variables)
            opt.apply_gradients(zip(grads, variables))
            return per_frame

        loss_init = objective(pose, rot_re, scale_re, shift, heat).numpy()
        for _ in range(steps):
            step()
        loss_opt = objective(pose, rot_re, scale_re, shift, heat).numpy()
        rot = inf_opt.vec6d_to_rot_mat(tf.reshape(rot_re, (-1, 6))).numpy().reshape(stop - start, cams, 3, 3)
        out["pose_opt"].append(pose.numpy())
        out["rot_opt"].append(rot)
        out["scale_opt"].append(tf.math.softplus(scale_re).numpy())
        out["shift_opt"].append(shift.numpy())
        out["loss_init"].append(loss_init)
        out["loss_opt"].append(loss_opt)
    return {k: np.concatenate(v) for k, v in out.items()}


def run(inputs: Path, output: Path, *, steps: int = 100, learning_rate: float = 1e-2, batch: int = 4096, limit: int | None = None) -> Path:
    data = np.load(inputs)
    heat, pose2d, bboxes, epi, valid = data["heatmaps"], data["pose2d"], data["bboxes"], data["epi"], data["valid"]
    if limit:
        heat, pose2d, bboxes, epi, valid = heat[:limit], pose2d[:limit], bboxes[:limit], epi[:limit], valid[:limit]
    n = len(epi)
    usable = np.asarray(valid, dtype=bool).all(axis=1)  # a frame without 2D in some view has no estimate
    _, heat_bbox = to_bbox_axis(pose2d, heat, bboxes)
    pose0 = np.full((n, epi.shape[2], 3), np.nan, np.float32)
    cams = epi.shape[1]
    rot0 = np.full((n, cams, 3, 3), np.nan, np.float32)
    scale0 = np.full((n, cams), np.nan, np.float32)
    shift0 = np.full((n, cams, 3), np.nan, np.float32)
    result = {k: np.full_like(v, np.nan) for k, v in (("pose_opt", pose0), ("rot_opt", rot0), ("scale_opt", scale0), ("shift_opt", shift0))}
    result["loss_init"] = np.full(n, np.nan, np.float32)
    result["loss_opt"] = np.full(n, np.nan, np.float32)
    t0 = time.time()
    if usable.any():
        p0, r0, s0, sh0 = initial_estimate(epi[usable])
        sub = optimise(p0, r0, s0, sh0, heat_bbox[usable], steps=steps, learning_rate=learning_rate, batch=batch)
        pose0[usable], rot0[usable], scale0[usable], shift0[usable] = p0, r0, s0, sh0
        for k, v in sub.items():
            result[k][usable] = v
    print(f"[metapose-s1] {int(usable.sum())}/{n} frames, {steps} steps: {time.time() - t0:.0f} s; loss {np.nanmean(result['loss_init']):.4f} -> {np.nanmean(result['loss_opt']):.4f}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, pose_init=pose0, rot_init=rot0, scale_init=scale0, shift_init=shift0, heatmaps_bbox=heat_bbox.astype(np.float32), usable=usable, **result)
    return output


# ----------------------------------------------------------------------------- validation against the official solver
def validate(release_root: Path, n_records: int = 8, steps: int = 100) -> None:
    """Run the official per-frame solver and this port on released H36M records; print the differences."""
    inf_opt = _import_metapose()
    from metapose import data_utils  # type: ignore

    ds = data_utils.read_tfrec_feature_dict_ds(str(release_root / "data" / "h36m" / "pre" / "test"))
    records = [rec for _, rec in ds.take(n_records)]
    official = []
    for rec in records:
        stats = inf_opt.run_inference_optimization(dict(rec), opt_steps=steps, report_n_results=2, cam_subset=[0, 1])
        official.append((stats["pose3d_opt_preds"][0], stats["pose3d_opt_preds"][-1], stats["loss"][0], stats["loss"][-1]))
    stack = lambda key: np.stack([rec[key].numpy()[:2] for rec in records])  # noqa: E731
    heat, pose2d, bboxes, epi = stack("heatmaps"), stack("pose2d_pred"), stack("bboxes"), stack("pose3d_epi_pred")
    _, heat_bbox = to_bbox_axis(pose2d, heat, bboxes)
    pose0, rot0, scale0, shift0 = initial_estimate(epi)
    ported = optimise(pose0, rot0, scale0, shift0, heat_bbox, steps=steps)
    init_diff = max(np.abs(pose0[i] - official[i][0]).max() for i in range(len(records)))
    opt_diff = max(np.abs(ported["pose_opt"][i] - official[i][1]).max() for i in range(len(records)))
    loss_diff = max(abs(float(ported["loss_opt"][i]) - float(official[i][3])) for i in range(len(records)))
    print(f"[metapose-s1] validation on {len(records)} released records: max |init pose diff| {init_diff:.2e}, max |optimised pose diff| {opt_diff:.2e}, max |final loss diff| {loss_diff:.2e}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--inputs", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--steps", type=int, default=100)
    r.add_argument("--learning-rate", type=float, default=1e-2)
    r.add_argument("--batch", type=int, default=4096)
    r.add_argument("--limit", type=int, default=None)
    v = sub.add_parser("validate")
    v.add_argument("--release-root", type=Path, required=True)
    v.add_argument("--records", type=int, default=8)
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args.inputs, args.output, steps=args.steps, learning_rate=args.learning_rate, batch=args.batch, limit=args.limit)
    else:
        validate(args.release_root, args.records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

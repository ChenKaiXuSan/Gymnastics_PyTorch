"""MetaPose stage 2 through the official ``train_metapose`` (runs in a TF env).

Writes the stage-1 results as ``opt``-format records in the layout
``launch_iterative_solver`` produces (two reported iterations: the initial
estimate and the optimum) and either

* ``--mode released``: runs the released 2-camera Human3.6M checkpoint
  (``--epochs_per_stage=0 --load_weights_from ckpt/h36m/cam2``) on every frame, or
* ``--mode shards``: writes the records once per subject (``shards/<person>/
  {train,test}``; ``train`` holds the frames with a stage-1 estimate, ``test``
  every frame in index order), in parallel processes -- the tfds serialiser
  is slow, and folds only recombine subjects;
* ``--mode train``: assembles the fold's ``opt_<fold>/{train,test}`` by
  concatenating subject shards (TFRecord files concatenate byte-wise; a
  64-record head of random training frames comes first because the script
  validates on the first ``valid_first_n`` records of its training split),
  trains stage 2 with the authors' script on the training subjects and
  predicts the test subjects. The training is label-free: the ``fwd`` loss
  regresses the 2D detections and the ``soln`` losses the stage-1 optimum;
  the record's ``pose3d`` field, which the script uses only for the
  early-stopping / checkpoint metric ``val_pred_pmpjpe``, holds the stage-1
  optimum, so model selection is label-free too. Epochs per stage, patience
  and the number of stages are capped by the caller (the released Human3.6M
  schedule is 300 epochs x up to 10 stages, patience 50).

Predicted 3D poses (H36M-17, bbox units of camera 0) are exported to
``s2.npz`` / ``s2_<fold>.npz`` together with the rows they belong to.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

VALID_FIRST_N = 64  # train_metapose --valid_first_n default
_DATA: dict[str, dict[str, np.ndarray]] = {}


def load_data(directory: Path) -> dict[str, dict[str, np.ndarray]]:
    """``inputs.npz`` and ``s1.npz`` materialised (NpzFile members decompress on every access), cached per process."""
    key = str(directory)
    if key not in _DATA:
        _DATA[key] = {"inputs": {k: v for k, v in np.load(directory / "inputs.npz").items()}, "s1": {k: v for k, v in np.load(directory / "s1.npz").items()}}
    return _DATA[key]


def person_blocks(directory: Path) -> list[tuple[str, int, int]]:
    """``(person, start, stop)`` of every subject's contiguous row block, in index order."""
    index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    blocks: list[tuple[str, int, int]] = []
    for trial in index["trials"]:
        person = str(trial["person"])
        if blocks and blocks[-1][0] == person and blocks[-1][2] == trial["start"]:
            blocks[-1] = (person, blocks[-1][1], trial["stop"])
        elif any(b[0] == person for b in blocks):
            raise ValueError(f"rows of person {person} are not contiguous in {directory / 'index.json'}")
        else:
            blocks.append((person, trial["start"], trial["stop"]))
    return blocks


def write_opt_records(directory: Path, third_party: Path, *, out: Path, rows: np.ndarray | None = None) -> Path:
    """Write the records of ``rows`` (default: all frames) below ``out``."""
    if str(third_party) not in sys.path:
        sys.path.insert(0, str(third_party))
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from metapose import data_utils  # type: ignore

    data = load_data(directory)
    inputs, s1 = dict(data["inputs"]), dict(data["s1"])
    if rows is not None:
        rows = np.asarray(rows, dtype=np.int64)
        total = len(inputs["pose2d"])
        inputs = {k: (v[rows] if v.ndim >= 1 and v.shape[0] == total else v) for k, v in inputs.items()}
        s1 = {k: (v[rows] if v.ndim >= 1 and v.shape[0] == total else v) for k, v in s1.items()}
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
    # A joint without a detection has no 2D point (the prepare stage stores 0, i.e. the image origin) and an
    # almost flat heatmap (variance (size/2)^2 instead of (0.02 size)^2). The release's 2D target / mean-heatmap
    # input has no notion of a missing joint, so those joints are imputed with the stage-1 solution's own
    # weak-perspective re-projection; the (uninformative) heatmap is kept as it is.
    var_detected = (0.02 * sizes / sizes) ** 2  # bbox units: (0.02)^2
    missing = heat_bbox[:, :, :, 0, 3] > 10.0 * var_detected[:, :, None]  # [N, C, 17]
    reprojected = (np.einsum("njd,nkdo->nkjo", filled["pose_opt"], filled["rot_opt"]) * filled["scale_opt"][:, :, None, None] + filled["shift_opt"][:, :, None, :])[..., :2].astype(np.float64)
    pose2d_bbox = np.where(missing[..., None], reprojected, pose2d_bbox)
    imputed = int(missing.any(axis=(1, 2)).sum())
    if imputed:
        print(f"[metapose-s2] {out.name}: 2D of missing joints imputed from the stage-1 re-projection on {imputed} of {n} frames", flush=True)
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
                # pose3d is the script's metric target only: the stage-1 optimum keeps model selection label-free.
                "pose3d": filled["pose_opt"][i].astype(np.float64), "cam_pose3d": np.zeros((cams, 3), np.float64), "cam_rot": np.tile(np.eye(3), (cams, 1, 1)),
                "cam_intr": np.zeros((cams, 4), np.float64), "cam_kd": np.zeros((cams, 5), np.float64),
                "pose2d_gt": pose2d_bbox[i], "pose2d_repr": pose2d_bbox[i], "heatmaps": heat_bbox[i], "pose2d_pred": pose2d_bbox[i],
                "keys": np.array([f"r{i:07d}_c{k:02d}".encode() for k in range(cams)], dtype=object), "bboxes": boxes_i32[i],
                "pose3d_epi_pred": epi[i], "cam_subset": np.arange(cams, dtype=np.int32),
            }

    if out.exists():
        shutil.rmtree(out)
    data_utils.write_tfrec_feature_dict_ds(records(), spec, str(out))
    return out


def _trainer_command() -> list[str]:
    """The launcher (``metapose_launch.py``: CPU SVD, private best-model path) unless ``METAPOSE_LAUNCHER=0``."""
    if os.environ.get("METAPOSE_LAUNCHER", "1") == "0":
        return [sys.executable, "-m", "metapose.train_metapose"]
    return [sys.executable, str(Path(__file__).with_name("metapose_launch.py"))]


COMMON_FLAGS = [
    "--dataset_warmup=false", "--n_cam=2", "--train_repeat_k=1", "--permute_cams_aug=false", "--use_equivariant_model=true",
    "--debug_show_single_frame_pmpjes=false", "--debug_enable_check_numerics=false", "--standardize_init_best=false",
    "--main_mlp_spec=512,512,ccat,512,512,ccat,512",
]


def _load_preds(preds: Path) -> np.ndarray:
    import tensorflow as tf

    loader = tf.data.Dataset.load if hasattr(tf.data.Dataset, "load") else tf.data.experimental.load
    return np.stack([d["pose3d"].numpy() for d in loader(str(preds))]).astype(np.float32)


def run_released(directory: Path, release_root: Path, third_party: Path) -> Path:
    """Released 2-camera checkpoint on every frame (no training)."""
    write_opt_records(directory, third_party, out=directory / "opt" / "test")
    log_dir, preds = directory / "s2_tb", directory / "s2_preds"
    if preds.exists():
        shutil.rmtree(preds)
    cmd = [*_trainer_command(), f"--data_root={directory}", "--experiment_name=strict_external", f"--tb_log_dir={log_dir}", "--dataset=opt", "--data_splits=test,test", *COMMON_FLAGS,
           f"--load_weights_from={release_root / 'ckpt' / 'h36m' / 'cam2' / 'model'}", "--load_stages_n=1", "--epochs_per_stage=0", "--max_stage_attempts=1", "--debug_take_n_train_batches=1", f"--save_preds_to={preds}"]
    env = {**os.environ, "PYTHONPATH": str(third_party) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    subprocess.run(cmd, check=True, env=env, cwd=str(third_party))
    pose = _load_preds(preds)
    np.savez_compressed(directory / "s2.npz", pose=pose, rows=np.arange(len(pose)))
    print(f"[metapose-s2] {len(pose)} predictions (released checkpoint) -> {directory / 's2.npz'}")
    return directory / "s2.npz"


# ----------------------------------------------------------------------------- subject shards
def _write_shard(args: tuple[str, str, str, np.ndarray]) -> tuple[str, int]:
    directory, third_party, out, rows = args
    write_opt_records(Path(directory), Path(third_party), out=Path(out), rows=rows)
    return out, int(len(rows))


def trainable_rows(directory: Path) -> np.ndarray:
    """Frames with a sane stage-1 solution: usable, weak-perspective scales in [0.25, 4] and a non-collapsed pose.

    The batched solver occasionally ends in a degenerate optimum (a collapsed
    pose with a huge scale; FreeMan: 1 of 710k frames); such targets are
    excluded from stage-2 training, never from the test rows.
    """
    s1 = np.load(directory / "s1.npz")
    usable = np.asarray(s1["usable"], dtype=bool)
    scale = np.asarray(s1["scale_opt"], dtype=np.float64)
    spread = np.asarray(s1["pose_opt"], dtype=np.float64).std(axis=1).mean(axis=1)
    sane = ((scale >= 0.25) & (scale <= 4.0)).all(axis=1) & (spread >= 0.05)
    return usable & sane


def write_shards(directory: Path, third_party: Path, *, workers: int = 8) -> Path:
    """``shards/<person>/train`` (trainable frames) and ``shards/<person>/test`` (all frames, index order) for every subject."""
    usable = trainable_rows(directory)
    shards = directory / "shards"
    if shards.exists():
        shutil.rmtree(shards)
    tasks, manifest = [], {}
    for person, start, stop in person_blocks(directory):
        rows = np.arange(start, stop, dtype=np.int64)
        manifest[person] = {"start": start, "stop": stop, "train_rows": int(usable[rows].sum()), "test_rows": int(len(rows))}
        tasks.append((str(directory), str(third_party), str(shards / person / "test"), rows))
        tasks.append((str(directory), str(third_party), str(shards / person / "train"), rows[usable[rows]]))
    # The writer is serial per shard; subjects are written in parallel processes (spawned: TensorFlow does not fork well).
    with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=multiprocessing.get_context("spawn")) as pool:
        for out, count in pool.map(_write_shard, tasks):
            print(f"[metapose-s2] shard {Path(out).relative_to(shards)}: {count} records", flush=True)
    (shards / "manifest.json").write_text(json.dumps({"persons": manifest, "valid_first_n": VALID_FIRST_N}, indent=1), encoding="utf-8")
    return shards


def _concatenate(parts: list[Path], out: Path) -> None:
    """A TFRecord file is a plain sequence of framed records: byte-wise concatenation is a valid file."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as handle:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, handle, 16 * 1024 * 1024)


def assemble_fold(directory: Path, third_party: Path, *, fold: str, train_persons: list[str], test_persons: list[str], seed: int) -> tuple[Path, np.ndarray, np.ndarray]:
    """``opt_<fold>/{train,test}`` from the subject shards; returns the directory and the (train, test) rows."""
    shards = directory / "shards"
    manifest = json.loads((shards / "manifest.json").read_text(encoding="utf-8"))["persons"]
    usable = trainable_rows(directory)
    blocks = {person: (int(v["start"]), int(v["stop"])) for person, v in manifest.items()}
    missing = [p for p in train_persons + test_persons if p not in blocks]
    if missing:
        # Fold subjects without any kept session (e.g. FreeMan's action / cycle filters), like the DataModule's tolerance.
        print(f"[metapose-s2] {fold}: subjects without records skipped: {missing}", flush=True)
    train_persons = [p for p in train_persons if p in blocks and manifest[p]["train_rows"] > 0]
    test_persons = [p for p in test_persons if p in blocks]
    rng = np.random.default_rng(seed)
    order = [train_persons[i] for i in rng.permutation(len(train_persons))]
    train_rows = np.concatenate([np.arange(*blocks[p]) for p in order])
    train_rows = train_rows[usable[train_rows]]
    test_persons = sorted(test_persons, key=lambda p: blocks[p][0])
    test_rows = np.concatenate([np.arange(*blocks[p]) for p in test_persons])
    if len(train_rows) == 0 or len(test_rows) == 0:
        raise ValueError(f"{fold}: {len(train_rows)} training / {len(test_rows)} test frames")
    dataset_dir = directory / f"opt_{fold}"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    # Validation head: the script validates on the first VALID_FIRST_N records of its training split.
    head_rows = rng.choice(train_rows, size=min(VALID_FIRST_N, len(train_rows)), replace=False)
    head = write_opt_records(directory, third_party, out=dataset_dir / "train_head", rows=head_rows)
    _concatenate([head / "dataset.tfrec"] + [shards / p / "train" / "dataset.tfrec" for p in order], dataset_dir / "train" / "dataset.tfrec")
    _concatenate([shards / p / "test" / "dataset.tfrec" for p in test_persons], dataset_dir / "test" / "dataset.tfrec")
    for split in ("train", "test"):
        for item in head.iterdir():
            if item.name != "dataset.tfrec":
                shutil.copy(item, dataset_dir / split / item.name)
    shutil.rmtree(head)
    print(f"[metapose-s2] {fold}: {len(train_rows)} training frames from {len(order)} subjects (+{len(head_rows)} validation head), {len(test_rows)} test frames from {len(test_persons)} subjects", flush=True)
    return dataset_dir, train_rows, test_rows


# The README's three stage-2 objectives (all label-free here: the 2D target is the SAM3D detection, not 2D ground truth).
LOSS_FLAGS = {
    "fwd": ["--lambda_fwd_loss=1.0", "--lambda_logp_loss=0.0", "--lambda_xopt_loss=0.0"],  # default: reprojection MSE to the 2D target
    "ss": ["--lambda_fwd_loss=0.0", "--lambda_logp_loss=1.0", "--lambda_xopt_loss=0.0"],  # "S1+S2/SS": heatmap log-likelihood
    "ts": ["--lambda_fwd_loss=0.0", "--lambda_logp_loss=0.0", "--lambda_xopt_loss=1.0"],  # "S1+S2/TS": student of the stage-1 solution
}


def run_name(fold: str, loss: str) -> str:
    return f"s2_{fold}" if loss == "fwd" else f"s2_{loss}_{fold}"


def run_trained(directory: Path, third_party: Path, *, fold: str, train_persons: list[str], test_persons: list[str], epochs_per_stage: int, patience: int, max_stages: int, learning_rate: float = 1e-4, seed: int = 0, loss: str = "fwd") -> Path:
    """Train stage 2 on the training subjects with the authors' script and predict the test subjects."""
    if loss not in LOSS_FLAGS:
        raise ValueError(f"loss must be one of {sorted(LOSS_FLAGS)}")
    dataset_dir, _, test_rows = assemble_fold(directory, third_party, fold=fold, train_persons=train_persons, test_persons=test_persons, seed=seed)
    name = run_name(fold, loss)
    log_dir, preds = directory / f"{name}_tb", directory / f"{name}_preds"
    for path in (log_dir, preds):
        if path.exists():
            shutil.rmtree(path)
    cmd = [*_trainer_command(), f"--data_root={directory}", f"--experiment_name=strict_external_{name}", f"--tb_log_dir={log_dir}", f"--dataset=opt_{fold}", "--data_splits=train,test", *COMMON_FLAGS, *LOSS_FLAGS[loss],
           # The script trains stages 0..max_n_stages and retrains a stage whose validation metric got worse; two retries are allowed.
           f"--epochs_per_stage={int(epochs_per_stage)}", f"--early_stopping_patience={int(patience)}", f"--max_n_stages={int(max_stages)}", f"--max_stage_attempts={int(max_stages) + 3}",
           f"--learning_rate={learning_rate}", f"--save_preds_to={preds}"]
    best = directory / f"{name}_best"
    if best.exists():
        shutil.rmtree(best)
    best.mkdir(parents=True)
    env = {**os.environ, "PYTHONPATH": str(third_party) + os.pathsep + os.environ.get("PYTHONPATH", ""), "METAPOSE_BEST_MODEL": str(best / "model")}
    subprocess.run(cmd, check=True, env=env, cwd=str(third_party))
    pose = _load_preds(preds)
    if len(pose) != len(test_rows):
        raise RuntimeError(f"stage 2 returned {len(pose)} predictions for {len(test_rows)} test frames")
    finite = np.isfinite(pose).all(axis=(1, 2)).mean()
    if finite < 0.99:
        raise RuntimeError(f"stage 2 of {fold} produced non-finite poses on {100 * (1 - finite):.1f} % of the test frames")
    out = directory / f"{name}.npz"
    np.savez_compressed(out, pose=pose, rows=np.asarray(test_rows, dtype=np.int64), loss=np.array(loss))
    print(f"[metapose-s2] {name}: {len(pose)} test predictions -> {out}")
    return out


def run_predict(directory: Path, third_party: Path, *, fold: str, train_persons: list[str], test_persons: list[str], loss: str, weights: Path, stages: int = 1, seed: int = 0) -> Path:
    """Predict the test subjects with an already trained stage-2 checkpoint (``--epochs_per_stage=0`` path)."""
    dataset_dir, _, test_rows = assemble_fold(directory, third_party, fold=fold, train_persons=train_persons, test_persons=test_persons, seed=seed)
    name = run_name(fold, loss)
    log_dir, preds = directory / f"{name}_predict_tb", directory / f"{name}_preds"
    for path in (log_dir, preds):
        if path.exists():
            shutil.rmtree(path)
    # No training: the test split serves as both splits like the released inference path.
    cmd = [*_trainer_command(), f"--data_root={directory}", f"--experiment_name=predict_{name}", f"--tb_log_dir={log_dir}", f"--dataset=opt_{fold}", "--data_splits=test,test", *COMMON_FLAGS,
           f"--load_weights_from={weights}", f"--load_stages_n={int(stages)}", "--epochs_per_stage=0", "--max_stage_attempts=1", "--debug_take_n_train_batches=1", f"--save_preds_to={preds}"]
    env = {**os.environ, "PYTHONPATH": str(third_party) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    subprocess.run(cmd, check=True, env=env, cwd=str(third_party))
    pose = _load_preds(preds)
    if len(pose) != len(test_rows):
        raise RuntimeError(f"prediction returned {len(pose)} poses for {len(test_rows)} test frames")
    finite = np.isfinite(pose).all(axis=(1, 2)).mean()
    if finite < 0.99:
        raise RuntimeError(f"{name}: non-finite poses on {100 * (1 - finite):.1f} % of the test frames")
    out = directory / f"{name}.npz"
    np.savez_compressed(out, pose=pose, rows=np.asarray(test_rows, dtype=np.int64), loss=np.array(loss), stages=np.array(stages))
    print(f"[metapose-s2] {name}: {len(pose)} test predictions from {weights} ({stages} stage(s)) -> {out}")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--third-party", type=Path, required=True)
    parser.add_argument("--mode", choices=("released", "shards", "train", "predict"), default="train")
    parser.add_argument("--weights", type=Path, default=None, help="predict mode: trained checkpoint prefix (e.g. s2_ts_fold_01_best/model)")
    parser.add_argument("--stages", type=int, default=1, help="predict mode: number of stage models in the checkpoint")
    parser.add_argument("--fold", default=None, help="fold name (train mode)")
    parser.add_argument("--fold-json", type=Path, default=None, help="fold file with train / test subject lists (train mode)")
    parser.add_argument("--workers", type=int, default=8, help="shard writer processes")
    parser.add_argument("--epochs-per-stage", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-stages", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loss", choices=sorted(LOSS_FLAGS), default="fwd", help="stage-2 objective (README variants; fwd = default)")
    args = parser.parse_args(argv)
    directory, release_root, third_party = (Path(a).resolve() for a in (args.directory, args.release_root, args.third_party))
    if args.mode == "released":
        run_released(directory, release_root, third_party)
    elif args.mode == "shards":
        write_shards(directory, third_party, workers=args.workers)
    else:
        if not args.fold or not args.fold_json:
            raise SystemExit("--fold and --fold-json are required in train mode")
        if not (directory / "shards" / "manifest.json").is_file():
            write_shards(directory, third_party, workers=args.workers)
        fold = json.loads(Path(args.fold_json).read_text(encoding="utf-8"))
        persons = dict(train_persons=[str(p) for p in fold["train"]], test_persons=[str(p) for p in fold["test"]])
        if args.mode == "predict":
            if not args.weights:
                raise SystemExit("--weights is required in predict mode")
            run_predict(directory, third_party, fold=args.fold, loss=args.loss, weights=Path(args.weights).resolve(), stages=args.stages, seed=args.seed, **persons)
        else:
            run_trained(directory, third_party, fold=args.fold, epochs_per_stage=args.epochs_per_stage, patience=args.patience, max_stages=args.max_stages, seed=args.seed, loss=args.loss, **persons)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

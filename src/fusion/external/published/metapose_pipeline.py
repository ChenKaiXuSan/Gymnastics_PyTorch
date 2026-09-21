"""MetaPose (Usman et al., CVPR 2022) as released, on this project's two views.

Pipeline (the authors' data flow, ``docs/research`` records the deviations):

    prepare   (this env)  per trial and view: SAM3D 2D keypoints -> Human3.6M-17
              2D layout, a square bounding box, a one-component Gaussian per
              joint standing in for the heatmap GMM (SAM3D emits keypoints, not
              heatmaps; sigma = 2 % of the box), and the monocular 3D
              initialisation ``pose3d_epi_pred`` from the official VideoPose3D
              lifter scaled to bbox units with the pelvis at (0.5, 0.5, 0.5)
              (the release uses EpipolarPose; any monocular lifter fills that
              role) -> ``inputs.npz`` + ``index.json``
    s1        (metapose env) stage-1 probabilistic bundle adjustment
              (:mod:`metapose_s1`, batched port validated against the official
              solver) -> ``s1.npz``
    train     (metapose env) per fold: the stage-2 network trained with the
              authors' ``train_metapose`` on the fold's training subjects
              (label-free losses: reprojection ``fwd`` + ``soln`` losses to the
              stage-1 optimum; model selection on the stage-1 optimum too),
              then applied to the fold's test subjects -> ``s2_<fold>.npz``.
              The Human3.6M schedule (300 epochs x up to 10 stages, patience
              50) is capped (``--epochs-per-stage``, ``--patience``,
              ``--max-stages``); the cap is recorded in the summary.
    s2        (metapose env) alternative to ``train``: the released 2-camera
              Human3.6M checkpoint ``ckpt/h36m/cam2`` on every frame (zero-shot,
              appendix only) -> ``s2.npz``
    evaluate  (this env)  the stage's 3D pose (H36M-17, bbox units of camera 0,
              scale-free) replaces both views on the trial's frames and is
              scored through the model protocol.

Outputs live below ``local/runs/external_published/metapose/<dataset>``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from common.paths import CHECKPOINT_ROOT, PROJECT_ROOT
from fusion.keypoints.schema import PosePairTrial

from .mapping import coco17_to_h36m17_2d, h36m17_to_mhr70, mhr70_to_coco17_2d
from .transform import LiftedTrialTransform, ViewSource

RELEASE_ROOT = CHECKPOINT_ROOT / "metapose" / "metapose"
THIRD_PARTY = Path(__file__).resolve().parents[1] / "third_party"
METAPOSE_PYTHON_ENV = "GYMNASTICS_METAPOSE_PYTHON"
HEATMAP_SIGMA_FRACTION = 0.02  # of the bounding-box size (the released heatmaps' dominant component)
BBOX_MARGIN = 0.15
N_MIX = 4


def metapose_python() -> Path:
    """Interpreter of the MetaPose TF env: ``$GYMNASTICS_METAPOSE_PYTHON``, else the
    sibling env ``metapose_gpu`` (TF 2.15 + CUDA) when present, else ``metapose`` (TF 2.8 CPU)."""
    configured = os.environ.get(METAPOSE_PYTHON_ENV)
    if configured:
        return Path(configured)
    envs = Path(sys.executable).resolve().parents[2]
    for name in ("metapose_gpu", "metapose"):
        candidate = envs / name / "bin" / "python"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no metapose conda env next to {envs}; set {METAPOSE_PYTHON_ENV}")


def _tf_env() -> dict[str, str]:
    # PYTHONNOUSERSITE: the user-site packages of the main env must not leak into the TF env.
    return {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src"), "PYTHONNOUSERSITE": "1"}


# ----------------------------------------------------------------------------- record construction
def square_bbox(points: np.ndarray, valid: np.ndarray, width: int, height: int, *, margin: float = BBOX_MARGIN) -> np.ndarray:
    """``[y0, y1, x0, x1]`` square box around the valid joints (release convention)."""
    pts = points[valid] if valid.any() else np.array([[width / 2, height / 2]], dtype=np.float32)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    size = max(x1 - x0, y1 - y0, 1.0) * (1.0 + 2 * margin)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    return np.array([round(cy - size / 2), round(cy + size / 2), round(cx - size / 2), round(cx + size / 2)], dtype=np.int32)


def gaussian_heatmaps(pose2d: np.ndarray, valid: np.ndarray, size: float) -> np.ndarray:
    """``[17, 4, 4]`` GMM parameters ``[weight, mu_x, mu_y, var]`` per joint (one component)."""
    heat = np.zeros((17, N_MIX, 4), dtype=np.float64)
    sigma = HEATMAP_SIGMA_FRACTION * size
    heat[:, 0, 0] = 1.0
    heat[:, 0, 1:3] = pose2d
    heat[:, 0, 3] = np.where(valid, sigma ** 2, (0.5 * size) ** 2)  # a missing joint is nearly uninformative
    heat[:, 1:, 1:3] = pose2d[:, None, :]
    heat[:, 1:, 3] = sigma ** 2
    return heat


def epi_init(h36m3d: np.ndarray, pose2d_norm: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Monocular 3D scaled to bbox units, pelvis at (0.5, 0.5, 0.5) (release convention)."""
    xy = h36m3d[:, :2]
    if valid.sum() >= 3:
        a = np.concatenate([xy[valid].reshape(-1, 1), np.tile(np.eye(2), (int(valid.sum()), 1))], axis=1)
        solution = np.linalg.lstsq(a, pose2d_norm[valid].reshape(-1), rcond=None)[0]
        scale = float(abs(solution[0]))
    else:
        scale = 1.0
    scale = scale if np.isfinite(scale) and scale > 1e-6 else 1.0
    return (scale * (h36m3d - h36m3d[0])) + 0.5


class RecordingTransform:
    """Collects MetaPose inputs for every trial instead of transforming it."""

    def __init__(self, lifted: LiftedTrialTransform, source: ViewSource) -> None:
        self.lifted, self.source = lifted, source
        self.rows: list[dict[str, np.ndarray]] = []
        self.index: list[dict[str, Any]] = []

    def __call__(self, trial: PosePairTrial) -> PosePairTrial:
        view_a, view_b = self.source.views(trial)
        start = len(self.rows)
        per_view = []
        for view, frame_map in ((view_a, trial.face_map), (view_b, trial.side_map)):
            points, valid = view.select(frame_map)
            coco, coco_valid = mhr70_to_coco17_2d(points, valid)
            h2d, h2d_valid = coco17_to_h36m17_2d(coco, coco_valid)
            lifted, frame_valid = self.lifted.lift_view_h36m(view)
            wanted = np.asarray(frame_map, dtype=np.int64)
            position = np.clip(np.searchsorted(view.frame_ids, wanted), 0, len(view.frame_ids) - 1)
            hit = view.frame_ids[position] == wanted
            h3d = np.where(hit[:, None, None], lifted[position], 0.0)
            per_view.append((h2d, h2d_valid, h3d, frame_valid[position] & hit, view.width, view.height))
        for t in range(len(trial.face_map)):
            row = {"pose2d": [], "bboxes": [], "heatmaps": [], "epi": [], "valid": []}
            for h2d, h2d_valid, h3d, ok, width, height in per_view:
                bbox = square_bbox(h2d[t], h2d_valid[t], width, height)
                size = float(max(bbox[1] - bbox[0], bbox[3] - bbox[2]))
                norm2d = (h2d[t] - np.array([bbox[2], bbox[0]], dtype=np.float32)) / size
                row["pose2d"].append(h2d[t])
                row["bboxes"].append(bbox)
                row["heatmaps"].append(gaussian_heatmaps(h2d[t], h2d_valid[t], size))
                row["epi"].append(epi_init(h3d[t], norm2d, h2d_valid[t]) if ok[t] else np.tile([0.5, 0.5, 0.5], (17, 1)))
                row["valid"].append(bool(ok[t]) and bool(h2d_valid[t].sum() >= 8))
            self.rows.append({k: np.stack(v) for k, v in row.items()})
        self.index.append({"person": trial.person_id, "trial": trial.trial_id, "start": start, "stop": len(self.rows), "frames": len(trial.face_map)})
        return trial

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        stacked = {k: np.stack([r[k] for r in self.rows]) for k in ("pose2d", "bboxes", "heatmaps", "epi", "valid")}
        np.savez_compressed(directory / "inputs.npz", **stacked)
        (directory / "index.json").write_text(json.dumps({"trials": self.index, "frames": len(self.rows)}, indent=1), encoding="utf-8")
        return directory / "inputs.npz"


# ----------------------------------------------------------------------------- stages in the TF env
def run_stage1(directory: Path, *, steps: int = 100, batch: int = 4096) -> Path:
    script = Path(__file__).with_name("metapose_s1.py")
    subprocess.run([str(metapose_python()), str(script), "run", "--inputs", str(directory / "inputs.npz"), "--output", str(directory / "s1.npz"), "--steps", str(steps), "--batch", str(batch)], check=True, env=_tf_env())
    return directory / "s1.npz"


def run_stage2_released(directory: Path) -> Path:
    """Released ``ckpt/h36m/cam2`` on every frame -> ``s2.npz`` (zero-shot; appendix only)."""
    script = Path(__file__).with_name("metapose_s2.py")
    subprocess.run([str(metapose_python()), str(script), "--mode", "released", "--directory", str(directory), "--release-root", str(RELEASE_ROOT), "--third-party", str(THIRD_PARTY)], check=True, env=_tf_env())
    return directory / "s2.npz"


def run_shards(directory: Path, *, workers: int = 8) -> Path:
    """Per-subject record shards (``shards/<person>/{train,test}``) for the per-fold training runs."""
    script = Path(__file__).with_name("metapose_s2.py")
    subprocess.run([str(metapose_python()), str(script), "--mode", "shards", "--directory", str(directory), "--release-root", str(RELEASE_ROOT), "--third-party", str(THIRD_PARTY), "--workers", str(workers)], check=True, env=_tf_env())
    return directory / "shards"


def stage2_name(fold: str, loss: str = "fwd") -> str:
    """``s2_<fold>`` for the default objective, ``s2_<loss>_<fold>`` for the README variants (ss, ts)."""
    return f"s2_{fold}" if loss == "fwd" else f"s2_{loss}_{fold}"


def run_stage2_trained(directory: Path, fold_json: Path, *, epochs_per_stage: int, patience: int, max_stages: int, seed: int = 0, loss: str = "fwd") -> Path:
    """Train stage 2 on the fold's training subjects, predict its test subjects -> ``s2[_<loss>]_<fold>.npz``
    (``metapose_s2.py`` writes per-subject record shards once and recombines them per fold)."""
    script = Path(__file__).with_name("metapose_s2.py")
    subprocess.run([str(metapose_python()), str(script), "--mode", "train", "--directory", str(directory), "--release-root", str(RELEASE_ROOT), "--third-party", str(THIRD_PARTY),
                    "--fold", fold_json.stem, "--fold-json", str(fold_json), "--epochs-per-stage", str(epochs_per_stage), "--patience", str(patience), "--max-stages", str(max_stages), "--seed", str(seed), "--loss", loss], check=True, env=_tf_env())
    return directory / f"{stage2_name(fold_json.stem, loss)}.npz"


def run_stage2_predict(directory: Path, fold_json: Path, *, loss: str = "fwd", stages: int = 1, seed: int = 0) -> Path:
    """Export the fold's test predictions from the saved best checkpoint of a (partially) trained run
    (``s2[_<loss>]_<fold>_best/model``), e.g. when a later refinement stage failed."""
    script = Path(__file__).with_name("metapose_s2.py")
    weights = directory / f"{stage2_name(fold_json.stem, loss)}_best" / "model"
    subprocess.run([str(metapose_python()), str(script), "--mode", "predict", "--directory", str(directory), "--release-root", str(RELEASE_ROOT), "--third-party", str(THIRD_PARTY),
                    "--fold", fold_json.stem, "--fold-json", str(fold_json), "--loss", loss, "--weights", str(weights), "--stages", str(stages), "--seed", str(seed)], check=True, env=_tf_env())
    return directory / f"{stage2_name(fold_json.stem, loss)}.npz"


# ----------------------------------------------------------------------------- evaluation transform
class MetaPoseTrialTransform:
    """Replace both views with the stage's 3D pose (H36M-17 -> MHR70 layout) on the trial's frames.

    Args:
        directory: The dataset's MetaPose directory (``inputs.npz``, ``s1.npz`` ...).
        stage: ``s2`` (stage-2 network), ``s1`` (stage-1 optimum) or ``init``
            (monocular initialisation).
        fold: With ``stage="s2"``: use the network trained for this fold
            (``s2_<fold>.npz``; only its test frames carry predictions).
            ``None`` uses the released checkpoint's ``s2.npz``.
        loss: Stage-2 objective variant of the trained run (``fwd``, ``ss``, ``ts``).
    """

    def __init__(self, directory: Path, stage: str, *, fold: str | None = None, loss: str = "fwd") -> None:
        if stage not in {"s1", "s2", "init"}:
            raise ValueError("stage must be s1, s2 or init")
        index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        self.slices = {(t["person"], t["trial"]): (t["start"], t["stop"]) for t in index["trials"]}
        s1 = np.load(directory / "s1.npz")
        self.valid = np.asarray(s1["usable"], dtype=bool)
        if stage == "s2" and fold is not None:
            trained = np.load(directory / f"{stage2_name(fold, loss)}.npz")
            self.pose = np.full(s1["pose_opt"].shape, np.nan, dtype=np.float32)
            self.pose[trained["rows"]] = trained["pose"]
        elif stage == "s2":
            self.pose = np.load(directory / "s2.npz")["pose"]
        else:
            self.pose = s1["pose_opt"] if stage == "s1" else s1["pose_init"]
        self.valid &= np.isfinite(self.pose).all(axis=(1, 2))
        self.stage = stage
        self.fold = fold
        self.loss = loss

    def __call__(self, trial: PosePairTrial) -> PosePairTrial:
        key = (trial.person_id, trial.trial_id)
        if key not in self.slices:
            raise KeyError(f"no MetaPose records for trial {key}; run prepare on the same data")
        start, stop = self.slices[key]
        if stop - start != len(trial.face_map):
            raise ValueError(f"MetaPose records for {key} cover {stop - start} frames, trial has {len(trial.face_map)}")
        pose, valid = h36m17_to_mhr70(self.pose[start:stop])
        valid &= self.valid[start:stop][:, None]
        pose = np.where(valid[..., None], pose, 0.0)
        metadata = {**dict(trial.source_metadata), "external_method": f"metapose_{self.stage}" + (f"_{self.loss}_{self.fold}" if self.fold else "")}
        return replace(trial, face=pose, side=pose, valid_face=valid, valid_side=valid, source_metadata=metadata)


def prepare_dataset(dataset: str, lifter, source: ViewSource, *, output_dir: Path, lifted_cache: Path, extra_overrides: Sequence[str] = ()) -> Path:
    """Run every trial of ``dataset`` through :class:`RecordingTransform` and save the inputs."""
    from omegaconf import OmegaConf

    from fusion.data import build_datamodule
    from fusion.train import compose_config

    recorder = RecordingTransform(LiftedTrialTransform(lifter, source, cache_dir=lifted_cache, method="videopose3d"), source)
    overrides = [f"data={dataset}", "data.fold_json=null", "data.cache_dir=null", "data.attach_reference=false", "data.num_workers=0", "data.cycle_target.enabled=false", *extra_overrides]
    cfg = compose_config(overrides)
    datamodule = build_datamodule(OmegaConf.to_container(cfg.data, resolve=True), trial_transform=recorder)  # type: ignore[arg-type]
    datamodule.load_samples()
    path = recorder.save(output_dir)
    print(f"[metapose] {dataset}: {len(recorder.index)} trials, {len(recorder.rows)} frames -> {path}")
    return path

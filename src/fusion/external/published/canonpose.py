"""CanonPose (Wandt et al., CVPR 2021) trained with its own recipe on this project's two views.

CanonPose is a self-supervised lifter: from the 2D detections of several
uncalibrated cameras it learns a canonical 3D pose and a per-view camera
rotation with a reprojection loss, a view-consistency loss (every view's
canonical pose reprojected into the other views) and a camera-consistency
loss (relative rotations shuffled between frames of the same subject). It
needs no 3D labels and no calibration, i.e. exactly this project's training
regime, so it is trained per fold on the training subjects' SAM3D 2D
keypoints of the two views and evaluated on the held-out subjects.

What is the authors' and what is ours:

* model (``model_confidences.Lifter``), the three losses, optimiser,
  schedule, batch size and epochs: the released code / defaults
  (git submodule ``fusion/external/third_party/CanonPose``; the training loop
  is a top-level script there and is reproduced verbatim as a function here);
* Rodrigues map: ``pytorch3d.transforms.so3_exponential_map`` when
  installed, else the same closed form in torch;
* 2D input: SAM3D keypoints mapped to the MPII-16 layout the model uses
  (pelvis / thorax / upper neck / head derived as in :mod:`mapping`),
  root-centred and L2-normalised like ``utils/data.py``; joint confidences
  are the SAM3D validity (1 / 0) because SAM3D emits no detection scores;
* the detector-specific skeleton-morphing network of Sec. 4.2 is skipped
  (it must be trained with 2D ground truth, which does not exist here);
* two-view result: the canonical poses of both views averaged (the
  canonical frame is shared by construction), also scored per view.
"""

from __future__ import annotations

import importlib.util
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from common.paths import PROJECT_ROOT
from common.skeletons.mhr70 import MHR70_INDEX
from fusion.keypoints.schema import PosePairTrial

from .fuse import procrustes_average
from .mapping import mhr70_to_coco17_2d
from .transform import ViewSource

THIRD_PARTY = Path(__file__).resolve().parents[1] / "third_party" / "CanonPose"

# MPII-16 layout used by CanonPose (Human3.6M "mpi skeleton" order).
MPII16_NAMES = ("right-ankle", "right-knee", "right-hip", "left-hip", "left-knee", "left-ankle", "pelvis", "thorax", "upper-neck", "head-top", "right-wrist", "right-elbow", "right-shoulder", "left-shoulder", "left-elbow", "left-wrist")
ROOT_INDEX = 6
_COCO = {"nose": 0, "left-shoulder": 5, "right-shoulder": 6, "left-elbow": 7, "right-elbow": 8, "left-wrist": 9, "right-wrist": 10, "left-hip": 11, "right-hip": 12, "left-knee": 13, "right-knee": 14, "left-ankle": 15, "right-ankle": 16}
MPII16_TO_MHR70 = {0: MHR70_INDEX["right-ankle"], 1: MHR70_INDEX["right-knee"], 2: MHR70_INDEX["right-hip"], 3: MHR70_INDEX["left-hip"], 4: MHR70_INDEX["left-knee"], 5: MHR70_INDEX["left-ankle"], 7: MHR70_INDEX["neck"], 10: MHR70_INDEX["right-wrist"], 11: MHR70_INDEX["right-elbow"], 12: MHR70_INDEX["right-shoulder"], 13: MHR70_INDEX["left-shoulder"], 14: MHR70_INDEX["left-elbow"], 15: MHR70_INDEX["left-wrist"]}

# Released defaults (train.py).
CONFIG = {"learning_rate": 1e-4, "batch_size": 32, "epochs": 100, "weight_rep": 1.0, "weight_view": 1.0, "weight_camera": 0.1, "milestones": (30, 60, 90), "gamma": 0.1, "weight_decay": 1e-5}


def _load_lifter_class():
    path = THIRD_PARTY / "model_confidences.py"
    if not path.is_file():
        raise FileNotFoundError(f"CanonPose submodule missing at {THIRD_PARTY}; run `git submodule update --init`")
    spec = importlib.util.spec_from_file_location("canonpose_model_confidences", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.Lifter


def rodrigues(angles):
    """``so3_exponential_map`` (axis-angle -> rotation matrix); pytorch3d when available."""
    try:
        from pytorch3d.transforms import so3_exponential_map  # type: ignore

        return so3_exponential_map(angles)
    except ImportError:
        import torch

        theta = torch.linalg.vector_norm(angles, dim=-1, keepdim=True).clamp_min(1e-8)
        axis = angles / theta
        k = torch.zeros(angles.shape[0], 3, 3, dtype=angles.dtype, device=angles.device)
        k[:, 0, 1], k[:, 0, 2], k[:, 1, 0], k[:, 1, 2], k[:, 2, 0], k[:, 2, 1] = -axis[:, 2], axis[:, 1], axis[:, 2], -axis[:, 0], -axis[:, 1], axis[:, 0]
        eye = torch.eye(3, dtype=angles.dtype, device=angles.device)[None]
        theta = theta[..., None]
        return eye + torch.sin(theta) * k + (1 - torch.cos(theta)) * (k @ k)


# ----------------------------------------------------------------------------- 2D preparation
def coco17_to_mpii16_2d(points: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``[T, 17, 2]`` COCO17 -> ``[T, 16, 2]`` MPII-16 (+ validity)."""
    src = np.asarray(points, dtype=np.float32)
    ok = np.asarray(valid, dtype=bool)
    out = np.zeros((src.shape[0], 16, 2), np.float32)
    out_valid = np.zeros((src.shape[0], 16), bool)
    for index, name in enumerate(MPII16_NAMES):
        if name in _COCO:
            out[:, index], out_valid[:, index] = src[:, _COCO[name]], ok[:, _COCO[name]]
    pelvis = 0.5 * (src[:, _COCO["left-hip"]] + src[:, _COCO["right-hip"]])
    thorax = 0.5 * (src[:, _COCO["left-shoulder"]] + src[:, _COCO["right-shoulder"]])
    hips_ok = ok[:, _COCO["left-hip"]] & ok[:, _COCO["right-hip"]]
    shoulders_ok = ok[:, _COCO["left-shoulder"]] & ok[:, _COCO["right-shoulder"]]
    out[:, 6], out_valid[:, 6] = pelvis, hips_ok
    out[:, 7], out_valid[:, 7] = thorax, shoulders_ok
    nose = src[:, _COCO["nose"]]
    out[:, 8], out_valid[:, 8] = 0.5 * (thorax + nose), shoulders_ok & ok[:, 0]
    out[:, 9], out_valid[:, 9] = nose + 0.5 * (nose - thorax), shoulders_ok & ok[:, 0]
    out[~out_valid] = 0.0
    return out, out_valid


def normalise_2d(points: np.ndarray) -> np.ndarray:
    """``utils/data.py``: root-centre at the pelvis, then L2-normalise the 32-vector (x16, y16)."""
    centred = points - points[:, ROOT_INDEX : ROOT_INDEX + 1]
    flat = np.concatenate([centred[:, :, 0], centred[:, :, 1]], axis=1)  # [T, 32] = (x..., y...)
    norm = np.linalg.norm(flat, ord=2, axis=1, keepdims=True)
    return flat / np.maximum(norm, 1e-8)


class RecordingTransform:
    """Collects the normalised 2D of both views (+ subject ids) for every trial."""

    def __init__(self, source: ViewSource) -> None:
        self.source = source
        self.rows: list[dict[str, np.ndarray]] = []
        self.index: list[dict[str, Any]] = []

    def __call__(self, trial: PosePairTrial) -> PosePairTrial:
        view_a, view_b = self.source.views(trial)
        start = len(self.rows)
        columns = []
        for view, frame_map in ((view_a, trial.face_map), (view_b, trial.side_map)):
            points, valid = view.select(frame_map)
            coco, coco_valid = mhr70_to_coco17_2d(points, valid)
            mpii, mpii_valid = coco17_to_mpii16_2d(coco, coco_valid)
            columns.append((normalise_2d(mpii), mpii_valid))
        for t in range(len(trial.face_map)):
            self.rows.append({"p2d": np.stack([c[0][t] for c in columns]), "conf": np.stack([c[1][t].astype(np.float32) for c in columns])})
        self.index.append({"person": trial.person_id, "trial": trial.trial_id, "start": start, "stop": len(self.rows), "frames": len(trial.face_map)})
        return trial

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "inputs.npz", p2d=np.stack([r["p2d"] for r in self.rows]), conf=np.stack([r["conf"] for r in self.rows]))
        (directory / "index.json").write_text(json.dumps({"trials": self.index, "frames": len(self.rows)}, indent=1), encoding="utf-8")
        return directory / "inputs.npz"


def prepare_dataset(dataset: str, source: ViewSource, *, output_dir: Path, extra_overrides: Sequence[str] = ()) -> Path:
    from omegaconf import OmegaConf

    from fusion.data import build_datamodule
    from fusion.train import compose_config

    recorder = RecordingTransform(source)
    cfg = compose_config([f"data={dataset}", "data.fold_json=null", "data.cache_dir=null", "data.attach_reference=false", "data.num_workers=0", "data.cycle_target.enabled=false", *extra_overrides])
    build_datamodule(OmegaConf.to_container(cfg.data, resolve=True), trial_transform=recorder).load_samples()  # type: ignore[arg-type]
    path = recorder.save(output_dir)
    print(f"[canonpose] {dataset}: {len(recorder.index)} trials, {len(recorder.rows)} frames -> {path}")
    return path


# ----------------------------------------------------------------------------- training (train.py, as a function)
def loss_weighted_rep_no_scale(p2d, p3d, confs):
    """Equation 5 of the paper, copied from ``train.py``."""
    import torch

    scale_p2d = torch.sqrt(p2d[:, 0:32].square().sum(axis=1, keepdim=True) / 32)
    p2d_scaled = p2d[:, 0:32] / scale_p2d
    scale_p3d = torch.sqrt(p3d[:, 0:32].square().sum(axis=1, keepdim=True) / 32)
    p3d_scaled = p3d[:, 0:32] / scale_p3d
    return ((p2d_scaled - p3d_scaled).abs().reshape(-1, 2, 16).sum(axis=1) * confs).sum() / (p2d_scaled.shape[0] * p2d_scaled.shape[1])


def train(inputs: Path, index: Path, train_persons: Sequence[str], output: Path, *, device: str = "cuda", epochs: int | None = None, seed: int = 0, log_every: int = 200) -> Path:
    """Train the lifter on the frames of ``train_persons`` with the released loop and defaults."""
    import torch
    import torch.optim as optim

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = np.load(inputs)
    meta = json.loads(Path(index).read_text(encoding="utf-8"))
    wanted = set(str(p) for p in train_persons)
    rows, subjects = [], []
    for entry in meta["trials"]:
        if entry["person"] in wanted:
            rows.append(np.arange(entry["start"], entry["stop"]))
            subjects.extend([entry["person"]] * (entry["stop"] - entry["start"]))
    rows = np.concatenate(rows)
    p2d = data["p2d"][rows]  # [N, C, 32]
    conf = data["conf"][rows]  # [N, C, 16]
    keep = (conf.sum(axis=2) >= 8).all(axis=1)  # frames with a usable pose in every view
    p2d, conf = p2d[keep], conf[keep]
    subject_ids = np.array([hash(s) % (1 << 31) for s in np.array(subjects)[keep]])
    n_cam = p2d.shape[1]
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    Lifter = _load_lifter_class()
    model = Lifter().to(dev)
    optimizer = optim.Adam(list(model.parameters()), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(CONFIG["milestones"]), gamma=CONFIG["gamma"])
    batch_size = CONFIG["batch_size"]
    n_epochs = int(epochs or CONFIG["epochs"])
    p2d_t = torch.from_numpy(p2d.astype(np.float32)).to(dev)
    conf_t = torch.from_numpy(conf.astype(np.float32)).to(dev)
    subj_t = torch.from_numpy(subject_ids).to(dev)
    n = p2d_t.shape[0]
    steps = math.ceil(n / batch_size)
    history = []
    t0 = time.time()
    for epoch in range(n_epochs):
        order = torch.from_numpy(rng.permutation(n)).to(dev)
        sums = {"loss": 0.0, "rep": 0.0, "view": 0.0, "camera": 0.0}
        for step in range(steps):
            batch = order[step * batch_size : (step + 1) * batch_size]
            inp_poses = p2d_t[batch].reshape(-1, 32)  # frame-major, camera-minor like train.py
            inp_confidences = conf_t[batch].reshape(-1, 16)
            sample_subjects = subj_t[batch]
            pred_poses, pred_cam_angles = model(inp_poses, inp_confidences)
            pred_rot = rodrigues(pred_cam_angles)
            rot_poses = pred_rot.matmul(pred_poses.reshape(-1, 3, 16)).reshape(-1, 48)
            loss_rep = loss_weighted_rep_no_scale(inp_poses, rot_poses, inp_confidences)
            pred_poses_rs = pred_poses.reshape((-1, n_cam, 48))
            pred_rot_rs = pred_rot.reshape(-1, n_cam, 3, 3)
            confidences_rs = inp_confidences.reshape(-1, n_cam, 16)
            inp_poses_rs = inp_poses.reshape(-1, n_cam, 32)
            rot_poses_rs = rot_poses.reshape(-1, n_cam, 48)
            loss_view = 0.0
            loss_camera = 0.0
            for c_cnt in range(n_cam):
                coi = np.delete(np.arange(n_cam), c_cnt)
                projected = pred_rot_rs[:, coi].matmul(pred_poses_rs.reshape(-1, n_cam, 3, 16)[:, c_cnt : c_cnt + 1].repeat(1, n_cam - 1, 1, 1)).reshape(-1, n_cam - 1, 48)
                loss_view = loss_view + loss_weighted_rep_no_scale(inp_poses_rs[:, coi].reshape(-1, 32), projected.reshape(-1, 48), confidences_rs[:, coi].reshape(-1, 16))
                relative_rotations = pred_rot_rs[:, coi].matmul(pred_rot_rs[:, [c_cnt]].permute(0, 1, 3, 2))
                for subject in sample_subjects.unique():
                    mask = sample_subjects == subject
                    count = int(mask.sum())
                    if count > 1:
                        perm = torch.from_numpy(rng.choice(count, size=count, replace=False)).to(dev)
                        samp_rel = relative_rotations[mask]
                        samp_rot_poses = rot_poses_rs[mask]
                        samp_inp = inp_poses_rs[mask][:, coi].reshape(-1, 32)
                        samp_conf = confidences_rs[mask][:, coi].reshape(-1, 16)
                        shuffled = samp_rel[perm].matmul(samp_rot_poses.reshape(-1, n_cam, 3, 16)[:, c_cnt : c_cnt + 1].repeat(1, n_cam - 1, 1, 1)).reshape(-1, n_cam - 1, 48)
                        loss_camera = loss_camera + loss_weighted_rep_no_scale(samp_inp, shuffled.reshape(-1, 48), samp_conf)
            loss = CONFIG["weight_rep"] * loss_rep + CONFIG["weight_view"] * loss_view + CONFIG["weight_camera"] * loss_camera
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sums["loss"] += float(loss)
            sums["rep"] += float(loss_rep)
            sums["view"] += float(loss_view)
            sums["camera"] += float(loss_camera) if torch.is_tensor(loss_camera) else float(loss_camera)
        scheduler.step()
        record = {"epoch": epoch, **{k: v / steps for k, v in sums.items()}, "seconds": time.time() - t0}
        history.append(record)
        if epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1:
            print(f"[canonpose] epoch {epoch + 1}/{n_epochs} loss {record['loss']:.4f} rep {record['rep']:.4f} view {record['view']:.4f} cam {record['camera']:.4f} ({record['seconds']:.0f} s)")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": CONFIG, "train_persons": sorted(wanted), "frames": int(n), "history": history}, output)
    return output


# ----------------------------------------------------------------------------- inference transform
class CanonPoseTrialTransform:
    """Replace both views with the trained lifter's canonical poses (averaged or per view)."""

    def __init__(self, checkpoint: Path, source: ViewSource, *, mode: str = "canonical_average", device: str = "cuda") -> None:
        import torch

        if mode not in {"canonical_average", "procrustes_average", "per_view"}:
            raise ValueError("mode must be canonical_average, procrustes_average or per_view")
        self.mode, self.source = mode, source
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = _load_lifter_class()()
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device).eval()

    def lift(self, view, frame_map) -> tuple[np.ndarray, np.ndarray]:
        import torch

        points, valid = view.select(frame_map)
        coco, coco_valid = mhr70_to_coco17_2d(points, valid)
        mpii, mpii_valid = coco17_to_mpii16_2d(coco, coco_valid)
        flat = torch.from_numpy(normalise_2d(mpii)).to(self.device)
        conf = torch.from_numpy(mpii_valid.astype(np.float32)).to(self.device)
        with torch.no_grad():
            pred, _ = self.model(flat, conf)
        canonical = pred.reshape(-1, 3, 16).permute(0, 2, 1).cpu().numpy()  # [T, 16, 3]
        frame_ok = mpii_valid.sum(axis=1) >= 8
        pose = np.zeros((len(frame_map), 70, 3), np.float32)
        pose_valid = np.zeros((len(frame_map), 70), bool)
        for mpii_index, mhr_index in MPII16_TO_MHR70.items():
            pose[:, mhr_index] = canonical[:, mpii_index]
            pose_valid[:, mhr_index] = mpii_valid[:, mpii_index] & frame_ok
        return np.where(pose_valid[..., None], pose, 0.0), pose_valid

    def __call__(self, trial: PosePairTrial) -> PosePairTrial:
        view_a, view_b = self.source.views(trial)
        pose_a, valid_a = self.lift(view_a, trial.face_map)
        pose_b, valid_b = self.lift(view_b, trial.side_map)
        if self.mode == "canonical_average":
            both = valid_a & valid_b
            fused = np.where(both[..., None], 0.5 * (pose_a + pose_b), np.where(valid_a[..., None], pose_a, pose_b))
            valid = valid_a | valid_b
            pose_a = pose_b = np.where(valid[..., None], fused, 0.0)
            valid_a = valid_b = valid
        elif self.mode == "procrustes_average":
            fused, valid = procrustes_average(pose_a, valid_a, pose_b, valid_b)
            pose_a = pose_b = fused
            valid_a = valid_b = valid
        metadata = {**dict(trial.source_metadata), "external_method": "canonpose", "external_mode": self.mode}
        return replace(trial, face=pose_a, side=pose_b, valid_face=valid_a, valid_side=valid_b, source_metadata=metadata)


def fold_train_persons(fold_json: Path) -> list[str]:
    fold = json.loads(Path(fold_json).read_text(encoding="utf-8"))
    return [str(s) for s in fold["train"]]


def output_root(dataset: str) -> Path:
    return PROJECT_ROOT / "local" / "runs" / "external_published" / "canonpose" / dataset

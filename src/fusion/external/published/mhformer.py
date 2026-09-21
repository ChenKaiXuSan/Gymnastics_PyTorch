"""MHFormer (Li et al., CVPR 2022) trained with its recipe on the reference joints (supervised).

MHFormer is a monocular video lifter: a window of 2D detections in, the 3D
pose of the centre frame in camera coordinates (root-relative) out, trained
with the MPJPE to 3D ground truth. It is a *supervised* baseline, so it is
trained only on the datasets that carry an independent 3D reference (FreeMan,
SportsPose): per fold, on the training subjects' two views (each view is a
monocular sample), with the reference joints rotated into each camera, and
evaluated on the held-out subjects. The private recordings have no such
reference, so there is no MHFormer row for them.

What is the authors' and what is ours:

* model (``model/mhformer.py``, git submodule
  ``fusion/external/third_party/MHFormer``), loss (MPJPE with the root at
  zero, all window frames supervised, ``out_all``), optimiser (Adam amsgrad
  1e-3), schedule (x0.95 per epoch, x0.5 every 5 epochs, 19 epochs =
  ``range(1, 20)``), flip augmentation, flip test-time augmentation, centre
  frame output, edge padding and the 81-frame configuration
  (``--frames 81 --batch_size 256``; the 351-frame variant is 4x the cost):
  the released ``main.py`` / ``common/`` reproduced here as functions;
* input: SAM3D 2D keypoints in the Human3.6M-17 layout (instead of the CPN
  detections), normalised with ``normalize_screen_coordinates``; joints
  without a detection are linearly interpolated in time (the release has no
  notion of a missing joint), and frames without a reference are masked out
  of the loss;
* target: the dataset's reference joints (COCO17) in the H36M-17 layout
  (pelvis / thorax / spine / head derived as in :mod:`mapping`), rotated into
  the view's camera with the calibrated world-to-camera rotation;
* model selection: best validation MPJPE on the fold's *validation*
  subjects (the release selects on the Human3.6M test set);
* two-view result: the two monocular poses Procrustes-averaged
  (``procrustes_average``), also scored per view.
"""

from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from common.paths import PROJECT_ROOT
from fusion.keypoints.schema import PosePairTrial

from .mapping import H36M17_LEFT, H36M17_RIGHT, coco17_to_h36m17, fill_missing_joints, mhr70_to_coco17_2d, named_to_h36m17
from .transform import ViewSource
from .videopose3d import normalize_screen_coordinates

THIRD_PARTY = Path(__file__).resolve().parents[1] / "third_party" / "MHFormer"

# Released defaults (common/opt.py, README: 81-frame model).
CONFIG: dict[str, Any] = {"frames": 81, "layers": 3, "channel": 512, "d_hid": 1024, "n_joints": 17, "out_joints": 17, "batch_size": 256, "lr": 1e-3, "lr_decay": 0.95, "lr_decay_large": 0.5, "large_decay_epoch": 5, "nepoch": 20, "amsgrad": True, "data_augmentation": True, "test_augmentation": True, "seed": 1, "workers": 8}
LEFT, RIGHT = list(H36M17_LEFT), list(H36M17_RIGHT)


def _load_model_class():
    """``model.mhformer.Model`` from the submodule (its package is ``model``; appended so nothing of ours is shadowed)."""
    path = THIRD_PARTY / "model" / "mhformer.py"
    if not path.is_file():
        raise FileNotFoundError(f"MHFormer submodule missing at {THIRD_PARTY}; run `git submodule update --init`")
    if str(THIRD_PARTY) not in sys.path:
        sys.path.append(str(THIRD_PARTY))
    spec = importlib.util.spec_from_file_location("mhformer_model", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.Model


def build_model(config: dict[str, Any] = CONFIG):
    args = SimpleNamespace(**{k: config[k] for k in ("frames", "layers", "channel", "d_hid", "n_joints", "out_joints")})
    return _load_model_class()(args)


# ----------------------------------------------------------------------------- records
class RecordingTransform:
    """Collects, per trial and view, the 2D input (H36M-17, normalised) and the camera rotation."""

    def __init__(self, source: ViewSource) -> None:
        self.source = source
        self.views: dict[tuple[str, str], list[dict[str, Any]]] = {}

    def __call__(self, trial: PosePairTrial) -> PosePairTrial:
        view_a, view_b = self.source.views(trial)
        rot_a, rot_b = self.source.rotations(trial)
        entries = []
        for name, view, frame_map, rotation in (("a", view_a, trial.face_map, rot_a), ("b", view_b, trial.side_map, rot_b)):
            points, valid = view.select(frame_map)
            coco, coco_valid = mhr70_to_coco17_2d(points, valid)
            h2d, h2d_valid = coco17_to_h36m17(coco, coco_valid)
            entries.append({"view": name, "pose2d": normalize_screen_coordinates(fill_missing_joints(h2d, h2d_valid), view.width, view.height), "valid2d": h2d_valid, "rotation": np.asarray(rotation, dtype=np.float64), "size": (view.width, view.height)})
        self.views[(trial.person_id, trial.trial_id)] = entries
        return trial


def prepare_dataset(dataset: str, source: ViewSource, *, output_dir: Path, extra_overrides: Sequence[str] = ()) -> Path:
    """Inputs and camera-frame targets of every trial and view -> ``inputs.npz`` + ``index.json``."""
    from omegaconf import OmegaConf

    from fusion.data import build_datamodule
    from fusion.train import compose_config

    recorder = RecordingTransform(source)
    cfg = compose_config([f"data={dataset}", "data.fold_json=null", "data.cache_dir=null", "data.attach_reference=true", "data.num_workers=0", "data.cycle_target.enabled=false", *extra_overrides])
    datamodule = build_datamodule(OmegaConf.to_container(cfg.data, resolve=True), trial_transform=recorder)  # type: ignore[arg-type]
    samples = datamodule.load_samples()
    names = tuple(datamodule.skeleton.joint_names)
    rows: dict[str, list[np.ndarray]] = {"pose2d": [], "valid2d": [], "pose3d": [], "valid3d": []}
    index: list[dict[str, Any]] = []
    start = 0
    for sample in samples:
        key = (str(sample.subject_id), str(sample.sequence_id))
        if key not in recorder.views:
            raise KeyError(f"no 2D views recorded for {key}")
        if sample.reference is None:
            continue
        world, world_valid = named_to_h36m17(sample.reference, sample.reference_valid, names)
        for entry in recorder.views[key]:
            camera = np.einsum("ij,tkj->tki", entry["rotation"], world.astype(np.float64)).astype(np.float32)
            valid3d = world_valid & world_valid[:, :1]  # root-relative targets need the pelvis
            camera = np.where(valid3d[..., None], camera - camera[:, :1], 0.0).astype(np.float32)
            frames = camera.shape[0]
            rows["pose2d"].append(entry["pose2d"].astype(np.float32))
            rows["valid2d"].append(entry["valid2d"])
            rows["pose3d"].append(camera)
            rows["valid3d"].append(valid3d)
            index.append({"person": key[0], "trial": key[1], "view": entry["view"], "start": start, "stop": start + frames, "size": list(entry["size"])})
            start += frames
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "inputs.npz", **{k: np.concatenate(v) for k, v in rows.items()})
    (output_dir / "index.json").write_text(json.dumps({"sequences": index, "frames": start, "joint_layout": "h36m17"}, indent=1), encoding="utf-8")
    print(f"[mhformer] {dataset}: {len(index)} view sequences, {start} frames -> {output_dir / 'inputs.npz'}")
    return output_dir / "inputs.npz"


# ----------------------------------------------------------------------------- windows
def _pad_edges(sequence: np.ndarray, pad: int) -> np.ndarray:
    return np.pad(sequence, ((pad, pad),) + ((0, 0),) * (sequence.ndim - 1), "edge")


def _flip_2d(window: np.ndarray) -> np.ndarray:
    out = window.copy()
    out[..., 0] *= -1
    out[..., LEFT + RIGHT, :] = out[..., RIGHT + LEFT, :]
    return out


class WindowDataset:
    """Every frame of every sequence as a centred window (``chunk_length`` 1, edge padded), flip copies when augmenting."""

    def __init__(self, sequences: list[dict[str, np.ndarray]], *, frames: int, augment: bool) -> None:
        import torch

        self.torch = torch
        self.pad = (frames - 1) // 2
        self.sequences = [{k: (_pad_edges(v, self.pad) if k in ("pose2d", "pose3d", "valid3d") else v) for k, v in s.items()} for s in sequences]
        self.items = [(i, t, flip) for i, s in enumerate(sequences) for t in range(s["pose2d"].shape[0]) for flip in ((False, True) if augment else (False,))]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, item: int):
        i, t, flip = self.items[item]
        s = self.sequences[i]
        window = slice(t, t + 2 * self.pad + 1)
        pose2d, pose3d, valid3d = s["pose2d"][window], s["pose3d"][window], s["valid3d"][window]
        if flip:
            pose2d, pose3d = _flip_2d(pose2d), _flip_2d(pose3d)
            valid3d = valid3d.copy()
            valid3d[:, LEFT + RIGHT] = valid3d[:, RIGHT + LEFT]
        return self.torch.from_numpy(np.ascontiguousarray(pose2d)), self.torch.from_numpy(np.ascontiguousarray(pose3d)), self.torch.from_numpy(np.ascontiguousarray(valid3d))


def load_sequences(inputs: Path, index: Path, persons: Sequence[str]) -> list[dict[str, np.ndarray]]:
    data = np.load(inputs)
    meta = json.loads(Path(index).read_text(encoding="utf-8"))
    wanted = {str(p) for p in persons}
    arrays = {k: data[k] for k in ("pose2d", "pose3d", "valid3d")}
    return [{k: arrays[k][s["start"]:s["stop"]] for k in arrays} for s in meta["sequences"] if str(s["person"]) in wanted]


def masked_mpjpe(predicted, target, valid):
    """MPJPE over the valid (frame, joint) pairs; the root target is zero like ``out_target[:, :, 0] = 0``."""
    target = target.clone()
    target[:, :, 0] = 0
    error = (predicted - target).norm(dim=-1)
    weight = valid.to(error.dtype)
    return (error * weight).sum() / weight.sum().clamp_min(1.0)


def predict_centre(model, pose2d, *, pad: int, flip_tta: bool):
    """Centre-frame prediction with the release's flip test-time augmentation."""
    out = model(pose2d)[:, pad]
    if flip_tta:
        flipped = pose2d.clone()
        flipped[..., 0] *= -1
        flipped[..., LEFT + RIGHT, :] = flipped[..., RIGHT + LEFT, :]
        out_flip = model(flipped)[:, pad]
        out_flip[..., 0] *= -1
        out_flip[:, LEFT + RIGHT] = out_flip[:, RIGHT + LEFT]
        out = (out + out_flip) / 2
    out[:, 0] = 0
    return out


def evaluate_sequences(model, sequences: list[dict[str, np.ndarray]], *, config: dict[str, Any], device) -> float:
    """MPJPE (mm) of the centre frame over every frame of ``sequences`` (release protocol, validity-masked)."""
    import torch

    dataset = WindowDataset(sequences, frames=config["frames"], augment=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["workers"], pin_memory=True)
    pad = (config["frames"] - 1) // 2
    total, count = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for pose2d, pose3d, valid3d in loader:
            pose2d, pose3d, valid3d = pose2d.to(device), pose3d.to(device)[:, pad], valid3d.to(device)[:, pad]
            predicted = predict_centre(model, pose2d, pad=pad, flip_tta=config["test_augmentation"])
            pose3d[:, 0] = 0
            error = (predicted - pose3d).norm(dim=-1)
            total += float((error * valid3d).sum())
            count += float(valid3d.sum())
    return 1000.0 * total / max(count, 1.0)


def train(inputs: Path, index: Path, train_persons: Sequence[str], val_persons: Sequence[str], output: Path, *, device: str = "cuda", config: dict[str, Any] | None = None, epochs: int | None = None) -> Path:
    """The released training loop (``main.py``) on the fold's training subjects; best validation epoch kept."""
    import torch

    cfg = {**CONFIG, **(config or {})}
    if epochs is not None:
        cfg["nepoch"] = epochs
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    train_sequences = load_sequences(inputs, index, train_persons)
    val_sequences = load_sequences(inputs, index, val_persons)
    if not train_sequences or not val_sequences:
        raise ValueError(f"{len(train_sequences)} training / {len(val_sequences)} validation sequences")
    dataset = WindowDataset(train_sequences, frames=cfg["frames"], augment=cfg["data_augmentation"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["workers"], pin_memory=True, drop_last=False)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], amsgrad=cfg["amsgrad"])
    lr = cfg["lr"]
    best, best_epoch = math.inf, 0
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[mhformer] {len(train_sequences)} training sequences ({len(dataset)} windows incl. flips), {len(val_sequences)} validation sequences, {cfg['nepoch'] - 1} epochs, device {device}")
    for epoch in range(1, cfg["nepoch"]):
        model.train()
        started, total, count = time.time(), 0.0, 0
        for pose2d, pose3d, valid3d in loader:
            pose2d, pose3d, valid3d = pose2d.to(device, non_blocking=True), pose3d.to(device, non_blocking=True), valid3d.to(device, non_blocking=True)
            loss = masked_mpjpe(model(pose2d), pose3d, valid3d)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * pose2d.shape[0]
            count += pose2d.shape[0]
        p1 = evaluate_sequences(model, val_sequences, config=cfg, device=device)
        if p1 < best:
            best, best_epoch = p1, epoch
            torch.save({"state_dict": model.state_dict(), "config": cfg, "epoch": epoch, "val_mpjpe_mm": p1}, output)
        print(f"[mhformer] epoch {epoch} lr {lr:.7f} loss {1000 * total / max(count, 1):.2f} mm val {p1:.2f} mm best {best:.2f} (epoch {best_epoch}) {time.time() - started:.0f}s")
        factor = cfg["lr_decay_large"] if epoch % cfg["large_decay_epoch"] == 0 else cfg["lr_decay"]
        lr *= factor
        for group in optimizer.param_groups:
            group["lr"] *= factor
    return output


# ----------------------------------------------------------------------------- inference (Lifter protocol)
class MHFormerLifter:
    """``lift(coco17 [T, 17, 2], width, height) -> h36m17 [T, 17, 3]`` with a trained checkpoint."""

    def __init__(self, checkpoint: Path, device: str = "cuda", *, batch_size: int = 256) -> None:
        import torch

        payload = torch.load(checkpoint, map_location="cpu")
        self.config = {**CONFIG, **payload.get("config", {})}
        self.model = build_model(self.config)
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(device).eval()
        self.device, self.batch_size = device, batch_size
        self.pad = (self.config["frames"] - 1) // 2

    def lift(self, coco17: np.ndarray, width: int, height: int) -> np.ndarray:
        import torch

        points = np.asarray(coco17, dtype=np.float32)
        h2d, h2d_valid = coco17_to_h36m17(points)
        normalised = normalize_screen_coordinates(fill_missing_joints(h2d, h2d_valid), width, height)
        padded = _pad_edges(normalised, self.pad)
        frames = points.shape[0]
        windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * self.pad + 1, axis=0)  # [T, 17, 2, W]
        windows = np.ascontiguousarray(np.moveaxis(windows, -1, 1))  # [T, W, 17, 2]
        out = np.zeros((frames, 17, 3), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, frames, self.batch_size):
                batch = torch.from_numpy(windows[start:start + self.batch_size]).to(self.device)
                out[start:start + self.batch_size] = predict_centre(self.model, batch, pad=self.pad, flip_tta=self.config["test_augmentation"]).cpu().numpy()
        return out


def fold_persons(fold_json: Path, split: str) -> list[str]:
    fold = json.loads(Path(fold_json).read_text(encoding="utf-8"))
    return [str(s) for s in fold[split]]


def output_root(dataset: str) -> Path:
    return PROJECT_ROOT / "local" / "runs" / "external_published" / "mhformer" / dataset

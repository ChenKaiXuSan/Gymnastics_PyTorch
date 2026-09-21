"""VideoPose3D (Pavllo et al., CVPR 2019) trained with its recipe on the reference joints (supervised).

The zero-shot use of the released Human3.6M checkpoint lives in
:mod:`videopose3d` (appendix only). This module trains the official temporal
convolutional model per fold on FreeMan / SportsPose -- the datasets with an
independent 3D reference -- exactly like :mod:`mhformer`: both views of the
training subjects are monocular samples, the input is the SAM3D 2D in the
Human3.6M-17 layout and the target is the reference rotated into the camera
(shared records, ``mhformer.prepare_dataset``).

What is the authors' and what is ours:

* model (``common/model.py`` of the submodule: ``TemporalModelOptimized1f``
  for training with single-frame targets, ``TemporalModel`` for inference,
  interchangeable weights), the README recipe of the 243-frame model
  (``run.py -e 80 -arc 3,3,3,3,3``: Adam amsgrad 1e-3, x0.95 per epoch,
  80 epochs, 1024 target frames per batch, dropout 0.25, 1024 channels,
  BatchNorm momentum decayed 0.1 -> 0.001, flip augmentation, flip
  test-time augmentation, MPJPE with the root at zero on the target frame,
  the final epoch's weights -- the release selects no epoch);
* input: SAM3D 2D keypoints in the H36M-17 layout (the release trains on
  CPN detections), missing joints interpolated in time, frames without a
  reference masked out of the loss;
* two-view result: the two monocular poses Procrustes-averaged
  (``procrustes_average``), also scored per view.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from common.paths import PROJECT_ROOT

from .mhformer import LEFT, RIGHT, WindowDataset, load_sequences, masked_mpjpe
from .videopose3d import normalize_screen_coordinates
from .mapping import coco17_to_h36m17, fill_missing_joints

# README recipe of the 243-frame model (run.py defaults + "-e 80 -arc 3,3,3,3,3").
CONFIG: dict[str, Any] = {"filter_widths": (3, 3, 3, 3, 3), "channels": 1024, "dropout": 0.25, "epochs": 80, "batch_size": 1024, "lr": 1e-3, "lr_decay": 0.95, "amsgrad": True,
                          "bn_momentum_initial": 0.1, "bn_momentum_final": 0.001, "data_augmentation": True, "test_augmentation": True, "seed": 1234, "workers": 8}


def _model_classes():
    import importlib.util

    from .videopose3d import THIRD_PARTY_ROOT

    path = THIRD_PARTY_ROOT / "common" / "model.py"
    spec = importlib.util.spec_from_file_location("videopose3d_common_model", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.TemporalModel, module.TemporalModelOptimized1f


def build_model(config: dict[str, Any] = CONFIG, *, optimized: bool = False):
    TemporalModel, TemporalModelOptimized1f = _model_classes()
    cls = TemporalModelOptimized1f if optimized else TemporalModel
    return cls(17, 2, 17, filter_widths=list(config["filter_widths"]), causal=False, dropout=config["dropout"], channels=config["channels"])


def receptive_field(config: dict[str, Any] = CONFIG) -> int:
    frames = 1
    for width in config["filter_widths"]:
        frames *= width
    return frames


def evaluate_sequences(model, sequences: list[dict[str, np.ndarray]], *, config: dict[str, Any], device) -> float:
    """MPJPE (mm) of the centre frame over every frame of ``sequences`` with the release's flip TTA, validity-masked."""
    import torch

    frames = receptive_field(config)
    dataset = WindowDataset(sequences, frames=frames, augment=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["workers"], pin_memory=True)
    pad = (frames - 1) // 2
    total, count = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for pose2d, pose3d, valid3d in loader:
            pose2d, target, valid = pose2d.to(device), pose3d.to(device)[:, pad], valid3d.to(device)[:, pad]
            predicted = model(pose2d)[:, 0]
            if config["test_augmentation"]:
                flipped = pose2d.clone()
                flipped[..., 0] *= -1
                flipped[..., LEFT + RIGHT, :] = flipped[..., RIGHT + LEFT, :]
                out_flip = model(flipped)[:, 0]
                out_flip[..., 0] *= -1
                out_flip[:, LEFT + RIGHT] = out_flip[:, RIGHT + LEFT]
                predicted = (predicted + out_flip) / 2
            target = target.clone()
            target[:, 0] = 0
            error = (predicted - target).norm(dim=-1)
            total += float((error * valid).sum())
            count += float(valid.sum())
    return 1000.0 * total / max(count, 1.0)


def train(inputs: Path, index: Path, train_persons: Sequence[str], val_persons: Sequence[str], output: Path, *, device: str = "cuda", config: dict[str, Any] | None = None, epochs: int | None = None) -> Path:
    """The released training loop (``run.py``, supervised branch) on the fold's training subjects; final epoch kept."""
    import torch

    cfg = {**CONFIG, **(config or {})}
    if epochs is not None:
        cfg["epochs"] = epochs
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    frames = receptive_field(cfg)
    pad = (frames - 1) // 2
    train_sequences = load_sequences(inputs, index, train_persons)
    val_sequences = load_sequences(inputs, index, val_persons)
    if not train_sequences:
        raise ValueError("no training sequences")
    dataset = WindowDataset(train_sequences, frames=frames, augment=cfg["data_augmentation"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["workers"], pin_memory=True, drop_last=False)
    model_train = build_model(cfg, optimized=True).to(device)
    model_eval = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model_train.parameters(), lr=cfg["lr"], amsgrad=cfg["amsgrad"])
    lr = cfg["lr"]
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[videopose3d] {len(train_sequences)} training sequences ({len(dataset)} windows incl. flips), {len(val_sequences)} validation sequences, {cfg['epochs']} epochs, receptive field {frames}, device {device}")
    for epoch in range(cfg["epochs"]):
        model_train.train()
        started, total, count = time.time(), 0.0, 0
        for pose2d, pose3d, valid3d in loader:
            pose2d = pose2d.to(device, non_blocking=True)
            target, valid = pose3d[:, pad : pad + 1].to(device, non_blocking=True), valid3d[:, pad : pad + 1].to(device, non_blocking=True)
            loss = masked_mpjpe(model_train(pose2d), target, valid)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * pose2d.shape[0]
            count += pose2d.shape[0]
        model_eval.load_state_dict(model_train.state_dict())
        val = evaluate_sequences(model_eval, val_sequences, config=cfg, device=device) if val_sequences else float("nan")
        print(f"[videopose3d] epoch {epoch + 1}/{cfg['epochs']} lr {lr:.6f} loss {1000 * total / max(count, 1):.2f} mm val {val:.2f} mm {time.time() - started:.0f}s")
        lr *= cfg["lr_decay"]
        for group in optimizer.param_groups:
            group["lr"] *= cfg["lr_decay"]
        momentum = cfg["bn_momentum_initial"] * math.exp(-(epoch + 1) / cfg["epochs"] * math.log(cfg["bn_momentum_initial"] / cfg["bn_momentum_final"]))
        model_train.set_bn_momentum(momentum)
    torch.save({"model_pos": model_eval.state_dict(), "config": cfg, "epoch": cfg["epochs"], "val_mpjpe_mm": val}, output)
    return output


class VideoPose3DTrainedLifter:
    """``lift(coco17 [T, 17, 2], width, height) -> h36m17 [T, 3D]`` with a fold's trained weights (H36M-17 input)."""

    def __init__(self, checkpoint: Path, device: str = "cuda") -> None:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.config = {**CONFIG, **payload.get("config", {})}
        self.model = build_model(self.config)
        self.model.load_state_dict(payload["model_pos"])
        self.model.to(device).eval()
        self.device = device
        self.pad = (receptive_field(self.config) - 1) // 2

    def lift(self, coco17: np.ndarray, width: int, height: int) -> np.ndarray:
        import torch

        points = np.asarray(coco17, dtype=np.float32)
        h2d, h2d_valid = coco17_to_h36m17(points)
        normalised = normalize_screen_coordinates(fill_missing_joints(h2d, h2d_valid), width, height)
        padded = np.pad(normalised, ((self.pad, self.pad), (0, 0), (0, 0)), "edge")
        batch = padded[None]
        if self.config["test_augmentation"]:
            flipped = padded.copy()
            flipped[:, :, 0] *= -1
            flipped[:, LEFT + RIGHT] = flipped[:, RIGHT + LEFT]
            batch = np.concatenate([batch, flipped[None]], axis=0)
        with torch.no_grad():
            predicted = self.model(torch.from_numpy(batch).to(self.device))
            if self.config["test_augmentation"]:
                predicted[1, :, :, 0] *= -1
                predicted[1, :, LEFT + RIGHT] = predicted[1, :, RIGHT + LEFT]
                predicted = predicted.mean(dim=0, keepdim=True)
        return predicted[0].cpu().numpy().astype(np.float32)


def output_root(dataset: str) -> Path:
    return PROJECT_ROOT / "local" / "runs" / "external_published" / "videopose3d_trained" / dataset

"""MDVPose (Ma et al.): MotionBERT fine-tuned with multi-view consistency, trained per fold (supervised).

"3D Human Pose Estimation from Multiple Dynamic Views via Single-view
Pretraining with Procrustes Alignment" fine-tunes the single-view MotionBERT
lifter (DSTformer, 243-frame clips) on the target multi-view dataset with the
MotionBERT 3D losses plus a Procrustes multi-view consistency term, one
uncalibrated camera per batch item. It is a supervised baseline (3D MPJPE is
the main loss), so it is trained on FreeMan / Fit3D only, per fold on
the training subjects' two views, and evaluated on the held-out subjects.

What is the authors' and what is ours:

* backbone (``lib/model/DSTformer.py`` of the git submodule
  ``fusion/external/third_party/MDVPose``), the released multi-view config
  (``configs/multi_view/MB_ft_mv_skipose.yaml``: 60 epochs, AdamW 3e-4,
  weight decay 0.01, lr x0.97 per epoch, 243-frame clips with stride 81,
  ``rootrel``, losses MPJPE + 0.5 scale-normalised MPJPE + 20 velocity +
  0.002 multi-view Procrustes, flip augmentation, flip test-time
  augmentation) and the MotionBERT Human3.6M checkpoint it starts from
  (``FT_MB_release_MB_ft_h36m``, the authors' Hugging Face mirror);
  the losses are re-implemented from ``lib/model/loss.py`` with the same
  formulas (frame validity masks replace their ``cam > 100`` padding marker),
  the multi-view term with their ``procrustes`` helper;
* one batch item = the two views of the same 243-frame clip (the release
  batches the cameras of one clip, six for Ski-Pose); ``pairs_per_batch`` 3
  keeps the released batch size of six clips, and the multi-view term is
  computed inside each pair;
* input: SAM3D 2D keypoints in the H36M-17 layout normalised like
  MotionBERT/VideoPose3D, confidence = SAM3D validity (the release feeds
  detector confidences); target = the reference joints rotated into each
  camera, root-relative; frames without a full reference are masked;
* flip augmentation is drawn per clip pair, i.e. both views flipped
  together (the released loader flips each camera item independently, which
  contradicts its own multi-view term);
* model selection: best validation-subject MPJPE (the release keeps the
  best test epoch);
* two-view result: the two monocular clips Procrustes-averaged
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
from typing import Any, Sequence

import numpy as np

from common.paths import CHECKPOINT_ROOT, PROJECT_ROOT

from .mapping import H36M17_LEFT, H36M17_RIGHT, coco17_to_h36m17, fill_missing_joints
from .videopose3d import normalize_screen_coordinates

THIRD_PARTY = Path(__file__).resolve().parents[1] / "third_party" / "MDVPose"
PRETRAINED = CHECKPOINT_ROOT / "motionbert" / "FT_MB_release_MB_ft_h36m_best_epoch.bin"
LEFT, RIGHT = list(H36M17_LEFT), list(H36M17_RIGHT)

# configs/multi_view/MB_ft_mv_skipose.yaml (model block = MotionBERT release).
CONFIG: dict[str, Any] = {
    "epochs": 60, "learning_rate": 3e-4, "weight_decay": 0.01, "lr_decay": 0.97,
    "maxlen": 243, "dim_feat": 512, "mlp_ratio": 2, "depth": 5, "dim_rep": 512, "num_heads": 8, "att_fuse": True,
    "clip_len": 243, "data_stride": 81, "num_joints": 17,
    "lambda_3d_velocity": 20.0, "lambda_scale": 0.5, "lambda_mv": 0.002, "flip": True, "seed": 0, "workers": 4,
    "pairs_per_batch": 3,  # 6 clips per batch, the released batch_size
}


def _third_party():
    if not (THIRD_PARTY / "lib" / "model" / "DSTformer.py").is_file():
        raise FileNotFoundError(f"MDVPose submodule missing at {THIRD_PARTY}; run `git submodule update --init`")
    if str(THIRD_PARTY) not in sys.path:
        sys.path.append(str(THIRD_PARTY))  # its package is ``lib``; appended so nothing of ours is shadowed


def build_model(config: dict[str, Any] = CONFIG):
    """``DSTformer`` exactly as ``lib.utils.learning.load_backbone`` builds it."""
    from functools import partial

    import torch.nn as nn

    _third_party()
    spec = importlib.util.spec_from_file_location("mdvpose_dstformer", THIRD_PARTY / "lib" / "model" / "DSTformer.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.DSTformer(dim_in=3, dim_out=3, dim_feat=config["dim_feat"], dim_rep=config["dim_rep"], depth=config["depth"], num_heads=config["num_heads"], mlp_ratio=config["mlp_ratio"],
                            norm_layer=partial(nn.LayerNorm, eps=1e-6), maxlen=config["maxlen"], num_joints=config["num_joints"], att_fuse=config["att_fuse"])


def load_pretrained(model, checkpoint: Path = PRETRAINED):
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model_pos"] if "model_pos" in payload else payload
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model


def _procrustes():
    _third_party()
    spec = importlib.util.spec_from_file_location("mdvpose_procrust", THIRD_PARTY / "lib" / "utils" / "procrust.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.procrustes


# ----------------------------------------------------------------------------- losses (lib/model/loss.py formulas)
def masked_norm_mean(diff, frame_valid):
    """``compute_valid_mean_norm``: mean joint-wise norm over the valid frames."""
    error = diff.norm(dim=-1)  # [N, T, J]
    weight = frame_valid.to(error.dtype)[..., None].expand_as(error)
    return (error * weight).sum() / weight.sum().clamp_min(1.0)


def loss_mpjpe(predicted, target, frame_valid):
    return masked_norm_mean(predicted - target, frame_valid)


def n_mpjpe(predicted, target, frame_valid):
    import torch

    norm_predicted = torch.mean(torch.sum(predicted ** 2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target * predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return loss_mpjpe(scale * predicted, target, frame_valid)


def loss_velocity(predicted, target, frame_valid):
    if predicted.shape[1] <= 1:
        return predicted.sum() * 0.0
    return masked_norm_mean((predicted[:, 1:] - predicted[:, :-1]) - (target[:, 1:] - target[:, :-1]), frame_valid[:, 1:])


def loss_multi_view(predicted, frame_valid, procrustes, pair_size: int = 2) -> Any:
    """``loss_multi_view`` inside every clip pair: each other view's frame Procrustes-aligned
    (R, s detached, the release's helper) onto the first valid view's; mean over all frame terms."""
    import torch

    terms = []
    n, t = predicted.shape[:2]
    valid = frame_valid.cpu().numpy()
    detached = predicted.detach().cpu().numpy()
    rotations, scales, pairs = [], [], []
    for first in range(0, n, pair_size):
        members = list(range(first, min(first + pair_size, n)))
        for f in range(t):
            views = [v for v in members if valid[v, f]]
            for other in views[1:]:
                rotation, scale = procrustes(detached[views[0], f], detached[other, f])
                rotations.append(rotation)
                scales.append(scale)
                pairs.append((views[0], other, f))
    if not pairs:
        return predicted.sum() * 0.0
    base_index = torch.tensor([p[0] for p in pairs], device=predicted.device)
    other_index = torch.tensor([p[1] for p in pairs], device=predicted.device)
    frame_index = torch.tensor([p[2] for p in pairs], device=predicted.device)
    rotation = torch.tensor(np.stack(rotations), dtype=predicted.dtype, device=predicted.device)  # [M, 3, 3]
    scale = torch.tensor(np.asarray(scales), dtype=predicted.dtype, device=predicted.device)  # [M]
    aligned = torch.matmul(predicted[other_index, frame_index], rotation.transpose(1, 2)) * scale[:, None, None]
    terms = torch.abs(aligned - predicted[base_index, frame_index]).sum(dim=(1, 2))
    return terms.mean()


def flip_data(data):
    """``lib.utils.utils_data.flip_data`` (numpy or torch, x is channel 0)."""
    flipped = data.clone() if hasattr(data, "clone") else data.copy()
    flipped[..., 0] *= -1
    flipped[..., LEFT + RIGHT, :] = flipped[..., RIGHT + LEFT, :]
    return flipped


# ----------------------------------------------------------------------------- clips
def clip_starts(frames: int, clip_len: int, stride: int) -> list[int]:
    """Clip starts covering the sequence: stride ``stride`` plus a final clip ending at the last frame."""
    if frames <= clip_len:
        return [0]
    starts = list(range(0, frames - clip_len + 1, stride))
    if starts[-1] + clip_len < frames:
        starts.append(frames - clip_len)
    return starts


def pair_sequences(index_path: Path, persons: Sequence[str]) -> list[tuple[dict, dict]]:
    """The (view a, view b) index entries of every trial of ``persons``."""
    meta = json.loads(Path(index_path).read_text(encoding="utf-8"))
    wanted = {str(p) for p in persons}
    by_trial: dict[tuple[str, str], dict[str, dict]] = {}
    for entry in meta["sequences"]:
        if str(entry["person"]) in wanted:
            by_trial.setdefault((entry["person"], entry["trial"]), {})[entry["view"]] = entry
    return [(views["a"], views["b"]) for views in by_trial.values() if "a" in views and "b" in views]


class ClipPairDataset:
    """243-frame clips of both views of a trial (same frames), padded with the last frame (invalid)."""

    def __init__(self, inputs: Path, index: Path, persons: Sequence[str], *, clip_len: int, stride: int, flip: bool) -> None:
        import torch

        self.torch = torch
        data = np.load(inputs)
        arrays = {k: data[k] for k in ("pose2d", "valid2d", "pose3d", "valid3d")}
        self.pairs = []
        self.items: list[tuple[int, int]] = []
        for a, b in pair_sequences(index, persons):
            views = []
            for entry in (a, b):
                sl = slice(entry["start"], entry["stop"])
                pose2d = np.concatenate([arrays["pose2d"][sl], arrays["valid2d"][sl][..., None].astype(np.float32)], axis=-1)  # x, y, confidence
                views.append({"input": pose2d, "target": arrays["pose3d"][sl], "valid": arrays["valid3d"][sl].all(axis=1)})
            frames = views[0]["input"].shape[0]
            if frames != views[1]["input"].shape[0]:
                raise ValueError(f"views of {a['trial']} differ in length")
            pair_index = len(self.pairs)
            self.pairs.append(views)
            self.items.extend((pair_index, start) for start in clip_starts(frames, clip_len, stride))
        self.clip_len, self.flip = clip_len, flip

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, item: int):
        pair_index, start = self.items[item]
        inputs, targets, valid = [], [], []
        for view in self.pairs[pair_index]:
            stop = min(start + self.clip_len, view["input"].shape[0])
            pad = self.clip_len - (stop - start)
            x, y, ok = view["input"][start:stop], view["target"][start:stop], view["valid"][start:stop]
            if pad:
                x = np.concatenate([x, np.repeat(x[-1:], pad, axis=0)])
                y = np.concatenate([y, np.repeat(y[-1:], pad, axis=0)])
                ok = np.concatenate([ok, np.zeros(pad, dtype=bool)])
            inputs.append(x)
            targets.append(y)
            valid.append(ok)
        x, y, ok = np.stack(inputs), np.stack(targets), np.stack(valid)
        if self.flip and random.random() > 0.5:
            x, y = flip_data(x), flip_data(y)
        return self.torch.from_numpy(np.ascontiguousarray(x)), self.torch.from_numpy(np.ascontiguousarray(y)), self.torch.from_numpy(ok)


# ----------------------------------------------------------------------------- training (train_mv_finetune.py as a function)
def _predict(model, batch_input, *, flip_tta: bool):
    if flip_tta:
        out = (model(batch_input) + flip_data(model(flip_data(batch_input)))) / 2
    else:
        out = model(batch_input)
    out[:, :, 0, :] = 0
    return out


def evaluate_clips(model, dataset: ClipPairDataset, *, device, flip_tta: bool) -> float:
    """MPJPE (mm) over the valid frames of the (non-overlapping) evaluation clips, root-relative."""
    import torch

    total, count = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for item in range(len(dataset)):
            x, y, ok = dataset[item]
            x, y, ok = x.to(device), y.to(device), ok.to(device)
            predicted = _predict(model, x, flip_tta=flip_tta)
            target = y - y[:, :, 0:1, :]
            error = (predicted - target).norm(dim=-1).mean(dim=-1)  # [N, T]
            total += float((error * ok).sum())
            count += float(ok.sum())
    return 1000.0 * total / max(count, 1.0)


def train(inputs: Path, index: Path, train_persons: Sequence[str], val_persons: Sequence[str], output: Path, *, device: str = "cuda", config: dict[str, Any] | None = None, epochs: int | None = None, pretrained: Path = PRETRAINED) -> Path:
    import torch

    cfg = {**CONFIG, **(config or {})}
    if epochs is not None:
        cfg["epochs"] = epochs
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    train_set = ClipPairDataset(inputs, index, train_persons, clip_len=cfg["clip_len"], stride=cfg["data_stride"], flip=cfg["flip"])
    val_set = ClipPairDataset(inputs, index, val_persons, clip_len=cfg["clip_len"], stride=cfg["clip_len"], flip=False)
    if not len(train_set) or not len(val_set):
        raise ValueError(f"{len(train_set)} training / {len(val_set)} validation clips")
    # A batch is ``pairs_per_batch`` clip pairs (the release: the six cameras of one clip, no shuffling);
    # the pair order is shuffled per epoch.
    def collate(items):
        return tuple(torch.cat([item[k] for item in items]) for k in range(3))

    loader = torch.utils.data.DataLoader(train_set, batch_size=cfg["pairs_per_batch"], shuffle=True, num_workers=cfg["workers"], pin_memory=True, collate_fn=collate, drop_last=False)
    model = load_pretrained(build_model(cfg), pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    procrustes = _procrustes()
    lr = cfg["learning_rate"]
    best, best_epoch = math.inf, -1
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[mdvpose] {len(train_set)} training clip pairs ({cfg['pairs_per_batch']} per batch), {len(val_set)} validation clip pairs, {cfg['epochs']} epochs, device {device}")
    for epoch in range(cfg["epochs"]):
        model.train()
        started, sums, steps = time.time(), {"3d_pos": 0.0, "3d_scale": 0.0, "3d_velocity": 0.0, "multi_view": 0.0, "total": 0.0}, 0
        for x, y, ok in loader:
            x, y, ok = x.to(device, non_blocking=True), y.to(device, non_blocking=True), ok.to(device, non_blocking=True)
            target = y - y[:, :, 0:1, :]  # rootrel
            predicted = model(x)
            l_pos = loss_mpjpe(predicted, target, ok)
            l_scale = n_mpjpe(predicted, target, ok)
            l_vel = loss_velocity(predicted, target, ok)
            l_mv = loss_multi_view(predicted, ok, procrustes, pair_size=2)
            loss = l_pos + cfg["lambda_scale"] * l_scale + cfg["lambda_3d_velocity"] * l_vel + cfg["lambda_mv"] * l_mv
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for key, value in (("3d_pos", l_pos), ("3d_scale", l_scale), ("3d_velocity", l_vel), ("multi_view", l_mv), ("total", loss)):
                sums[key] += float(value.detach())
            steps += 1
        e1 = evaluate_clips(model, val_set, device=device, flip_tta=cfg["flip"])
        if e1 < best:
            best, best_epoch = e1, epoch
            torch.save({"state_dict": model.state_dict(), "config": cfg, "epoch": epoch, "val_mpjpe_mm": e1}, output)
        means = {k: v / max(steps, 1) for k, v in sums.items()}
        print(f"[mdvpose] epoch {epoch + 1}/{cfg['epochs']} lr {lr:.6f} 3d_pos {1000 * means['3d_pos']:.2f} mm scale {1000 * means['3d_scale']:.2f} vel {1000 * means['3d_velocity']:.2f} mv {means['multi_view']:.3f} total {means['total']:.4f} val {e1:.2f} mm best {best:.2f} (epoch {best_epoch + 1}) {time.time() - started:.0f}s")
        lr *= cfg["lr_decay"]
        for group in optimizer.param_groups:
            group["lr"] *= cfg["lr_decay"]
    return output


# ----------------------------------------------------------------------------- inference (Lifter protocol)
class MDVPoseLifter:
    """``lift(coco17 [T, 17, 2], width, height) -> h36m17 [T, 17, 3]``: non-overlapping 243-frame clips, flip TTA."""

    def __init__(self, checkpoint: Path, device: str = "cuda") -> None:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.config = {**CONFIG, **payload.get("config", {})}
        self.model = build_model(self.config)
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(device).eval()
        self.device = device

    def lift(self, coco17: np.ndarray, width: int, height: int) -> np.ndarray:
        import torch

        points = np.asarray(coco17, dtype=np.float32)
        h2d, h2d_valid = coco17_to_h36m17(points)
        normalised = normalize_screen_coordinates(fill_missing_joints(h2d, h2d_valid), width, height)
        x = np.concatenate([normalised, np.ones(normalised.shape[:2] + (1,), np.float32)], axis=-1)
        frames, clip_len = x.shape[0], self.config["clip_len"]
        out = np.zeros((frames, 17, 3), dtype=np.float32)
        with torch.no_grad():
            for start in clip_starts(frames, clip_len, clip_len):
                stop = min(start + clip_len, frames)
                clip = x[start:stop]
                if stop - start < clip_len:
                    clip = np.concatenate([clip, np.repeat(clip[-1:], clip_len - (stop - start), axis=0)])
                predicted = _predict(self.model, torch.from_numpy(clip[None]).to(self.device), flip_tta=self.config["flip"])
                out[start:stop] = predicted[0, : stop - start].cpu().numpy()
        return out


def output_root(dataset: str) -> Path:
    return PROJECT_ROOT / "local" / "runs" / "external_published" / "mdvpose" / dataset


__all__ = ["CONFIG", "ClipPairDataset", "MDVPoseLifter", "build_model", "load_pretrained", "output_root", "train"]

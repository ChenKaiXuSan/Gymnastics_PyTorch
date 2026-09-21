"""VideoPose3D (Pavllo et al., CVPR 2019) as released: official model and weights.

The ``pretrained_h36m_detectron_coco`` checkpoint lifts COCO17 2D keypoints
(normalised screen coordinates) to Human3.6M-17 3D joints in metres, root
relative. Inference follows ``run.py`` of the official repository: the 2D
sequence is edge-padded by half the receptive field (243 frames -> 121 per
side), run through the temporal model, and averaged with the horizontally
flipped sequence (test-time augmentation, the authors' default).

Code: git submodule ``fusion/external/third_party/VideoPose3D``; weights:
``local/checkpoints/videopose3d/pretrained_h36m_detectron_coco.bin``
(https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np

from common.paths import CHECKPOINT_ROOT

from .mapping import COCO17_LEFT, COCO17_RIGHT, H36M17_LEFT, H36M17_RIGHT

THIRD_PARTY_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "VideoPose3D"
CHECKPOINT_ENV = "GYMNASTICS_VIDEOPOSE3D_CHECKPOINT"
DEFAULT_CHECKPOINT = CHECKPOINT_ROOT / "videopose3d" / "pretrained_h36m_detectron_coco.bin"
# Architecture of the released checkpoint (README: "-arc 3,3,3,3,3", 1024 channels).
FILTER_WIDTHS = (3, 3, 3, 3, 3)
CHANNELS = 1024


def _load_temporal_model_class():
    """Import ``TemporalModel`` from the submodule by file path.

    VideoPose3D's package is called ``common`` like this repository's shared
    library, so it cannot go on ``sys.path``; ``common/model.py`` only depends
    on torch and is loaded as a standalone module.
    """
    path = THIRD_PARTY_ROOT / "common" / "model.py"
    if not path.is_file():
        raise FileNotFoundError(f"VideoPose3D submodule missing at {THIRD_PARTY_ROOT}; run `git submodule update --init`")
    spec = importlib.util.spec_from_file_location("videopose3d_common_model", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.TemporalModel


def resolve_checkpoint(path: str | Path | None = None) -> Path:
    candidate = Path(path or os.environ.get(CHECKPOINT_ENV, DEFAULT_CHECKPOINT))
    if not candidate.is_file():
        raise FileNotFoundError(f"VideoPose3D checkpoint not found at {candidate}; download pretrained_h36m_detectron_coco.bin from the official release or set {CHECKPOINT_ENV}")
    return candidate


def normalize_screen_coordinates(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """The official normalisation: x in [0, w] -> [-1, 1], y scaled by the same factor."""
    return points / width * 2.0 - np.array([1.0, height / width], dtype=np.float32)


class VideoPose3DLifter:
    """Thin wrapper around the official ``TemporalModel`` with the released weights."""

    def __init__(self, checkpoint: str | Path | None = None, device: str = "cuda", *, test_time_augmentation: bool = True) -> None:
        import torch

        TemporalModel = _load_temporal_model_class()
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model = TemporalModel(17, 2, 17, filter_widths=list(FILTER_WIDTHS), causal=False, dropout=0.25, channels=CHANNELS, dense=False)
        state = torch.load(resolve_checkpoint(checkpoint), map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model_pos"])
        self.model.to(self.device).eval()
        self.receptive_field = int(self.model.receptive_field())
        self.pad = (self.receptive_field - 1) // 2
        self.test_time_augmentation = bool(test_time_augmentation)

    def lift(self, coco17: np.ndarray, width: int, height: int) -> np.ndarray:
        """``[T, 17, 2]`` COCO17 image keypoints -> ``[T, 17, 3]`` H36M joints (metres, root relative)."""
        import torch

        points = np.asarray(coco17, dtype=np.float32)
        if points.ndim != 3 or points.shape[1:] != (17, 2):
            raise ValueError("expected [T, 17, 2] COCO17 keypoints")
        normalised = normalize_screen_coordinates(points, width, height)
        padded = np.pad(normalised, ((self.pad, self.pad), (0, 0), (0, 0)), "edge")
        batch = padded[None]
        if self.test_time_augmentation:
            flipped = padded.copy()
            flipped[:, :, 0] *= -1
            flipped[:, list(COCO17_LEFT) + list(COCO17_RIGHT)] = flipped[:, list(COCO17_RIGHT) + list(COCO17_LEFT)]
            batch = np.concatenate([batch, flipped[None]], axis=0)
        with torch.no_grad():
            predicted = self.model(torch.from_numpy(batch).to(self.device))
            if self.test_time_augmentation:
                predicted[1, :, :, 0] *= -1
                predicted[1, :, list(H36M17_LEFT) + list(H36M17_RIGHT)] = predicted[1, :, list(H36M17_RIGHT) + list(H36M17_LEFT)]
                predicted = predicted.mean(dim=0, keepdim=True)
        output = predicted[0].cpu().numpy().astype(np.float32)
        if output.shape[0] != points.shape[0]:
            raise RuntimeError(f"VideoPose3D returned {output.shape[0]} frames for {points.shape[0]} inputs")
        return output

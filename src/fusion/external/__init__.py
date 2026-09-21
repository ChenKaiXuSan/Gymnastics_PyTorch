"""External learned fusion baselines trained with the project's own protocol.

Every architecture here consumes exactly what the proposed model consumes (the
two canonical views, their validity, the phase and the camera-depth axes), is
trained with the same folds, windows and objectives (``python -m fusion train
model=external_<name>``) and returns a :class:`~fusion.outputs.PoseFusionOutput`,
so the comparison isolates the learned architecture.  Public checkpoints are
never used (no zero-shot rows).

    tcn           VideoPose3D-style dilated temporal convolution (Pavllo et al., CVPR 2019)
    smoothnet     SmoothNet temporal-only MLP refiner (Zeng et al., ECCV 2022)
    metapose_mlp  MetaPose-style per-frame aggregation MLP (Usman et al., CVPR 2022)
    muc_weights   MUC-style learned per-view per-joint weights (Zhu et al., AAAI 2025)
"""

from .model import EXTERNAL_BACKBONES, ExternalFusionModel, ExternalModelConfig

__all__ = ["EXTERNAL_BACKBONES", "ExternalFusionModel", "ExternalModelConfig"]

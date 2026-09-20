"""Stage 4 - the proposed cycle-aware dual-view 3D pose fusion network (Architecture v1.0).

This package implements a learned fusion model that merges two independent
monocular 3D pose estimates of the same person (View A and View B, in this
project the *face* and *side* cameras) into one refined 3D pose sequence, and
it does so while explicitly modelling the *cyclic* structure of the observed
motion.

Research Motivation:
    The existing rotation-aware fusion model of this repository
    (``fusion.archive.rotation_aware``) treats a window of poses as an
    unstructured temporal signal.  Gymnastics trunk-rotation exercises, gait,
    and most rehabilitation movements are repeated cycles.  Cycle structure
    carries two kinds of information that a plain temporal model cannot
    exploit directly:

    * *Phase*: where inside a cycle each frame lies.  Two frames at the same
      phase in consecutive cycles should show approximately the same pose,
      which is a strong self-supervised prior for repairing a corrupted view.
    * *Time scale separation*: short-term dynamics (velocity, local
      trajectory bends) and long-term structure (periodic recurrence,
      bilateral alternation) live at different temporal scales and are
      encoded by separate branches.

    The model never predicts a pose from scratch.  It (i) predicts, for every
    joint and time step, how much to trust each view, (ii) forms a weighted
    average of the *original* view predictions, and (iii) adds a small learned
    residual correction.  The two view encoders share their parameters so the
    model is symmetric in its inputs.

Architecture (one shared encoder per view, applied to A and B):

    P_A, P_B  [B, T, J, 3]                (canonical body-frame 3D keypoints)
        |
        +-- Spatial Transformer (over joints, per frame)   -> F_pose   [B, T, J, D]
        +-- Short Motion Transformer (local temporal)      -> F_short  [B, T, J, D]
        +-- Long Motion Transformer (cycle-scale temporal) -> F_long   [B, T, J, D]
                                       |
                        Motion Fusion: F_motion = MLP([F_short ; F_long])
                                       |
                 FiLM: H = (1 + gamma(F_motion)) * F_pose + beta(F_motion)
                                       |
                        Bidirectional Cross-View Attention (H_A <-> H_B)
                                       |
                        Joint-wise reliability: [w_A, w_B] = softmax(R)
                                       |
                        Weighted fusion: P_base = w_A * P_A + w_B * P_B
                                       |
                        Residual refinement: P_hat = P_base + Delta_P

Tensor convention:
    Every intermediate representation is ``[B, T, J, C]`` with ``B`` batch,
    ``T`` temporal samples, ``J`` joints and ``C`` channels.  Modules that
    call PyTorch attention APIs temporarily fold ``B`` with ``T`` (attention
    over joints) or ``B`` with ``J`` (attention over time) and document the
    reshape in place.

Package layout:
    ``skeleton``            common joint set, bones, bilateral pairs
    ``sample``              unified dual-view sample and batch contracts
    ``outputs``             structured :class:`PoseFusionOutput`
    ``phase``               phase normalisation and phase encoding
    ``velocity``            physical joint velocity from timestamps
    ``modules``             the neural building blocks listed above
    ``model``               :class:`CycleAwareFusionModel`
    ``losses``              self-supervised objectives
    ``corruptions``         synthetic input corruption for recovery training
    ``data``                one DataModule per dataset plus shared windowing
    ``lightning_module``    PyTorch Lightning wrapper around the pure model
    ``train``               Hydra entry point

Hydra configuration lives under ``src/configs/fusion`` and is described in
``docs/cycle_aware_fusion.md``.
"""

from __future__ import annotations

ARCHITECTURE_VERSION = "1.0"

__all__ = ["ARCHITECTURE_VERSION"]

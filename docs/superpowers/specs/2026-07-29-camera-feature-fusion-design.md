# Fitted-Camera Feature Fusion Design

## Objective

Test whether fitted two-view camera geometry provides information beyond the
existing trunk-rotation features in the A6 self-supervised fusion network.
The camera-guided methods use fitted camera features during both training and
inference and are therefore reported separately from camera-free A-series
methods.

## Fixed research boundary

- Keep the A6 pose, trunk rotation, angular velocity, angular acceleration,
  cross-view disagreement, temporal TCN, bounded residual, and self-supervised
  losses unchanged.
- Do not train against Unity-native 3D or private triangulated 3D.
- Fit the relative camera from synchronized SAM3D 2D keypoints only.
- Use a direction-transfer split on Unity: fit and train on one continuous
  direction, then evaluate on the opposite direction and the static diagnostic.
- Use Unity-native 3D only after checkpoint creation for evaluation.
- Preserve the existing camera-free model API and checkpoint compatibility when
  camera conditioning is disabled.

## Camera fitting

For each Unity direction-transfer fold:

1. Decompose the simulator projection matrices to obtain fixed intrinsics.
2. Fit the relative rotation and unit translation direction with a RANSAC
   essential matrix using only the training direction's SAM3D 2D
   correspondences.
3. Split the training frames into even fit frames and odd audit frames.
4. Store fit inlier ratio and held-out reprojection error.
5. Refit the relative pose using all training-direction frames after the audit.
6. Apply this one fitted rig to the training, held-out-direction, and static
   sequences.

Translation magnitude is excluded because essential-matrix translation is
scale-ambiguous.

## Feature contract

`CameraFeatureBundle` contains finite tensors:

- `global_features [B,Cg]`: rotation 6D, translation direction, normalized
  intrinsics, inlier ratio, and robust held-out reprojection quality.
- `joint_features [B,T,J,Cj]`: symmetric epipolar residual, triangulation angle,
  per-view validity, and per-view normalized image coordinates.
- `valid [B,T,J]`: both 2D observations are finite and the fitted rig is valid.

Feature subsets define the experiment matrix:

- G0: no camera features.
- G1: fitted global pose and intrinsics only.
- G2: G1 plus camera-fit quality.
- G3: G2 plus per-frame/per-joint geometry.
- G4: G3 with camera-motion FiLM conditioning of the existing A6 fused
  features; this is the candidate method.
- G5: the G4 architecture with a deterministic wrong-camera perturbation,
  serving as a negative control.

## Model integration

The unmodified A6 pipeline first creates symmetric view features and existing
trunk-motion features. Camera conditioning is optional:

```text
existing symmetric A6 features
          +
camera global/joint encoder
          |
camera-motion FiLM (G4/G5) or additive conditioning (G1-G3)
          |
existing temporal TCN and bounded residual
          +
independently bounded camera-motion residual bypass
```

The camera encoder is zero-initialized at its final projection so loading an A6
checkpoint initially reproduces the source model exactly. G0 has no camera
parameters. Camera-enabled checkpoints record the feature schema and ablation.
The direct camera-motion residual is also zero-initialized. It preserves the
exact A6 initialization while retaining a gradient path when the transferred
A6 `tanh` residual is saturated under domain shift.

## Swap behavior

The physical input convention remains face reference and side secondary for
the calibrated G-series. This series is not claimed to be camera-free
view-swap invariant. A diagnostic swaps the two inputs while inverting the
relative pose and checks that the geometric feature builder transforms
consistently. The original A-series swap-invariance claim remains unchanged.

## Training and evaluation

Each G cell uses two direction-transfer folds and seeds 0, 1, and 2. The source
checkpoint is the same A6 checkpoint. Training uses only existing A6
self-supervised losses and deterministic corruptions. The evaluation stage is
separate and loads Unity-native 3D only after training has completed.

Primary outcome:

- held-out continuous-sequence Unity-native MPJPE after one Sim3 per sequence.

Secondary outcomes:

- static MPJPE;
- axial-rotation angle error;
- ROM error and peak-timing error;
- gate/camera-feature diagnostics;
- change relative to G0, with fold-then-seed macro aggregation.

## Interpretation rules

- G4 improves native-GT MPJPE and motion metrics over G0: camera features add
  useful information beyond existing trunk features.
- G4 improves motion metrics but not MPJPE: report task-specific motion
  preservation, not general pose accuracy.
- G5 matches or beats G4: the network is not using correct camera geometry, so
  no camera-information claim is allowed.
- Only private pseudo-reference improves: treat the result as evaluation
  coupling, not accuracy evidence.
- Fitted-camera methods remain a separate calibrated-input block in the paper.

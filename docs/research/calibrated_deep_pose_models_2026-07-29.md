# Calibrated Deep Multi-View Pose Models

Date: 2026-07-29

## Scope

This scan targets learning-based multi-view 3D human-pose methods that consume
known camera projection geometry, especially extrinsics. It distinguishes
methods that are close to the present two-view, single-person gymnastics
pipeline from multi-person scene methods that would require substantial
adaptation.

The central finding is that the calibrated deep-learning literature mostly
fuses images, 2D heatmaps, or 2D keypoints through projective geometry. It does
not directly fuse two already-complete monocular 3D pose streams. Consequently,
the models below are valid external-parameter comparators only if the paper
clearly separates their input setting from the proposed post-estimation
3D-to-3D fusion setting. An extrinsic-conditioned network that consumes the two
SAM3D 3D streams would instead be a new model designed for this project, not a
standard published baseline.

## Shortlist

| Model | How camera parameters enter | Supervision and input | Official code | Fit to this project |
|---|---|---|---|---|
| Learnable Triangulation, algebraic variant (ICCV 2019) | Known projection matrices enter a differentiable weighted triangulation layer; image networks predict 2D heatmaps and camera-joint confidence | Synchronized images and 3D training labels | [Official PyTorch code](https://github.com/karfly/learnable-triangulation-pytorch) | **Best first deep calibrated baseline.** It is single-person, explicitly supports any number of cameras, and the project page illustrates two views. Adapt the output head from the paper skeleton to the desired MHR70 subset. |
| Learnable Triangulation, volumetric variant (ICCV 2019) | Projection matrices unproject per-view feature maps into a shared 3D volume | Synchronized images and 3D training labels; needs a rough pelvis volume centre | Same repository | Stronger but heavier second-stage option. It supports arbitrary camera counts, but its 3D grid and skeleton-specific output make it more expensive than the algebraic model. |
| Generalizable Human Pose Triangulation (CVPR 2022) | Camera matrices generate random multi-view triangulation hypotheses; a neural scorer ranks camera-arrangement-independent 3D hypotheses | 2D detections, calibration, and 3D supervision | No official implementation located in this scan | **Best conceptual match to the existing 2D-keypoint path.** It avoids image-backbone retraining and explicitly targets unseen camera arrangements, but implementation cost is higher without released code. |
| Lightweight Camera-Disentangled Representation (CVPR 2020) | Known camera transformations condition encoders/decoders; a differentiable DLT layer reconstructs 3D | Synchronized images and 3D supervision | No official implementation located in this scan | Scientifically relevant and efficient, but reproduction risk is high without code. Use as related work unless an implementation is found. |
| VoxelPose (ECCV 2020) | Calibration warps per-view 2D heatmaps/features into a common 3D voxel volume | Multi-person images/heatmaps and supervised 3D poses | [Official PyTorch code](https://github.com/microsoft/voxelpose-pytorch) | Runnable code exists, but it is designed for multi-person scene localization and commonly uses 3–5 cameras. For a single gymnast and two cameras, its root-proposal network and 3D scene volume are mostly overhead. |
| MVGFormer (CVPR 2024) | Learning-free geometry modules project 3D queries into each calibrated view and triangulate refined 2D poses inside a Transformer decoder | Multi-person images, calibration, and supervised 3D poses | [Official code](https://github.com/XunshanMan/MVGFormer) | A strong modern reference and camera-generalization experiment, but its custom CUDA attention build, multi-person queries, Panoptic-style data, and skeleton conversion make it a high-cost baseline. |
| SelfPose3D (CVPR 2024) | Known calibration projects synthetic 3D roots and predicted 3D poses into all camera views for self-supervised losses | Calibrated images plus off-the-shelf 2D pseudo poses; no 2D/3D ground truth required | [Official code](https://github.com/CAMMA-public/SelfPose3D) | **Most relevant no-3D-GT training direction.** It avoids target 3D labels, but remains a multi-person volumetric system built around Panoptic/Shelf/Campus and normally multiple cameras. Two-view stability must be established experimentally. |

## Recommended experiment order

1. **Learnable Triangulation--Algebraic.** Start with the standard body-joint
   subset shared by MHR70 and the public model skeleton. Train on Unity, where
   true 3D joints and exact cameras exist, then evaluate zero-shot on the human
   dataset. Any human-data adaptation should use reprojection/self-supervision
   and person-level splits. Do not train against the triangulated
   pseudo-reference and evaluate on the same people.
2. **Generalizable Human Pose Triangulation.** Implement only if a
   keypoint-input deep model is desired. It is a cleaner comparison with the
   current SAM3D 2D detections than image-feature volumetric models.
3. **SelfPose3D.** Use when the scientific question is specifically whether
   calibrated self-supervision can learn from the 137-person videos without
   3D labels.
4. Treat **VoxelPose/MVGFormer** as higher-cost multi-person baselines. They are
   suitable for a later, broader paper comparison, not the smallest defensible
   baseline set.

## Fair-comparison constraints

- Report these image/2D-to-3D systems separately from the present
  3D-keypoint-to-3D-keypoint fusion models; they consume different information.
- Keep the camera source explicit: exact Unity cameras versus estimated
  per-person human cameras.
- Use person-level splits. A network trained on human triangulated
  pseudo-reference targets must never be evaluated on the same people and
  presented as independent validation of that reference.
- Report both exact-camera Unity results and estimated-camera human results;
  otherwise model error and calibration error cannot be separated.
- Begin with a common body-joint subset. Training a 17/19-joint public model and
  evaluating it as if it predicted all 70 MHR joints would not be comparable.

## Primary sources

- Iskakov et al., [Learnable Triangulation of Human Pose](https://openaccess.thecvf.com/content_ICCV_2019/html/Iskakov_Learnable_Triangulation_of_Human_Pose_ICCV_2019_paper.html), ICCV 2019; [official project page](https://saic-violet.github.io/learnable-triangulation/).
- Remelli et al., [Lightweight Multi-View 3D Pose Estimation Through Camera-Disentangled Representation](https://openaccess.thecvf.com/content_CVPR_2020/html/Remelli_Lightweight_Multi-View_3D_Pose_Estimation_Through_Camera-Disentangled_Representation_CVPR_2020_paper.html), CVPR 2020.
- Tu et al., [VoxelPose](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460188.pdf), ECCV 2020.
- Bartol et al., [Generalizable Human Pose Triangulation](https://openaccess.thecvf.com/content/CVPR2022/html/Bartol_Generalizable_Human_Pose_Triangulation_CVPR_2022_paper.html), CVPR 2022.
- Liao et al., [Multiple View Geometry Transformers for 3D Human Pose Estimation](https://openaccess.thecvf.com/content/CVPR2024/papers/Liao_Multiple_View_Geometry_Transformers_for_3D_Human_Pose_Estimation_CVPR_2024_paper.pdf), CVPR 2024.
- Srivastav et al., [SelfPose3D](https://openaccess.thecvf.com/content/CVPR2024/html/Srivastav_SelfPose3d_Self-Supervised_Multi-Person_Multi-View_3d_Pose_Estimation_CVPR_2024_paper.html), CVPR 2024.

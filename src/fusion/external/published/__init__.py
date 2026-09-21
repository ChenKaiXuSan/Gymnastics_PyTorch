"""Published external methods run as released (strict external baselines).

Unlike :mod:`fusion.external.model` (published *architectures* trained inside
this project's protocol, i.e. backbone ablations), everything here runs the
authors' own code and weights on the authors' own inputs, and this project
only supplies the data and the evaluation:

    videopose3d   Pavllo et al., CVPR 2019 -- official dilated TCN lifter with the
                  released ``pretrained_h36m_detectron_coco`` weights, applied to
                  the SAM3D 2D keypoints (COCO17) of each view; the two monocular
                  3D sequences are then combined by per-frame Procrustes
                  alignment and averaging (no learning, no depth prior).
    (muc, metapose: see ``docs/research`` for their status)

Every method produces a ``PosePairTrial`` on the same frames as the SAM3D
trial it replaces (``trial_transform``), so it is evaluated through the
standard DataModules: same folds, same phase windows, same per-frame
PA-MPJPE on the 20 major joints (joints the method does not predict are
invalid and excluded, and the joint coverage is reported next to the number).

Third-party code lives under ``fusion/external/third_party`` (git
submodules); weights under ``local/checkpoints``.
"""

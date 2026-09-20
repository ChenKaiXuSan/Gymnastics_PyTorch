# Archived fusion models

## rotation_aware (archived 2026-09-19; package `fusion.archive.rotation_aware`)

The self-supervised rotation-aware residual TCN that underlies the Sports
Engineering manuscript (ablations A4–A11, the plain-TCN controls B1/B2, the
real-camera pilot and the cross-view-attention variants). It is frozen: bug
fixes only, no new experiments. The active model is
`fusion`.

It remains importable and runnable so the paper artefacts can be regenerated:

```bash
python -m fusion rotation-aware {prepare,train,infer,evaluate} --config src/configs/archive/rotation_aware.yaml ...
python -m fusion benchmark-freeman-train ...        # FreeMan subject-disjoint folds of the archived model
```

Its dataset-independent pieces (trial schema, skeleton spec, canonical body
frame, trunk / quality features, quality-weighted base fusion, person cache)
were extracted to `fusion.keypoints`, which is what every other package
imports. Tests live under `tests/fusion/archive/rotation_aware/`.

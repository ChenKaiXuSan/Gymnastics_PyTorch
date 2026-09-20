# Archived fusion models

## rotation_aware (archived 2026-09-19; package `gymnastics.archive.rotation_aware`)

The self-supervised rotation-aware residual TCN that underlies the Sports
Engineering manuscript (ablations A4–A11, the plain-TCN controls B1/B2, the
real-camera pilot and the cross-view-attention variants). It is frozen: bug
fixes only, no new experiments. The active model is
`gymnastics.fusion`.

It remains importable and runnable so the paper artefacts can be regenerated:

```bash
gymnastics fuse rotation-aware {prepare,train,infer,evaluate} --config configs/fusion/rotation_aware.yaml ...
gymnastics benchmark freeman-train ...        # FreeMan subject-disjoint folds of the archived model
```

Its dataset-independent pieces (trial schema, skeleton spec, canonical body
frame, trunk / quality features, quality-weighted base fusion, person cache)
were extracted to `gymnastics.keypoints`, which is what every other package
imports. Tests live under `tests/archive/rotation_aware/`.

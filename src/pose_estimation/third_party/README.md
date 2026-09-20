# Third-party dependencies of the pose-estimation stage

Pinned upstream repositories (git submodules), not project-owned code. Only
`pose_estimation` imports them, through `pose_estimation._third_party`.

```bash
git submodule update --init --recursive
```

Pinned repositories:

- `sam-3d-body`: `b5c765a0d89d789985e186d396315e7590887b94`

Do not copy upstream packages into `src/pose_estimation`; the adapter adds the
checkout to `sys.path` at import time.

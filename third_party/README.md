# Third-party dependencies

This directory contains pinned upstream repositories, not project-owned code.

```bash
git submodule update --init --recursive
```

Pinned repositories:

- `sam-3d-body`: `b5c765a0d89d789985e186d396315e7590887b94`

Project-specific inference orchestration and adapters live in
`src/gymnastics/sam3d/`. Do not copy upstream packages into that directory.

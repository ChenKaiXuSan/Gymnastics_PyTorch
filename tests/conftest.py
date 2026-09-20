from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def tiny_smoothnet_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Point the SmoothNet baseline at a small random checkpoint during tests.

    The public Human3.6M checkpoint is not part of the repository; tests only
    need a syntactically valid state dict with the same layout.
    """
    import torch

    from gymnastics.baselines.classical_baselines import (
        SMOOTHNET_CHECKPOINT_ENV,
        build_smoothnet,
    )

    torch.manual_seed(0)
    model = build_smoothnet(4, hidden_size=16, res_hidden_size=8, num_blocks=2)
    path = tmp_path_factory.mktemp("smoothnet") / "tiny_checkpoint.pth.tar"
    torch.save({"state_dict": model.state_dict()}, path)
    import os

    previous = os.environ.get(SMOOTHNET_CHECKPOINT_ENV)
    os.environ[SMOOTHNET_CHECKPOINT_ENV] = str(path)
    yield path
    if previous is None:
        os.environ.pop(SMOOTHNET_CHECKPOINT_ENV, None)
    else:
        os.environ[SMOOTHNET_CHECKPOINT_ENV] = previous

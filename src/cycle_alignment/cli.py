"""Entry point of stage 2: ``python -m cycle_alignment [align|cycles ...]``."""

from __future__ import annotations

from typing import Sequence

from common.cli import dispatch

COMMANDS = {
    "align": ("cycle_alignment.main", "cli_main", True, "estimate the side-to-face offset and segment cycles"),
    "cycles": ("cycle_alignment.annotate_cycles", "main", True, "annotate cycle middles / detect cycles for private, freeman, unity; build the index"),
}


def main(argv: Sequence[str] | None = None) -> int:
    return dispatch("cycle_alignment", "Stage 2: face/side alignment and cycle segmentation", COMMANDS, argv, default="align")

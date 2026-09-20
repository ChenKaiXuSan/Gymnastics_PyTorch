"""FreeMan public external benchmark (zero-shot and subject-disjoint training).

``cli`` is the ``python -m fusion benchmark-freeman`` entry point with the
production ``DefaultStageOperations``; ``runner`` holds the stage protocol, run
state and multi-GPU orchestration; ``stages`` the per-subject building blocks;
``training_cli`` the subject-disjoint training command.
"""

from .download import (
    download_release,
    fetch_hub_inventory,
    load_config,
    run_preflight,
    validate_downloads,
)
from .schema import ArchiveEntry, PreflightReport

__all__ = [
    "ArchiveEntry",
    "PreflightReport",
    "download_release",
    "fetch_hub_inventory",
    "load_config",
    "run_preflight",
    "validate_downloads",
]

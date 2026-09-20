"""FreeMan action labels and the action groups used to select sessions.

The FreeMan release ships one action label per session in
``FreeMan_actions.csv`` (a copy is tracked as
``src/configs/shared/freeman_actions.csv``).  The label is not part of the
benchmark manifests, so this module joins the two through the session hash
(``20220618_<hash>_subj12`` <-> ``<hash>_video``).

The groups below partition the 122 labels by whether the motion repeats.
They drive ``data.options.actions`` of the FreeMan DataModule: the private
gymnastics motion is a repeated cycle, and the cycle detector run on every
FreeMan session also fires one or two spurious "cycles" on daily activities
(drink, sit, phone call), which is what made the periodicity objective harmful
on the unfiltered release.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping

from common.paths import CONFIG_ROOT

DEFAULT_ACTION_TABLE = CONFIG_ROOT / "shared" / "freeman_actions.csv"

DANCE_ACTIONS: tuple[str, ...] = (
    "hiphop", "jazz", "latin", "urban", "breaking", "kpop", "popping",
    "ballet", "folk dance", "dance", "wacking", "aerobics",
)
EXERCISE_ACTIONS: tuple[str, ...] = (
    "strentch", "strentch upper", "strentch lower", "warm up", "jump",
    "skipping rope", "skip rope", "skipping ropo", "squat", "arm lifting",
    "leg lifting", "arm curl", "lateral raise", "flybird", "step high", "row",
)
BALL_ACTIONS: tuple[str, ...] = (
    "pass ball", "dribble", "shoot ball", "shoot", "kick shuttlecock", "throw", "pass",
)

ACTION_GROUPS: Mapping[str, tuple[str, ...]] = {
    "dance": DANCE_ACTIONS,
    "exercise": EXERCISE_ACTIONS,
    "ball": BALL_ACTIONS,
    # Everything whose motion repeats; the default training selection.
    "repetitive": DANCE_ACTIONS + EXERCISE_ACTIONS + BALL_ACTIONS,
}


def session_hash(session_id: str) -> str:
    """``20220618_3f97b19c01_subj12`` -> ``3f97b19c01``."""
    parts = session_id.split("_")
    if len(parts) < 2:
        raise ValueError(f"unexpected FreeMan session id: {session_id}")
    return parts[1]


def load_action_table(path: str | Path = DEFAULT_ACTION_TABLE) -> dict[str, str]:
    """Map session hash -> action label from ``FreeMan_actions.csv``."""
    table: dict[str, str] = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = str(row["Session"]).replace("_video", "").strip()
            table[key] = str(row["Actions"]).strip()
    if not table:
        raise ValueError(f"empty FreeMan action table: {path}")
    return table


def expand_actions(selection: Iterable[str]) -> frozenset[str]:
    """Expand group names (``dance``, ``repetitive``, ...) and plain labels."""
    labels: set[str] = set()
    for item in selection:
        name = str(item).strip()
        labels.update(ACTION_GROUPS.get(name, (name,)))
    return frozenset(labels)


def session_action(session_id: str, table: Mapping[str, str]) -> str | None:
    """Action label of a session, ``None`` when the release lists none."""
    return table.get(session_hash(session_id))

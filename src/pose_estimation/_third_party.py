"""Load the pinned SAM-3D-Body checkout without vendoring it into our package.

The upstream repository is a git submodule at
``src/pose_estimation/third_party/sam-3d-body``; it is the only external
source tree the project imports, and only this stage imports it.
"""

from pathlib import Path
import sys

THIRD_PARTY_ROOT = Path(__file__).resolve().parent / "third_party"


def ensure_sam3d_body_importable() -> Path:
    repository_root = Path(__file__).resolve().parents[2]
    candidates = [THIRD_PARTY_ROOT / "sam-3d-body"]
    git_entry = repository_root / ".git"
    if git_entry.is_file():
        # Linked worktree: the submodule checkout lives next to the main worktree.
        line = git_entry.read_text(encoding="utf-8").strip()
        if line.startswith("gitdir:"):
            worktree_git_dir = Path(line.split(":", 1)[1].strip()).resolve()
            common_file = worktree_git_dir / "commondir"
            if common_file.is_file():
                common_git_dir = (
                    worktree_git_dir
                    / common_file.read_text(encoding="utf-8").strip()
                ).resolve()
                candidates.append(
                    common_git_dir.parent / "src" / "pose_estimation" / "third_party" / "sam-3d-body"
                )
    checkout = next(
        (
            candidate
            for candidate in candidates
            if (candidate / "sam_3d_body/__init__.py").is_file()
        ),
        None,
    )
    if checkout is None:
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise RuntimeError(
            "SAM-3D-Body checkout is missing or uninitialized; searched: "
            f"{searched}"
        )
    checkout_text = str(checkout)
    if checkout_text not in sys.path:
        sys.path.insert(0, checkout_text)
    return checkout

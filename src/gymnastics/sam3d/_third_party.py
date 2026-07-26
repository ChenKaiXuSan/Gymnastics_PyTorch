"""Load the pinned SAM-3D-Body checkout without vendoring it into our package."""

from pathlib import Path
import sys


def ensure_sam3d_body_importable() -> Path:
    repository_root = Path(__file__).resolve().parents[3]
    checkout = repository_root / "third_party" / "sam-3d-body"
    if not checkout.is_dir():
        raise RuntimeError(
            "SAM-3D-Body checkout is missing; initialize third-party dependencies"
        )
    checkout_text = str(checkout)
    if checkout_text not in sys.path:
        sys.path.insert(0, checkout_text)
    return checkout

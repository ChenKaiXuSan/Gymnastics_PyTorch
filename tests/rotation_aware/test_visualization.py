from pathlib import Path

import numpy as np

from fuse.metadata.mhr70 import mhr_names
from fuse.rotation_aware.visualization import visualize_saved_sequence


def test_visualization_writes_curves_without_mutating_saved_arrays(
    tmp_path: Path,
) -> None:
    kpts = np.ones((5, len(mhr_names), 3), dtype=np.float32)
    source = tmp_path / "sequence.npz"
    np.savez_compressed(
        source,
        kpts_world=kpts,
        kpts_face_canonical=kpts.copy(),
        kpts_side_canonical=kpts.copy(),
        kpts_base_canonical=kpts.copy(),
        kpts_fused_canonical=kpts.copy(),
        theta_fused_rad=np.arange(5),
        omega_fused_rad_s=np.arange(5),
        quality_face=np.ones(5),
        quality_side=np.ones(5),
        frame_valid=np.ones(5, dtype=bool),
    )
    before = source.read_bytes()

    outputs = visualize_saved_sequence(source, tmp_path / "figures", animation=True)

    assert outputs.theta_omega_path.exists()
    assert outputs.quality_path.exists()
    assert outputs.animation_path is not None and outputs.animation_path.exists()
    assert source.read_bytes() == before

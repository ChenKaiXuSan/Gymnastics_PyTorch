from gymnastics.sam3d._third_party import ensure_sam3d_body_importable


def test_worktree_resolves_initialized_sam3d_checkout() -> None:
    checkout = ensure_sam3d_body_importable()

    assert (checkout / "sam_3d_body/__init__.py").is_file()

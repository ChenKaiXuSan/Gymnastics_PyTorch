from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_active_python_code_uses_single_src_package():
    package_root = PROJECT_ROOT / "src" / "gymnastics"

    assert package_root.is_dir()
    assert (package_root / "__init__.py").is_file()
    assert (package_root / "cli.py").is_file()


def test_runtime_assets_have_one_local_root():
    assert (PROJECT_ROOT / "local").is_dir()
    assert not (PROJECT_ROOT / "checkpoint").exists()
    assert not (PROJECT_ROOT / "ckpt").exists()
    assert not (PROJECT_ROOT / "camera_calibration" / "input_video").exists()


def test_legacy_top_level_packages_are_removed():
    legacy_packages = (
        "SAM3Dbody",
        "analysis",
        "camera_calibration",
        "fuse",
        "project",
        "split_cycle",
        "triangulation",
    )

    assert not [name for name in legacy_packages if (PROJECT_ROOT / name).exists()]

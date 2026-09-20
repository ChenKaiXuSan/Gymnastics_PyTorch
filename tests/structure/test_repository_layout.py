from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_active_python_code_is_split_into_the_four_pipeline_stages():
    src = PROJECT_ROOT / "src"
    for stage in ("pose_estimation", "cycle_alignment", "pseudo_gt", "fusion"):
        assert (src / stage / "__init__.py").is_file()
        assert (src / stage / "cli.py").is_file()
        assert (src / stage / "__main__.py").is_file()
    assert (src / "common" / "paths.py").is_file()
    assert (src / "configs").is_dir() and not (src / "configs" / "__init__.py").exists()
    assert not (src / "gymnastics").exists()


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

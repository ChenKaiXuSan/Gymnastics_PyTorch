from pathlib import Path


DATA_ROOT = "/home/data/xchen/gymnastics"


def test_main_configs_default_to_current_data_root():
    config_files = [
        "configs/classification/train.yaml",
        "configs/legacy/pose_estimation/inference.yaml",
        "configs/sam3d/sam3d_body.yaml",
        "configs/triangulation/legacy.yaml",
    ]

    for config_file in config_files:
        text = Path(config_file).read_text(encoding="utf-8")
        assert DATA_ROOT in text
        assert "/workspace/data" not in text

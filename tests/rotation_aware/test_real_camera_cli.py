from pathlib import Path

from gymnastics.fusion.rotation_aware.real_camera_cli import build_parser


def test_train_matrix_cli_collects_seed_subset_and_device() -> None:
    args = build_parser().parse_args(
        [
            "train-matrix",
            "--config",
            "configs/fusion/real_camera_pilot.yaml",
            "--seed",
            "0",
            "--seed",
            "2",
            "--device",
            "cuda:0",
        ]
    )

    assert args.command == "train-matrix"
    assert args.config == Path("configs/fusion/real_camera_pilot.yaml")
    assert args.seed == [0, 2]
    assert args.device == "cuda:0"


def test_evaluate_cli_has_no_training_device_argument() -> None:
    args = build_parser().parse_args(
        [
            "evaluate",
            "--config",
            "configs/fusion/real_camera_pilot.yaml",
        ]
    )

    assert args.command == "evaluate"
    assert args.config == Path("configs/fusion/real_camera_pilot.yaml")
    assert not hasattr(args, "device")


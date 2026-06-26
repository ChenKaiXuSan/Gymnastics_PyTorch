from pathlib import Path

from split_cycle import main as split_main


def test_resolve_person_ids_filters_requested_people(tmp_path):
    person_root = tmp_path / "person"
    for person_id in ["1", "2", "10"]:
        (person_root / person_id).mkdir(parents=True)

    person_ids = split_main.resolve_person_ids(person_root, ["10", "2"])

    assert person_ids == ["2", "10"]


def test_resolve_person_ids_defaults_to_all_people(tmp_path):
    person_root = tmp_path / "person"
    for person_id in ["10", "1", "2"]:
        (person_root / person_id).mkdir(parents=True)

    person_ids = split_main.resolve_person_ids(person_root, None)

    assert person_ids == ["1", "2", "10"]


def test_parse_args_keeps_legacy_positional_threads():
    args = split_main.parse_args(["7"])

    assert args.threads == 7
    assert args.person is None


def test_parse_args_accepts_explicit_people_and_roots():
    args = split_main.parse_args(
        [
            "--threads",
            "3",
            "--person",
            "46",
            "47",
            "--raw-root",
            "/raw",
            "--kpt-root",
            "/kpt",
            "--log-root",
            "/logs",
        ]
    )

    assert args.threads == 3
    assert args.person == ["46", "47"]
    assert args.raw_root == Path("/raw")
    assert args.kpt_root == Path("/kpt")
    assert args.log_root == Path("/logs")

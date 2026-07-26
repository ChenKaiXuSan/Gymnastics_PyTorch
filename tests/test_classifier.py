from importlib.util import find_spec


def test_classification_models_are_part_of_the_gymnastics_package():
    assert find_spec("gymnastics.classification.models.st_gcn") is not None
    assert find_spec("gymnastics.classification.models.tcn") is not None
    assert find_spec("gymnastics.classification.models.skeleton_mamba") is not None

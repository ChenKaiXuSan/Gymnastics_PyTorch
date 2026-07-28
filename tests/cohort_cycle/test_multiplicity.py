import numpy as np

from gymnastics.analysis.cohort_cycle.multiplicity import (
    benjamini_hochberg,
    holm_adjust,
)


def test_holm_adjust_controls_family_in_original_order():
    """Using independent raw p-values would inflate the core family error."""
    adjusted = holm_adjust(np.array([0.01, 0.04, 0.03]))
    np.testing.assert_allclose(adjusted, np.array([0.03, 0.06, 0.06]))


def test_benjamini_hochberg_is_monotone_in_rank():
    """Exploratory FDR values must obey the step-up monotonicity rule."""
    adjusted = benjamini_hochberg(np.array([0.01, 0.04, 0.03]))
    np.testing.assert_allclose(adjusted, np.array([0.03, 0.04, 0.04]))

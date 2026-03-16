from __future__ import annotations

import pandas as pd

from march_madness.models.baseline import apply_probability_calibrator, fit_probability_calibrator


def test_probability_calibrator_preserves_probability_bounds() -> None:
    y_true = pd.Series([0, 0, 1, 1, 0, 1, 0, 1])
    probabilities = pd.Series([0.08, 0.22, 0.61, 0.79, 0.35, 0.67, 0.41, 0.88])

    calibrator = fit_probability_calibrator(y_true, probabilities, random_state=7)
    calibrated = apply_probability_calibrator(calibrator, probabilities)

    assert calibrator is not None
    assert calibrated.between(0.0, 1.0).all()
    assert calibrated.index.tolist() == probabilities.index.tolist()


def test_identity_calibration_returns_clipped_probabilities() -> None:
    probabilities = pd.Series([0.0, 0.5, 1.0])

    calibrated = apply_probability_calibrator(None, probabilities)

    assert calibrated.tolist()[0] > 0.0
    assert calibrated.tolist()[-1] < 1.0

from __future__ import annotations

from typing import Any

import pandas as pd

from march_madness.features.matchup_builder import build_custom_matchup_frame
from march_madness.models.baseline import apply_probability_calibrator, blend_probabilities, predict_probabilities


def predict_single_matchup(
    model_bundle: dict[str, Any],
    team_features: pd.DataFrame,
    season: int,
    team_one_id: int,
    team_two_id: int,
    division: str,
) -> tuple[float, pd.DataFrame]:
    frame = build_custom_matchup_frame(division, season, team_one_id, team_two_id, team_features)
    logistic_probability = predict_probabilities(model_bundle["models"]["logistic"], frame, model_bundle["feature_columns"]).iloc[0]
    hist_gbm_probability = predict_probabilities(model_bundle["models"]["hist_gbm"], frame, model_bundle["feature_columns"]).iloc[0]
    xgboost_probability = predict_probabilities(model_bundle["models"]["xgboost"], frame, model_bundle["feature_columns"]).iloc[0]
    canonical_probability = float(
        blend_probabilities(
            {
                "logistic": pd.Series([logistic_probability]),
                "hist_gbm": pd.Series([hist_gbm_probability]),
                "xgboost": pd.Series([xgboost_probability]),
            },
            model_bundle["blend_weights"],
        ).iloc[0]
    )
    canonical_probability = float(
        apply_probability_calibrator(
            model_bundle.get("calibrator"),
            pd.Series([canonical_probability]),
        ).iloc[0]
    )
    market_probability = frame.iloc[0].get("market_probability_a")
    if pd.notna(market_probability):
        canonical_probability = float(
            (1.0 - model_bundle.get("market_weight", 0.0)) * canonical_probability
            + model_bundle.get("market_weight", 0.0) * float(market_probability)
        )
    selected_probability = canonical_probability if int(team_one_id) < int(team_two_id) else 1.0 - canonical_probability
    return selected_probability, frame

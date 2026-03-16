from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from march_madness.config import ARTIFACTS_DIR
from march_madness.features.matchup_builder import build_submission_feature_frame
from march_madness.models.baseline import apply_probability_calibrator, blend_probabilities, predict_probabilities


def model_bundle_path(division: str) -> Path:
    return ARTIFACTS_DIR / "models" / f"{division.lower()}_baseline_model.joblib"


def feature_table_path(division: str) -> Path:
    return ARTIFACTS_DIR / "features" / f"{division.lower()}_team_features.csv"


def top25_context_path(division: str) -> Path:
    return ARTIFACTS_DIR / "features" / f"{division.lower()}_top25_context.csv"


def metrics_path(division: str) -> Path:
    return ARTIFACTS_DIR / "metrics" / f"{division.lower()}_cv_metrics.csv"


def save_model_bundle(division: str, bundle: dict[str, Any]) -> None:
    joblib.dump(bundle, model_bundle_path(division))


def load_model_bundle(division: str) -> dict[str, Any]:
    return joblib.load(model_bundle_path(division))


def save_feature_table(division: str, team_features: pd.DataFrame) -> None:
    team_features.to_csv(feature_table_path(division), index=False)


def save_top25_context_table(division: str, top25_context: pd.DataFrame) -> None:
    top25_context.to_csv(top25_context_path(division), index=False)


def load_feature_table(division: str) -> pd.DataFrame:
    return pd.read_csv(feature_table_path(division))


def load_top25_context_table(division: str) -> pd.DataFrame:
    path = top25_context_path(division)
    if not path.exists():
        return pd.DataFrame(columns=["season", "team_id", "top25_elo_wins", "top25_elo_games"])
    return pd.read_csv(path)


def save_metrics_table(division: str, metrics_frame: pd.DataFrame) -> None:
    metrics_frame.to_csv(metrics_path(division), index=False)


def generate_division_submission(
    division: str,
    model_bundle: dict[str, Any],
    team_features: pd.DataFrame,
    stage: int = 2,
    refresh_external: bool = False,
) -> pd.DataFrame:
    submission_features = build_submission_feature_frame(division, team_features, stage=stage, refresh_external=refresh_external)
    prediction_map = {
        "logistic": predict_probabilities(model_bundle["models"]["logistic"], submission_features, model_bundle["feature_columns"]),
        "hist_gbm": predict_probabilities(model_bundle["models"]["hist_gbm"], submission_features, model_bundle["feature_columns"]),
        "xgboost": predict_probabilities(model_bundle["models"]["xgboost"], submission_features, model_bundle["feature_columns"]),
    }
    predictions = blend_probabilities(prediction_map, model_bundle["blend_weights"])
    predictions = apply_probability_calibrator(model_bundle.get("calibrator"), predictions)
    if "market_probability_a" in submission_features.columns:
        market_mask = submission_features["market_probability_a"].notna()
        predictions.loc[market_mask] = (
            (1.0 - model_bundle.get("market_weight", 0.0)) * predictions.loc[market_mask]
            + model_bundle.get("market_weight", 0.0) * submission_features.loc[market_mask, "market_probability_a"].astype(float)
        )
    output = submission_features[["ID"]].copy()
    output["Pred"] = predictions.clip(0.0, 1.0)
    return output


def save_submission(stage: int, submission_frame: pd.DataFrame) -> Path:
    output_path = ARTIFACTS_DIR / "submissions" / f"stage{stage}_baseline_submission.csv"
    submission_frame.sort_values("ID").to_csv(output_path, index=False)
    return output_path

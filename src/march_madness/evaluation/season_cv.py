from __future__ import annotations

import pandas as pd

from march_madness.evaluation.metrics import probability_metrics
from march_madness.models.baseline import (
    HIST_GBM_CONFIGS,
    LOGISTIC_CONFIGS,
    XGBOOST_CONFIGS,
    apply_probability_calibrator,
    blend_probabilities,
    fit_probability_calibrator,
    fit_candidate_models,
    optimize_blend_weights,
    predict_probabilities,
)


MODEL_CONFIG_SPACE = {
    "logistic": list(LOGISTIC_CONFIGS.keys()),
    "hist_gbm": list(HIST_GBM_CONFIGS.keys()),
    "xgboost": list(XGBOOST_CONFIGS.keys()),
}


def _oof_predictions_for_single_family(
    training_frame: pd.DataFrame,
    feature_columns: list[str],
    family_name: str,
    config_name: str,
    min_train_seasons: int,
    random_state: int,
) -> pd.DataFrame:
    seasons = sorted(training_frame["season"].unique().tolist())
    rows: list[pd.DataFrame] = []
    for holdout_season in seasons:
        train_frame = training_frame[training_frame["season"] < holdout_season]
        test_frame = training_frame[training_frame["season"] == holdout_season]
        if len(train_frame["season"].unique()) < min_train_seasons or test_frame.empty:
            continue
        selected_configs = {"logistic": "baseline", "hist_gbm": "balanced", "xgboost": "balanced"}
        selected_configs[family_name] = config_name
        models = fit_candidate_models(
            train_frame,
            feature_columns,
            random_state=random_state,
            selected_configs=selected_configs,
        )
        predictions = predict_probabilities(models[family_name], test_frame, feature_columns)
        rows.append(pd.DataFrame({"target": test_frame["target"], "prediction": predictions}))
    if not rows:
        return pd.DataFrame(columns=["target", "prediction"])
    return pd.concat(rows, ignore_index=True)


def select_model_configs(
    training_frame: pd.DataFrame,
    feature_columns: list[str],
    min_train_seasons: int,
    random_state: int,
) -> dict[str, str]:
    selected = {"logistic": "baseline", "hist_gbm": "balanced", "xgboost": "balanced"}
    for family_name, configs in MODEL_CONFIG_SPACE.items():
        best_score = float("inf")
        best_name = selected[family_name]
        for config_name in configs:
            oof = _oof_predictions_for_single_family(
                training_frame,
                feature_columns,
                family_name,
                config_name,
                min_train_seasons=min_train_seasons,
                random_state=random_state,
            )
            if oof.empty:
                continue
            score = float(((oof["prediction"] - oof["target"]) ** 2).mean())
            if score < best_score:
                best_score = score
                best_name = config_name
        selected[family_name] = best_name
    return selected


def run_rolling_season_cv(
    training_frame: pd.DataFrame,
    feature_columns: list[str],
    min_train_seasons: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], object, dict[str, str]]:
    seasons = sorted(training_frame["season"].unique().tolist())
    metric_rows: list[dict[str, float | int | str]] = []
    prediction_rows: list[pd.DataFrame] = []
    oof_rows: list[pd.DataFrame] = []
    selected_configs = select_model_configs(
        training_frame,
        feature_columns,
        min_train_seasons=min_train_seasons,
        random_state=random_state,
    )

    for holdout_season in seasons:
        train_frame = training_frame[training_frame["season"] < holdout_season]
        test_frame = training_frame[training_frame["season"] == holdout_season]
        train_seasons = sorted(train_frame["season"].unique().tolist())
        if len(train_seasons) < min_train_seasons or test_frame.empty:
            continue

        models = fit_candidate_models(train_frame, feature_columns, random_state=random_state, selected_configs=selected_configs)
        logistic_probabilities = predict_probabilities(models["logistic"], test_frame, feature_columns)
        hist_gbm_probabilities = predict_probabilities(models["hist_gbm"], test_frame, feature_columns)
        xgboost_probabilities = predict_probabilities(models["xgboost"], test_frame, feature_columns)

        holdout_predictions = test_frame[["division", "season", "team_a", "team_b", "team_a_name", "team_b_name", "target"]].copy()
        holdout_predictions["logistic_prediction"] = logistic_probabilities
        holdout_predictions["hist_gbm_prediction"] = hist_gbm_probabilities
        holdout_predictions["xgboost_prediction"] = xgboost_probabilities
        prediction_rows.append(holdout_predictions)
        oof_rows.append(holdout_predictions[["target", "logistic_prediction", "hist_gbm_prediction", "xgboost_prediction"]])

    oof_frame = pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame()
    blend_weights = {"logistic": 1 / 3, "hist_gbm": 1 / 3, "xgboost": 1 / 3}
    calibrator = None
    if not oof_frame.empty:
        blend_weights, _ = optimize_blend_weights(
            oof_frame["target"],
            {
                "logistic": oof_frame["logistic_prediction"],
                "hist_gbm": oof_frame["hist_gbm_prediction"],
                "xgboost": oof_frame["xgboost_prediction"],
            },
        )
        raw_oof_probabilities = blend_probabilities(
            {
                "logistic": oof_frame["logistic_prediction"],
                "hist_gbm": oof_frame["hist_gbm_prediction"],
                "xgboost": oof_frame["xgboost_prediction"],
            },
            blend_weights,
        )
        calibrator = fit_probability_calibrator(oof_frame["target"], raw_oof_probabilities, random_state=random_state)

    for holdout_predictions in prediction_rows:
        raw_probabilities = blend_probabilities(
            {
                "logistic": holdout_predictions["logistic_prediction"],
                "hist_gbm": holdout_predictions["hist_gbm_prediction"],
                "xgboost": holdout_predictions["xgboost_prediction"],
            },
            blend_weights,
        )
        probabilities = apply_probability_calibrator(calibrator, raw_probabilities)
        metrics = probability_metrics(holdout_predictions["target"], probabilities)
        metric_rows.append(
            {
                "division": str(holdout_predictions["division"].iloc[0]),
                "season": int(holdout_predictions["season"].iloc[0]),
                "train_season_count": int(len(training_frame[training_frame["season"] < holdout_predictions["season"].iloc[0]]["season"].unique())),
                "logistic_weight": float(blend_weights["logistic"]),
                "hist_gbm_weight": float(blend_weights["hist_gbm"]),
                "xgboost_weight": float(blend_weights["xgboost"]),
                **metrics,
            }
        )
        holdout_predictions["raw_prediction"] = raw_probabilities
        holdout_predictions["prediction"] = probabilities

    metrics_frame = pd.DataFrame(metric_rows)
    predictions_frame = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    return metrics_frame, predictions_frame, blend_weights, calibrator, selected_configs

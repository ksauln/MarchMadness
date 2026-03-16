from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


LOGISTIC_CONFIGS = {
    "balanced": {"C": 0.5},
    "baseline": {"C": 1.0},
    "loose": {"C": 2.0},
}

HIST_GBM_CONFIGS = {
    "balanced": {"learning_rate": 0.04, "max_depth": 4, "max_iter": 500, "min_samples_leaf": 10, "l2_regularization": 0.2},
    "patient": {"learning_rate": 0.03, "max_depth": 3, "max_iter": 700, "min_samples_leaf": 8, "l2_regularization": 0.1},
    "aggressive": {"learning_rate": 0.06, "max_depth": 5, "max_iter": 350, "min_samples_leaf": 15, "l2_regularization": 0.4},
}

XGBOOST_CONFIGS = {
    "balanced": {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 4, "min_child_weight": 2, "subsample": 0.85, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.0},
    "patient": {"n_estimators": 700, "learning_rate": 0.02, "max_depth": 3, "min_child_weight": 1, "subsample": 0.9, "colsample_bytree": 0.75, "reg_lambda": 1.5, "reg_alpha": 0.0},
    "aggressive": {"n_estimators": 350, "learning_rate": 0.05, "max_depth": 5, "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.85, "reg_lambda": 2.0, "reg_alpha": 0.0},
}


def build_logistic_pipeline(random_state: int = 42, config_name: str = "baseline") -> Pipeline:
    config = LOGISTIC_CONFIGS[config_name]
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=4000,
                    solver="lbfgs",
                    C=config["C"],
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_hist_gbm_pipeline(random_state: int = 42, config_name: str = "balanced") -> Pipeline:
    config = HIST_GBM_CONFIGS[config_name]
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=config["learning_rate"],
                    max_depth=config["max_depth"],
                    max_iter=config["max_iter"],
                    min_samples_leaf=config["min_samples_leaf"],
                    l2_regularization=config["l2_regularization"],
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_xgboost_pipeline(random_state: int = 42, config_name: str = "balanced") -> Pipeline:
    config = XGBOOST_CONFIGS[config_name]
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBClassifier(
                    n_estimators=config["n_estimators"],
                    learning_rate=config["learning_rate"],
                    max_depth=config["max_depth"],
                    min_child_weight=config["min_child_weight"],
                    subsample=config["subsample"],
                    colsample_bytree=config["colsample_bytree"],
                    reg_lambda=config["reg_lambda"],
                    reg_alpha=config["reg_alpha"],
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_candidate_models(
    random_state: int = 42,
    selected_configs: dict[str, str] | None = None,
) -> dict[str, Pipeline]:
    selected_configs = selected_configs or {"logistic": "baseline", "hist_gbm": "balanced", "xgboost": "balanced"}
    return {
        "logistic": build_logistic_pipeline(random_state=random_state, config_name=selected_configs["logistic"]),
        "hist_gbm": build_hist_gbm_pipeline(random_state=random_state, config_name=selected_configs["hist_gbm"]),
        "xgboost": build_xgboost_pipeline(random_state=random_state, config_name=selected_configs["xgboost"]),
    }


def fit_candidate_models(
    training_frame: pd.DataFrame,
    feature_columns: list[str],
    random_state: int = 42,
    selected_configs: dict[str, str] | None = None,
) -> dict[str, Pipeline]:
    fitted: dict[str, Pipeline] = {}
    for name, pipeline in build_candidate_models(random_state=random_state, selected_configs=selected_configs).items():
        model = clone(pipeline)
        model.fit(training_frame[feature_columns], training_frame["target"])
        fitted[name] = model
    return fitted


def predict_probabilities(model: Pipeline, frame: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
    probabilities = model.predict_proba(frame[feature_columns])[:, 1]
    return pd.Series(probabilities, index=frame.index, dtype=float)


def blend_probabilities(probability_map: dict[str, pd.Series], blend_weights: dict[str, float]) -> pd.Series:
    output = None
    for model_name, probabilities in probability_map.items():
        weight = blend_weights.get(model_name, 0.0)
        component = probabilities * weight
        output = component if output is None else output + component
    if output is None:
        raise ValueError("No probabilities provided for blending.")
    return output


def _logit_features(probabilities: pd.Series | np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def fit_probability_calibrator(
    y_true: pd.Series,
    probabilities: pd.Series,
    random_state: int = 42,
) -> LogisticRegression | None:
    target = pd.Series(y_true).astype(int)
    if target.nunique() < 2:
        return None
    calibrator = LogisticRegression(
        C=0.7,
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )
    calibrator.fit(_logit_features(probabilities), target)
    return calibrator


def apply_probability_calibrator(
    calibrator: LogisticRegression | None,
    probabilities: pd.Series,
) -> pd.Series:
    clipped = pd.Series(probabilities, index=probabilities.index, dtype=float).clip(1e-6, 1.0 - 1e-6)
    if calibrator is None:
        return clipped
    calibrated = calibrator.predict_proba(_logit_features(clipped))[:, 1]
    return pd.Series(calibrated, index=clipped.index, dtype=float).clip(1e-6, 1.0 - 1e-6)


def optimize_blend_weights(y_true: pd.Series, probability_map: dict[str, pd.Series], step: float = 0.02) -> tuple[dict[str, float], float]:
    model_names = list(probability_map)
    best_score = float("inf")
    best_weights = {name: 1.0 / len(model_names) for name in model_names}
    scale = int(round(1.0 / step))
    for first in range(scale + 1):
        for second in range(scale + 1 - first):
            third = scale - first - second
            weights = {
                model_names[0]: first / scale,
                model_names[1]: second / scale,
                model_names[2]: third / scale,
            }
            blended = blend_probabilities(probability_map, weights)
            score = float(((blended - y_true) ** 2).mean())
            if score < best_score:
                best_weights = weights
                best_score = score
    return best_weights, best_score


def build_model_bundle(
    division: str,
    models: dict[str, Pipeline],
    feature_columns: list[str],
    training_frame: pd.DataFrame,
    blend_weights: dict[str, float],
    calibrator: LogisticRegression | None = None,
    market_weight: float = 0.2,
    selected_configs: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "division": division,
        "models": models,
        "feature_columns": feature_columns,
        "trained_seasons": sorted(training_frame["season"].unique().tolist()),
        "training_rows": int(len(training_frame)),
        "blend_weights": {name: float(weight) for name, weight in blend_weights.items()},
        "calibrator": calibrator,
        "calibration_strategy": "platt_logit" if calibrator is not None else "identity",
        "market_weight": float(market_weight),
        "selected_configs": selected_configs or {},
    }

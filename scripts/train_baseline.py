from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from march_madness.config import ARTIFACTS_DIR, DIVISION_LABELS, DIVISIONS, ensure_artifact_dirs
from march_madness.data.external import build_espn_bracket_games
from march_madness.evaluation.season_cv import run_rolling_season_cv
from march_madness.features.matchup_builder import build_tournament_training_frame, matchup_feature_columns
from march_madness.features.team_aggregates import build_team_features, build_top25_context
from march_madness.inference.submission import (
    generate_division_submission,
    save_feature_table,
    save_metrics_table,
    save_model_bundle,
    save_submission,
    save_top25_context_table,
)
from march_madness.models.baseline import build_model_bundle, fit_candidate_models
from march_madness.simulation import run_bracket_simulation, save_bracket_games


def _division_simulation_count(
    division: str,
    default_n_simulations: int,
    men_n_simulations: int | None,
    women_n_simulations: int | None,
) -> int:
    if division == "M":
        return men_n_simulations if men_n_simulations is not None else 500_000
    if division == "W" and women_n_simulations is not None:
        return women_n_simulations
    return default_n_simulations


def run(
    stage: int,
    min_train_seasons: int,
    random_state: int,
    refresh_external: bool,
    n_simulations: int,
    men_n_simulations: int | None = None,
    women_n_simulations: int | None = None,
) -> dict[str, object]:
    ensure_artifact_dirs()
    feature_columns = matchup_feature_columns()
    combined_submissions: list[pd.DataFrame] = []
    summary: dict[str, object] = {"stage": stage, "divisions": {}}

    for division in DIVISIONS:
        print(f"[{DIVISION_LABELS[division]}] refreshing ESPN bracket data")
        bracket_games = build_espn_bracket_games(division, refresh=refresh_external)
        save_bracket_games(division, bracket_games)

        print(f"[{DIVISION_LABELS[division]}] building team features")
        team_features = build_team_features(division)
        save_feature_table(division, team_features)
        save_top25_context_table(division, build_top25_context(division, team_features))

        print(f"[{DIVISION_LABELS[division]}] building tournament training frame")
        training_frame = build_tournament_training_frame(division, team_features)

        print(f"[{DIVISION_LABELS[division]}] running rolling season validation")
        metrics_frame, prediction_frame, blend_weights, calibrator, selected_configs = run_rolling_season_cv(
            training_frame,
            feature_columns=feature_columns,
            min_train_seasons=min_train_seasons,
            random_state=random_state,
        )
        save_metrics_table(division, metrics_frame)
        prediction_frame.to_csv(ARTIFACTS_DIR / "metrics" / f"{division.lower()}_cv_predictions.csv", index=False)

        print(f"[{DIVISION_LABELS[division]}] fitting final ensemble")
        models = fit_candidate_models(training_frame, feature_columns, random_state=random_state, selected_configs=selected_configs)
        bundle = build_model_bundle(
            division,
            models,
            feature_columns,
            training_frame,
            blend_weights=blend_weights,
            calibrator=calibrator,
            market_weight=0.2,
            selected_configs=selected_configs,
        )
        save_model_bundle(division, bundle)

        print(f"[{DIVISION_LABELS[division]}] generating Stage {stage} predictions")
        submission_frame = generate_division_submission(division, bundle, team_features, stage=stage, refresh_external=False)
        combined_submissions.append(submission_frame)

        print(f"[{DIVISION_LABELS[division]}] running bracket simulation")
        division_n_simulations = _division_simulation_count(
            division,
            default_n_simulations=n_simulations,
            men_n_simulations=men_n_simulations,
            women_n_simulations=women_n_simulations,
        )
        bracket_simulation = run_bracket_simulation(
            division,
            bundle,
            team_features,
            n_simulations=division_n_simulations,
            market_weight=bundle["market_weight"],
            refresh_external=False,
        )

        summary["divisions"][division] = {
            "training_rows": int(len(training_frame)),
            "seasons": [int(season) for season in sorted(training_frame["season"].unique().tolist())],
            "cv_rows": int(len(metrics_frame)),
            "blend_weights": {name: float(weight) for name, weight in blend_weights.items()},
            "selected_configs": selected_configs,
            "calibration_strategy": bundle["calibration_strategy"],
            "market_weight": float(bundle["market_weight"]),
            "n_simulations": int(division_n_simulations),
            "avg_brier_score": float(metrics_frame["brier_score"].mean()) if not metrics_frame.empty else None,
            "avg_log_loss": float(metrics_frame["log_loss"].mean()) if not metrics_frame.empty else None,
            "sim_title_favorite": str(bracket_simulation.iloc[0]["team_name"]) if not bracket_simulation.empty else None,
            "sim_title_probability": float(bracket_simulation.iloc[0]["title"]) if not bracket_simulation.empty else None,
        }

    combined_submission = pd.concat(combined_submissions, ignore_index=True).sort_values("ID").reset_index(drop=True)
    output_path = save_submission(stage, combined_submission)
    summary["submission_path"] = str(output_path)
    summary_path = ARTIFACTS_DIR / "metrics" / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved submission to {output_path}")
    print(f"Saved summary to {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the baseline March Madness models and generate a submission.")
    parser.add_argument("--stage", type=int, default=2, choices=(1, 2))
    parser.add_argument("--min-train-seasons", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--refresh-external", action="store_true")
    parser.add_argument("--n-simulations", type=int, default=10000)
    parser.add_argument("--men-n-simulations", type=int, default=None)
    parser.add_argument("--women-n-simulations", type=int, default=None)
    args = parser.parse_args()
    run(
        stage=args.stage,
        min_train_seasons=args.min_train_seasons,
        random_state=args.random_state,
        refresh_external=args.refresh_external,
        n_simulations=args.n_simulations,
        men_n_simulations=args.men_n_simulations,
        women_n_simulations=args.women_n_simulations,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import pandas as pd

from march_madness.data.external import build_market_lines_table, load_seed_table
from march_madness.data.loaders import infer_division_from_team_id, load_sample_submission, load_tournament_results
from march_madness.features.team_aggregates import BASE_TEAM_FEATURE_COLUMNS


def parse_submission_ids(submission: pd.DataFrame | None = None, stage: int = 2) -> pd.DataFrame:
    source = load_sample_submission(stage=stage) if submission is None else submission.copy()
    parts = source["ID"].str.split("_", expand=True).astype(int)
    parsed = source.copy()
    parsed["season"] = parts[0]
    parsed["team_a"] = parts[1]
    parsed["team_b"] = parts[2]
    parsed["division"] = parsed["team_a"].map(infer_division_from_team_id)
    return parsed


def _merge_team_features(matchups: pd.DataFrame, team_features: pd.DataFrame) -> pd.DataFrame:
    feature_columns = BASE_TEAM_FEATURE_COLUMNS
    left_features = team_features[["season", "team_id", "team_name", *feature_columns]].rename(
        columns={"team_id": "team_a", "team_name": "team_a_name", **{column: f"{column}_a" for column in feature_columns}}
    )
    right_features = team_features[["season", "team_id", "team_name", *feature_columns]].rename(
        columns={"team_id": "team_b", "team_name": "team_b_name", **{column: f"{column}_b" for column in feature_columns}}
    )

    merged = matchups.merge(left_features, on=["season", "team_a"], how="left").merge(
        right_features, on=["season", "team_b"], how="left"
    )
    for column in feature_columns:
        merged[f"{column}_diff"] = merged[f"{column}_a"] - merged[f"{column}_b"]
    merged["seedless_feature_coverage"] = merged["games_a"].notna().astype(int) + merged["games_b"].notna().astype(int)
    return merged


def _merge_seed_features(matchups: pd.DataFrame, division: str, refresh_external: bool = False) -> pd.DataFrame:
    seeds = load_seed_table(division, refresh=refresh_external)
    left = seeds[["season", "team_id", "seed_num", "seed_missing"]].rename(
        columns={"team_id": "team_a", "seed_num": "seed_num_a", "seed_missing": "seed_missing_a"}
    )
    right = seeds[["season", "team_id", "seed_num", "seed_missing"]].rename(
        columns={"team_id": "team_b", "seed_num": "seed_num_b", "seed_missing": "seed_missing_b"}
    )
    merged = matchups.merge(left, on=["season", "team_a"], how="left").merge(right, on=["season", "team_b"], how="left")
    merged["seed_num_a"] = merged["seed_num_a"].fillna(20.0)
    merged["seed_num_b"] = merged["seed_num_b"].fillna(20.0)
    merged["seed_missing_a"] = merged["seed_missing_a"].fillna(1.0)
    merged["seed_missing_b"] = merged["seed_missing_b"].fillna(1.0)
    merged["seed_diff"] = merged["seed_num_b"] - merged["seed_num_a"]
    merged["seed_sum"] = merged["seed_num_a"] + merged["seed_num_b"]
    return merged


def _merge_market_features(matchups: pd.DataFrame, division: str, refresh_external: bool = False) -> pd.DataFrame:
    lines = build_market_lines_table(division, refresh=refresh_external)
    if lines.empty:
        matchups["market_probability_a"] = pd.NA
        matchups["official_game_flag"] = 0
        matchups["market_round_id"] = pd.NA
        return matchups
    merged = matchups.merge(lines, on=["season", "division", "team_a", "team_b"], how="left")
    merged = merged.rename(columns={"round_id": "market_round_id"})
    merged["official_game_flag"] = merged["official_game_flag"].fillna(0)
    return merged


def matchup_feature_columns() -> list[str]:
    columns = ["season"]
    for source_column in BASE_TEAM_FEATURE_COLUMNS:
        columns.extend([f"{source_column}_a", f"{source_column}_b", f"{source_column}_diff"])
    columns.extend(
        [
            "seedless_feature_coverage",
            "seed_num_a",
            "seed_num_b",
            "seed_missing_a",
            "seed_missing_b",
            "seed_diff",
            "seed_sum",
        ]
    )
    return columns


def build_tournament_training_frame(division: str, team_features: pd.DataFrame) -> pd.DataFrame:
    results = load_tournament_results(division)
    available_seasons = set(team_features["season"].unique())
    results = results[results["Season"].isin(available_seasons)].copy()

    winner_view = pd.DataFrame(
        {
            "division": division,
            "season": results["Season"].astype(int),
            "team_a": results["WTeamID"].astype(int),
            "team_b": results["LTeamID"].astype(int),
            "target": 1,
            "day_num": results["DayNum"].astype(int),
        }
    )
    loser_view = winner_view.rename(columns={"team_a": "team_b", "team_b": "team_a"}).copy()
    loser_view["target"] = 0
    matchups = pd.concat([winner_view, loser_view], ignore_index=True)
    frame = _merge_team_features(matchups, team_features)
    frame = _merge_seed_features(frame, division)
    frame = _merge_market_features(frame, division)
    return frame.sort_values(["season", "team_a", "team_b", "target"], ascending=[True, True, True, False]).reset_index(drop=True)


def build_submission_feature_frame(division: str, team_features: pd.DataFrame, stage: int = 2, refresh_external: bool = False) -> pd.DataFrame:
    parsed = parse_submission_ids(stage=stage)
    division_rows = parsed[parsed["division"] == division].copy()
    frame = _merge_team_features(division_rows, team_features)
    frame = _merge_seed_features(frame, division, refresh_external=refresh_external)
    frame = _merge_market_features(frame, division, refresh_external=refresh_external)
    return frame.sort_values("ID").reset_index(drop=True)


def build_custom_matchup_frame(division: str, season: int, team_a: int, team_b: int, team_features: pd.DataFrame, refresh_external: bool = False) -> pd.DataFrame:
    canonical_a, canonical_b = sorted((int(team_a), int(team_b)))
    matchups = pd.DataFrame(
        [{"division": division, "season": int(season), "team_a": canonical_a, "team_b": canonical_b}]
    )
    frame = _merge_team_features(matchups, team_features)
    frame = _merge_seed_features(frame, division, refresh_external=refresh_external)
    frame = _merge_market_features(frame, division, refresh_external=refresh_external)
    return frame

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from march_madness.data.canonicalize import build_regular_season_long
from march_madness.data.loaders import load_massey_ordinals, load_regular_season_results, load_teams


BASE_TEAM_FEATURE_COLUMNS = [
    "games",
    "wins",
    "losses",
    "win_pct",
    "avg_points_for",
    "avg_points_against",
    "avg_margin",
    "avg_num_ot",
    "fg_pct",
    "fg3_pct",
    "ft_pct",
    "efg_pct",
    "opp_efg_pct",
    "ft_rate",
    "fg3_rate",
    "turnover_rate",
    "assist_rate",
    "ast_to_ratio",
    "orb_per_game",
    "drb_per_game",
    "orb_pct",
    "drb_pct",
    "stl_per_game",
    "blk_per_game",
    "pf_per_game",
    "avg_possessions",
    "off_rating",
    "def_rating",
    "net_rating",
    "last10_win_pct",
    "last10_avg_margin",
    "last10_avg_points_for",
    "last10_avg_points_against",
    "last5_win_pct",
    "last5_avg_margin",
    "opp_avg_win_pct",
    "opp_avg_margin",
    "elo_rating",
    "massey_mean_rank",
    "massey_median_rank",
    "massey_best_rank",
    "massey_worst_rank",
    "massey_rank_std",
    "massey_system_count",
]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.astype(float) / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def compute_elo_ratings(division: str, base_rating: float = 1500.0, home_advantage: float = 60.0) -> pd.DataFrame:
    results = load_regular_season_results(division).sort_values(["Season", "DayNum", "WTeamID", "LTeamID"])
    rows: list[dict[str, float | int]] = []

    for season, season_games in results.groupby("Season", sort=True):
        ratings: defaultdict[int, float] = defaultdict(lambda: base_rating)
        for game in season_games.itertuples(index=False):
            winner_id = int(game.WTeamID)
            loser_id = int(game.LTeamID)
            winner_rating = ratings[winner_id]
            loser_rating = ratings[loser_id]

            winner_loc_boost = home_advantage if game.WLoc == "H" else 0.0
            loser_loc_boost = home_advantage if game.WLoc == "A" else 0.0
            expected_winner = 1.0 / (
                1.0 + 10.0 ** (-((winner_rating + winner_loc_boost) - (loser_rating + loser_loc_boost)) / 400.0)
            )
            score_margin = abs(int(game.WScore) - int(game.LScore))
            k_factor = 20.0 + min(score_margin, 25) * 0.2
            delta = k_factor * (1.0 - expected_winner)
            ratings[winner_id] = winner_rating + delta
            ratings[loser_id] = loser_rating - delta

        for team_id, rating in ratings.items():
            rows.append({"season": int(season), "team_id": int(team_id), "elo_rating": float(rating)})

    return pd.DataFrame(rows)


def compute_massey_features() -> pd.DataFrame:
    massey = load_massey_ordinals()
    latest = massey.groupby("Season")["RankingDayNum"].transform("max")
    latest = massey[massey["RankingDayNum"] == latest].copy()
    features = (
        latest.groupby(["Season", "TeamID"], sort=True)
        .agg(
            massey_mean_rank=("OrdinalRank", "mean"),
            massey_median_rank=("OrdinalRank", "median"),
            massey_best_rank=("OrdinalRank", "min"),
            massey_worst_rank=("OrdinalRank", "max"),
            massey_rank_std=("OrdinalRank", "std"),
            massey_system_count=("SystemName", "nunique"),
        )
        .reset_index()
        .rename(columns={"Season": "season", "TeamID": "team_id"})
    )
    return features


def build_team_features(division: str) -> pd.DataFrame:
    long_games = build_regular_season_long(division)

    grouped = long_games.groupby(["season", "team_id"], sort=True)
    aggregates = grouped.agg(
        games=("is_win", "size"),
        wins=("is_win", "sum"),
        avg_points_for=("team_score", "mean"),
        avg_points_against=("opp_score", "mean"),
        avg_margin=("score_margin", "mean"),
        avg_num_ot=("num_ot", "mean"),
        total_fgm=("fgm", "sum"),
        total_fga=("fga", "sum"),
        total_fgm3=("fgm3", "sum"),
        total_fga3=("fga3", "sum"),
        total_ftm=("ftm", "sum"),
        total_fta=("fta", "sum"),
        total_or=("or", "sum"),
        total_dr=("dr", "sum"),
        total_opp_or=("opp_or", "sum"),
        total_ast=("ast", "sum"),
        total_to=("to", "sum"),
        total_stl=("stl", "sum"),
        total_blk=("blk", "sum"),
        total_pf=("pf", "sum"),
        total_opp_fgm=("opp_fgm", "sum"),
        total_opp_fga=("opp_fga", "sum"),
        total_opp_dr=("opp_dr", "sum"),
    )
    aggregates["losses"] = aggregates["games"] - aggregates["wins"]
    aggregates["win_pct"] = _safe_divide(aggregates["wins"], aggregates["games"])
    aggregates["fg_pct"] = _safe_divide(aggregates["total_fgm"], aggregates["total_fga"])
    aggregates["fg3_pct"] = _safe_divide(aggregates["total_fgm3"], aggregates["total_fga3"])
    aggregates["ft_pct"] = _safe_divide(aggregates["total_ftm"], aggregates["total_fta"])
    aggregates["efg_pct"] = _safe_divide(aggregates["total_fgm"] + 0.5 * aggregates["total_fgm3"], aggregates["total_fga"])
    aggregates["opp_efg_pct"] = _safe_divide(
        aggregates["total_opp_fgm"] + 0.5 * grouped["opp_fgm3"].sum(),
        aggregates["total_opp_fga"],
    )
    aggregates["ft_rate"] = _safe_divide(aggregates["total_fta"], aggregates["total_fga"])
    aggregates["fg3_rate"] = _safe_divide(aggregates["total_fga3"], aggregates["total_fga"])
    aggregates["turnover_rate"] = _safe_divide(aggregates["total_to"], aggregates["total_fga"] + 0.44 * aggregates["total_fta"])
    aggregates["assist_rate"] = _safe_divide(aggregates["total_ast"], aggregates["total_fgm"])
    aggregates["ast_to_ratio"] = _safe_divide(aggregates["total_ast"], aggregates["total_to"])
    aggregates["orb_per_game"] = _safe_divide(aggregates["total_or"], aggregates["games"])
    aggregates["drb_per_game"] = _safe_divide(aggregates["total_dr"], aggregates["games"])
    aggregates["orb_pct"] = _safe_divide(aggregates["total_or"], aggregates["total_or"] + aggregates["total_opp_dr"])
    aggregates["drb_pct"] = _safe_divide(aggregates["total_dr"], aggregates["total_dr"] + aggregates["total_opp_or"])
    aggregates["stl_per_game"] = _safe_divide(aggregates["total_stl"], aggregates["games"])
    aggregates["blk_per_game"] = _safe_divide(aggregates["total_blk"], aggregates["games"])
    aggregates["pf_per_game"] = _safe_divide(aggregates["total_pf"], aggregates["games"])
    team_possessions = aggregates["total_fga"] - aggregates["total_or"] + aggregates["total_to"] + 0.475 * aggregates["total_fta"]
    opp_possessions = aggregates["total_opp_fga"] - aggregates["total_opp_or"] + grouped["opp_to"].sum() + 0.475 * grouped["opp_fta"].sum()
    aggregates["avg_possessions"] = _safe_divide(team_possessions + opp_possessions, 2.0 * aggregates["games"])
    aggregates["off_rating"] = _safe_divide(100.0 * grouped["team_score"].sum(), team_possessions)
    aggregates["def_rating"] = _safe_divide(100.0 * grouped["opp_score"].sum(), opp_possessions)
    aggregates["net_rating"] = aggregates["off_rating"] - aggregates["def_rating"]

    recent_games = long_games.sort_values(["season", "team_id", "day_num"]).groupby(["season", "team_id"], group_keys=False).tail(10)
    recent = recent_games.groupby(["season", "team_id"], sort=True).agg(
        last10_games=("is_win", "size"),
        last10_wins=("is_win", "sum"),
        last10_avg_margin=("score_margin", "mean"),
        last10_avg_points_for=("team_score", "mean"),
        last10_avg_points_against=("opp_score", "mean"),
    )
    recent["last10_win_pct"] = _safe_divide(recent["last10_wins"], recent["last10_games"])

    recent5_games = long_games.sort_values(["season", "team_id", "day_num"]).groupby(["season", "team_id"], group_keys=False).tail(5)
    recent5 = recent5_games.groupby(["season", "team_id"], sort=True).agg(
        last5_games=("is_win", "size"),
        last5_wins=("is_win", "sum"),
        last5_avg_margin=("score_margin", "mean"),
    )
    recent5["last5_win_pct"] = _safe_divide(recent5["last5_wins"], recent5["last5_games"])

    feature_frame = aggregates.merge(recent, on=["season", "team_id"], how="left").merge(recent5, on=["season", "team_id"], how="left")

    opponent_quality = feature_frame.reset_index()[["season", "team_id", "win_pct", "avg_margin"]].rename(
        columns={"team_id": "opp_team_id", "win_pct": "opp_win_pct_feature", "avg_margin": "opp_margin_feature"}
    )
    schedule_strength = (
        long_games.merge(opponent_quality, on=["season", "opp_team_id"], how="left")
        .groupby(["season", "team_id"], sort=True)
        .agg(
            opp_avg_win_pct=("opp_win_pct_feature", "mean"),
            opp_avg_margin=("opp_margin_feature", "mean"),
        )
    )

    elo = compute_elo_ratings(division)
    teams = load_teams(division)

    feature_frame = (
        feature_frame
        .merge(schedule_strength, on=["season", "team_id"], how="left")
        .merge(elo, on=["season", "team_id"], how="left")
        .reset_index()
        .merge(teams[["team_id", "team_name"]], on="team_id", how="left")
    )
    if division == "M":
        feature_frame = feature_frame.merge(compute_massey_features(), on=["season", "team_id"], how="left")
    else:
        for column in [
            "massey_mean_rank",
            "massey_median_rank",
            "massey_best_rank",
            "massey_worst_rank",
            "massey_rank_std",
            "massey_system_count",
        ]:
            feature_frame[column] = 0.0
    feature_frame["division"] = division

    ordered_columns = ["division", "season", "team_id", "team_name", *BASE_TEAM_FEATURE_COLUMNS]
    return feature_frame[ordered_columns].sort_values(["season", "team_id"]).reset_index(drop=True)

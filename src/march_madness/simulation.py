from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from march_madness.config import ARTIFACTS_DIR
from march_madness.data.external import build_espn_bracket_games
from march_madness.inference.predict import predict_single_matchup


ROUND_NAMES = {
    "first_four": "First Four",
    "round_of_64": "Round of 64",
    "round_of_32": "Round of 32",
    "sweet_16": "Sweet 16",
    "elite_8": "Elite 8",
    "final_four": "Final Four",
    "championship": "Championship",
    "title": "Champion",
}


def bracket_simulation_path(division: str) -> Path:
    return ARTIFACTS_DIR / "metrics" / f"{division.lower()}_bracket_simulation.csv"


def bracket_games_path(division: str) -> Path:
    return ARTIFACTS_DIR / "external" / f"{division.lower()}_espn_bracket_games.csv"


def save_bracket_games(division: str, games: pd.DataFrame) -> None:
    games.to_csv(bracket_games_path(division), index=False)


def load_bracket_games(division: str) -> pd.DataFrame:
    return pd.read_csv(bracket_games_path(division))


def save_bracket_simulation(division: str, table: pd.DataFrame) -> None:
    table.to_csv(bracket_simulation_path(division), index=False)


def load_bracket_simulation(division: str) -> pd.DataFrame:
    return pd.read_csv(bracket_simulation_path(division))


def _official_round_probability(row: pd.Series) -> float | None:
    if pd.isna(row.get("market_probability_a")):
        return None
    return float(row["market_probability_a"])


def run_bracket_simulation(
    division: str,
    model_bundle: dict[str, Any],
    team_features: pd.DataFrame,
    n_simulations: int = 10000,
    market_weight: float = 0.2,
    refresh_external: bool = False,
) -> pd.DataFrame:
    games = build_espn_bracket_games(division, refresh=refresh_external)
    save_bracket_games(division, games)

    first_four = games[games["round_id"] == 0].copy()
    round_one = games[games["round_id"] == 1].copy()
    round_one["region_index"] = ((round_one["bracket_location"] - 1) // 8).astype(int)
    first_four_lookup: dict[tuple[str, int], list[int]] = {}
    for row in first_four.itertuples(index=False):
        seed_value = int(row.team_one_seed)
        participants = [int(row.team_one_id), int(row.team_two_id)]
        first_four_lookup[(row.region_label, seed_value)] = participants

    region_labels = round_one.groupby("region_index")["region_label"].first().to_dict()

    all_team_ids = sorted(
        {
            int(team_id)
            for column in ("team_one_id", "team_two_id")
            for team_id in games[column].fillna(0).astype(int).tolist()
            if int(team_id) > 0
        }
    )
    advancement_counts = {team_id: {key: 0 for key in ROUND_NAMES} for team_id in all_team_ids}

    @lru_cache(maxsize=None)
    def pair_probability(team_one_id: int, team_two_id: int) -> float:
        probability, matchup_frame = predict_single_matchup(
            model_bundle,
            team_features,
            2026,
            team_one_id,
            team_two_id,
            division,
        )
        market_probability = _official_round_probability(matchup_frame.iloc[0])
        if market_probability is not None:
            probability = (1.0 - market_weight) * probability + market_weight * market_probability
        return float(probability)

    rng = np.random.default_rng(42)

    def simulate_game(team_one_id: int, team_two_id: int) -> int:
        probability = pair_probability(team_one_id, team_two_id)
        return team_one_id if rng.random() < probability else team_two_id

    def resolve_first_four(region_label: str, seed_num: int) -> int:
        teams = first_four_lookup[(region_label, seed_num)]
        for team_id in teams:
            advancement_counts[team_id]["first_four"] += 1
        return simulate_game(teams[0], teams[1])

    for _ in range(n_simulations):
        region_winners: list[int] = []
        for region_index in sorted(region_labels):
            region_label = region_labels[region_index]
            region_games = round_one[round_one["region_index"] == region_index].sort_values("bracket_location")
            round64_winners: list[int] = []
            for row in region_games.itertuples(index=False):
                team_one_id = int(row.team_one_id)
                team_two_id = int(row.team_two_id)
                if team_one_id == 0:
                    team_one_id = resolve_first_four(region_label, int(row.team_one_seed))
                if team_two_id == 0:
                    team_two_id = resolve_first_four(region_label, int(row.team_two_seed))
                advancement_counts[team_one_id]["round_of_64"] += 1
                advancement_counts[team_two_id]["round_of_64"] += 1
                round64_winners.append(simulate_game(team_one_id, team_two_id))

            round32_winners: list[int] = []
            for index in range(0, len(round64_winners), 2):
                team_one_id = round64_winners[index]
                team_two_id = round64_winners[index + 1]
                advancement_counts[team_one_id]["round_of_32"] += 1
                advancement_counts[team_two_id]["round_of_32"] += 1
                round32_winners.append(simulate_game(team_one_id, team_two_id))

            sweet16_winners: list[int] = []
            for index in range(0, len(round32_winners), 2):
                team_one_id = round32_winners[index]
                team_two_id = round32_winners[index + 1]
                advancement_counts[team_one_id]["sweet_16"] += 1
                advancement_counts[team_two_id]["sweet_16"] += 1
                sweet16_winners.append(simulate_game(team_one_id, team_two_id))

            team_one_id = sweet16_winners[0]
            team_two_id = sweet16_winners[1]
            advancement_counts[team_one_id]["elite_8"] += 1
            advancement_counts[team_two_id]["elite_8"] += 1
            region_winners.append(simulate_game(team_one_id, team_two_id))

        semifinal_one = region_winners[0:2]
        semifinal_two = region_winners[2:4]
        for team_id in semifinal_one:
            advancement_counts[team_id]["final_four"] += 1
        for team_id in semifinal_two:
            advancement_counts[team_id]["final_four"] += 1
        finalist_one = simulate_game(semifinal_one[0], semifinal_one[1])
        finalist_two = simulate_game(semifinal_two[0], semifinal_two[1])
        advancement_counts[finalist_one]["championship"] += 1
        advancement_counts[finalist_two]["championship"] += 1
        champion = simulate_game(finalist_one, finalist_two)
        advancement_counts[champion]["title"] += 1

    team_lookup = team_features[team_features["season"] == 2026][["team_id", "team_name"]].drop_duplicates()
    rows: list[dict[str, object]] = []
    for team_id, counts in advancement_counts.items():
        team_name = team_lookup.loc[team_lookup["team_id"] == team_id, "team_name"].iloc[0]
        row = {"team_id": team_id, "team_name": team_name}
        for key in ROUND_NAMES:
            row[key] = counts[key] / float(n_simulations)
        rows.append(row)

    table = pd.DataFrame(rows).sort_values(["title", "championship", "final_four"], ascending=False).reset_index(drop=True)
    save_bracket_simulation(division, table)
    return table

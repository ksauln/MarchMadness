from __future__ import annotations

import pandas as pd

from march_madness.config import CURRENT_SEASON
from march_madness.data import external


def test_load_seed_table_handles_womens_city_only_region_labels(monkeypatch) -> None:
    monkeypatch.setattr(
        external,
        "load_tournament_seeds",
        lambda division: pd.DataFrame(columns=["Season", "TeamID", "Seed"]),
    )
    monkeypatch.setattr(
        external,
        "build_espn_bracket_games",
        lambda division, refresh=False: pd.DataFrame(
            [
                {
                    "season": CURRENT_SEASON,
                    "region_label": "Fort Worth",
                    "team_one_id": 3101,
                    "team_one_name": "Alpha",
                    "team_one_seed": 1,
                    "team_two_id": 3102,
                    "team_two_name": "Beta",
                    "team_two_seed": 16,
                },
                {
                    "season": CURRENT_SEASON,
                    "region_label": "Regional 3 - Fort Worth",
                    "team_one_id": 3103,
                    "team_one_name": "Gamma",
                    "team_one_seed": 2,
                    "team_two_id": 3104,
                    "team_two_name": "Delta",
                    "team_two_seed": 15,
                },
                {
                    "season": CURRENT_SEASON,
                    "region_label": "Sacramento",
                    "team_one_id": 3105,
                    "team_one_name": "Epsilon",
                    "team_one_seed": 1,
                    "team_two_id": 3106,
                    "team_two_name": "Zeta",
                    "team_two_seed": 16,
                },
            ]
        ),
    )
    monkeypatch.setattr(external, "_payload_path", lambda division: external.ARTIFACTS_DIR / "external" / "stub.json")

    seeds = external.load_seed_table("W", refresh=True)

    assert seeds.loc[seeds["team_id"] == 3101, "seed_code"].item() == "W01"
    assert seeds.loc[seeds["team_id"] == 3103, "seed_code"].item() == "Z02"
    assert seeds.loc[seeds["team_id"] == 3105, "seed_code"].item() == "X01"


def test_load_seed_table_prefers_saved_artifact(tmp_path, monkeypatch) -> None:
    artifact_path = tmp_path / "m_seed_table.csv"
    pd.DataFrame(
        [
            {
                "season": 2026,
                "team_id": 1101,
                "seed_code": "W01",
                "division": "M",
                "seed_num": 1,
                "seed_missing": 0,
            }
        ]
    ).to_csv(artifact_path, index=False)
    monkeypatch.setattr(external, "_seed_table_path", lambda division: artifact_path)

    seeds = external.load_seed_table("M", refresh=False)

    assert seeds.loc[0, "team_id"] == 1101
    assert seeds.loc[0, "seed_code"] == "W01"


def test_build_market_lines_table_prefers_saved_artifact(tmp_path, monkeypatch) -> None:
    artifact_path = tmp_path / "m_market_lines.csv"
    pd.DataFrame(
        [
            {
                "season": 2026,
                "division": "M",
                "team_a": 1101,
                "team_b": 1102,
                "market_probability_a": 0.61,
                "official_game_flag": 1,
                "odds": "TEAM -4.5",
                "round_id": 1,
                "region_label": "South",
            }
        ]
    ).to_csv(artifact_path, index=False)
    monkeypatch.setattr(external, "_market_lines_path", lambda division: artifact_path)

    lines = external.build_market_lines_table("M", refresh=False)

    assert lines.loc[0, "team_a"] == 1101
    assert float(lines.loc[0, "market_probability_a"]) == 0.61

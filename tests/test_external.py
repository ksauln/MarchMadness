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

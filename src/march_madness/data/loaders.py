from __future__ import annotations

from pathlib import Path

import pandas as pd

from march_madness.config import DIVISIONS, TEAM_ID_DIVISION_CUTOFF, get_data_dir


def _validate_division(division: str) -> str:
    division = division.upper()
    if division not in DIVISIONS:
        raise ValueError(f"Unsupported division: {division}")
    return division


def _csv_path(filename: str) -> Path:
    return get_data_dir() / filename


def load_regular_season_results(division: str) -> pd.DataFrame:
    division = _validate_division(division)
    return pd.read_csv(_csv_path(f"{division}RegularSeasonDetailedResults.csv"))


def load_tournament_results(division: str) -> pd.DataFrame:
    division = _validate_division(division)
    return pd.read_csv(_csv_path(f"{division}NCAATourneyCompactResults.csv"))


def load_teams(division: str) -> pd.DataFrame:
    division = _validate_division(division)
    frame = pd.read_csv(_csv_path(f"{division}Teams.csv"))
    columns = {"TeamID": "team_id", "TeamName": "team_name"}
    return frame.rename(columns=columns)


def load_team_spellings(division: str) -> pd.DataFrame:
    division = _validate_division(division)
    return pd.read_csv(_csv_path(f"{division}TeamSpellings.csv"))


def load_tournament_seeds(division: str) -> pd.DataFrame:
    division = _validate_division(division)
    return pd.read_csv(_csv_path(f"{division}NCAATourneySeeds.csv"))


def load_sample_submission(stage: int = 2) -> pd.DataFrame:
    if stage not in (1, 2):
        raise ValueError("stage must be 1 or 2")
    return pd.read_csv(_csv_path(f"SampleSubmissionStage{stage}.csv"))


def load_massey_ordinals() -> pd.DataFrame:
    return pd.read_csv(_csv_path("MMasseyOrdinals.csv"))


def infer_division_from_team_id(team_id: int) -> str:
    return "M" if int(team_id) < TEAM_ID_DIVISION_CUTOFF else "W"

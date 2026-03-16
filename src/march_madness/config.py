from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR_CANDIDATES = (REPO_ROOT / "data", REPO_ROOT / "Data")
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
CURRENT_SEASON = 2026
DIVISIONS = ("M", "W")
DIVISION_LABELS = {"M": "Men", "W": "Women"}
TEAM_ID_DIVISION_CUTOFF = 3000


def get_data_dir() -> Path:
    for candidate in DATA_DIR_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find a data directory. Expected ./data or ./Data.")


def ensure_artifact_dirs() -> None:
    for relative in (
        "models",
        "metrics",
        "submissions",
        "features",
        "external",
    ):
        (ARTIFACTS_DIR / relative).mkdir(parents=True, exist_ok=True)

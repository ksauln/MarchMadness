from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from march_madness.config import DIVISIONS, ensure_artifact_dirs
from march_madness.inference.submission import generate_division_submission, load_feature_table, load_model_bundle, save_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Kaggle submission from saved model artifacts.")
    parser.add_argument("--stage", type=int, default=2, choices=(1, 2))
    parser.add_argument("--refresh-external", action="store_true")
    args = parser.parse_args()

    ensure_artifact_dirs()
    submissions: list[pd.DataFrame] = []
    for division in DIVISIONS:
        bundle = load_model_bundle(division)
        team_features = load_feature_table(division)
        submissions.append(generate_division_submission(division, bundle, team_features, stage=args.stage, refresh_external=args.refresh_external))

    output_path = save_submission(args.stage, pd.concat(submissions, ignore_index=True))
    print(f"Saved submission to {output_path}")


if __name__ == "__main__":
    main()

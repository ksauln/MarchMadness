from __future__ import annotations

import pandas as pd

from march_madness.features.matchup_builder import parse_submission_ids


def test_parse_submission_ids_extracts_division_and_teams() -> None:
    frame = pd.DataFrame({"ID": ["2026_1101_1102", "2026_3101_3102"], "Pred": [0.5, 0.5]})
    parsed = parse_submission_ids(frame)
    assert parsed["division"].tolist() == ["M", "W"]
    assert parsed["team_a"].tolist() == [1101, 3101]
    assert parsed["team_b"].tolist() == [1102, 3102]

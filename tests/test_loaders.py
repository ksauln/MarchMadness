from __future__ import annotations

from march_madness.data.loaders import load_regular_season_results, load_sample_submission, load_teams


def test_load_teams_contains_expected_columns() -> None:
    men = load_teams("M")
    women = load_teams("W")
    assert {"team_id", "team_name"} <= set(men.columns)
    assert {"team_id", "team_name"} <= set(women.columns)
    assert len(men) >= 300
    assert len(women) >= 300


def test_regular_season_and_submission_have_current_data() -> None:
    regular = load_regular_season_results("M")
    submission = load_sample_submission(stage=2)
    assert int(regular["Season"].max()) >= 2026
    assert submission["ID"].str.startswith("2026_").all()

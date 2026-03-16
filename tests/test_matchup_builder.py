from __future__ import annotations

from march_madness.features.matchup_builder import build_tournament_training_frame, matchup_feature_columns
from march_madness.features.team_aggregates import build_team_features


def test_training_frame_builds_for_women() -> None:
    team_features = build_team_features("W")
    training = build_tournament_training_frame("W", team_features)
    assert not training.empty
    assert set(training["target"].unique()).issubset({0, 1})
    for column in matchup_feature_columns()[:5]:
        assert column in training.columns

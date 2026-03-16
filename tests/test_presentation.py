from __future__ import annotations

from march_madness.ui.presentation import build_matchup_pick, build_upset_signal


def test_build_upset_signal_flags_model_upset_pick() -> None:
    signal = build_upset_signal(
        team_one_name="Favorite",
        team_two_name="Underdog",
        team_one_seed=3,
        team_two_seed=11,
        team_one_probability=0.46,
        team_two_probability=0.54,
    )

    assert signal["flagged"] is True
    assert signal["level"] == "Model upset pick"
    assert signal["team"] == "Underdog"
    assert signal["seed_gap"] == 8


def test_build_upset_signal_ignores_close_or_even_seed_games() -> None:
    signal = build_upset_signal(
        team_one_name="Team A",
        team_two_name="Team B",
        team_one_seed=8,
        team_two_seed=9,
        team_one_probability=0.44,
        team_two_probability=0.56,
    )

    assert signal["flagged"] is False
    assert signal["summary"] is None


def test_build_matchup_pick_can_take_a_live_upset_shot() -> None:
    upset = build_upset_signal(
        team_one_name="Favorite",
        team_two_name="Underdog",
        team_one_seed=4,
        team_two_seed=10,
        team_one_probability=0.56,
        team_two_probability=0.44,
    )

    pick = build_matchup_pick(
        team_one_name="Favorite",
        team_two_name="Underdog",
        team_one_probability=0.56,
        team_two_probability=0.44,
        upset_signal=upset,
        team_one_market_probability=0.59,
        team_two_market_probability=0.41,
    )

    assert pick["team"] == "Underdog"
    assert pick["strategy"] == "Upset pick"


def test_build_matchup_pick_defaults_to_higher_probability_team() -> None:
    pick = build_matchup_pick(
        team_one_name="Team A",
        team_two_name="Team B",
        team_one_probability=0.61,
        team_two_probability=0.39,
        upset_signal={"flagged": False},
    )

    assert pick["team"] == "Team A"
    assert pick["strategy"] == "Chalk pick"

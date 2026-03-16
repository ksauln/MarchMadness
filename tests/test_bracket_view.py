from __future__ import annotations

from march_madness.ui.bracket import BRACKET_VARIANTS, GamePick, TeamPick, render_bracket_svg


def _game(team_one: TeamPick, team_two: TeamPick, winner: TeamPick, *, note: str) -> GamePick:
    return GamePick(
        round_id=1,
        region_label="EAST",
        bracket_location=1,
        team_one=team_one,
        team_two=team_two,
        winner=winner,
        winner_probability=0.62,
        note=note,
    )


def test_bracket_variants_expose_four_options() -> None:
    assert set(BRACKET_VARIANTS) == {"model_only", "likely_upsets", "market_consensus", "title_equity"}


def test_render_bracket_svg_includes_variant_and_champion() -> None:
    alpha = TeamPick(team_id=1, team_name="Alpha", seed=1)
    beta = TeamPick(team_id=2, team_name="Beta", seed=16)
    game = _game(alpha, beta, alpha, note="Model lean 62.0%")
    variant = {
        "label": "Model Only",
        "description": "Take the higher blended matchup win probability in every game.",
        "first_four": [],
        "regions": {
            "EAST": {"round_one": [game] * 8, "round_two": [game] * 4, "sweet_16": [game] * 2, "elite_8": [game]},
            "SOUTH": {"round_one": [game] * 8, "round_two": [game] * 4, "sweet_16": [game] * 2, "elite_8": [game]},
            "WEST": {"round_one": [game] * 8, "round_two": [game] * 4, "sweet_16": [game] * 2, "elite_8": [game]},
            "MIDWEST": {"round_one": [game] * 8, "round_two": [game] * 4, "sweet_16": [game] * 2, "elite_8": [game]},
        },
        "semifinals": [game, game],
        "championship": game,
        "champion": alpha,
    }

    svg = render_bracket_svg("Men", variant)

    assert 'aria-label="Men bracket"' in svg
    assert "First" in svg
    assert "Round" in svg
    assert "Alpha" in svg
    assert "Champion" in svg

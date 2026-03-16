from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from html import escape
from typing import Any

import pandas as pd

from march_madness.inference.predict import predict_single_matchup
from march_madness.ui.presentation import build_matchup_pick, build_upset_signal


BRACKET_VARIANTS: dict[str, dict[str, str]] = {
    "model_only": {
        "label": "Model Only",
        "description": "Take the higher blended matchup win probability in every game.",
    },
    "likely_upsets": {
        "label": "Likely Upsets",
        "description": "Use the existing upset-pick heuristic to chase credible bracket breakers while staying mostly chalk.",
    },
    "market_consensus": {
        "label": "Market Consensus",
        "description": "Lean on official market prices when they exist, then fall back to the model deeper in the bracket.",
    },
    "title_equity": {
        "label": "Title Equity",
        "description": "Break close games with title and championship odds so the bracket favors teams with stronger deep-run equity.",
    },
}

_REGION_DISPLAY = {
    "EAST": "East",
    "SOUTH": "South",
    "WEST": "West",
    "MIDWEST": "Midwest",
}
_ROUND_HEADERS = (
    ("First", "Round"),
    ("Second", "Round"),
    ("Sweet", "16"),
    ("Elite", "8"),
    ("Final", "Four"),
    ("National", "Championship"),
    ("Final", "Four"),
    ("Elite", "8"),
    ("Sweet", "16"),
    ("Second", "Round"),
    ("First", "Round"),
)


@dataclass(frozen=True)
class TeamPick:
    team_id: int
    team_name: str
    seed: int | None


@dataclass(frozen=True)
class GamePick:
    round_id: int
    region_label: str
    bracket_location: int
    team_one: TeamPick
    team_two: TeamPick
    winner: TeamPick
    winner_probability: float
    note: str


def build_bracket_variant(
    division: str,
    games: pd.DataFrame,
    team_features: pd.DataFrame,
    model_bundle: dict[str, Any],
    simulation: pd.DataFrame,
    variant_key: str,
) -> dict[str, Any]:
    if variant_key not in BRACKET_VARIANTS:
        raise ValueError(f"Unknown bracket variant: {variant_key}")

    season = 2026
    simulation_lookup = (
        simulation[["team_id", "title", "championship", "final_four"]].drop_duplicates().set_index("team_id").to_dict("index")
        if not simulation.empty
        else {}
    )

    @lru_cache(maxsize=None)
    def matchup_context(team_one_id: int, team_two_id: int) -> tuple[float, float | None]:
        probability, matchup_frame = predict_single_matchup(
            model_bundle,
            team_features,
            season,
            team_one_id,
            team_two_id,
            division,
        )
        market_probability = matchup_frame.iloc[0].get("market_probability_a")
        market_probability_value = float(market_probability) if pd.notna(market_probability) else None
        if market_probability_value is not None:
            probability = (1.0 - float(model_bundle.get("market_weight", 0.0))) * float(probability) + float(
                model_bundle.get("market_weight", 0.0)
            ) * market_probability_value
        return float(probability), market_probability_value

    def decide_winner(team_one: TeamPick, team_two: TeamPick) -> tuple[TeamPick, float, str]:
        team_one_probability, market_probability = matchup_context(team_one.team_id, team_two.team_id)
        team_two_probability = 1.0 - team_one_probability
        model_winner = team_one if team_one_probability >= team_two_probability else team_two
        model_probability = max(team_one_probability, team_two_probability)

        if variant_key == "model_only":
            return model_winner, model_probability, f"Model lean {model_probability:.1%}"

        upset_signal = build_upset_signal(
            team_one_name=team_one.team_name,
            team_two_name=team_two.team_name,
            team_one_seed=team_one.seed,
            team_two_seed=team_two.seed,
            team_one_probability=team_one_probability,
            team_two_probability=team_two_probability,
        )

        if variant_key == "likely_upsets":
            pick = build_matchup_pick(
                team_one_name=team_one.team_name,
                team_two_name=team_two.team_name,
                team_one_probability=team_one_probability,
                team_two_probability=team_two_probability,
                upset_signal=upset_signal,
                team_one_market_probability=market_probability,
                team_two_market_probability=None if market_probability is None else 1.0 - market_probability,
            )
            winner = team_one if pick["team"] == team_one.team_name else team_two
            probability = team_one_probability if winner == team_one else team_two_probability
            return winner, probability, str(pick["strategy"])

        if variant_key == "market_consensus" and market_probability is not None:
            winner = team_one if market_probability >= 0.5 else team_two
            probability = max(market_probability, 1.0 - market_probability)
            return winner, probability, f"Market lean {probability:.1%}"

        if variant_key == "title_equity":
            title_one = float(simulation_lookup.get(team_one.team_id, {}).get("title", 0.0))
            title_two = float(simulation_lookup.get(team_two.team_id, {}).get("title", 0.0))
            championship_one = float(simulation_lookup.get(team_one.team_id, {}).get("championship", 0.0))
            championship_two = float(simulation_lookup.get(team_two.team_id, {}).get("championship", 0.0))
            equity_one = title_one * 0.7 + championship_one * 0.3
            equity_two = title_two * 0.7 + championship_two * 0.3
            if abs(team_one_probability - 0.5) <= 0.08 and abs(equity_one - equity_two) >= 0.0075:
                if equity_one > equity_two and team_one_probability >= 0.42:
                    return team_one, team_one_probability, f"Title equity {equity_one:.1%}"
                if equity_two > equity_one and team_two_probability >= 0.42:
                    return team_two, team_two_probability, f"Title equity {equity_two:.1%}"

        return model_winner, model_probability, f"Model lean {model_probability:.1%}"

    def build_team(team_id: Any, team_name: Any, seed: Any) -> TeamPick:
        return TeamPick(
            team_id=int(team_id),
            team_name=str(team_name),
            seed=int(seed) if pd.notna(seed) else None,
        )

    games_frame = games.copy()
    first_four_games = games_frame[games_frame["round_id"] == 0].sort_values("bracket_location")
    round_one = games_frame[games_frame["round_id"] == 1].sort_values("bracket_location").copy()

    first_four_results: dict[tuple[str, int], TeamPick] = {}
    rendered_first_four: list[GamePick] = []
    for row in first_four_games.itertuples(index=False):
        team_one = build_team(row.team_one_id, row.team_one_name, row.team_one_seed)
        team_two = build_team(row.team_two_id, row.team_two_name, row.team_two_seed)
        winner, probability, note = decide_winner(team_one, team_two)
        game = GamePick(
            round_id=0,
            region_label=str(row.region_label),
            bracket_location=int(row.bracket_location),
            team_one=team_one,
            team_two=team_two,
            winner=winner,
            winner_probability=probability,
            note=note,
        )
        first_four_results[(str(row.region_label), int(row.team_one_seed))] = winner
        rendered_first_four.append(game)

    round_one["region_index"] = ((round_one["bracket_location"] - 1) // 8).astype(int)
    region_order = round_one.groupby("region_index")["region_label"].first().tolist()
    regions: dict[str, dict[str, list[GamePick]]] = {}
    region_champions: list[TeamPick] = []

    def play_round(teams: list[TeamPick], round_id: int, region_label: str) -> tuple[list[GamePick], list[TeamPick]]:
        round_games: list[GamePick] = []
        winners: list[TeamPick] = []
        for index in range(0, len(teams), 2):
            team_one = teams[index]
            team_two = teams[index + 1]
            winner, probability, note = decide_winner(team_one, team_two)
            round_games.append(
                GamePick(
                    round_id=round_id,
                    region_label=region_label,
                    bracket_location=(index // 2) + 1,
                    team_one=team_one,
                    team_two=team_two,
                    winner=winner,
                    winner_probability=probability,
                    note=note,
                )
            )
            winners.append(winner)
        return round_games, winners

    for region_label in region_order:
        region_games = round_one[round_one["region_label"] == region_label].sort_values("bracket_location")
        round_one_results: list[GamePick] = []
        round_one_winners: list[TeamPick] = []
        for row in region_games.itertuples(index=False):
            team_one = build_team(row.team_one_id, row.team_one_name, row.team_one_seed)
            team_two = build_team(row.team_two_id, row.team_two_name, row.team_two_seed)
            if team_one.team_id == 0:
                team_one = first_four_results[(str(row.region_label), int(row.team_one_seed))]
            if team_two.team_id == 0:
                team_two = first_four_results[(str(row.region_label), int(row.team_two_seed))]
            winner, probability, note = decide_winner(team_one, team_two)
            round_one_results.append(
                GamePick(
                    round_id=1,
                    region_label=str(region_label),
                    bracket_location=int(row.bracket_location),
                    team_one=team_one,
                    team_two=team_two,
                    winner=winner,
                    winner_probability=probability,
                    note=note,
                )
            )
            round_one_winners.append(winner)

        round_two_results, round_two_winners = play_round(round_one_winners, 2, str(region_label))
        round_three_results, round_three_winners = play_round(round_two_winners, 3, str(region_label))
        round_four_results, round_four_winners = play_round(round_three_winners, 4, str(region_label))
        regions[str(region_label)] = {
            "round_one": round_one_results,
            "round_two": round_two_results,
            "sweet_16": round_three_results,
            "elite_8": round_four_results,
        }
        region_champions.append(round_four_winners[0])

    semifinal_left_winner, semifinal_left_probability, semifinal_left_note = decide_winner(region_champions[0], region_champions[1])
    semifinal_right_winner, semifinal_right_probability, semifinal_right_note = decide_winner(region_champions[2], region_champions[3])
    semifinal_left = GamePick(
        round_id=5,
        region_label="Final Four Left",
        bracket_location=1,
        team_one=region_champions[0],
        team_two=region_champions[1],
        winner=semifinal_left_winner,
        winner_probability=semifinal_left_probability,
        note=semifinal_left_note,
    )
    semifinal_right = GamePick(
        round_id=5,
        region_label="Final Four Right",
        bracket_location=2,
        team_one=region_champions[2],
        team_two=region_champions[3],
        winner=semifinal_right_winner,
        winner_probability=semifinal_right_probability,
        note=semifinal_right_note,
    )
    champion, championship_probability, championship_note = decide_winner(semifinal_left_winner, semifinal_right_winner)
    championship = GamePick(
        round_id=6,
        region_label="Championship",
        bracket_location=1,
        team_one=semifinal_left_winner,
        team_two=semifinal_right_winner,
        winner=champion,
        winner_probability=championship_probability,
        note=championship_note,
    )

    return {
        "key": variant_key,
        "label": BRACKET_VARIANTS[variant_key]["label"],
        "description": BRACKET_VARIANTS[variant_key]["description"],
        "first_four": rendered_first_four,
        "regions": regions,
        "region_order": region_order,
        "semifinals": [semifinal_left, semifinal_right],
        "championship": championship,
        "champion": champion,
    }


def render_bracket_svg(division_label: str, variant: dict[str, Any]) -> str:
    region_order = [label for label in ("EAST", "WEST", "SOUTH", "MIDWEST") if label in variant["regions"]]
    html_parts = [
        f'<div class="bracket-view" aria-label="{escape(division_label)} bracket">',
        "<style>"
        ".bracket-view{padding:0.5rem 0 1rem;}"
        ".bracket-first-four{margin-bottom:1.25rem;}"
        ".bracket-section-title{font:700 0.8rem \"IBM Plex Sans\", sans-serif; text-transform:uppercase; letter-spacing:0.08em; color:#5f675f; margin-bottom:0.55rem;}"
        ".bracket-first-four-grid{display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:0.9rem 1rem;}"
        ".bracket-region-grid{display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1.2rem; align-items:start;}"
        ".bracket-region{background:rgba(255,252,246,0.92); border:1px solid rgba(24,33,27,0.12); border-radius:24px; padding:1rem; box-shadow:0 12px 40px rgba(24,33,27,0.06);}"
        ".bracket-region-head{display:flex; align-items:flex-end; justify-content:space-between; gap:1rem; margin-bottom:0.85rem;}"
        ".bracket-region-title{font:700 2rem \"DM Serif Display\", serif; color:#18211b; line-height:1;}"
        ".bracket-region-sub{font:600 0.82rem \"IBM Plex Sans\", sans-serif; color:#5f675f; text-transform:uppercase; letter-spacing:0.06em;}"
        ".bracket-rounds{display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:0.9rem;}"
        ".bracket-round-col{display:flex; flex-direction:column;}"
        ".bracket-round-label{font:700 0.78rem \"IBM Plex Sans\", sans-serif; color:#18211b; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.6rem; min-height:2rem;}"
        ".bracket-round-stack{display:flex; flex-direction:column; gap:0.75rem;}"
        ".round-gap-1 .bracket-game{margin-bottom:0;}"
        ".round-gap-2 .bracket-game{margin-bottom:1.1rem;}"
        ".round-gap-3 .bracket-game{margin-bottom:3.2rem;}"
        ".round-gap-4 .bracket-game{margin-bottom:8.6rem;}"
        ".bracket-game{border:1px solid #2b3931; border-radius:14px; background:#fffdf9; overflow:hidden;}"
        ".bracket-game-row{padding:0.38rem 0.65rem; font:700 0.96rem \"IBM Plex Sans\", sans-serif; color:#18211b; line-height:1.15; overflow-wrap:anywhere;}"
        ".bracket-game-row + .bracket-game-row{border-top:1px solid rgba(24,33,27,0.14);}"
        ".bracket-game-row.winner{background:#e7f4eb;}"
        ".bracket-game-row.loser{color:#637064; font-weight:600;}"
        ".bracket-game-note{padding:0.38rem 0.65rem 0.52rem; border-top:1px solid rgba(24,33,27,0.1); font:600 0.72rem \"IBM Plex Sans\", sans-serif; color:#5f675f; line-height:1.25;}"
        ".bracket-finals{margin-top:1.4rem; background:rgba(255,252,246,0.92); border:1px solid rgba(24,33,27,0.12); border-radius:24px; padding:1rem; box-shadow:0 12px 40px rgba(24,33,27,0.06);}"
        ".bracket-finals-grid{display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:1rem; align-items:start;}"
        ".bracket-finals-card{border:1px solid rgba(24,33,27,0.14); border-radius:18px; background:#fffdf9; padding:0.9rem; min-height:100%;}"
        ".bracket-finals-label{font:700 0.8rem \"IBM Plex Sans\", sans-serif; color:#5f675f; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.65rem;}"
        ".bracket-finals-winner{margin-top:0.8rem; padding-top:0.8rem; border-top:1px solid rgba(24,33,27,0.12); font:700 0.92rem \"IBM Plex Sans\", sans-serif; color:#18211b;}"
        ".bracket-champion{background:#fff1d6; border:2px solid #c45b32;}"
        ".bracket-champion-name{font:700 1.55rem \"DM Serif Display\", serif; color:#18211b; line-height:1.1;}"
        ".bracket-champion-note{margin-top:0.7rem; font:600 0.78rem \"IBM Plex Sans\", sans-serif; color:#5f675f; line-height:1.3;}"
        "@media (max-width: 1200px){.bracket-region-grid{grid-template-columns:1fr;}.bracket-finals-grid{grid-template-columns:1fr;}}"
        "@media (max-width: 900px){.bracket-rounds{grid-template-columns:repeat(2,minmax(0,1fr));}.bracket-first-four-grid{grid-template-columns:1fr;}}"
        "</style>",
    ]

    if variant["first_four"]:
        html_parts.append('<section class="bracket-first-four">')
        html_parts.append('<div class="bracket-section-title">First Four</div>')
        html_parts.append('<div class="bracket-first-four-grid">')
        for game in variant["first_four"]:
            html_parts.append(_game_card_html(game))
        html_parts.append("</div></section>")

    html_parts.append('<section class="bracket-region-grid">')
    for region_label in region_order:
        region = variant["regions"][region_label]
        html_parts.append('<article class="bracket-region">')
        html_parts.append(
            f'<div class="bracket-region-head"><div class="bracket-region-title">{escape(_REGION_DISPLAY.get(region_label, region_label.title()))}</div>'
            f'<div class="bracket-region-sub">Regional path</div></div>'
        )
        html_parts.append('<div class="bracket-rounds">')
        rounds = [
            ("First Round", region["round_one"], "round-gap-1"),
            ("Second Round", region["round_two"], "round-gap-2"),
            ("Sweet 16", region["sweet_16"], "round-gap-3"),
            ("Elite 8", region["elite_8"], "round-gap-4"),
        ]
        for label, games, gap_class in rounds:
            html_parts.append(f'<div class="bracket-round-col {gap_class}">')
            html_parts.append(f'<div class="bracket-round-label">{escape(label)}</div>')
            html_parts.append('<div class="bracket-round-stack">')
            for game in games:
                html_parts.append(_game_card_html(game))
            html_parts.append("</div></div>")
        html_parts.append("</div></article>")
    html_parts.append("</section>")

    semifinal_left, semifinal_right = variant["semifinals"]
    championship = variant["championship"]
    html_parts.append('<section class="bracket-finals">')
    html_parts.append('<div class="bracket-section-title">Final Four And Champion</div>')
    html_parts.append('<div class="bracket-finals-grid">')
    html_parts.append(
        '<div class="bracket-finals-card">'
        '<div class="bracket-finals-label">Semifinal 1</div>'
        f'{_game_card_html(semifinal_left)}'
        f'<div class="bracket-finals-winner">Winner: {escape(_team_label(semifinal_left.winner, max_length=40))}</div>'
        '</div>'
    )
    html_parts.append(
        '<div class="bracket-finals-card">'
        '<div class="bracket-finals-label">Semifinal 2</div>'
        f'{_game_card_html(semifinal_right)}'
        f'<div class="bracket-finals-winner">Winner: {escape(_team_label(semifinal_right.winner, max_length=40))}</div>'
        '</div>'
    )
    html_parts.append(
        '<div class="bracket-finals-card bracket-champion">'
        '<div class="bracket-finals-label">Championship</div>'
        f'{_game_card_html(championship)}'
        f'<div class="bracket-finals-winner">Champion</div>'
        f'<div class="bracket-champion-name">{escape(_team_label(variant["champion"], max_length=40))}</div>'
        f'<div class="bracket-champion-note">{escape(_display_note(championship) or championship.note)}</div>'
        '</div>'
    )
    html_parts.append("</div></section></div>")
    return "".join(html_parts)


def _team_label(team: TeamPick, max_length: int = 24) -> str:
    seed_prefix = f"({team.seed}) " if team.seed is not None else ""
    label = f"{seed_prefix}{team.team_name}"
    return label if len(label) <= max_length else f"{label[: max_length - 1].rstrip()}…"


def _display_note(game: GamePick) -> str | None:
    note = game.note.strip()
    if not note:
        return None
    if note.startswith("Model lean") and game.round_id < 4:
        return None
    return note


def _game_card_html(game: GamePick) -> str:
    note = _display_note(game)
    team_one_class = "winner" if game.winner.team_id == game.team_one.team_id else "loser"
    team_two_class = "winner" if game.winner.team_id == game.team_two.team_id else "loser"
    parts = [
        '<div class="bracket-game">',
        f'<div class="bracket-game-row {team_one_class}">{escape(_team_label(game.team_one, max_length=48))}</div>',
        f'<div class="bracket-game-row {team_two_class}">{escape(_team_label(game.team_two, max_length=48))}</div>',
    ]
    if note:
        parts.append(f'<div class="bracket-game-note">{escape(note)}</div>')
    parts.append("</div>")
    return "".join(parts)

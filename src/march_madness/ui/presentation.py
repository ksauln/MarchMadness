from __future__ import annotations

from typing import Any

import pandas as pd


def _seed_value(value: Any) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_upset_signal(
    team_one_name: str,
    team_two_name: str,
    team_one_seed: Any,
    team_two_seed: Any,
    team_one_probability: float,
    team_two_probability: float,
) -> dict[str, Any]:
    seed_one = _seed_value(team_one_seed)
    seed_two = _seed_value(team_two_seed)
    empty = {
        "flagged": False,
        "level": None,
        "team": None,
        "seed": None,
        "favorite_seed": None,
        "win_probability": None,
        "seed_gap": None,
        "summary": None,
    }
    if seed_one is None or seed_two is None or seed_one == seed_two:
        return empty

    if seed_one > seed_two:
        underdog_name = team_one_name
        underdog_seed = seed_one
        favorite_seed = seed_two
        underdog_probability = float(team_one_probability)
    else:
        underdog_name = team_two_name
        underdog_seed = seed_two
        favorite_seed = seed_one
        underdog_probability = float(team_two_probability)

    seed_gap = int(abs(underdog_seed - favorite_seed))
    level: str | None = None
    if seed_gap >= 2 and underdog_probability >= 0.50:
        level = "Model upset pick"
    elif seed_gap >= 3 and underdog_probability >= 0.42:
        level = "Strong upset watch"
    elif seed_gap >= 5 and underdog_probability >= 0.35:
        level = "Upset watch"
    elif seed_gap >= 8 and underdog_probability >= 0.28:
        level = "Long-shot upset flyer"

    if level is None:
        return empty

    return {
        "flagged": True,
        "level": level,
        "team": underdog_name,
        "seed": int(underdog_seed),
        "favorite_seed": int(favorite_seed),
        "win_probability": underdog_probability,
        "seed_gap": seed_gap,
        "summary": f"{level}: No. {int(underdog_seed)} {underdog_name} over No. {int(favorite_seed)} at {underdog_probability:.1%}",
    }


def build_matchup_pick(
    team_one_name: str,
    team_two_name: str,
    team_one_probability: float,
    team_two_probability: float,
    upset_signal: dict[str, Any],
    team_one_market_probability: float | None = None,
    team_two_market_probability: float | None = None,
) -> dict[str, Any]:
    favorite_name = team_one_name if float(team_one_probability) >= float(team_two_probability) else team_two_name
    favorite_probability = max(float(team_one_probability), float(team_two_probability))

    if upset_signal.get("flagged"):
        underdog_name = str(upset_signal["team"])
        underdog_probability = float(upset_signal["win_probability"])
        if underdog_name == team_one_name:
            underdog_market = team_one_market_probability
        else:
            underdog_market = team_two_market_probability

        if upset_signal["level"] == "Model upset pick":
            return {
                "team": underdog_name,
                "probability": underdog_probability,
                "strategy": "Upset pick",
                "reason": upset_signal["summary"],
            }
        if upset_signal["level"] == "Strong upset watch" and underdog_probability >= 0.44:
            if underdog_market is None or float(underdog_market) >= 0.40:
                return {
                    "team": underdog_name,
                    "probability": underdog_probability,
                    "strategy": "Upset pick",
                    "reason": f'Bracket upside over chalk: {upset_signal["summary"]}',
                }
        if upset_signal["level"] == "Upset watch" and underdog_probability >= 0.40:
            if underdog_market is None or float(underdog_market) >= 0.36:
                return {
                    "team": underdog_name,
                    "probability": underdog_probability,
                    "strategy": "Upset pick",
                    "reason": f'High-variance bracket stab: {upset_signal["summary"]}',
                }

    return {
        "team": favorite_name,
        "probability": favorite_probability,
        "strategy": "Chalk pick",
        "reason": f"Highest blended win probability at {favorite_probability:.1%}",
    }

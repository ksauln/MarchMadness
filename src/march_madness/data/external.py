from __future__ import annotations

import json
import re
from html import unescape
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from march_madness.config import ARTIFACTS_DIR, CURRENT_SEASON, get_data_dir
from march_madness.data.loaders import load_team_spellings, load_teams, load_tournament_seeds


ESPN_BRACKET_URLS = {
    "M": "https://www.espn.com/mens-college-basketball/bracket",
    "W": "https://www.espn.com/womens-college-basketball/bracket",
}

MANUAL_NAME_ALIASES = {
    "M": {
        "ca baptist": "cal baptist",
        "miami": "miami fl",
        "queens": "queens nc",
    },
    "W": {
        "ca baptist": "cal baptist",
    },
}

REGION_CODE_MAP = {
    "M": {"EAST": "W", "WEST": "X", "SOUTH": "Y", "MIDWEST": "Z"},
    "W": {
        "Regional 1 - Fort Worth": "W",
        "Regional 4 - Sacramento": "X",
        "Regional 2 - Sacramento": "Y",
        "Regional 3 - Fort Worth": "Z",
        "Fort Worth": "W",
        "Sacramento": "X",
    },
}


def _normalize_name(value: str) -> str:
    normalized = unescape(str(value)).lower()
    normalized = normalized.replace("&", "and")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def _build_team_name_map(division: str) -> dict[str, int]:
    teams = load_teams(division)
    spellings = load_team_spellings(division)
    mapping: dict[str, int] = {}
    for row in teams.itertuples(index=False):
        mapping[_normalize_name(row.team_name)] = int(row.team_id)
    for row in spellings.itertuples(index=False):
        mapping[_normalize_name(row.TeamNameSpelling)] = int(row.TeamID)
    for alias, canonical in MANUAL_NAME_ALIASES.get(division, {}).items():
        team_id = mapping.get(_normalize_name(canonical))
        if team_id is not None:
            mapping[_normalize_name(alias)] = team_id
    return mapping


def _extract_bracket_payload(html: str) -> dict:
    key = '"bracket":'
    index = html.find(key)
    if index == -1:
        raise ValueError("Could not locate bracket payload in ESPN page.")
    start = index + len(key)
    depth = 0
    in_string = False
    escape = False
    end = start
    for position, character in enumerate(html[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif character == "\\":
                escape = True
            elif character == '"':
                in_string = False
        else:
            if character == '"':
                in_string = True
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    end = position + 1
                    break
    return json.loads(html[start:end])


def _payload_path(division: str) -> Path:
    return ARTIFACTS_DIR / "external" / f"{division.lower()}_espn_bracket.json"


def load_espn_bracket_payload(division: str, refresh: bool = False) -> dict:
    path = _payload_path(division)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    if not refresh and not path.exists():
        raise FileNotFoundError(f"Cached ESPN bracket payload not found for {division}. Refresh first.")

    request = Request(ESPN_BRACKET_URLS[division], headers={"User-Agent": "Mozilla/5.0"})
    html = urlopen(request).read().decode("utf-8", "ignore")
    payload = _extract_bracket_payload(html)
    path.write_text(json.dumps(payload))
    return payload


def _matchup_region_label(payload: dict, matchup: dict) -> str:
    if matchup["roundId"] == 0:
        return matchup["label"]
    if matchup["roundId"] == 1:
        region_order = payload["regions"]
        region_index = (int(matchup["bracketLocation"]) - 1) // 8
        return region_order[region_index]["labelPrimary"]
    if matchup["roundId"] in (2, 3, 4):
        region_order = payload["regions"]
        slots_per_region = {2: 4, 3: 2, 4: 1}[int(matchup["roundId"])]
        region_index = (int(matchup["bracketLocation"]) - 1) // slots_per_region
        return region_order[region_index]["labelPrimary"]
    if matchup["roundId"] == 5:
        return "Final Four Left" if int(matchup["bracketLocation"]) == 1 else "Final Four Right"
    return "Championship"


def build_espn_bracket_games(division: str, refresh: bool = False) -> pd.DataFrame:
    payload = load_espn_bracket_payload(division, refresh=refresh)
    name_map = _build_team_name_map(division)
    rows: list[dict[str, object]] = []

    for matchup in payload["matchups"]:
        region_label = _matchup_region_label(payload, matchup)
        for side_name, prefix in (("competitorOne", "team_one"), ("competitorTwo", "team_two")):
            competitor = matchup.get(side_name, {})
            team_name = competitor.get("name", "TBD")
            normalized_name = _normalize_name(team_name)
            team_id = name_map.get(normalized_name, 0) if team_name != "TBD" else 0
            matchup[f"{prefix}_id"] = team_id
            matchup[f"{prefix}_name"] = team_name

        rows.append(
            {
                "season": CURRENT_SEASON,
                "division": division,
                "round_id": int(matchup["roundId"]),
                "bracket_location": int(matchup["bracketLocation"]),
                "region_label": region_label,
                "game_id": str(matchup.get("id", "")),
                "status": matchup.get("statusDesc"),
                "status_detail": matchup.get("statusDetail"),
                "location": matchup.get("location"),
                "odds": matchup.get("odds"),
                "team_one_id": matchup["team_one_id"],
                "team_one_name": matchup["team_one_name"],
                "team_one_seed": competitor_seed(matchup.get("competitorOne", {})),
                "team_one_abbreviation": matchup.get("competitorOne", {}).get("abbreviation"),
                "team_two_id": matchup["team_two_id"],
                "team_two_name": matchup["team_two_name"],
                "team_two_seed": competitor_seed(matchup.get("competitorTwo", {})),
                "team_two_abbreviation": matchup.get("competitorTwo", {}).get("abbreviation"),
            }
        )

    return pd.DataFrame(rows)


def competitor_seed(competitor: dict) -> int | None:
    seed = competitor.get("seed")
    if seed in (None, "", "0"):
        return None
    return int(seed)


def _region_code_for_label(division: str, region_label: str) -> str:
    direct = REGION_CODE_MAP[division].get(region_label)
    if direct is not None:
        return direct
    normalized = str(region_label).strip()
    if division == "W":
        if "Fort Worth" in normalized:
            return "W"
        if "Sacramento" in normalized:
            return "X"
    raise KeyError(f"Unsupported region label for division {division}: {region_label!r}")


def _seed_numeric(seed_value: str) -> int:
    match = re.search(r"(\d{2})", str(seed_value))
    if not match:
        raise ValueError(f"Could not parse numeric seed from {seed_value!r}")
    return int(match.group(1))


def load_seed_table(division: str, refresh: bool = False) -> pd.DataFrame:
    historical = load_tournament_seeds(division).rename(columns={"Season": "season", "TeamID": "team_id", "Seed": "seed_code"})
    historical["division"] = division
    historical["seed_num"] = historical["seed_code"].map(_seed_numeric)
    historical["seed_missing"] = 0

    if refresh or _payload_path(division).exists():
        current_games = build_espn_bracket_games(division, refresh=refresh)
    else:
        current_games = pd.DataFrame()
    current_rows: list[dict[str, object]] = []
    for row in current_games.itertuples(index=False):
        for prefix in ("team_one", "team_two"):
            team_id = getattr(row, f"{prefix}_id")
            seed_num = getattr(row, f"{prefix}_seed")
            team_name = getattr(row, f"{prefix}_name")
            if not team_id or seed_num is None or team_name == "TBD":
                continue
            current_rows.append(
                {
                    "season": CURRENT_SEASON,
                    "team_id": int(team_id),
                    "seed_code": f"{_region_code_for_label(division, row.region_label)}{int(seed_num):02d}",
                    "division": division,
                    "seed_num": int(seed_num),
                    "seed_missing": 0,
                }
            )
    current = pd.DataFrame(current_rows).drop_duplicates(["season", "team_id"])
    return pd.concat([historical[["season", "team_id", "seed_code", "division", "seed_num", "seed_missing"]], current], ignore_index=True)


def tournament_margin_scale(division: str) -> float:
    data_dir = get_data_dir()
    if division == "M":
        path = data_dir / "MNCAATourneyCompactResults.csv"
    else:
        path = data_dir / "WNCAATourneyCompactResults.csv"
    frame = pd.read_csv(path)
    margins = (frame["WScore"] - frame["LScore"]).abs()
    return float(max(margins.std(ddof=0), 8.0))


def _spread_to_probability(spread: float, sigma: float) -> float:
    from statistics import NormalDist

    return float(NormalDist().cdf(spread / sigma))


def _parse_odds(matchup_row: pd.Series, sigma: float) -> tuple[float | None, int | None]:
    odds = matchup_row.get("odds")
    if not isinstance(odds, str) or not odds.strip():
        return None, None
    text = odds.strip()
    if text.upper() == "PK":
        favorite_team_id = matchup_row["team_one_id"]
        return 0.5, int(favorite_team_id) if favorite_team_id else None
    match = re.match(r"([A-Z0-9\- ]+)\s+([+-]?\d+(?:\.\d+)?)", text)
    if not match:
        return None, None
    favorite_token = match.group(1).strip().upper()
    spread = abs(float(match.group(2)))
    candidates = [
        (matchup_row["team_one_id"], str(matchup_row.get("team_one_abbreviation", "")).upper(), _normalize_name(matchup_row["team_one_name"]).upper()),
        (matchup_row["team_two_id"], str(matchup_row.get("team_two_abbreviation", "")).upper(), _normalize_name(matchup_row["team_two_name"]).upper()),
    ]
    favorite_team_id = None
    for team_id, abbreviation, normalized_name in candidates:
        if not team_id:
            continue
        if favorite_token == abbreviation or favorite_token == normalized_name or favorite_token.replace(" ", "") == abbreviation.replace(" ", ""):
            favorite_team_id = int(team_id)
            break
    if favorite_team_id is None:
        return None, None
    return _spread_to_probability(spread, sigma), favorite_team_id


def build_market_lines_table(division: str, refresh: bool = False) -> pd.DataFrame:
    if refresh or _payload_path(division).exists():
        games = build_espn_bracket_games(division, refresh=refresh)
    else:
        return pd.DataFrame(columns=["season", "division", "team_a", "team_b", "market_probability_a", "official_game_flag", "odds", "round_id", "region_label"])
    sigma = tournament_margin_scale(division)
    rows: list[dict[str, object]] = []
    for row in games.itertuples(index=False):
        if int(row.round_id) not in (0, 1):
            continue
        if not row.team_one_id or not row.team_two_id:
            continue
        series = pd.Series(row._asdict())
        favorite_probability, favorite_team_id = _parse_odds(series, sigma)
        team_a = min(int(row.team_one_id), int(row.team_two_id))
        team_b = max(int(row.team_one_id), int(row.team_two_id))
        market_prob_a = None
        if favorite_probability is not None and favorite_team_id is not None:
            market_prob_a = favorite_probability if favorite_team_id == team_a else 1.0 - favorite_probability
        rows.append(
            {
                "season": CURRENT_SEASON,
                "division": division,
                "team_a": team_a,
                "team_b": team_b,
                "market_probability_a": market_prob_a,
                "official_game_flag": 1,
                "odds": row.odds,
                "round_id": int(row.round_id),
                "region_label": row.region_label,
            }
        )
    return pd.DataFrame(rows).drop_duplicates(["season", "team_a", "team_b"])

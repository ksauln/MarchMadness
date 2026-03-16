from __future__ import annotations

import pandas as pd

from march_madness.data.loaders import load_regular_season_results


def _reverse_locations(values: pd.Series) -> pd.Series:
    return values.map({"H": "A", "A": "H", "N": "N"}).fillna("N")


def build_regular_season_long(division: str) -> pd.DataFrame:
    results = load_regular_season_results(division)

    winner_rows = pd.DataFrame(
        {
            "season": results["Season"],
            "day_num": results["DayNum"],
            "team_id": results["WTeamID"],
            "opp_team_id": results["LTeamID"],
            "team_score": results["WScore"],
            "opp_score": results["LScore"],
            "score_margin": results["WScore"] - results["LScore"],
            "loc": results["WLoc"],
            "num_ot": results["NumOT"],
            "is_win": 1,
            "fgm": results["WFGM"],
            "fga": results["WFGA"],
            "fgm3": results["WFGM3"],
            "fga3": results["WFGA3"],
            "ftm": results["WFTM"],
            "fta": results["WFTA"],
            "or": results["WOR"],
            "dr": results["WDR"],
            "ast": results["WAst"],
            "to": results["WTO"],
            "stl": results["WStl"],
            "blk": results["WBlk"],
            "pf": results["WPF"],
            "opp_fgm": results["LFGM"],
            "opp_fga": results["LFGA"],
            "opp_fgm3": results["LFGM3"],
            "opp_fga3": results["LFGA3"],
            "opp_ftm": results["LFTM"],
            "opp_fta": results["LFTA"],
            "opp_or": results["LOR"],
            "opp_dr": results["LDR"],
            "opp_ast": results["LAst"],
            "opp_to": results["LTO"],
            "opp_stl": results["LStl"],
            "opp_blk": results["LBlk"],
            "opp_pf": results["LPF"],
        }
    )

    loser_rows = pd.DataFrame(
        {
            "season": results["Season"],
            "day_num": results["DayNum"],
            "team_id": results["LTeamID"],
            "opp_team_id": results["WTeamID"],
            "team_score": results["LScore"],
            "opp_score": results["WScore"],
            "score_margin": results["LScore"] - results["WScore"],
            "loc": _reverse_locations(results["WLoc"]),
            "num_ot": results["NumOT"],
            "is_win": 0,
            "fgm": results["LFGM"],
            "fga": results["LFGA"],
            "fgm3": results["LFGM3"],
            "fga3": results["LFGA3"],
            "ftm": results["LFTM"],
            "fta": results["LFTA"],
            "or": results["LOR"],
            "dr": results["LDR"],
            "ast": results["LAst"],
            "to": results["LTO"],
            "stl": results["LStl"],
            "blk": results["LBlk"],
            "pf": results["LPF"],
            "opp_fgm": results["WFGM"],
            "opp_fga": results["WFGA"],
            "opp_fgm3": results["WFGM3"],
            "opp_fga3": results["WFGA3"],
            "opp_ftm": results["WFTM"],
            "opp_fta": results["WFTA"],
            "opp_or": results["WOR"],
            "opp_dr": results["WDR"],
            "opp_ast": results["WAst"],
            "opp_to": results["WTO"],
            "opp_stl": results["WStl"],
            "opp_blk": results["WBlk"],
            "opp_pf": results["WPF"],
        }
    )

    long_frame = (
        pd.concat([winner_rows, loser_rows], ignore_index=True)
        .sort_values(["season", "day_num", "team_id", "opp_team_id"])
        .reset_index(drop=True)
    )
    return long_frame

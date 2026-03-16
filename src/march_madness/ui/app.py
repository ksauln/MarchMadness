from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from march_madness.config import ARTIFACTS_DIR, DIVISION_LABELS
from march_madness.inference.predict import predict_single_matchup
from march_madness.inference.submission import (
    feature_table_path,
    load_feature_table,
    load_model_bundle,
    load_top25_context_table,
    metrics_path,
    model_bundle_path,
)
from march_madness.simulation import bracket_games_path, bracket_simulation_path
from march_madness.ui.presentation import build_matchup_pick, build_upset_signal


def _load_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
        :root {
          --bg: #f4efe4;
          --paper: rgba(255, 252, 246, 0.92);
          --ink: #18211b;
          --muted: #5f675f;
          --accent: #c45b32;
          --accent-2: #154734;
          --line: rgba(24, 33, 27, 0.12);
        }
        .stApp {
          background:
            radial-gradient(circle at top left, rgba(196, 91, 50, 0.16), transparent 28%),
            radial-gradient(circle at top right, rgba(21, 71, 52, 0.15), transparent 32%),
            linear-gradient(180deg, #f8f2e8 0%, #f0eadf 100%);
          color: var(--ink);
        }
        html, body, [class*="css"] {
          font-family: "IBM Plex Sans", sans-serif;
        }
        h1, h2, h3 {
          font-family: "DM Serif Display", serif !important;
          letter-spacing: 0.02em;
          color: var(--ink);
        }
        .block-container {
          padding-top: 2rem;
          padding-bottom: 3rem;
        }
        .hero-card, .panel-card {
          background: var(--paper);
          border: 1px solid var(--line);
          border-radius: 22px;
          padding: 1.15rem 1.25rem;
          box-shadow: 0 12px 40px rgba(24, 33, 27, 0.08);
        }
        .hero-title {
          font-family: "DM Serif Display", serif;
          font-size: 3rem;
          line-height: 1;
          margin-bottom: 0.35rem;
        }
        .hero-copy {
          color: var(--muted);
          max-width: 58rem;
          font-size: 1rem;
        }
        .kpi-label {
          color: var(--muted);
          text-transform: uppercase;
          font-size: 0.72rem;
          letter-spacing: 0.08em;
          margin-bottom: 0.35rem;
        }
        .kpi-value {
          font-size: 1.8rem;
          font-weight: 700;
          color: var(--accent-2);
        }
        .subtle {
          color: var(--muted);
        }
        .accent-chip {
          display: inline-block;
          border-radius: 999px;
          padding: 0.22rem 0.6rem;
          background: rgba(196, 91, 50, 0.12);
          color: var(--accent);
          font-size: 0.78rem;
          margin-right: 0.45rem;
        }
        .stTabs [data-baseweb="tab-list"] {
          gap: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
          background: rgba(255,255,255,0.55);
          border-radius: 999px;
          border: 1px solid var(--line);
          padding: 0.55rem 1rem;
        }
        .stTabs [aria-selected="true"] {
          background: #fffaf2 !important;
          border-color: rgba(196, 91, 50, 0.35) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model(division: str):
    return load_model_bundle(division)


@st.cache_data
def load_team_features(division: str) -> pd.DataFrame:
    return load_feature_table(division)


@st.cache_data
def load_metrics(division: str) -> pd.DataFrame:
    path = metrics_path(division)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_bracket_simulation(division: str) -> pd.DataFrame:
    path = bracket_simulation_path(division)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_bracket_games(division: str) -> pd.DataFrame:
    path = bracket_games_path(division)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_training_summary() -> dict:
    path = ARTIFACTS_DIR / "metrics" / "training_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@st.cache_data
def load_submission_snapshot() -> pd.DataFrame:
    path = ARTIFACTS_DIR / "submissions" / "stage2_baseline_submission.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _artifacts_ready() -> bool:
    return all(model_bundle_path(division).exists() and feature_table_path(division).exists() for division in ("M", "W"))


def _top_feature_diffs(frame: pd.DataFrame) -> pd.DataFrame:
    diff_columns = [column for column in frame.columns if column.endswith("_diff")]
    values = frame.iloc[0][diff_columns].astype(float)
    table = pd.DataFrame(
        {
            "feature": [column.replace("_diff", "") for column in diff_columns],
            "difference": values.values,
        }
    )
    table["abs_difference"] = table["difference"].abs()
    return table.sort_values("abs_difference", ascending=False).head(14).drop(columns="abs_difference")


@st.cache_data
def load_top25_context(division: str) -> pd.DataFrame:
    return load_top25_context_table(division)


def _format_pct(value: float | int | None, digits: int = 1) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}%}"


def _format_number(value: float | int | None, digits: int = 1) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def _team_snapshot(
    frame: pd.DataFrame,
    team_one_id: int,
    team_two_id: int,
    team_one_name: str,
    team_two_name: str,
    top25_lookup: dict[int, tuple[int, int]],
) -> pd.DataFrame:
    row = frame.iloc[0]
    first_suffix = "a" if int(row["team_a"]) == int(team_one_id) else "b"
    second_suffix = "a" if int(row["team_a"]) == int(team_two_id) else "b"

    def value(metric: str, suffix: str):
        return row.get(f"{metric}_{suffix}")

    def record(suffix: str) -> str:
        wins = int(round(float(value("wins", suffix))))
        losses = int(round(float(value("losses", suffix))))
        return f"{wins}-{losses}"

    def recent_record(suffix: str, window: int) -> str:
        win_pct = value(f"last{window}_win_pct", suffix)
        if pd.isna(win_pct):
            return "N/A"
        wins = int(round(float(win_pct) * window))
        return f"{wins}-{max(window - wins, 0)}"

    def top25_record(team_id: int) -> str:
        wins, games = top25_lookup.get(int(team_id), (0, 0))
        return f"{wins} wins in {games} games"

    market_probability = row.get("market_probability_a")
    if pd.notna(market_probability):
        market_team_one = float(market_probability) if first_suffix == "a" else 1.0 - float(market_probability)
        market_team_two = 1.0 - market_team_one
    else:
        market_team_one = None
        market_team_two = None

    return pd.DataFrame(
        [
            {"Metric": "Record", team_one_name: record(first_suffix), team_two_name: record(second_suffix)},
            {"Metric": "Win %", team_one_name: _format_pct(value("win_pct", first_suffix)), team_two_name: _format_pct(value("win_pct", second_suffix))},
            {
                "Metric": "Wins vs Top 25 (Elo)",
                team_one_name: top25_record(team_one_id),
                team_two_name: top25_record(team_two_id),
            },
            {"Metric": "Seed", team_one_name: _format_number(value("seed_num", first_suffix), 0), team_two_name: _format_number(value("seed_num", second_suffix), 0)},
            {"Metric": "PPG", team_one_name: _format_number(value("avg_points_for", first_suffix)), team_two_name: _format_number(value("avg_points_for", second_suffix))},
            {
                "Metric": "Points Against / Game",
                team_one_name: _format_number(value("avg_points_against", first_suffix)),
                team_two_name: _format_number(value("avg_points_against", second_suffix)),
            },
            {"Metric": "Avg Margin", team_one_name: _format_number(value("avg_margin", first_suffix)), team_two_name: _format_number(value("avg_margin", second_suffix))},
            {"Metric": "Off Rating", team_one_name: _format_number(value("off_rating", first_suffix)), team_two_name: _format_number(value("off_rating", second_suffix))},
            {"Metric": "Def Rating", team_one_name: _format_number(value("def_rating", first_suffix)), team_two_name: _format_number(value("def_rating", second_suffix))},
            {"Metric": "Net Rating", team_one_name: _format_number(value("net_rating", first_suffix)), team_two_name: _format_number(value("net_rating", second_suffix))},
            {"Metric": "Elo Rating", team_one_name: _format_number(value("elo_rating", first_suffix), 0), team_two_name: _format_number(value("elo_rating", second_suffix), 0)},
            {"Metric": "3PT %", team_one_name: _format_pct(value("fg3_pct", first_suffix)), team_two_name: _format_pct(value("fg3_pct", second_suffix))},
            {"Metric": "eFG %", team_one_name: _format_pct(value("efg_pct", first_suffix)), team_two_name: _format_pct(value("efg_pct", second_suffix))},
            {"Metric": "Last 5", team_one_name: recent_record(first_suffix, 5), team_two_name: recent_record(second_suffix, 5)},
            {"Metric": "Last 10", team_one_name: recent_record(first_suffix, 10), team_two_name: recent_record(second_suffix, 10)},
            {"Metric": "Market Win %", team_one_name: _format_pct(market_team_one), team_two_name: _format_pct(market_team_two)},
        ]
    )


def _official_matchups_with_predictions(division: str, team_features: pd.DataFrame, model_bundle: dict) -> pd.DataFrame:
    games = load_bracket_games(division)
    if games.empty:
        return games
    rows: list[dict[str, object]] = []
    for row in games.itertuples(index=False):
        if int(row.round_id) not in (0, 1):
            continue
        if not row.team_one_id or not row.team_two_id:
            continue
        probability, matchup_frame = predict_single_matchup(
            model_bundle,
            team_features,
            2026,
            int(row.team_one_id),
            int(row.team_two_id),
            division,
        )
        upset = build_upset_signal(
            team_one_name=row.team_one_name,
            team_two_name=row.team_two_name,
            team_one_seed=row.team_one_seed,
            team_two_seed=row.team_two_seed,
            team_one_probability=probability,
            team_two_probability=1.0 - probability,
        )
        rows.append(
            {
                "round": "First Four" if int(row.round_id) == 0 else "Round of 64",
                "region": row.region_label,
                "team_one": row.team_one_name,
                "team_two": row.team_two_name,
                "team_one_seed": row.team_one_seed,
                "team_two_seed": row.team_two_seed,
                "team_one_win_prob": probability,
                "team_two_win_prob": 1.0 - probability,
                "upset_flag": upset["summary"] if upset["flagged"] else "",
                "odds": matchup_frame.iloc[0].get("odds"),
                "market_probability_a": matchup_frame.iloc[0].get("market_probability_a"),
            }
        )
    return pd.DataFrame(rows)


st.set_page_config(page_title="March Madness Predictor", layout="wide")
_load_css()

if not _artifacts_ready():
    st.error("Model artifacts are missing. Run `./.venv/bin/python scripts/train_baseline.py --stage 2 --refresh-external` first.")
    st.stop()

division = st.sidebar.selectbox("Division", options=["M", "W"], format_func=lambda value: DIVISION_LABELS[value])
team_features = load_team_features(division)
top25_context = load_top25_context(division)
model_bundle = load_model(division)
metrics = load_metrics(division)
simulation = load_bracket_simulation(division)
summary = load_training_summary().get("divisions", {}).get(division, {})
submission_snapshot = load_submission_snapshot()

st.markdown(
    f"""
    <div class="hero-card">
      <div class="hero-title">Tournament Intelligence Lab</div>
      <div class="hero-copy">
        Seed-aware, bracket-aware NCAA forecasting for the 2026 tournaments. This build blends logistic regression,
        histogram gradient boosting, XGBoost, and ESPN bracket context to score matchups, inspect edges, and simulate the field.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

top_row = st.columns(4)
top_row[0].markdown(
    f'<div class="panel-card"><div class="kpi-label">Avg Brier</div><div class="kpi-value">{summary.get("avg_brier_score", float("nan")):.4f}</div></div>',
    unsafe_allow_html=True,
)
top_row[1].markdown(
    f'<div class="panel-card"><div class="kpi-label">Avg Log Loss</div><div class="kpi-value">{summary.get("avg_log_loss", float("nan")):.4f}</div></div>',
    unsafe_allow_html=True,
)
top_row[2].markdown(
    f'<div class="panel-card"><div class="kpi-label">Blend</div><div class="subtle">{", ".join(f"{k}: {v:.2f}" for k, v in model_bundle.get("blend_weights", {}).items())}</div><div class="kpi-label" style="margin-top:0.75rem;">Selected Configs</div><div class="subtle">{", ".join(f"{k}: {v}" for k, v in model_bundle.get("selected_configs", {}).items())}</div><div class="kpi-label" style="margin-top:0.75rem;">Market Weight</div><div class="kpi-value">{model_bundle.get("market_weight", 0.0):.2f}</div></div>',
    unsafe_allow_html=True,
)
favorite_name = summary.get("sim_title_favorite", "Unavailable")
favorite_prob = summary.get("sim_title_probability", 0.0)
top_row[3].markdown(
    f'<div class="panel-card"><div class="kpi-label">Simulation Favorite</div><div class="kpi-value">{favorite_name}</div><div class="subtle">Title odds {favorite_prob:.1%}</div></div>',
    unsafe_allow_html=True,
)

tabs = st.tabs(["Matchup Lab", "Bracket Outlook", "Model Health", "Submission Snapshot"])

with tabs[0]:
    available_seasons = sorted(team_features["season"].unique().tolist(), reverse=True)
    season = st.selectbox("Season", options=available_seasons, index=0)
    season_frame = team_features[team_features["season"] == season].sort_values("team_name")
    team_lookup = dict(zip(season_frame["team_name"], season_frame["team_id"]))
    team_names = season_frame["team_name"].dropna().tolist()

    pick_columns = st.columns(2)
    team_one_name = pick_columns[0].selectbox("Team One", options=team_names, index=0)
    team_two_default = 1 if len(team_names) > 1 else 0
    team_two_name = pick_columns[1].selectbox("Team Two", options=team_names, index=team_two_default)

    if team_one_name != team_two_name:
        team_one_id = int(team_lookup[team_one_name])
        team_two_id = int(team_lookup[team_two_name])
        probability, matchup_frame = predict_single_matchup(model_bundle, team_features, season, team_one_id, team_two_id, division)
        row = matchup_frame.iloc[0]
        team_one_suffix = "a" if int(row["team_a"]) == team_one_id else "b"
        team_two_suffix = "a" if int(row["team_a"]) == team_two_id else "b"
        market_probability = row.get("market_probability_a")
        if pd.notna(market_probability):
            team_one_market_probability = float(market_probability) if team_one_suffix == "a" else 1.0 - float(market_probability)
            team_two_market_probability = 1.0 - team_one_market_probability
            sportsbook_value = f'{team_one_name if team_one_market_probability >= team_two_market_probability else team_two_name} {max(team_one_market_probability, team_two_market_probability):.1%}'
        else:
            team_one_market_probability = None
            team_two_market_probability = None
            sportsbook_value = "N/A"
        upset = build_upset_signal(
            team_one_name=team_one_name,
            team_two_name=team_two_name,
            team_one_seed=row.get(f"seed_num_{team_one_suffix}"),
            team_two_seed=row.get(f"seed_num_{team_two_suffix}"),
            team_one_probability=probability,
            team_two_probability=1.0 - probability,
        )
        pick = build_matchup_pick(
            team_one_name=team_one_name,
            team_two_name=team_two_name,
            team_one_probability=probability,
            team_two_probability=1.0 - probability,
            upset_signal=upset,
            team_one_market_probability=team_one_market_probability,
            team_two_market_probability=team_two_market_probability,
        )

        metric_columns = st.columns(5)
        metric_columns[0].metric(
            label=f"{team_one_name} Win %",
            value=f"{probability:.1%}",
            help="Model-estimated chance that this selected team wins the matchup after any live market adjustment.",
        )
        metric_columns[1].metric(
            label=f"{team_two_name} Win %",
            value=f"{1.0 - probability:.1%}",
            help="Model-estimated chance that this selected team wins the matchup after any live market adjustment.",
        )
        metric_columns[2].metric(
            label="Sportsbook Lean",
            value=sportsbook_value,
            help="Live sportsbook-implied favorite and win probability, derived from ESPN odds when they are available for an official current bracket game.",
        )
        if upset["flagged"]:
            metric_columns[3].metric(
                label="Upset Flag",
                value=upset["level"],
                delta=f'{upset["team"]} at {float(upset["win_probability"]):.1%}',
                help="Heuristic flag for a lower-seeded team that still has a credible path to win based on the model and seed gap.",
            )
        else:
            metric_columns[3].metric(
                label="Upset Flag",
                value="No signal",
                help="Heuristic flag for a lower-seeded team that still has a credible path to win based on the model and seed gap.",
            )
        metric_columns[4].metric(
            label="Bracket Pick",
            value=pick["team"],
            delta=f'{pick["strategy"]} at {float(pick["probability"]):.1%}',
            help="Suggested pick for bracket selection. This can favor a credible upset instead of simply following the highest raw win probability.",
        )

        chips = []
        if pd.notna(matchup_frame.iloc[0].get("seed_num_a")):
            chips.append(f'Seeds {int(matchup_frame.iloc[0]["seed_num_a"])} vs {int(matchup_frame.iloc[0]["seed_num_b"])}')
        if pd.notna(matchup_frame.iloc[0].get("odds")):
            chips.append(f'Odds {matchup_frame.iloc[0]["odds"]}')
        if int(matchup_frame.iloc[0].get("official_game_flag", 0)) == 1:
            chips.append("Official 2026 bracket game")
        if upset["flagged"]:
            chips.append(upset["summary"])
        chips.append(f'Pick {pick["team"]}')
        st.markdown("".join(f'<span class="accent-chip">{chip}</span>' for chip in chips), unsafe_allow_html=True)

        inner_left, inner_right = st.columns([1.15, 0.85])
        with inner_left:
            st.subheader("Feature Edge")
            st.dataframe(_top_feature_diffs(matchup_frame), use_container_width=True, hide_index=True)
        with inner_right:
            st.subheader("Team Snapshot")
            top25_lookup = {
                int(row.team_id): (int(row.top25_elo_wins), int(row.top25_elo_games))
                for row in top25_context[top25_context["season"] == season].itertuples(index=False)
            }
            snapshot = _team_snapshot(
                matchup_frame,
                team_one_id=team_one_id,
                team_two_id=team_two_id,
                team_one_name=team_one_name,
                team_two_name=team_two_name,
                top25_lookup=top25_lookup,
            )
            pick_column = pick["team"]
            if pick_column in snapshot.columns:
                snapshot = snapshot.rename(columns={pick_column: f"{pick_column} ★"})
                styled_snapshot = snapshot.style.set_properties(
                    subset=[f"{pick_column} ★"],
                    **{"background-color": "rgba(21, 71, 52, 0.12)", "font-weight": "700"},
                )
                st.dataframe(styled_snapshot, use_container_width=True, hide_index=True)
            else:
                st.dataframe(snapshot, use_container_width=True, hide_index=True)
    else:
        st.info("Choose two different teams to inspect the matchup.")

with tabs[1]:
    if simulation.empty:
        st.info("Bracket simulation artifacts are not available yet.")
    else:
        left, right = st.columns([0.95, 1.05])
        with left:
            st.subheader("Top Title Paths")
            display = simulation[["team_name", "title", "championship", "final_four", "elite_8"]].head(16).copy()
            for column in ("title", "championship", "final_four", "elite_8"):
                display[column] = display[column].map(lambda value: f"{value:.1%}")
            st.dataframe(display, use_container_width=True, hide_index=True)
        with right:
            st.subheader("Live Bracket Matchups")
            official_games = _official_matchups_with_predictions(division, team_features, model_bundle)
            if official_games.empty:
                st.info("No official current bracket matchups were loaded.")
            else:
                official_display = official_games.copy()
                official_display = official_display.sort_values(
                    by=["upset_flag", "team_one_win_prob"],
                    ascending=[False, False],
                )
                official_display["team_one_win_prob"] = official_display["team_one_win_prob"].map(lambda value: f"{value:.1%}")
                official_display["team_two_win_prob"] = official_display["team_two_win_prob"].map(lambda value: f"{value:.1%}")
                if "market_probability_a" in official_display:
                    official_display["market_probability_a"] = official_display["market_probability_a"].map(
                        lambda value: f"{value:.1%}" if pd.notna(value) else "N/A"
                    )
                st.dataframe(official_display, use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Rolling Validation")
    if not metrics.empty:
        chart_data = metrics.sort_values("season").set_index("season")[["brier_score", "log_loss", "accuracy"]]
        st.line_chart(chart_data)
        st.dataframe(metrics.sort_values("season"), use_container_width=True, hide_index=True)
    else:
        st.info("Validation metrics are not available.")

with tabs[3]:
    if submission_snapshot.empty:
        st.info("No submission artifact was found.")
    else:
        snapshot_row = submission_snapshot["Pred"]
        summary_cols = st.columns(4)
        summary_cols[0].metric("Rows", f"{len(submission_snapshot):,}")
        summary_cols[1].metric("Mean Pred", f"{snapshot_row.mean():.4f}")
        summary_cols[2].metric("Min Pred", f"{snapshot_row.min():.6f}")
        summary_cols[3].metric("Max Pred", f"{snapshot_row.max():.6f}")
        st.subheader("Current Submission Sample")
        st.dataframe(submission_snapshot.head(20), use_container_width=True, hide_index=True)

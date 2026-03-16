# March Madness Predictor

This repo contains a working NCAA tournament prediction system built around the Kaggle March Machine Learning Mania 2026 dataset in [Data](/Users/kyle/Documents/MyProjects/MarchMadness/Data). It trains separate men's and women's models, pulls live 2026 bracket context from ESPN, generates Kaggle submissions, simulates the bracket, and serves a Streamlit UI for matchup analysis.

## Current Build
- Separate men's and women's pipelines
- Feature set built from regular-season detailed box scores
- Elo ratings and opponent-strength features
- Men's late-season Massey ordinal summaries
- Historical tournament-seed features plus live 2026 ESPN seed/bracket data
- Ensemble of logistic regression, histogram gradient boosting, and XGBoost
- Out-of-fold blend selection using rolling season validation
- Conservative market-aware adjustment for current official bracket games
- Monte Carlo simulation for the live 2026 bracket
- Streamlit app with matchup analysis, bracket outlook, validation metrics, submission snapshot, and upset flags

## Repo Layout
- [src/march_madness](/Users/kyle/Documents/MyProjects/MarchMadness/src/march_madness): core package
- [scripts](/Users/kyle/Documents/MyProjects/MarchMadness/scripts): training, submission, and app entrypoints
- [tests](/Users/kyle/Documents/MyProjects/MarchMadness/tests): regression and pipeline tests
- [artifacts](/Users/kyle/Documents/MyProjects/MarchMadness/artifacts): generated models, metrics, submissions, and external snapshots
- [docs](/Users/kyle/Documents/MyProjects/MarchMadness/docs): planning and architecture docs

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -e . pytest
```

## Main Commands
Train models, refresh live bracket data, generate artifacts:
```bash
.venv/bin/python scripts/train_baseline.py --stage 2 --refresh-external
```

Default simulation counts during training:
- Men: `500,000`
- Women: `10,000`

Optional overrides:
```bash
.venv/bin/python scripts/train_baseline.py --stage 2 --men-n-simulations 500000 --women-n-simulations 10000
```

Generate a submission from saved artifacts:
```bash
.venv/bin/python scripts/generate_submission.py --stage 2
```

Launch the Streamlit app:
```bash
.venv/bin/python scripts/launch_app.py
```

Run tests:
```bash
.venv/bin/python -m pytest
```

## Community Cloud Deploy
This repo includes a root [app.py](/Users/kyle/Documents/MyProjects/MarchMadness/app.py) entrypoint plus a root [requirements.txt](/Users/kyle/Documents/MyProjects/MarchMadness/requirements.txt) with explicit runtime dependencies so the app can be deployed directly on Streamlit Community Cloud.

Recommended deploy flow:
- Commit the app code and the generated [artifacts](/Users/kyle/Documents/MyProjects/MarchMadness/artifacts) directory.
- Do not commit raw Kaggle training data in [Data](/Users/kyle/Documents/MyProjects/MarchMadness/Data); the deployed app does not need it.
- In Streamlit Community Cloud, point the app to `app.py`.
- Let Community Cloud install dependencies from [requirements.txt](/Users/kyle/Documents/MyProjects/MarchMadness/requirements.txt).
- Use a Python runtime compatible with [pyproject.toml](/Users/kyle/Documents/MyProjects/MarchMadness/pyproject.toml) (`>=3.11`).

If the deploy boots without the committed runtime artifacts, the app now reports the exact missing files under [artifacts](/Users/kyle/Documents/MyProjects/MarchMadness/artifacts) so the failure is actionable.

Local deploy-equivalent smoke test:
```bash
.venv/bin/python -m streamlit run app.py
```

## Pipeline Summary
1. Load Kaggle regular-season and tournament data from [Data](/Users/kyle/Documents/MyProjects/MarchMadness/Data).
2. Build team-level season features from detailed regular-season box scores.
3. Compute Elo ratings and late-season form indicators.
4. Add men's Massey ordinal summaries.
5. Merge historical tournament seeds and current live ESPN bracket seeds.
6. Train candidate models on mirrored tournament matchups.
7. Choose blend weights from rolling-season out-of-fold predictions.
8. Fit final models on all available training seasons.
9. Generate Stage 2 predictions and run bracket simulations.

## Key Features Used
- Record, win rate, points scored, points allowed, margin
- Shooting efficiency: FG%, 3PT%, FT%, eFG%
- Possession and efficiency metrics: offensive rating, defensive rating, net rating
- Ball security and playmaking: turnover rate, assist rate, assist-to-turnover ratio
- Rebounding rates and per-game rebounding
- Recent form over the last 5 and last 10 games
- Opponent quality summaries
- Elo ratings
- Men's Massey summary ranks
- Tournament seed features
- Live market context for official current bracket games

## Streamlit App
The app is at [src/march_madness/ui/app.py](/Users/kyle/Documents/MyProjects/MarchMadness/src/march_madness/ui/app.py) and currently exposes five tabs:

- `Matchup Lab`
  - Compare any two teams from a selected season
  - View team win probabilities, `Sportsbook Lean`, `Upset Flag`, and `Bracket Pick`
  - Inspect feature-difference leaders
  - Review a team snapshot with record, top-25 wins, scoring, efficiency, form, seed, and market win rate
- `Bracket Outlook`
  - See simulated title odds and advancement rates
  - Review official live bracket games with model probabilities and upset-watch labels
- `Bracket Builder`
  - Render a full filled-in tournament bracket in-app
  - Switch among four deterministic bracket variants: `Model Only`, `Likely Upsets`, `Market Consensus`, and `Title Equity`
- `Model Health`
  - Inspect rolling validation metrics
- `Submission Snapshot`
  - Inspect the generated Kaggle submission sample

## Upset Flag Logic
The upset indicator is a presentation-layer heuristic in [src/march_madness/ui/presentation.py](/Users/kyle/Documents/MyProjects/MarchMadness/src/march_madness/ui/presentation.py). It flags games when the lower-seeded team has enough modeled win probability to matter.

Current thresholds:
- `Model upset pick`: lower seed has at least a 2-seed disadvantage and at least 50% win probability
- `Strong upset watch`: seed gap at least 3 and win probability at least 42%
- `Upset watch`: seed gap at least 5 and win probability at least 35%
- `Long-shot upset flyer`: seed gap at least 8 and win probability at least 28%

## UI Terms
- `Sportsbook Lean`: live sportsbook-implied favorite and win probability, derived from ESPN odds when available
- `Upset Flag`: a heuristic alert that a lower-seeded team has enough modeled win probability to be worth serious bracket attention
- `Bracket Pick`: the suggested team to advance in a bracket entry; this can intentionally take a credible upset instead of always following the highest raw win probability

## Generated Artifacts
- Models: [artifacts/models](/Users/kyle/Documents/MyProjects/MarchMadness/artifacts/models)
- Metrics: [artifacts/metrics](/Users/kyle/Documents/MyProjects/MarchMadness/artifacts/metrics)
- Team features: [artifacts/features](/Users/kyle/Documents/MyProjects/MarchMadness/artifacts/features)
- External bracket snapshots: [artifacts/external](/Users/kyle/Documents/MyProjects/MarchMadness/artifacts/external)
- Submissions: [artifacts/submissions](/Users/kyle/Documents/MyProjects/MarchMadness/artifacts/submissions)

Important files:
- Summary metrics: [artifacts/metrics/training_summary.json](/Users/kyle/Documents/MyProjects/MarchMadness/artifacts/metrics/training_summary.json)
- Final submission: [artifacts/submissions/stage2_baseline_submission.csv](/Users/kyle/Documents/MyProjects/MarchMadness/artifacts/submissions/stage2_baseline_submission.csv)

## Data Sources
- Kaggle competition data in the local [Data](/Users/kyle/Documents/MyProjects/MarchMadness/Data) directory
- Live ESPN men's and women's bracket data cached under [artifacts/external](/Users/kyle/Documents/MyProjects/MarchMadness/artifacts/external)

For a public app deploy, the artifacts are the important runtime assets. Raw Kaggle training data is only needed to retrain the models locally.

## Verification
The current codebase is intended to be verified with:
```bash
.venv/bin/python -m pytest
.venv/bin/python scripts/train_baseline.py --stage 2 --refresh-external
.venv/bin/python scripts/generate_submission.py --stage 2
.venv/bin/python scripts/launch_app.py
```

## Known Limits
- The exact Kaggle evaluation text should still be confirmed against the live competition page before treating offline ranking as final.
- Live market adjustments only apply to official current bracket games where odds are available.
- The upset flag is a UI heuristic layered on top of the model, not a separate classifier.
- ESPN label and naming changes can require maintenance in the external mapping layer.

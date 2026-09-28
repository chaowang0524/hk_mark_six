# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hong Kong Mark Six number prediction — a Python CLI tool that scrapes historical lottery data, performs statistical analysis, and trains ML models to predict all 7 drawn numbers. Mark Six draws 7 numbers from 1–49: 6 winning (main) numbers + 1 extra number. The pipeline predicts both: a multi-label model ranks main-number candidates, and a separate multi-class model ranks extra-number candidates.

## Setup

```bash
# Install dependencies into the existing .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## CLI Commands

All commands run via `python main.py <subcommand>`:

```bash
python main.py scrape                    # Scrape historical draws → data/raw_draws.csv
python main.py preprocess                # Feature engineering → data/features.csv
python main.py analyze                   # Print statistical report to terminal
python main.py train [--model rf|xgb|lr|all]  # Train main + extra models → data/models/
python main.py evaluate                  # Backtesting + metrics comparison
python main.py predict [--top N] [--model xgb] [--both] [--top-main N]  # Predict next draw's extra number (add --both for main numbers too)
```

## Architecture

Data flows linearly through the pipeline:

```
scraper.py → data/raw_draws.csv
           ↓
preprocessing.py → data/features.csv
           ↓
models.py → data/models/  (RF, XGBoost, LogReg — one set per target: extra_*.joblib, main_*.joblib)
           ↓
evaluate.py (extra: top-1/3/5/10 hit rate; main: hit-rate@k)
           ↓
predictor.py → ranked candidates with confidence scores, for both targets
```

**`src/scraper.py`** — Web scraping with requests/BeautifulSoup. Supports incremental updates (only fetches new draws).

**`src/preprocessing.py`** — Validates numbers (1–49), engineers features. **The 6 winning numbers of the draw being predicted are never available as input** — all features must use strictly lagged (past-draw) data:
- Extra-number history: rolling frequency (windows 10/30/100/300/500), all-time count, EWM, hot/cold trend, z-score vs. uniform, gap since last appearance, raw lag values (1–10 draws back), Markov-style transition frequency from the previous extra number
- Main-number history (lagged): the same treatment as extra numbers — multi-window rolling frequency, all-time count, EWM, trend, z-score, gap, presence in the last 1/3/5/10 draws, sorted 6-number lag values (1–10 draws back), and a Markov-style co-occurrence transition from the previous draw's 6 main numbers — plus parity/sum/consecutive stats of the *previous* draw's winning numbers
- Date seasonality: day-of-week, month

**`src/statistics.py`** — Frequency distribution, hot/cold analysis, gap analysis, chi-square uniformity test, autocorrelation. Outputs to terminal only.

**`src/models.py`** — Two sub-problems, trained separately: extra number is **multi-class classification** (49 classes, one model); main numbers are **multi-label binary classification** (49 independent binary outputs via `MultiOutputClassifier` — no joint modeling of the 6-number set). Uses **chronological train/test split** (last 5% as test, `TEST_FRACTION` in `models.py`) — never random split, to avoid data leakage. Saves models to `data/models/` as `extra_*.joblib` and `main_*.joblib`.

**`src/evaluate.py`** — Extra number: top-1/3/5/10 hit rate, per-number precision/recall, confusion matrix, side-by-side model comparison. Main numbers: hit-rate@k (mean true positives in the top-k picks) vs. the uniform baseline. `walk_forward_evaluate` (honest retrain-and-predict backtest) currently covers the extra number only.

**`src/predictor.py`** — Loads latest trained models, optionally ensembles model probabilities, outputs top-N candidates for the extra number (default) and, with `--both`, main numbers too, with confidence scores and supporting statistics. Recency/last-draw penalties and the `--wildcard` echo heuristic apply to the extra number only.

**`main.py`** — CLI entry point (argparse), routes subcommands to the appropriate module.

## Key Design Decisions

- **Chronological split only** — random splits leak future data into training; never use `shuffle=True` for the train/test split.
- **Extra number: multi-class classification** (49 classes) rather than regression, because the output is a discrete set.
- **Main numbers: multi-label classification** (49 independent binary classifiers, top-6 by probability at inference) rather than a joint model over 6-number combinations — simpler, and the combinatorial target space (49 choose 6) is intractable to model directly.
- **Ensemble option** — weighted average of model probability outputs for final prediction, applied independently to each target.

## Note:
- This project is for education purpose to learn statistics, based on the idea of the lottery result is not uniformly distributed and predictable.
- Do not add the signature by 'Claude' in the commit message
- When explaining, use simple language, avoid using jargon to explain jargons.
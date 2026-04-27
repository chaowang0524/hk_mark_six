# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hong Kong Mark Six extra number prediction — a Python CLI tool that scrapes historical lottery data, performs statistical analysis, and trains ML models to predict the extra (special) number. Mark Six draws 7 numbers from 1–49: 6 winning numbers + 1 extra number. This project targets the extra number exclusively.

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
python main.py train [--model rf|xgb|lr|all]  # Train models → data/models/
python main.py evaluate                  # Backtesting + metrics comparison
python main.py predict [--top N] [--model xgb]  # Predict next extra number
```

## Architecture

Data flows linearly through the pipeline:

```
scraper.py → data/raw_draws.csv
           ↓
preprocessing.py → data/features.csv
           ↓
models.py → data/models/  (RF, XGBoost, LogReg)
           ↓
evaluate.py (backtesting, top-1/3/5/10 hit rates)
           ↓
predictor.py → ranked candidates with confidence scores
```

**`src/scraper.py`** — Web scraping with requests/BeautifulSoup. Supports incremental updates (only fetches new draws).

**`src/preprocessing.py`** — Validates numbers (1–49), engineers features. **The 6 winning numbers of the draw being predicted are never available as input** — all features must use strictly lagged (past-draw) data:
- Extra-number history: rolling frequency (windows 10/30/100), gap since last extra appearance, trend
- Main-number history (lagged): rolling frequency as main number, parity/sum/consecutive stats of the *previous* draw's winning numbers, co-occurrence between past main numbers and past extra numbers
- Date seasonality: day-of-week, month

**`src/statistics.py`** — Frequency distribution, hot/cold analysis, gap analysis, chi-square uniformity test, autocorrelation. Outputs to terminal only.

**`src/models.py`** — Multi-class classification (49 classes). Uses **chronological train/test split** (last 20% as test) — never random split, to avoid data leakage. Saves models to `data/models/`.

**`src/evaluate.py`** — Computes top-1/3/5/10 hit rate, per-number precision/recall, confusion matrix, side-by-side model comparison.

**`src/predictor.py`** — Loads latest trained model, optionally ensembles model probabilities, outputs top-N candidates with confidence scores and supporting statistics.

**`main.py`** — CLI entry point (argparse), routes subcommands to the appropriate module.

## Key Design Decisions

- **Chronological split only** — random splits leak future data into training; never use `shuffle=True` for the train/test split.
- **Multi-class classification** (49 classes, one per number) rather than regression, because the output is a discrete set.
- **Ensemble option** — weighted average of model probability outputs for final prediction.

## Note:
- This project is for education purpose to learn statistics, based on the idea of the lottery result is not uniformly distributed and predictable.
- Do not add the signature by 'Claude Opus 4.7 <noreply@anthropic.com>' in the commit message
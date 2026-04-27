# Hong Kong Mark Six Extra Number Prediction — Project Plan

## Context

Build a Python CLI tool that scrapes historical Hong Kong Mark Six lottery data, performs statistical analysis, and uses ML models to predict the extra (special) number. The Mark Six draws 7 numbers from 1-49: 6 winning numbers + 1 extra number. We focus exclusively on predicting the extra number.

## Project Structure

```
lottery/
├── src/
│   ├── __init__.py
│   ├── scraper.py        # Web scraping for historical data
│   ├── preprocessing.py  # Data cleaning & feature engineering
│   ├── statistics.py     # Statistical analysis (freq, gap, hot/cold)
│   ├── models.py         # ML models (RF, XGBoost, Logistic Regression)
│   ├── evaluate.py       # Backtesting & accuracy metrics
│   └── predictor.py      # Prediction logic
├── data/                  # Scraped & processed data (CSV)
│   └── models/            # Saved trained models
├── tests/
├── requirements.txt
└── main.py               # CLI entry point
```

## Implementation Steps

### Step 1: Project Setup
- Create directory structure
- `requirements.txt`: requests, beautifulsoup4, pandas, numpy, scikit-learn, xgboost, matplotlib
- Python 3.10+ with the existing `.venv`

### Step 2: Web Scraper (`src/scraper.py`)
- Scrape historical Mark Six results from a public lottery results site
- Parse draw date, draw number, 6 winning numbers, and extra number
- Store as CSV in `data/raw_draws.csv`
- Support incremental updates (only scrape new draws)
- CLI command: `python main.py scrape`

### Step 3: Data Preprocessing (`src/preprocessing.py`)
- Load raw CSV, validate numbers (1-49), handle missing data
- **Input constraint:** the 6 winning numbers of the draw being predicted are NOT available at prediction time. All features must be derived solely from past draws (lagged data). Features fall into two groups:
  1. **Extra-number history:** rolling frequency of each number as extra (windows: 10, 30, 100 draws), gap since last extra appearance, extra number moving average and trend
  2. **Main-number history (lagged):** rolling frequency of each number appearing as a main number in past draws, parity/sum/consecutive stats of the previous draw's 6 winning numbers, co-occurrence between past main numbers and extra numbers (all strictly from prior draws)
- Day-of-week and month features (draw date seasonality)
- Save processed features to `data/features.csv`
- CLI command: `python main.py preprocess`

### Step 4: Statistical Analysis (`src/statistics.py`)
- Extra number frequency distribution and histogram
- Hot/cold number analysis (most/least frequent extra numbers in recent N draws)
- Gap analysis: average gaps between appearances for each number
- Chi-square test for uniformity (is the distribution truly random?)
- Autocorrelation analysis (does the previous extra number predict the next?)
- Output summary report to md file
- CLI command: `python main.py analyze`

### Step 5: ML Models (`src/models.py`)
Train multiple models and compare:
1. **Random Forest**: Tabular features, handles non-linearity well
2. **XGBoost**: Gradient boosting, often best for tabular data
3. **Logistic Regression**: Baseline linear model (multinomial, 49 classes)

Target: multi-class classification (49 classes, numbers 1-49)
- Train/test split: chronological (e.g., last 20% as test)
- Save trained models to `data/models/`
- CLI command: `python main.py train [--model rf|xgb|lr|all]`

### Step 6: Evaluation (`src/evaluate.py`)
- Accuracy: top-1, top-3, top-5, top-10 hit rate
- Per-number precision/recall
- Confusion matrix
- Backtesting: simulate predictions on historical data, track cumulative accuracy
- Compare models side-by-side
- CLI command: `python main.py evaluate`

### Step 7: Prediction CLI (`main.py` + `src/predictor.py`)
- Load latest trained model, generate prediction for next draw
- Output top-N recommended extra numbers with confidence scores
- Show supporting statistics (frequency, gap, recent trend)
- CLI commands:
  - `python main.py predict` — predict next extra number
  - `python main.py predict --top 5` — top 5 candidates
  - `python main.py predict --model xgb` — use specific model

## Key Design Decisions
- **Chronological split** (not random) for train/test to avoid data leakage from time-series
- **Multi-class classification** (49 classes) rather than regression
- **Ensemble consideration**: weighted average of model probabilities for final prediction
- **Important caveat**: Lottery draws are independent random events. This tool is for educational/data-science purposes. No model can reliably predict lottery numbers.

## Verification
1. Run `python main.py scrape` — verify CSV is created with valid data
2. Run `python main.py preprocess` — verify features CSV has no NaN values
3. Run `python main.py analyze` — verify statistical report renders correctly
4. Run `python main.py train --model all` — verify all models train without errors
5. Run `python main.py evaluate` — verify metrics are computed and displayed
6. Run `python main.py predict --top 10` — verify prediction output with confidence scores

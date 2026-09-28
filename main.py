"""Hong Kong Mark Six 7-number prediction (6 main + 1 extra) — CLI entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mark-six",
        description="Mark Six 7-number prediction pipeline (6 main + 1 extra).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("scrape", help="Fetch historical draws → data/raw_draws.csv")
    sub.add_parser("preprocess", help="Build lagged features → data/features.csv")
    sub.add_parser("analyze", help="Write statistical report to data/reports/")

    p_update = sub.add_parser(
        "update",
        help="Scrape latest draws → retrain all models on full history → predict next draw",
    )
    p_update.add_argument(
        "--model",
        default="all",
        choices=["rf", "xgb", "lr", "all"],
        help="Which model(s) to retrain (default: all)",
    )
    p_update.add_argument("--top", type=int, default=10, help="Top-N extra candidates in prediction")
    p_update.add_argument("--top-main", type=int, default=10, help="Top-N main-number candidates in prediction")
    p_update.add_argument(
        "--both",
        action="store_true",
        help="Also predict main numbers, not just the extra number (passed through to predict)",
    )
    p_update.add_argument(
        "--penalty", type=float, default=0.0, metavar="STRENGTH",
        help="Recency penalty passed through to predict (0 = disabled)",
    )
    p_update.add_argument(
        "--penalize-last",
        type=float,
        nargs="?",
        const=0.5,
        default=None,
        metavar="FACTOR",
        help="Penalise the last 3 extra numbers with a linearly fading factor starting at FACTOR (0=exclude, 1=no change). Omit FACTOR to use 0.5.",
    )
    p_update.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Override the guessed next-draw date passed through to predict.",
    )

    p_train = sub.add_parser("train", help="Train main-number and extra-number models")
    p_train.add_argument(
        "--model", required=True, choices=["rf", "xgb", "lr", "all"],
        help="Which model to train. Use 'all' to train rf+xgb+lr (slow with ~1400 features).",
    )
    p_train.add_argument(
        "--full",
        action="store_true",
        help="Train on the entire dataset (no held-out test set). "
        "Use after evaluating with the default chronological split.",
    )

    sub.add_parser("evaluate", help="Backtest + compare models")

    p_wf = sub.add_parser(
        "walk-forward",
        help="Honest backtest (extra number only): retrain on [0..t-1] and predict draw t for the last N draws",
    )
    p_wf.add_argument("--n-test", type=int, default=200, help="Trailing draws to backtest")
    p_wf.add_argument("--stride", type=int, default=20, help="Retrain every N draws")
    p_wf.add_argument(
        "--model", default="all", choices=["rf", "xgb", "lr", "all"]
    )

    p_pred = sub.add_parser(
        "predict", help="Predict next draw's extra number (use --both to also predict main numbers)"
    )
    p_pred.add_argument("--top", type=int, default=10, help="Top-N extra candidates")
    p_pred.add_argument("--top-main", type=int, default=10, help="Top-N main-number candidates (only used with --both)")
    p_pred.add_argument(
        "--both",
        action="store_true",
        help="Also predict main numbers, not just the extra number (default: extra number only)",
    )
    p_pred.add_argument(
        "--model",
        default="xgb",
        choices=["rf", "xgb", "lr", "ensemble"],
        help="Model to use (ensemble averages all available)",
    )
    p_pred.add_argument(
        "--penalty",
        type=float,
        default=0.0,
        metavar="STRENGTH",
        help="Recency penalty strength 0–1 (e.g. 0.5). Discounts numbers that "
             "appeared as extra recently: drawn 1 draw ago → ×STRENGTH, "
             "2 draws ago → ×√STRENGTH, fading over 10 draws. 0 = disabled.",
    )
    p_pred.add_argument(
        "--penalize-last",
        type=float,
        nargs="?",
        const=0.5,
        default=None,
        metavar="FACTOR",
        help="Penalise the last 3 extra numbers with a linearly fading factor starting at FACTOR (0=exclude, 1=no change). Omit FACTOR to use 0.5.",
    )
    p_pred.add_argument(
        "--wildcard",
        action="store_true",
        help="Show only the model's top 5, plus a separately-labelled 'wildcard' 5 "
        "echoing whichever numbers scored highest across reports issued 2 months ago "
        "(excludes anything already in the model's own top list). For fun only — "
        "draws are independent, this is not additional model signal.",
    )
    p_pred.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Override the guessed next-draw date (default: next Tue/Thu/Sat after "
        "the latest scraped draw). Use this when the real HKJC calendar deviates "
        "from the usual cadence.",
    )

    args = parser.parse_args()

    if args.command == "update":
        from scraper import scrape
        from preprocessing import preprocess
        from models import train
        from predictor import predict
        print("=== Step 1/3: Scraping latest draws ===")
        scrape()
        print("\n=== Step 2/3: Retraining extra + main models on full history ===")
        train(args.model, full=True)
        print("\n=== Step 3/3: Predicting next draw ===")
        predict(model_name="ensemble" if args.model == "all" else args.model,
                top_extra=args.top, top_main=args.top_main, both=args.both,
                penalty=args.penalty, penalize_last=args.penalize_last,
                date=args.date)

    elif args.command == "scrape":
        from scraper import scrape
        scrape()
    elif args.command == "preprocess":
        from preprocessing import preprocess
        preprocess()
    elif args.command == "analyze":
        from statistics import run_analysis
        run_analysis()
    elif args.command == "train":
        from models import train
        train(args.model, full=args.full)
    elif args.command == "evaluate":
        from evaluate import evaluate
        evaluate()
    elif args.command == "walk-forward":
        from evaluate import walk_forward_evaluate
        names = ("rf", "xgb", "lr") if args.model == "all" else (args.model,)
        walk_forward_evaluate(
            n_test=args.n_test, stride=args.stride, model_names=names
        )
    elif args.command == "predict":
        from predictor import predict
        predict(model_name=args.model, top_extra=args.top, top_main=args.top_main, both=args.both,
                penalty=args.penalty, penalize_last=args.penalize_last,
                wildcard=args.wildcard, date=args.date)


if __name__ == "__main__":
    main()
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

    p_train = sub.add_parser("train", help="Train ML models")
    p_train.add_argument(
        "--model", default="all", choices=["rf", "xgb", "lr", "all"]
    )
    p_train.add_argument(
        "--target", default="both", choices=["extra", "main", "both"],
        help="Which target to train (default: both)",
    )

    sub.add_parser("evaluate", help="Backtest + compare models")

    p_pred = sub.add_parser(
        "predict", help="Predict next draw's 6 main numbers + 1 extra number"
    )
    p_pred.add_argument("--top", type=int, default=10, help="Top-N extra candidates")
    p_pred.add_argument(
        "--top-main", type=int, default=10, help="Top-N main number candidates"
    )
    p_pred.add_argument(
        "--model",
        default="xgb",
        choices=["rf", "xgb", "lr", "ensemble"],
        help="Model to use (ensemble averages all available)",
    )

    args = parser.parse_args()

    if args.command == "scrape":
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
        train(args.model, args.target)
    elif args.command == "evaluate":
        from evaluate import evaluate
        evaluate()
    elif args.command == "predict":
        from predictor import predict
        predict(model_name=args.model, top_extra=args.top, top_main=args.top_main)


if __name__ == "__main__":
    main()
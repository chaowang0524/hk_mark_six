"""Predict the next Mark Six draw: 6 main numbers + 1 extra number.

Builds the lagged feature row for a hypothetical next draw, runs it
through trained models (or an ensemble average of all available models),
and reports a recommended ticket plus ranked candidate tables with
supporting statistics.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from models import MODELS_DIR
from preprocessing import POOL, build_features, load_raw

REPORTS_DIR = Path(__file__).resolve().parent.parent / "data" / "reports"
PREDICTION_LOG = Path(__file__).resolve().parent.parent / "data" / "prediction_log.csv"

EXTRA_MODEL_FILES = {
    "rf": MODELS_DIR / "extra_rf.joblib",
    "xgb": MODELS_DIR / "extra_xgb.joblib",
    "lr": MODELS_DIR / "extra_lr.joblib",
}
MAIN_MODEL_FILES = {
    "rf": MODELS_DIR / "main_rf.joblib",
    "xgb": MODELS_DIR / "main_xgb.joblib",
    "lr": MODELS_DIR / "main_lr.joblib",
}


def _load_model(name: str, target: str):
    files = EXTRA_MODEL_FILES if target == "extra" else MAIN_MODEL_FILES
    path = files.get(name)
    if path is None or not path.exists():
        return None
    return joblib.load(path)


def _available_models(target: str) -> list[str]:
    files = EXTRA_MODEL_FILES if target == "extra" else MAIN_MODEL_FILES
    return [n for n, p in files.items() if p.exists()]


def _next_draw_feature_row(override_date: str | None = None) -> tuple[pd.Series, pd.Timestamp, pd.DataFrame]:
    """Return (feature row, predicted draw date, raw history).

    By default the draw date is guessed as the next Tue/Thu/Sat after the
    latest scraped draw. Pass `override_date` (e.g. "2026-07-20") to pin it
    explicitly — needed when the real HKJC calendar deviates from the usual
    cadence, or when raw_draws.csv hasn't been rescraped yet.
    """
    raw = load_raw()
    last_date = raw["date"].iloc[-1]

    if override_date is not None:
        next_date = pd.Timestamp(override_date)
        if next_date <= last_date:
            raise ValueError(
                f"--date {next_date.date()} is not after the latest scraped draw "
                f"({last_date.date()}); run `python main.py scrape` first if your "
                "data is stale."
            )
        if next_date.dayofweek not in {1, 3, 5}:
            print(
                f"  NOTE: {next_date.date()} is a {next_date.strftime('%A')} — "
                "not a regular Mark Six draw day (Tue/Thu/Sat). Proceeding anyway "
                "since --date was explicitly given.\n"
            )
    else:
        draw_days = {1, 3, 5}  # Tue/Thu/Sat
        next_date = last_date
        for _ in range(7):
            next_date = next_date + pd.Timedelta(days=1)
            if next_date.dayofweek in draw_days:
                break
    placeholder = pd.DataFrame(
        [
            {
                "draw": "NEXT",
                "date": next_date,
                "n1": 1,
                "n2": 2,
                "n3": 3,
                "n4": 4,
                "n5": 5,
                "n6": 6,
                "extra": 1,
            }
        ]
    )
    extended = pd.concat([raw, placeholder], ignore_index=True)
    features = build_features(extended)
    return features.iloc[-1], next_date, raw


def _align_features(row: pd.Series) -> pd.DataFrame:
    cols_path = MODELS_DIR / "feature_columns.json"
    if not cols_path.exists():
        raise FileNotFoundError(
            "feature_columns.json missing; run `python main.py train` first."
        )
    import json
    cols = json.loads(cols_path.read_text())
    return pd.DataFrame([row[cols].to_dict()])[cols]


def _get_extra_proba(model_name: str, X_next: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Returns shape (49,) and a label describing the model used."""
    if model_name == "ensemble":
        names = _available_models("extra")
        if not names:
            raise FileNotFoundError("No trained extra models found in data/models/.")
        probas = []
        for n in names:
            clf = _load_model(n, "extra")
            if clf is not None:
                probas.append(clf.predict_proba(X_next)[0])
        return np.mean(probas, axis=0), f"ensemble({'+'.join(names)})"
    clf = _load_model(model_name, "extra")
    if clf is None:
        raise FileNotFoundError(
            f"extra_{model_name}.joblib not found; run `python main.py train` first."
        )
    return clf.predict_proba(X_next)[0], model_name



def _apply_prediction_penalty(
    proba: np.ndarray,
    strength: float = 0.5,
    window: int = 5,
) -> np.ndarray:
    """Discount numbers that appeared in recent predictions.

    Reads the last `window` rows from prediction_log.csv. For each row,
    numbers that were ranked higher and predicted more recently get a
    steeper discount using:

        factor = strength ^ (1 / (age * rank))

    where age=1 is the most recent prediction and rank=1 is top-1.

    Examples with strength=0.5:
      rank=1, age=1 → 0.50  (top-1 last prediction — steepest)
      rank=1, age=2 → 0.71
      rank=5, age=1 → 0.87
      rank=10, age=1 → 0.93 (nearly no effect)

    Result is renormalised so probabilities still sum to 1.
    """
    if not PREDICTION_LOG.exists():
        return proba

    df = pd.read_csv(PREDICTION_LOG)
    if df.empty:
        return proba

    recent = df.tail(window).reset_index(drop=True)
    proba = proba.copy()

    for i, row in recent.iterrows():
        age = len(recent) - i  # most recent row → age=1
        for rank, token in enumerate(str(row["predictions"]).split(), start=1):
            try:
                num = int(token.split("(")[0])
            except ValueError:
                continue
            proba[num - 1] *= strength ** (1.0 / (age * rank))

    total = proba.sum()
    if total > 0:
        proba /= total
    return proba


def _get_main_proba(model_name: str, X_next: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Returns shape (49,) — P(number k in main draw), k=1..49."""
    if model_name == "ensemble":
        names = _available_models("main")
        if not names:
            raise FileNotFoundError("No trained main models found in data/models/.")
        probas = []
        for n in names:
            clf = _load_model(n, "main")
            if clf is not None:
                proba_list = clf.predict_proba(X_next)  # list of 49 arrays (1, 2)
                probas.append(np.array([p[0, 1] for p in proba_list]))
        return np.mean(probas, axis=0), f"ensemble({'+'.join(names)})"
    clf = _load_model(model_name, "main")
    if clf is None:
        raise FileNotFoundError(
            f"main_{model_name}.joblib not found; run `python main.py train` first."
        )
    proba_list = clf.predict_proba(X_next)  # list of 49 arrays (1, 2)
    return np.array([p[0, 1] for p in proba_list]), model_name


def _supporting_stats_extra(raw: pd.DataFrame) -> dict[int, dict]:
    extra = raw["extra"].to_numpy()
    T = len(extra)
    last_seen: dict[int, int] = {}
    for i, v in enumerate(extra):
        last_seen[int(v)] = i
    freq = pd.Series(extra).value_counts()
    recent30 = pd.Series(extra[-30:]).value_counts()
    stats = {}
    for n in POOL:
        total = int(freq.get(n, 0))
        stats[n] = {
            "total": total,
            "pct": total / T * 100,
            "gap": T - 1 - last_seen[n] if n in last_seen else T,
            "recent30": int(recent30.get(n, 0)),
        }
    return stats


def _supporting_stats_main(raw: pd.DataFrame) -> dict[int, dict]:
    num_cols = [f"n{i}" for i in range(1, 7)]
    T = len(raw)
    all_main = raw[num_cols].to_numpy().flatten()
    freq = pd.Series(all_main).value_counts()
    recent30_flat = raw[num_cols].iloc[-30:].to_numpy().flatten()
    recent30 = pd.Series(recent30_flat).value_counts()
    stats = {}
    for n in POOL:
        total = int(freq.get(n, 0))
        stats[n] = {
            "total": total,
            "pct": total / (T * 6) * 100,
            "recent30": int(recent30.get(n, 0)),
        }
    return stats


def _check_model_staleness(model_name: str, raw: pd.DataFrame, both: bool = True) -> None:
    """Warn if saved models are stale or were trained with a held-out test split."""
    import json
    from models import MODELS_DIR, TEST_FRACTION

    latest_draw = raw["date"].iloc[-1]
    warned = False

    targets = (("extra", EXTRA_MODEL_FILES), ("main", MAIN_MODEL_FILES)) if both else (("extra", EXTRA_MODEL_FILES),)
    for target, files in targets:
        names = _available_models(target) if model_name == "ensemble" else [model_name]
        for n in names:
            path = files.get(n)
            if not (path and path.exists()):
                continue

            # Check if trained with full data or a held-out split
            metrics_path = MODELS_DIR / f"{target}_{n}_metrics.json"
            trained_full = False
            if metrics_path.exists():
                meta = json.loads(metrics_path.read_text())
                trained_full = meta.get("trained_on") == "full"

            # Count draws that occurred after the model file was last written
            trained_at = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").tz_localize(None)
            draws_since = int((raw["date"] > trained_at).sum())

            if not trained_full:
                holdout = int(len(raw) * TEST_FRACTION)
                print(
                    f"  WARNING [{target}/{n}]: trained with a {TEST_FRACTION:.0%} test split — "
                    f"the most recent ~{holdout} draws were held out and NOT used in training."
                )
                warned = True
            elif draws_since > 0:
                print(
                    f"  WARNING [{target}/{n}]: {draws_since} draw(s) since last retrain "
                    f"(latest draw: {latest_draw.date()}) are not in the training data."
                )
                warned = True

    if warned:
        print("  → Run `python main.py update` to scrape + retrain on all draws + predict.\n")


def _apply_last_draw_penalty(
    proba: np.ndarray, raw: pd.DataFrame, factor: float, lookback: int = 3
) -> tuple[np.ndarray, list[tuple[int, float]]]:
    """Penalise the last `lookback` extra numbers with a linearly fading factor.

    The most recent draw gets `factor`; each older draw is linearly interpolated
    toward 1.0 (no penalty) in equal steps.  E.g. factor=0.5, lookback=3:
        t-1 → 0.50,  t-2 → 0.67,  t-3 → 0.83
    """
    extras = raw["extra"].iloc[-lookback:].tolist()
    proba = proba.copy()
    penalized: list[tuple[int, float]] = []
    n = len(extras)
    for i, extra in enumerate(reversed(extras)):
        # i=0 is the most recent; linearly fade toward 1.0
        f = factor + (1.0 - factor) * i / lookback
        num = int(extra)
        proba[num - 1] *= f
        penalized.append((num, round(f, 4)))
    total = proba.sum()
    if total > 0:
        proba /= total
    return proba, penalized


_REPORT_CUTOFF_RE = re.compile(r"History:\s*\d+\s*draws through (\d{4}-\d{2}-\d{2})")
_REPORT_ROW_RE = re.compile(
    r"^\s*(\d{1,2})\s+(\d{1,2})\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+\d+\s+\d+\s*$", re.M
)


def _echo_scores_for_month(year: int, month: int) -> dict[int, float]:
    """Sum each number's raw confidence across every saved report cut off in year-month.

    This "echoes" whatever the model was recommending two months ago — it carries
    no new predictive information (draws are independent), it's a labelled wildcard
    heuristic, not a claim about the model's confidence in this draw.
    """
    scores: dict[int, float] = {}
    if not REPORTS_DIR.exists():
        return scores
    for f in sorted(REPORTS_DIR.glob("predict_*.txt")):
        text = f.read_text(encoding="utf-8")
        m = _REPORT_CUTOFF_RE.search(text)
        if not m:
            continue
        cy, cm, _ = m.group(1).split("-")
        if int(cy) != year or int(cm) != month:
            continue
        for rank, num, conf in _REPORT_ROW_RE.findall(text)[:10]:
            n = int(num)
            scores[n] = scores.get(n, 0.0) + float(conf)
    return scores


def predict(model_name: str = "xgb", top_extra: int = 10, top_main: int = 10,
            both: bool = False,
            penalty: float = 0.0, penalize_last: float | None = None,
            wildcard: bool = False, date: str | None = None) -> None:
    row, next_date, raw = _next_draw_feature_row(override_date=date)
    _check_model_staleness(model_name, raw, both=both)
    X_next = _align_features(row)

    extra_proba, extra_used = _get_extra_proba(model_name, X_next)

    if 0.0 < penalty < 1.0:
        extra_proba = _apply_prediction_penalty(extra_proba, strength=penalty)

    last_extra_penalized: list[tuple[int, float]] | None = None
    if penalize_last is not None:
        extra_proba, last_extra_penalized = _apply_last_draw_penalty(extra_proba, raw, factor=penalize_last)

    extra_order = np.argsort(-extra_proba)[:top_extra]

    extra_stats = _supporting_stats_extra(raw)

    if both:
        main_proba, main_used = _get_main_proba(model_name, X_next)
        main_order = np.argsort(-main_proba)[:top_main]
        main_stats = _supporting_stats_main(raw)
    else:
        main_proba, main_used, main_order, main_stats = None, None, None, None

    wildcard_info = None
    if wildcard:
        target = next_date.normalize().replace(day=1) - pd.DateOffset(months=2)
        echo = _echo_scores_for_month(target.year, target.month)
        excluded = {int(i) + 1 for i in extra_order}
        ranked_echo = sorted(
            ((n, s) for n, s in echo.items() if n not in excluded),
            key=lambda item: -item[1],
        )[:5]
        wildcard_info = {"month": f"{target.year}-{target.month:02d}", "picks": ranked_echo}

    en = _render_combined_report(
        extra_order, extra_proba, extra_stats,
        main_order, main_proba, main_stats,
        next_date, raw, extra_used, main_used,
        lang="en", penalty=penalty,
        last_extra_penalized=last_extra_penalized, penalize_last_factor=penalize_last,
        wildcard_info=wildcard_info,
    )
    zh = _render_combined_report(
        extra_order, extra_proba, extra_stats,
        main_order, main_proba, main_stats,
        next_date, raw, extra_used, main_used,
        lang="zh", penalty=penalty,
        last_extra_penalized=last_extra_penalized, penalize_last_factor=penalize_last,
        wildcard_info=wildcard_info,
    )

    print(en)
    print("\n" + "=" * 60 + "\n")
    print(zh)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS_DIR / f"predict_{stamp}.txt"
    out_path.write_text(
        en + "\n\n" + "=" * 60 + "\n\n" + zh + "\n",
        encoding="utf-8",
    )
    print(f"\nReport saved to {out_path}")


def _render_combined_report(
    extra_order: np.ndarray,
    extra_proba: np.ndarray,
    extra_stats: dict[int, dict],
    main_order: np.ndarray | None,
    main_proba: np.ndarray | None,
    main_stats: dict[int, dict] | None,
    next_date: pd.Timestamp,
    raw: pd.DataFrame,
    extra_used: str,
    main_used: str | None,
    lang: str,
    penalty: float = 0.0,
    last_extra_penalized: list[tuple[int, float]] | None = None,
    penalize_last_factor: float | None = None,
    wildcard_info: dict | None = None,
) -> str:
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    sep = "=" * 60
    model_order = extra_order[:5] if wildcard_info is not None else extra_order
    total_conf = extra_proba.sum()
    total_main_conf = main_proba.sum() if main_proba is not None else None

    def model_row(rank: int, idx) -> str:
        n = int(idx) + 1
        s = extra_stats[n]
        conf_pct = extra_proba[idx] / total_conf * 100
        return (
            f"{rank:<5}{n:<6}{extra_proba[idx]:<12.4f}"
            f"{conf_pct:<10.2f}{s['pct']:<10.2f}{s['gap']:<6}{s['recent30']}"
        )

    def main_row(rank: int, idx) -> str:
        n = int(idx) + 1
        s = main_stats[n]
        conf_pct = main_proba[idx] / total_main_conf * 100
        return (
            f"{rank:<5}{n:<6}{main_proba[idx]:<12.4f}"
            f"{conf_pct:<10.2f}{s['pct']:<10.2f}{s['recent30']}"
        )

    if lang == "en":
        w(sep)
        w(f"Mark Six Prediction — {next_date.date()}")
        w(f"History: {len(raw)} draws through {raw['date'].iloc[-1].date()}")
        model_line = f"Model: extra={extra_used}"
        if main_used is not None:
            model_line += f"  main={main_used}"
        w(model_line)
        if penalty > 0.0:
            w(f"Recency penalty: {penalty} (discounts numbers drawn recently as extra)")
        if last_extra_penalized is not None:
            parts = ", ".join(f"{num}×{f}" for num, f in last_extra_penalized)
            w(f"Last-draw penalty: {parts} (linear, renormalised)")
        w(sep)
        w()
        w(f"TOP {len(model_order)} SUGGESTED EXTRA NUMBERS (model)")
        w(f"{'Rank':<5}{'Num':<6}{'Confidence':<12}{'Conf%':<10}{'HistPct':<10}{'Gap':<6}{'Last30'}")
        w("-" * 55)
        for rank, idx in enumerate(model_order, start=1):
            w(model_row(rank, idx))
        w()
        w(f"Uniform baseline: extra=1/49={1/49:.4f}")
        w("Lottery draws are independent random events.")
        if main_order is not None:
            w()
            w(sep)
            w(f"TOP {len(main_order)} SUGGESTED MAIN NUMBERS (model)")
            w(f"{'Rank':<5}{'Num':<6}{'Confidence':<12}{'Conf%':<10}{'HistPct':<10}{'Last30'}")
            w("-" * 55)
            for rank, idx in enumerate(main_order, start=1):
                w(main_row(rank, idx))
            w()
            w(f"Uniform baseline: main=6/49={6/49:.4f}")
            w("Lottery draws are independent random events.")
        if wildcard_info is not None:
            w()
            w(sep)
            w(
                f"WILDCARD {len(wildcard_info['picks'])} — echo of {wildcard_info['month']} "
                "predictions (for fun, not a model signal)"
            )
            w(
                "These numbers ranked highest by summed confidence across all reports "
                f"issued in {wildcard_info['month']}, excluding anything already in the "
                "model's top list above. Draws are independent — this recycles old picks, "
                "it does not predict this draw."
            )
            w(f"{'Rank':<5}{'Num':<6}{'EchoScore':<12}{'HistPct':<10}{'Gap':<6}{'Last30'}")
            w("-" * 55)
            for rank, (n, score) in enumerate(wildcard_info["picks"], start=1):
                s = extra_stats[n]
                w(
                    f"{rank:<5}{n:<6}{score:<12.4f}"
                    f"{s['pct']:<10.2f}{s['gap']:<6}{s['recent30']}"
                )
    else:
        w(sep)
        w(f"马克六号预测 — {next_date.date()}")
        w(f"历史数据: {len(raw)} 期,截至 {raw['date'].iloc[-1].date()}")
        model_line = f"模型: 特别号={extra_used}"
        if main_used is not None:
            model_line += f"  正码={main_used}"
        w(model_line)
        if penalty > 0.0:
            w(f"近期惩罚系数: {penalty}（对近期出现为特别号的号码降权）")
        if last_extra_penalized is not None:
            parts = "、".join(f"{num}×{f}" for num, f in last_extra_penalized)
            w(f"近期惩罚: {parts}（线性衰减，再归一化）")
        w(sep)
        w()
        w(f"特别号推荐前 {len(model_order)} 名（模型）")
        w(f"{'排名':<5}{'号码':<6}{'置信度':<12}{'置信%':<10}{'历史占比':<10}{'间隔':<6}{'近30期'}")
        w("-" * 55)
        for rank, idx in enumerate(model_order, start=1):
            w(model_row(rank, idx))
        w()
        w(f"均匀基准: 特别号=1/49={1/49:.4f}")
        w("彩票开奖为独立随机事件,仅供参考。")
        if main_order is not None:
            w()
            w(sep)
            w(f"正码推荐前 {len(main_order)} 名（模型）")
            w(f"{'排名':<5}{'号码':<6}{'置信度':<12}{'置信%':<10}{'历史占比':<10}{'近30期'}")
            w("-" * 55)
            for rank, idx in enumerate(main_order, start=1):
                w(main_row(rank, idx))
            w()
            w(f"均匀基准: 正码=6/49={6/49:.4f}")
            w("彩票开奖为独立随机事件,仅供参考。")
        if wildcard_info is not None:
            w()
            w(sep)
            w(
                f"彩蛋 {len(wildcard_info['picks'])} 个 — 回声自 {wildcard_info['month']} "
                "的预测（纯属娱乐，非模型信号）"
            )
            w(
                "这些号码是在排除模型本期前列号码后，按"
                f"{wildcard_info['month']}期间所有报告置信度总和排序取前列。"
                "彩票开奖为独立随机事件——此列表只是重现旧预测，并非对本期的预测。"
            )
            w(f"{'排名':<5}{'号码':<6}{'回声分':<12}{'历史占比':<10}{'间隔':<6}{'近30期'}")
            w("-" * 55)
            for rank, (n, score) in enumerate(wildcard_info["picks"], start=1):
                s = extra_stats[n]
                w(
                    f"{rank:<5}{n:<6}{score:<12.4f}"
                    f"{s['pct']:<10.2f}{s['gap']:<6}{s['recent30']}"
                )

    return "\n".join(lines)


if __name__ == "__main__":
    predict()

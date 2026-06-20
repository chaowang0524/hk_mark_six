"""Predict the next Mark Six draw: 6 main numbers + 1 extra number.

Builds the lagged feature row for a hypothetical next draw, runs it
through trained models (or an ensemble average of all available models),
and reports a recommended ticket plus ranked candidate tables with
supporting statistics.
"""

from __future__ import annotations

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


def _next_draw_feature_row() -> tuple[pd.Series, pd.Timestamp, pd.DataFrame]:
    """Return (feature row, predicted draw date, raw history)."""
    raw = load_raw()
    last_date = raw["date"].iloc[-1]
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


def _check_model_staleness(model_name: str, raw: pd.DataFrame) -> None:
    """Warn if saved models are stale or were trained with a held-out test split."""
    import json
    from models import MODELS_DIR, TEST_FRACTION

    names = _available_models("extra") if model_name == "ensemble" else [model_name]
    latest_draw = raw["date"].iloc[-1]
    warned = False

    for n in names:
        path = EXTRA_MODEL_FILES.get(n)
        if not (path and path.exists()):
            continue

        # Check if trained with full data or a held-out split
        metrics_path = MODELS_DIR / f"extra_{n}_metrics.json"
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
                f"  WARNING [{n}]: trained with a {TEST_FRACTION:.0%} test split — "
                f"the most recent ~{holdout} draws were held out and NOT used in training."
            )
            warned = True
        elif draws_since > 0:
            print(
                f"  WARNING [{n}]: {draws_since} draw(s) since last retrain "
                f"(latest draw: {latest_draw.date()}) are not in the training data."
            )
            warned = True

    if warned:
        print("  → Run `python main.py update` to scrape + retrain on all draws + predict.\n")


def _apply_last_draw_penalty(proba: np.ndarray, raw: pd.DataFrame, factor: float) -> tuple[np.ndarray, int]:
    """Scale the last draw's extra number probability by factor, renormalise, return (proba, number)."""
    last_extra = int(raw["extra"].iloc[-1])
    proba = proba.copy()
    proba[last_extra - 1] *= factor
    total = proba.sum()
    if total > 0:
        proba /= total
    return proba, last_extra


def predict(model_name: str = "xgb", top_extra: int = 10, penalty: float = 0.0,
            penalize_last: float | None = None) -> None:
    row, next_date, raw = _next_draw_feature_row()
    _check_model_staleness(model_name, raw)
    X_next = _align_features(row)

    extra_proba, extra_used = _get_extra_proba(model_name, X_next)

    if 0.0 < penalty < 1.0:
        extra_proba = _apply_prediction_penalty(extra_proba, strength=penalty)

    last_extra_penalized: int | None = None
    if penalize_last is not None:
        extra_proba, last_extra_penalized = _apply_last_draw_penalty(extra_proba, raw, factor=penalize_last)

    extra_order = np.argsort(-extra_proba)[:top_extra]

    extra_stats = _supporting_stats_extra(raw)

    en = _render_combined_report(
        extra_order, extra_proba, extra_stats,
        next_date, raw, extra_used,
        lang="en", penalty=penalty,
        last_extra_penalized=last_extra_penalized, penalize_last_factor=penalize_last,
    )
    zh = _render_combined_report(
        extra_order, extra_proba, extra_stats,
        next_date, raw, extra_used,
        lang="zh", penalty=penalty,
        last_extra_penalized=last_extra_penalized, penalize_last_factor=penalize_last,
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
    next_date: pd.Timestamp,
    raw: pd.DataFrame,
    extra_used: str,
    lang: str,
    penalty: float = 0.0,
    last_extra_penalized: int | None = None,
    penalize_last_factor: float | None = None,
) -> str:
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    sep = "=" * 60

    if lang == "en":
        w(sep)
        w(f"Mark Six Prediction — {next_date.date()}")
        w(f"History: {len(raw)} draws through {raw['date'].iloc[-1].date()}")
        w(f"Model: {extra_used}")
        if penalty > 0.0:
            w(f"Recency penalty: {penalty} (discounts numbers drawn recently as extra)")
        if last_extra_penalized is not None:
            w(f"Last-draw penalty: extra number {last_extra_penalized} scaled by {penalize_last_factor} (then renormalised)")
        w(sep)
        w()
        w(f"TOP {len(extra_order)} SUGGESTED EXTRA NUMBERS")
        w(f"{'Rank':<5}{'Num':<6}{'Confidence':<12}{'Conf%':<10}{'HistPct':<10}{'Gap':<6}{'Last30'}")
        w("-" * 55)
        total_conf = extra_proba.sum()
        for rank, idx in enumerate(extra_order, start=1):
            n = int(idx) + 1
            s = extra_stats[n]
            conf_pct = extra_proba[idx] / total_conf * 100
            w(
                f"{rank:<5}{n:<6}{extra_proba[idx]:<12.4f}"
                f"{conf_pct:<10.2f}{s['pct']:<10.2f}{s['gap']:<6}{s['recent30']}"
            )
        w()
        w(f"Uniform baseline: extra=1/49={1/49:.4f}")
        w("Lottery draws are independent random events.")
    else:
        w(sep)
        w(f"马克六号预测 — {next_date.date()}")
        w(f"历史数据: {len(raw)} 期,截至 {raw['date'].iloc[-1].date()}")
        w(f"模型: {extra_used}")
        if penalty > 0.0:
            w(f"近期惩罚系数: {penalty}（对近期出现为特别号的号码降权）")
        if last_extra_penalized is not None:
            w(f"上期惩罚: 上期特别号 {last_extra_penalized} 概率乘以 {penalize_last_factor}（再归一化）")
        w(sep)
        w()
        w(f"特别号推荐前 {len(extra_order)} 名")
        w(f"{'排名':<5}{'号码':<6}{'置信度':<12}{'置信%':<10}{'历史占比':<10}{'间隔':<6}{'近30期'}")
        w("-" * 55)
        total_conf = extra_proba.sum()
        for rank, idx in enumerate(extra_order, start=1):
            n = int(idx) + 1
            s = extra_stats[n]
            conf_pct = extra_proba[idx] / total_conf * 100
            w(
                f"{rank:<5}{n:<6}{extra_proba[idx]:<12.4f}"
                f"{conf_pct:<10.2f}{s['pct']:<10.2f}{s['gap']:<6}{s['recent30']}"
            )
        w()
        w(f"均匀基准: 特别号=1/49={1/49:.4f}")
        w("彩票开奖为独立随机事件,仅供参考。")

    return "\n".join(lines)


if __name__ == "__main__":
    predict()

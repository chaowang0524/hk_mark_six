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


def predict(model_name: str = "xgb", top_extra: int = 10, top_main: int = 10) -> None:
    row, next_date, raw = _next_draw_feature_row()
    X_next = _align_features(row)

    extra_proba, extra_used = _get_extra_proba(model_name, X_next)
    main_proba, main_used = _get_main_proba(model_name, X_next)

    extra_order = np.argsort(-extra_proba)[:top_extra]
    main_order = np.argsort(-main_proba)[:top_main]

    extra_stats = _supporting_stats_extra(raw)
    main_stats = _supporting_stats_main(raw)

    en = _render_combined_report(
        main_order, main_proba, main_stats,
        extra_order, extra_proba, extra_stats,
        next_date, raw, main_used, extra_used,
        lang="en",
    )
    zh = _render_combined_report(
        main_order, main_proba, main_stats,
        extra_order, extra_proba, extra_stats,
        next_date, raw, main_used, extra_used,
        lang="zh",
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
    main_order: np.ndarray,
    main_proba: np.ndarray,
    main_stats: dict[int, dict],
    extra_order: np.ndarray,
    extra_proba: np.ndarray,
    extra_stats: dict[int, dict],
    next_date: pd.Timestamp,
    raw: pd.DataFrame,
    main_used: str,
    extra_used: str,
    lang: str,
) -> str:
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    sep = "=" * 60

    if lang == "en":
        w(sep)
        w(f"Mark Six Prediction — {next_date.date()}")
        w(f"History: {len(raw)} draws through {raw['date'].iloc[-1].date()}")
        w(f"Models: main={main_used}  extra={extra_used}")
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
        w(f"---------- MAIN NUMBERS (top {len(main_order)} candidates) ----------")
        w(f"{'Rank':<5}{'Num':<6}{'P(appear)':<12}{'HistPct':<10}{'Last30'}")
        w("-" * 42)
        for rank, idx in enumerate(main_order, start=1):
            n = int(idx) + 1
            s = main_stats[n]
            w(f"{rank:<5}{n:<6}{main_proba[idx]:<12.4f}{s['pct']:<10.2f}{s['recent30']}")
        w()
        w(
            f"Uniform baseline: main P(appear)=6/49={6/49:.4f},  "
            f"extra=1/49={1/49:.4f}"
        )
        w("Lottery draws are independent random events.")
    else:
        w(sep)
        w(f"马克六号预测 — {next_date.date()}")
        w(f"历史数据: {len(raw)} 期,截至 {raw['date'].iloc[-1].date()}")
        w(f"模型: 正码={main_used}  特别号={extra_used}")
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
        w(f"---------- 正码候选(前 {len(main_order)} 名) ----------")
        for rank, idx in enumerate(main_order, start=1):
            n = int(idx) + 1
            s = main_stats[n]
            w(
                f"第 {rank} 名  号码 {n:<2}  "
                f"出现概率 {main_proba[idx]:.4f}  "
                f"历史占比 {s['pct']:.2f}%  "
                f"近30期 {s['recent30']} 次"
            )
        w()
        w(f"均匀基准: 正码出现概率=6/49={6/49:.4f},  特别号=1/49={1/49:.4f}")
        w("彩票开奖为独立随机事件,仅供参考。")

    return "\n".join(lines)


if __name__ == "__main__":
    predict()

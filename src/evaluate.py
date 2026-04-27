"""Backtesting and metrics comparison for trained Mark Six models.

Evaluates both sub-problems on the same chronological test split:
  - Extra number: top-1/3/5/10 accuracy vs uniform baseline (k/49)
  - Main numbers: hit-rate@k (k=1..6) vs uniform baseline (k*6/49)

Writes a markdown report plus per-model CSV files.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)

from models import (
    MODELS_DIR,
    TEST_FRACTION,
    _chronological_split,
    _hit_rate_at_k,
    _load_features_extra,
    _load_features_main,
    _main_predict_proba,
)
from preprocessing import POOL

REPORT_PATH = Path("data/reports/evaluation.md")
CONFUSION_DIR = Path("data/reports/confusion")

EXTRA_TOP_K = (1, 3, 5, 10)
UNIFORM_EXTRA_BASELINE = {k: k / 49 for k in EXTRA_TOP_K}

MAIN_TOP_K = (1, 2, 3, 4, 5, 6)
UNIFORM_MAIN_BASELINE = {k: k * 6 / 49 for k in MAIN_TOP_K}


def _load_trained_model(name: str, target: str):
    path = MODELS_DIR / f"{target}_{name}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def _discover_models(target: str) -> list[str]:
    return [
        name
        for name in ("rf", "xgb", "lr")
        if (MODELS_DIR / f"{target}_{name}.joblib").exists()
    ]


def _top_k_scores(proba: np.ndarray, y: np.ndarray) -> dict[int, float]:
    return {
        k: top_k_accuracy_score(y, proba, labels=range(49), k=k)
        for k in EXTRA_TOP_K
    }


def _cumulative_topk(proba: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    order = np.argsort(-proba, axis=1)
    rows = []
    hits = {k: 0 for k in EXTRA_TOP_K}
    for i, true in enumerate(y, start=1):
        for k in EXTRA_TOP_K:
            if true in order[i - 1, :k]:
                hits[k] += 1
        rows.append({"n": i, **{f"top_{k}": hits[k] / i for k in EXTRA_TOP_K}})
    return pd.DataFrame(rows)


def evaluate() -> Path:
    extra_names = _discover_models("extra")
    main_names = _discover_models("main")

    if not extra_names and not main_names:
        raise FileNotFoundError(
            f"No trained models found in {MODELS_DIR}. Run `python main.py train` first."
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    w("# Mark Six Model Evaluation")
    w()

    # ------------------------------------------------------------------ #
    # Extra-number evaluation
    # ------------------------------------------------------------------ #
    if extra_names:
        X, y = _load_features_extra()
        _, X_test, _, y_test = _chronological_split(X, y)
        y_test_arr = y_test.to_numpy()

        w("## Extra Number — Top-k Accuracy")
        w()
        w(
            f"Test window: last {TEST_FRACTION:.0%} of draws "
            f"({len(X_test)} rows). Classes: 49. Uniform baseline = k/49."
        )
        w()

        per_extra: dict[str, dict] = {}
        for name in extra_names:
            clf = _load_trained_model(name, "extra")
            if clf is None:
                continue
            proba = clf.predict_proba(X_test)
            preds = np.argmax(proba, axis=1)
            per_extra[name] = {
                "topk": _top_k_scores(proba, y_test_arr),
                "preds": preds,
                "proba": proba,
            }

        header = "| Model | " + " | ".join(f"top-{k}" for k in EXTRA_TOP_K) + " |"
        sep = "|" + "-------|" * (len(EXTRA_TOP_K) + 1)
        w(header)
        w(sep)
        w(
            "| uniform | "
            + " | ".join(f"{UNIFORM_EXTRA_BASELINE[k]:.4f}" for k in EXTRA_TOP_K)
            + " |"
        )
        for name, res in per_extra.items():
            w(
                f"| {name} | "
                + " | ".join(f"{res['topk'][k]:.4f}" for k in EXTRA_TOP_K)
                + " |"
            )
        w()

        # Per-number precision / recall
        w("### Per-Number Precision & Recall")
        w()
        labels = list(range(49))
        for name, res in per_extra.items():
            prec = precision_score(
                y_test_arr, res["preds"], labels=labels, average=None, zero_division=0
            )
            rec = recall_score(
                y_test_arr, res["preds"], labels=labels, average=None, zero_division=0
            )
            support = np.bincount(y_test_arr, minlength=49)
            w(f"#### {name}")
            w()
            w("| Number | Support | Precision | Recall |")
            w("|--------|---------|-----------|--------|")
            for idx in labels:
                if support[idx] == 0 and prec[idx] == 0:
                    continue
                w(
                    f"| {idx + 1:<6} | {support[idx]:<7} | "
                    f"{prec[idx]:.3f}     | {rec[idx]:.3f}  |"
                )
            w()

        # Confusion matrices
        w("### Confusion Matrices")
        w()
        for name, res in per_extra.items():
            cm = confusion_matrix(y_test_arr, res["preds"], labels=labels)
            cm_df = pd.DataFrame(cm, index=POOL, columns=POOL)
            path = CONFUSION_DIR / f"extra_{name}.csv"
            cm_df.to_csv(path)
            w(f"- **{name}** → `{path}`")
        w()

        # Cumulative walk-forward
        w("### Cumulative Top-k on Test Window")
        w()
        w("| Model | " + " | ".join(f"top-{k} final" for k in EXTRA_TOP_K) + " |")
        w("|" + "-------|" * (len(EXTRA_TOP_K) + 1))
        for name, res in per_extra.items():
            cum = _cumulative_topk(res["proba"], y_test_arr)
            cum.to_csv(CONFUSION_DIR.parent / f"cumulative_extra_{name}.csv", index=False)
            final = cum.iloc[-1]
            w(
                f"| {name} | "
                + " | ".join(f"{final[f'top_{k}']:.4f}" for k in EXTRA_TOP_K)
                + " |"
            )
        w()

        w("### Verdict vs Uniform (Extra)")
        w()
        for name, res in per_extra.items():
            beats = [k for k in EXTRA_TOP_K if res["topk"][k] > UNIFORM_EXTRA_BASELINE[k]]
            w(f"- **{name}** beats uniform at: {beats if beats else 'none'}")
        w()

    # ------------------------------------------------------------------ #
    # Main-number evaluation
    # ------------------------------------------------------------------ #
    if main_names:
        X_m, Y_m = _load_features_main()
        _, X_m_test, _, Y_m_test = _chronological_split(X_m, Y_m)
        Y_m_arr = Y_m_test.to_numpy()

        w("## Main Numbers — Hit-Rate@k")
        w()
        w(
            f"Test window: {len(X_m_test)} rows. "
            "Metric: mean number of actual main numbers captured when selecting top-k candidates. "
            "Uniform baseline = k × 6/49."
        )
        w()

        header = "| Model | " + " | ".join(f"@{k}" for k in MAIN_TOP_K) + " |"
        sep = "|" + "-------|" * (len(MAIN_TOP_K) + 1)
        w(header)
        w(sep)
        w(
            "| uniform | "
            + " | ".join(f"{UNIFORM_MAIN_BASELINE[k]:.4f}" for k in MAIN_TOP_K)
            + " |"
        )

        for name in main_names:
            clf = _load_trained_model(name, "main")
            if clf is None:
                continue
            proba_m = _main_predict_proba(clf, X_m_test)
            hit_rates = {k: _hit_rate_at_k(proba_m, Y_m_arr, k) for k in MAIN_TOP_K}
            w(
                f"| {name} | "
                + " | ".join(f"{hit_rates[k]:.4f}" for k in MAIN_TOP_K)
                + " |"
            )
        w()

        w("### Verdict vs Uniform (Main)")
        w()
        for name in main_names:
            clf = _load_trained_model(name, "main")
            if clf is None:
                continue
            proba_m = _main_predict_proba(clf, X_m_test)
            beats = [
                k
                for k in MAIN_TOP_K
                if _hit_rate_at_k(proba_m, Y_m_arr, k) > UNIFORM_MAIN_BASELINE[k]
            ]
            w(f"- **{name}** beats uniform at: {beats if beats else 'none'}")
        w()

    w("---")
    w(
        "Lottery draws are designed to be independent and uniform; "
        "deviations from the uniform baseline are expected from finite-sample noise."
    )
    w()

    REPORT_PATH.write_text("\n".join(lines))
    print(f"Evaluation report → {REPORT_PATH}")
    return REPORT_PATH


if __name__ == "__main__":
    evaluate()

"""ML models for Mark Six 7-number prediction.

Two sub-problems:
  - Extra number: multi-class classification (49 classes).
  - Main numbers: multi-label binary classification (49 binary outputs),
    using MultiOutputClassifier; top-6 by probability at inference.

Both use a chronological train/test split (no shuffle, no leakage).
"""

from __future__ import annotations

import json
from pathlib import Path

from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from preprocessing import FEATURES_PATH, POOL, RAW_PATH, preprocess

MODELS_DIR = Path("data/models")
TEST_FRACTION = 0.05

EXTRA_MODEL_REGISTRY: dict[str, Callable] = {
    "rf": lambda: RandomForestClassifier(
        n_estimators=300, max_depth=15, n_jobs=-1, random_state=42
    ),
    "xgb": lambda: xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=49,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    ),
    "lr": lambda: CalibratedClassifierCV(
        estimator=Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        solver="lbfgs",
                        random_state=42,
                        C=0.01,
                    ),
                ),
            ]
        ),
        method="isotonic",
        cv=5,
    ),
}

MAIN_MODEL_REGISTRY: dict[str, Callable] = {
    "rf": lambda: MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=300, max_depth=15, n_jobs=-1, random_state=42
        )
    ),
    "xgb": lambda: MultiOutputClassifier(
        xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        )
    ),
    "lr": lambda: MultiOutputClassifier(
        Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        )
    ),
}


def _check_schema(df: pd.DataFrame) -> None:
    if "target_extra" not in df.columns:
        if "target" in df.columns:
            raise RuntimeError(
                "features.csv uses the old schema (column 'target'). "
                "Run `python main.py preprocess` to rebuild."
            )
        raise RuntimeError(
            "features.csv is missing required columns. "
            "Run `python main.py preprocess`."
        )


def _maybe_rebuild_features() -> None:
    if not FEATURES_PATH.exists():
        print("features.csv not found; building from raw data...")
        preprocess()
    elif RAW_PATH.exists() and RAW_PATH.stat().st_mtime > FEATURES_PATH.stat().st_mtime:
        print("raw_draws.csv is newer than features.csv; rebuilding features...")
        preprocess()


def _load_features_extra() -> tuple[pd.DataFrame, pd.Series]:
    _maybe_rebuild_features()
    df = pd.read_csv(FEATURES_PATH)
    _check_schema(df)
    drop_cols = {"draw", "date", "target_extra"} | {
        c for c in df.columns if c.startswith("main_is_")
    }
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df["target_extra"] - 1  # 0-indexed for sklearn/xgb
    return X, y


def _load_features_main() -> tuple[pd.DataFrame, pd.DataFrame]:
    _maybe_rebuild_features()
    df = pd.read_csv(FEATURES_PATH)
    _check_schema(df)
    target_cols = [f"main_is_{k}" for k in POOL]
    drop_cols = {"draw", "date", "target_extra"} | set(target_cols)
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    Y = df[target_cols].astype(int)
    return X, Y


def _chronological_split(X, y):
    split = int(len(X) * (1 - TEST_FRACTION))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def _main_predict_proba(clf, X: pd.DataFrame) -> np.ndarray:
    """Returns shape (n_samples, 49) — P(number k in main draw) for k=1..49."""
    proba_list = clf.predict_proba(X)  # list of 49 arrays each (n, 2)
    return np.column_stack([p[:, 1] for p in proba_list])


def _hit_rate_at_k(proba_matrix: np.ndarray, Y_true: np.ndarray, k: int) -> float:
    """Mean true positives when selecting top-k numbers by probability."""
    top_k_idx = np.argsort(-proba_matrix, axis=1)[:, :k]
    hits = []
    for i in range(len(Y_true)):
        predicted_set = set(top_k_idx[i])
        actual_set = set(np.where(Y_true[i] == 1)[0])
        hits.append(len(predicted_set & actual_set))
    return float(np.mean(hits))


def train_extra_model(name: str, full: bool = False) -> None:
    if name not in EXTRA_MODEL_REGISTRY:
        print(f"Unknown extra model '{name}'; skipping.")
        return
    X, y = _load_features_extra()
    metrics_path = MODELS_DIR / f"extra_{name}_metrics.json"

    if full:
        print(
            f"\n[extra/{name}] Train: {len(X)} (full, no test set)  "
            f"Features: {X.shape[1]}"
        )
        clf = EXTRA_MODEL_REGISTRY[name]()
        clf.fit(X, y)
        joblib.dump(clf, MODELS_DIR / f"extra_{name}.joblib")
        metrics_path.write_text(json.dumps({"trained_on": "full", "n_train": len(X)}, indent=2))
        print("  no test metrics — run `python main.py evaluate` for backtest results.")
        return

    X_train, X_test, y_train, y_test = _chronological_split(X, y)
    print(
        f"\n[extra/{name}] Train: {len(X_train)}  Test: {len(X_test)}  "
        f"Features: {X.shape[1]}"
    )

    clf = EXTRA_MODEL_REGISTRY[name]()
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)
    preds = np.argmax(proba, axis=1)
    acc1 = accuracy_score(y_test, preds)
    topk = {k: top_k_accuracy_score(y_test, proba, labels=range(49), k=k) for k in [3, 5, 10]}

    joblib.dump(clf, MODELS_DIR / f"extra_{name}.joblib")

    metrics = {"top_1": acc1, **{f"top_{k}": v for k, v in topk.items()}}
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(
        f"  top-1={acc1:.4f}  "
        + "  ".join(f"top-{k}={topk[k]:.4f}" for k in [3, 5, 10])
    )


def train_main_model(name: str, full: bool = False) -> None:
    if name not in MAIN_MODEL_REGISTRY:
        print(f"Unknown main model '{name}'; skipping.")
        return
    X, Y = _load_features_main()
    metrics_path = MODELS_DIR / f"main_{name}_metrics.json"

    if full:
        print(
            f"\n[main/{name}] Train: {len(X)} (full, no test set)  "
            f"Features: {X.shape[1]}"
        )
        clf = MAIN_MODEL_REGISTRY[name]()
        clf.fit(X, Y)
        joblib.dump(clf, MODELS_DIR / f"main_{name}.joblib")
        metrics_path.write_text(json.dumps({"trained_on": "full", "n_train": len(X)}, indent=2))
        print("  no test metrics — run `python main.py evaluate` for backtest results.")
        return

    X_train, X_test, Y_train, Y_test = _chronological_split(X, Y)
    print(
        f"\n[main/{name}] Train: {len(X_train)}  Test: {len(X_test)}  "
        f"Features: {X.shape[1]}"
    )

    clf = MAIN_MODEL_REGISTRY[name]()
    clf.fit(X_train, Y_train)

    proba_matrix = _main_predict_proba(clf, X_test)
    hit6 = _hit_rate_at_k(proba_matrix, Y_test.to_numpy(), k=6)

    joblib.dump(clf, MODELS_DIR / f"main_{name}.joblib")

    metrics = {f"hit_rate_{k}": _hit_rate_at_k(proba_matrix, Y_test.to_numpy(), k=k) for k in [3, 6, 10]}
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"  hit-rate@6={hit6:.4f}  (uniform baseline={6*6/49:.4f})")


def train(
    model_name: str = "all",
    full: bool = False,
    on_step: Callable[[str, str], None] | None = None,
) -> None:
    """on_step(target, name), called right before each sub-model trains —
    e.g. on_step("extra", "xgb") — so a caller (e.g. the web UI) can show
    live progress through a potentially long run."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    names = list(EXTRA_MODEL_REGISTRY.keys()) if model_name == "all" else [model_name]

    for name in names:
        if on_step:
            on_step("extra", name)
        train_extra_model(name, full=full)
    for name in names:
        if on_step:
            on_step("main", name)
        train_main_model(name, full=full)

    X, _ = _load_features_extra()
    (MODELS_DIR / "feature_columns.json").write_text(json.dumps(list(X.columns)))
    print(f"\nModels saved to {MODELS_DIR}/")


if __name__ == "__main__":
    train()

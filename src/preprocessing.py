"""Feature engineering for Mark Six 7-number prediction.

All features must be strictly lagged: no information from the draw being
predicted (neither its winning numbers nor its extra number) may appear as
input. Targets for row t are the 6 main numbers and the extra number of draw
t; every feature column is computed from draws [0 .. t-1].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw_draws.csv")
FEATURES_PATH = Path("data/features.csv")

NUM_MIN = 1
NUM_MAX = 49
POOL = list(range(NUM_MIN, NUM_MAX + 1))
ROLLING_WINDOWS = (10, 30, 100)

# Chinese New Year (month, day) for each calendar year — used to assign draws
# that fall before CNY to the previous zodiac year.
_CNY_DATES: dict[int, tuple[int, int]] = {
    2008: (2, 7),  2009: (1, 26), 2010: (2, 14), 2011: (2, 3),
    2012: (1, 23), 2013: (2, 10), 2014: (1, 31), 2015: (2, 19),
    2016: (2, 8),  2017: (1, 28), 2018: (2, 16), 2019: (2, 5),
    2020: (1, 25), 2021: (2, 12), 2022: (2, 1),  2023: (1, 22),
    2024: (2, 10), 2025: (1, 29), 2026: (2, 17),
}


def load_raw() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"{RAW_PATH} missing; run `python main.py scrape` first.")
    df = pd.read_csv(RAW_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    num_cols = [f"n{i}" for i in range(1, 7)]
    valid = df[num_cols + ["extra"]].apply(
        lambda col: col.between(NUM_MIN, NUM_MAX)
    ).all(axis=1)
    df = df[valid].reset_index(drop=True)
    return df


def _rolling_count_matrix(presence: np.ndarray, window: int) -> np.ndarray:
    """presence shape (T, 49). Returns counts over the previous `window`
    draws (not including the current row)."""
    T = presence.shape[0]
    cs = np.vstack([np.zeros((1, presence.shape[1])), np.cumsum(presence, axis=0)])
    out = np.zeros_like(presence, dtype=float)
    for t in range(T):
        lo = max(0, t - window)
        out[t] = cs[t] - cs[lo]
    return out


def _gap_since_last(presence: np.ndarray) -> np.ndarray:
    """For each (row t, number k), the number of draws since k last appeared,
    using only rows [0 .. t-1]. If never seen, return t (count from start)."""
    T, K = presence.shape
    last_idx = np.full(K, -1, dtype=int)
    out = np.zeros((T, K), dtype=float)
    for t in range(T):
        out[t] = np.where(last_idx >= 0, t - 1 - last_idx, t)
        appeared = np.where(presence[t] > 0)[0]
        last_idx[appeared] = t
    return out


def _zodiac_year_index(date: pd.Timestamp) -> int:
    """0=Rat(2008) … 11=Pig, adjusted for Chinese New Year."""
    yr = date.year
    m, d = _CNY_DATES.get(yr, (2, 4))
    if (date.month, date.day) < (m, d):
        yr -= 1
    return (yr - 2008) % 12


def _zodiac_features(df: pd.DataFrame) -> pd.DataFrame:
    """Zodiac-group features (all strictly lagged).

    Number n belongs to zodiac group (n-1)%12.  Group 0 always maps to the
    current Chinese-zodiac-year's animal (the 5-number group: 1,13,25,37,49).
    """
    T = len(df)
    num_cols = [f"n{i}" for i in range(1, 7)]
    extras = df["extra"].to_numpy()
    mains = df[num_cols].to_numpy()

    zodiac_year = np.array([_zodiac_year_index(d) for d in df["date"]])

    extra_grp = (extras - 1) % 12        # shape (T,)
    main_grp = (mains - 1) % 12          # shape (T, 6)

    # Zodiac-group presence matrix for extras: shape (T, 12)
    extra_zodiac_presence = np.zeros((T, 12), dtype=float)
    extra_zodiac_presence[np.arange(T), extra_grp] = 1.0

    # Shift so row t only contains data from rows 0..t-1
    shifted = np.vstack([np.zeros((1, 12)), extra_zodiac_presence[:-1]])

    result: dict[str, np.ndarray] = {"zodiac_year": zodiac_year}

    for w in ROLLING_WINDOWS:
        zc = _rolling_count_matrix(shifted, w)
        for g in range(12):
            result[f"zodiac_extra_freq_w{w}_g{g}"] = zc[:, g]

    result["prev_extra_zodiac_grp"] = np.concatenate(
        [[np.nan], extra_grp[:-1].astype(float)]
    )

    main_zodiac_counts = np.zeros((T, 12), dtype=float)
    for t in range(T):
        for g in main_grp[t]:
            main_zodiac_counts[t, int(g)] += 1
    prev_main_zodiac = np.vstack([np.full((1, 12), np.nan), main_zodiac_counts[:-1]])
    for g in range(12):
        result[f"prev_main_zodiac_count_g{g}"] = prev_main_zodiac[:, g]

    return pd.DataFrame(result)


def _prev_draw_stats(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [f"n{i}" for i in range(1, 7)]
    nums = df[num_cols].to_numpy()
    sums = nums.sum(axis=1)
    odd_counts = (nums % 2 == 1).sum(axis=1)
    mins = nums.min(axis=1)
    maxs = nums.max(axis=1)
    sorted_nums = np.sort(nums, axis=1)
    consec = (np.diff(sorted_nums, axis=1) == 1).sum(axis=1)

    prev = pd.DataFrame(
        {
            "prev_main_sum": np.concatenate([[np.nan], sums[:-1]]),
            "prev_main_odd": np.concatenate([[np.nan], odd_counts[:-1]]),
            "prev_main_min": np.concatenate([[np.nan], mins[:-1]]),
            "prev_main_max": np.concatenate([[np.nan], maxs[:-1]]),
            "prev_main_consec": np.concatenate([[np.nan], consec[:-1]]),
            "prev_extra": np.concatenate([[np.nan], df["extra"].to_numpy()[:-1]]),
        }
    )
    return prev


def build_features(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = load_raw()
    T = len(df)

    extra = df["extra"].to_numpy()
    extra_presence = np.zeros((T, NUM_MAX), dtype=float)
    extra_presence[np.arange(T), extra - 1] = 1.0

    num_cols = [f"n{i}" for i in range(1, 7)]
    main_presence = np.zeros((T, NUM_MAX), dtype=float)
    for c in num_cols:
        vals = df[c].to_numpy() - 1
        main_presence[np.arange(T), vals] = 1.0

    # IMPORTANT: rolling counts for row t must exclude row t itself.
    shifted_extra = np.vstack([np.zeros((1, NUM_MAX)), extra_presence[:-1]])
    shifted_main = np.vstack([np.zeros((1, NUM_MAX)), main_presence[:-1]])

    feature_frames: list[pd.DataFrame] = []

    for w in ROLLING_WINDOWS:
        ec = _rolling_count_matrix(shifted_extra, w)
        feature_frames.append(
            pd.DataFrame(ec, columns=[f"extra_freq_w{w}_n{k}" for k in POOL])
        )
        mc = _rolling_count_matrix(shifted_main, w)
        feature_frames.append(
            pd.DataFrame(mc, columns=[f"main_freq_w{w}_n{k}" for k in POOL])
        )

    gap = _gap_since_last(shifted_extra)
    feature_frames.append(
        pd.DataFrame(gap, columns=[f"extra_gap_n{k}" for k in POOL])
    )

    feature_frames.append(_prev_draw_stats(df))
    feature_frames.append(_zodiac_features(df))

    feature_frames.append(
        pd.DataFrame(
            {
                "dow": df["date"].dt.dayofweek.to_numpy(),
                "month": df["date"].dt.month.to_numpy(),
                "year": df["date"].dt.year.to_numpy(),
            }
        )
    )

    X = pd.concat(feature_frames, axis=1)
    meta = df[["draw", "date"]].reset_index(drop=True)

    main_target_df = pd.DataFrame(
        main_presence.astype(int),
        columns=[f"main_is_{k}" for k in POOL],
    )

    out = pd.concat(
        [meta, X, main_target_df, df[["extra"]].rename(columns={"extra": "target_extra"})],
        axis=1,
    )
    out = out.dropna().reset_index(drop=True)
    return out


def preprocess() -> Path:
    features = build_features()
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_PATH, index=False)
    print(
        f"Preprocess complete: {len(features)} rows, "
        f"{features.shape[1]} columns → {FEATURES_PATH}\n"
        f"  Targets: target_extra (1 column) + main_is_1..main_is_49 (49 columns)"
    )
    return FEATURES_PATH


if __name__ == "__main__":
    preprocess()

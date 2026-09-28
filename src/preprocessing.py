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
EXTRA_LONG_WINDOWS = (300, 500)
MAIN_FREQ_WINDOWS = (10, 30, 100, 300, 500)
EWM_ALPHAS = (0.05,)
EXTRA_LAGS = tuple(range(1, 11))
MAIN_LAGS = tuple(range(1, 11))
WAS_MAIN_LAST_K = (1, 3, 5, 10)
MARKOV_WINDOW = 100
LAST_K_STATS = 10
TREND_SHORT_W = 10
TREND_LONG_W = 100
MAIN_DRAW_SIZE = 6

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


def _ewm_matrix(presence_shifted: np.ndarray, alpha: float) -> np.ndarray:
    return (
        pd.DataFrame(presence_shifted)
        .ewm(alpha=alpha, adjust=False)
        .mean()
        .to_numpy()
    )


def _markov_window(extras_1based: np.ndarray, window: int) -> np.ndarray:
    """For row t with j = extras[t-1], count of past transition pairs
    (extras[i], extras[i+1]) with extras[i] == j, broken out per next value k,
    over the last `window` pairs i ∈ [max(0, t-1-window) .. t-2]."""
    T = len(extras_1based)
    M = np.zeros((NUM_MAX, NUM_MAX), dtype=float)
    out = np.zeros((T, NUM_MAX), dtype=float)
    for t in range(T):
        if t >= 1:
            j = int(extras_1based[t - 1]) - 1
            out[t] = M[j]
        if t >= 1:
            ji = int(extras_1based[t - 1]) - 1
            ki = int(extras_1based[t]) - 1
            M[ji, ki] += 1
            old = t - 1 - window
            if old >= 0:
                jo = int(extras_1based[old]) - 1
                ko = int(extras_1based[old + 1]) - 1
                M[jo, ko] -= 1
    return out


def _main_markov_window(main_presence: np.ndarray, window: int) -> np.ndarray:
    """For row t, sum over the (up to 6) numbers that were main numbers in
    draw t-1 of each candidate's co-occurrence count as a *next-draw* main
    number, using only transitions from the last `window` consecutive
    draw-pairs i -> i+1 with i in [max(0, t-1-window) .. t-2]."""
    T, K = main_presence.shape
    M = np.zeros((K, K), dtype=float)
    out = np.zeros((T, K), dtype=float)
    for t in range(T):
        if t >= 1:
            prev_idx = np.where(main_presence[t - 1] > 0)[0]
            out[t] = M[prev_idx].sum(axis=0)
            cur_idx = np.where(main_presence[t] > 0)[0]
            M[np.ix_(prev_idx, cur_idx)] += 1
            old = t - 1 - window
            if old >= 0:
                old_prev_idx = np.where(main_presence[old] > 0)[0]
                old_cur_idx = np.where(main_presence[old + 1] > 0)[0]
                M[np.ix_(old_prev_idx, old_cur_idx)] -= 1
    return out


def _last_k_extra_stats(extras_1based: np.ndarray, k: int) -> pd.DataFrame:
    T = len(extras_1based)
    mean = np.full(T, np.nan)
    odd = np.full(T, np.nan)
    le25 = np.full(T, np.nan)
    for t in range(k, T):
        window = extras_1based[t - k : t]
        mean[t] = window.mean()
        odd[t] = (window % 2 == 1).mean()
        le25[t] = float((window <= 25).sum())
    return pd.DataFrame(
        {
            f"extra_last{k}_mean": mean,
            f"extra_last{k}_odd_ratio": odd,
            f"extra_last{k}_le25_count": le25,
        }
    )


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
    extra_freqs: dict[int, np.ndarray] = {}

    # Extra-number rolling frequency: short + long windows
    for w in ROLLING_WINDOWS + EXTRA_LONG_WINDOWS:
        ec = _rolling_count_matrix(shifted_extra, w)
        extra_freqs[w] = ec
        feature_frames.append(
            pd.DataFrame(ec, columns=[f"extra_freq_w{w}_n{k}" for k in POOL])
        )

    # All-time cumulative count (uses only past data because of shifted_extra)
    extra_all = np.cumsum(shifted_extra, axis=0)
    feature_frames.append(
        pd.DataFrame(extra_all, columns=[f"extra_freq_wall_n{k}" for k in POOL])
    )

    # Exponentially-weighted extra frequency
    for alpha in EWM_ALPHAS:
        tag = f"{int(round(alpha * 100)):02d}"
        ew = _ewm_matrix(shifted_extra, alpha)
        feature_frames.append(
            pd.DataFrame(ew, columns=[f"extra_ewm_a{tag}_n{k}" for k in POOL])
        )

    # Hot/cold trend: short-window rate minus long-window rate
    trend = extra_freqs[TREND_SHORT_W] / TREND_SHORT_W - extra_freqs[TREND_LONG_W] / TREND_LONG_W
    feature_frames.append(
        pd.DataFrame(trend, columns=[f"extra_trend_n{k}" for k in POOL])
    )

    # Z-score of long-window count vs. uniform-extra null hypothesis
    expected = TREND_LONG_W / NUM_MAX
    std = np.sqrt(TREND_LONG_W * (1.0 / NUM_MAX) * ((NUM_MAX - 1) / NUM_MAX))
    zscore = (extra_freqs[TREND_LONG_W] - expected) / std
    feature_frames.append(
        pd.DataFrame(zscore, columns=[f"extra_zscore_n{k}" for k in POOL])
    )

    # Main-number rolling frequency: short + long windows (mirrors extra treatment)
    main_freqs: dict[int, np.ndarray] = {}
    for w in MAIN_FREQ_WINDOWS:
        mc = _rolling_count_matrix(shifted_main, w)
        main_freqs[w] = mc
        feature_frames.append(
            pd.DataFrame(mc, columns=[f"main_freq_w{w}_n{k}" for k in POOL])
        )

    # All-time cumulative main-number count
    main_all = np.cumsum(shifted_main, axis=0)
    feature_frames.append(
        pd.DataFrame(main_all, columns=[f"main_freq_wall_n{k}" for k in POOL])
    )

    # Exponentially-weighted main-number frequency
    for alpha in EWM_ALPHAS:
        tag = f"{int(round(alpha * 100)):02d}"
        mew = _ewm_matrix(shifted_main, alpha)
        feature_frames.append(
            pd.DataFrame(mew, columns=[f"main_ewm_a{tag}_n{k}" for k in POOL])
        )

    # Hot/cold trend: short-window rate minus long-window rate, as a main number
    main_trend = main_freqs[TREND_SHORT_W] / TREND_SHORT_W - main_freqs[TREND_LONG_W] / TREND_LONG_W
    feature_frames.append(
        pd.DataFrame(main_trend, columns=[f"main_trend_n{k}" for k in POOL])
    )

    # Z-score of long-window main count vs. uniform-main null (p = 6/49 per draw)
    p_main = MAIN_DRAW_SIZE / NUM_MAX
    main_expected = TREND_LONG_W * p_main
    main_std = np.sqrt(TREND_LONG_W * p_main * (1 - p_main))
    main_zscore = (main_freqs[TREND_LONG_W] - main_expected) / main_std
    feature_frames.append(
        pd.DataFrame(main_zscore, columns=[f"main_zscore_n{k}" for k in POOL])
    )

    # Gaps: extra and main
    extra_gap = _gap_since_last(shifted_extra)
    feature_frames.append(
        pd.DataFrame(extra_gap, columns=[f"extra_gap_n{k}" for k in POOL])
    )
    main_gap = _gap_since_last(shifted_main)
    feature_frames.append(
        pd.DataFrame(main_gap, columns=[f"main_gap_n{k}" for k in POOL])
    )

    # Co-occurrence: did number k appear as a main number in any of last K draws
    for K in WAS_MAIN_LAST_K:
        rc = _rolling_count_matrix(shifted_main, K)
        binary = (rc >= 1).astype(float)
        feature_frames.append(
            pd.DataFrame(binary, columns=[f"was_main_last_{K}_n{k}" for k in POOL])
        )

    # Recent extras: raw lags 1..10 + summary stats over last 10
    extras_arr = extra.astype(float)
    lag_cols: dict[str, np.ndarray] = {}
    for lag in EXTRA_LAGS:
        arr = np.full(T, np.nan)
        if T - lag > 0:
            arr[lag:] = extras_arr[: T - lag]
        lag_cols[f"extra_lag_{lag}"] = arr
    feature_frames.append(pd.DataFrame(lag_cols))
    feature_frames.append(_last_k_extra_stats(extras_arr, LAST_K_STATS))

    # Recent main draws: sorted 6-number lags 1..10 (mirrors extra_lag_*, one
    # column per lag per sorted position since main draws have 6 numbers)
    mains_sorted = np.sort(df[num_cols].to_numpy(), axis=1)
    main_lag_cols: dict[str, np.ndarray] = {}
    for lag in MAIN_LAGS:
        arr = np.full((T, MAIN_DRAW_SIZE), np.nan)
        if T - lag > 0:
            arr[lag:] = mains_sorted[: T - lag]
        for j in range(MAIN_DRAW_SIZE):
            main_lag_cols[f"main_lag_{lag}_pos{j + 1}"] = arr[:, j]
    feature_frames.append(pd.DataFrame(main_lag_cols))

    # Markov-style transition row: P(extra=k | prev_extra) over last 100 pairs
    markov = _markov_window(extra, MARKOV_WINDOW)
    feature_frames.append(
        pd.DataFrame(
            markov,
            columns=[f"prev_extra_to_n{k}_freq_w{MARKOV_WINDOW}" for k in POOL],
        )
    )

    # Markov-style transition row: co-occurrence freq of prev main numbers -> k
    main_markov = _main_markov_window(main_presence, MARKOV_WINDOW)
    feature_frames.append(
        pd.DataFrame(
            main_markov,
            columns=[f"prev_main_to_n{k}_freq_w{MARKOV_WINDOW}" for k in POOL],
        )
    )

    feature_frames.append(_prev_draw_stats(df))
    feature_frames.append(_zodiac_features(df))

    feature_frames.append(
        pd.DataFrame(
            {
                "dow": df["date"].dt.dayofweek.to_numpy(),
                "month": df["date"].dt.month.to_numpy(),
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

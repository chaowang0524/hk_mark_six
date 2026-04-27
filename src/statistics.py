"""Statistical analysis of Mark Six history: extra number and main numbers.

Outputs a markdown report to data/reports/statistics.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from preprocessing import RAW_PATH, NUM_MIN, NUM_MAX, POOL, load_raw

REPORT_PATH = Path("data/reports/statistics.md")


def _frequency_table(extra: pd.Series) -> pd.Series:
    counts = extra.value_counts().reindex(POOL, fill_value=0).sort_index()
    return counts


def _main_number_analysis(df: pd.DataFrame, lines: list[str]) -> None:
    num_cols = [f"n{i}" for i in range(1, 7)]
    T = len(df)
    all_main = df[num_cols].to_numpy().flatten()
    total_appearances = T * 6
    freq = pd.Series(all_main).value_counts().reindex(POOL, fill_value=0).sort_index()

    def w(s: str = "") -> None:
        lines.append(s)

    w("# Mark Six Main-Number Statistical Report")
    w()
    w(
        f"**Draws analyzed:** {T}  "
        f"({df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()})  "
        f"Total appearances: {total_appearances} ({T} draws × 6)"
    )
    w()

    w("## Main-Number Frequency")
    w()
    w("| Number | Count | Pct    |")
    w("|--------|-------|--------|")
    for n in POOL:
        pct = freq[n] / total_appearances * 100
        w(f"| {n:<6} | {freq[n]:<5} | {pct:>5.2f}% |")

    w()
    w("## Hot Numbers (top 10)")
    w()
    w("| Number | Count | Pct   |")
    w("|--------|-------|-------|")
    for n, c in freq.nlargest(10).items():
        w(f"| {n:<6} | {c:<5} | {c / total_appearances * 100:.1f}% |")

    w()
    w("## Cold Numbers (bottom 10)")
    w()
    w("| Number | Count | Pct   |")
    w("|--------|-------|-------|")
    for n, c in freq.nsmallest(10).items():
        w(f"| {n:<6} | {c:<5} | {c / total_appearances * 100:.1f}% |")

    w()
    w("## Gap Analysis (draws between appearances as main number)")
    w()
    w("| Number | Avg gap | Max gap | Current gap |")
    w("|--------|---------|---------|-------------|")
    last_seen: dict[int, int] = {}
    gaps: dict[int, list[int]] = {n: [] for n in POOL}
    for i, row_vals in enumerate(df[num_cols].to_numpy()):
        for val in row_vals:
            n = int(val)
            if n in last_seen:
                gaps[n].append(i - last_seen[n])
            last_seen[n] = i
    for n in POOL:
        vals = gaps[n]
        avg = np.mean(vals) if vals else T
        mx = max(vals) if vals else T
        cur = T - 1 - last_seen.get(n, T - 1) if n in last_seen else T
        w(f"| {n:<6} | {avg:<7.1f} | {mx:<7} | {cur:<11} |")

    w()
    w("## Chi-Square Uniformity Test (Main Numbers)")
    w()
    observed = freq.values
    expected = np.full(NUM_MAX, total_appearances / NUM_MAX)
    chi2, p = sp_stats.chisquare(observed, expected)
    w(f"- **χ²** = {chi2:.2f}")
    w(f"- **p-value** = {p:.4f}")
    if p < 0.05:
        w("- Distribution differs significantly from uniform (p < 0.05).")
    else:
        w("- No significant deviation from uniform distribution.")

    w()
    w("## Recent Trend (last 30 draws — main numbers)")
    w()
    w("| Number | Count |")
    w("|--------|-------|")
    recent_main = df[num_cols].iloc[-30:].to_numpy().flatten()
    rf_recent = pd.Series(recent_main).value_counts().reindex(POOL, fill_value=0)
    for n in POOL:
        if rf_recent[n] == 0:
            continue
        w(f"| {n:<6} | {rf_recent[n]:<5} |")


def run_analysis() -> Path:
    df = load_raw()
    extra = df["extra"]
    T = len(extra)
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    w("# Mark Six Extra-Number Statistical Report")
    w()
    w(
        f"**Draws analyzed:** {T}  "
        f"({df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()})"
    )
    w()

    # --- Frequency ---
    freq = _frequency_table(extra)
    w("## Extra-Number Frequency")
    w()
    w("| Number | Count | Pct    |")
    w("|--------|-------|--------|")
    for n in POOL:
        pct = freq[n] / T * 100
        w(f"| {n:<6} | {freq[n]:<5} | {pct:>5.2f}% |")

    # --- Hot / Cold ---
    w()
    w("## Hot Numbers (top 10)")
    w()
    w("| Number | Count | Pct   |")
    w("|--------|-------|-------|")
    for n, c in freq.nlargest(10).items():
        w(f"| {n:<6} | {c:<5} | {c / T * 100:.1f}% |")

    w()
    w("## Cold Numbers (bottom 10)")
    w()
    w("| Number | Count | Pct   |")
    w("|--------|-------|-------|")
    for n, c in freq.nsmallest(10).items():
        w(f"| {n:<6} | {c:<5} | {c / T * 100:.1f}% |")

    # --- Gap Analysis ---
    w()
    w("## Gap Analysis (draws between appearances)")
    w()
    w("| Number | Avg gap | Max gap | Current gap |")
    w("|--------|---------|---------|-------------|")
    last_seen: dict[int, int] = {}
    gaps: dict[int, list[int]] = {n: [] for n in POOL}
    for i, val in enumerate(extra):
        n = int(val)
        if n in last_seen:
            gaps[n].append(i - last_seen[n])
        last_seen[n] = i

    for n in POOL:
        vals = gaps[n]
        avg = np.mean(vals) if vals else T
        mx = max(vals) if vals else T
        cur = T - 1 - last_seen.get(n, T - 1) if n in last_seen else T
        w(f"| {n:<6} | {avg:<7.1f} | {mx:<7} | {cur:<11} |")

    # --- Chi-Square Uniformity Test ---
    w()
    w("## Chi-Square Uniformity Test")
    w()
    observed = freq.values
    expected = np.full(NUM_MAX, T / NUM_MAX)
    chi2, p = sp_stats.chisquare(observed, expected)
    w(f"- **χ²** = {chi2:.2f}")
    w(f"- **p-value** = {p:.4f}")
    if p < 0.05:
        w("- Distribution differs significantly from uniform (p < 0.05).")
    else:
        w("- No significant deviation from uniform distribution.")

    # --- Autocorrelation ---
    w()
    w("## Autocorrelation of Extra Numbers")
    w()
    w("| Lag | Correlation |")
    w("|-----|-------------|")
    series = extra.to_numpy().astype(float)
    for lag in (1, 2, 3, 5, 10):
        if lag >= T:
            continue
        r = np.corrcoef(series[:-lag], series[lag:])[0, 1]
        w(f"| {lag:<3} | {r:+.4f}      |")

    # --- Recent Trend (last 30 draws) ---
    w()
    w("## Recent Trend (last 30 draws)")
    w()
    w("| Number | Count |")
    w("|--------|-------|")
    recent = extra.iloc[-30:]
    rf = recent.value_counts().reindex(POOL, fill_value=0)
    for n in POOL:
        if rf[n] == 0:
            continue
        w(f"| {n:<6} | {rf[n]:<5} |")

    w()
    w("---")
    w()

    # ------------------------------------------------------------------ #
    # Main-number analysis
    # ------------------------------------------------------------------ #
    _main_number_analysis(df, lines)

    w()
    w("---")
    w("*Report generated by `python main.py analyze`*")
    w()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print(f"Report written to {REPORT_PATH}")
    return REPORT_PATH


if __name__ == "__main__":
    run_analysis()

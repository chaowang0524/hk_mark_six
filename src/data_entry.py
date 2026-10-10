"""Manual add/remove of Mark Six draws — the data-acquisition path while the
scraper is blocked (see STATUS.md). Reuses scraper's `Draw`/ID-assignment
logic so hand-entered rows are indistinguishable in format from scraped ones.
"""

from __future__ import annotations

import csv

import pandas as pd

import scraper
from scraper import CSV_HEADER, Draw

NUM_MIN, NUM_MAX = 1, 49


def read_tail(n: int = 10) -> pd.DataFrame:
    """Return the most recent `n` draws, oldest first."""
    if not scraper.DATA_PATH.exists():
        return pd.DataFrame(columns=CSV_HEADER)
    df = pd.read_csv(scraper.DATA_PATH)
    return df.tail(n).reset_index(drop=True)


def add_draw(date: str, mains: list[int], extra: int) -> str:
    """Validate and append a new draw. Returns the assigned draw ID."""
    mains = sorted(mains)
    if len(mains) != 6 or len(set(mains)) != 6:
        raise ValueError("Pick exactly 6 different main numbers.")
    if not all(NUM_MIN <= n <= NUM_MAX for n in [*mains, extra]):
        raise ValueError(f"All numbers must be between {NUM_MIN} and {NUM_MAX}.")
    if extra in mains:
        raise ValueError("The extra number must be different from the 6 main numbers.")

    existing_dates, last_num_by_year = scraper.load_existing_draws()
    if existing_dates and date <= max(existing_dates):
        raise ValueError(
            f"Date {date} must be after the most recent draw ({max(existing_dates)})."
        )

    draw = Draw(date=date, numbers=mains, extra=extra)
    scraper.assign_ids_and_write([draw], last_num_by_year)
    return draw.draw


def delete_last() -> dict[str, str] | None:
    """Remove the most recently added draw. Returns the removed row, or
    None if there was nothing to delete."""
    if not scraper.DATA_PATH.exists():
        return None
    with scraper.DATA_PATH.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    removed = rows.pop()
    with scraper.DATA_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return removed

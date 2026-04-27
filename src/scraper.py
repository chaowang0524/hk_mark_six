"""Hong Kong Mark Six historical draw scraper.

Source: https://en.lottolyzer.com/history/hong-kong/mark-six/
Each page returns 50 draws in a single HTML table. We iterate forward until
we hit a page with no rows or we see a draw that already exists in the local
CSV (incremental mode).
"""

from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = (
    "https://en.lottolyzer.com/history/hong-kong/mark-six"
    "/page/{page}/per-page/50/summary-view"
)
DATA_PATH = Path("data/raw_draws.csv")
CSV_HEADER = ["draw", "date", "n1", "n2", "n3", "n4", "n5", "n6", "extra"]
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


@dataclass
class Draw:
    draw: str
    date: str
    numbers: list[int]
    extra: int

    def as_row(self) -> list[str]:
        n = self.numbers + [0] * (6 - len(self.numbers))
        return [self.draw, self.date, *map(str, n[:6]), str(self.extra)]


def _fetch(page: int, session: requests.Session) -> str:
    url = BASE_URL.format(page=page)
    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    return resp.text


def _parse_page(html: str) -> list[Draw]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="summary-table")
    if table is None:
        return []
    draws: list[Draw] = []
    tbody = table.find("tbody")
    if tbody is None:
        return []
    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue
        draw_id = tds[0].get_text(strip=True)
        date = tds[1].get_text(strip=True)
        winning_raw = tds[2].get_text(strip=True)
        extra_raw = tds[3].get_text(strip=True)
        if not re.match(r"^\d{2}/\d{3}$", draw_id):
            continue
        try:
            numbers = [int(x) for x in winning_raw.split(",") if x.strip()]
            extra = int(extra_raw)
        except ValueError:
            continue
        if len(numbers) != 6:
            continue
        draws.append(Draw(draw=draw_id, date=date, numbers=numbers, extra=extra))
    return draws


def _load_existing_draw_ids() -> set[str]:
    if not DATA_PATH.exists():
        return set()
    with DATA_PATH.open() as f:
        reader = csv.DictReader(f)
        return {row["draw"] for row in reader}


def _write_csv(rows: list[Draw], existing_ids: set[str]) -> int:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not DATA_PATH.exists()
    new = [d for d in rows if d.draw not in existing_ids]
    if not new:
        return 0
    with DATA_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADER)
        for d in new:
            writer.writerow(d.as_row())
    return len(new)


def scrape(max_pages: int = 200, delay: float = 0.4, incremental: bool = True) -> int:
    """Scrape historical draws. Returns number of new rows appended."""
    existing = _load_existing_draw_ids() if incremental else set()
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    collected: list[Draw] = []
    stopped_early = False
    for page in range(1, max_pages + 1):
        try:
            html = _fetch(page, session)
        except requests.RequestException as exc:
            print(f"  page {page}: fetch failed ({exc}); stopping")
            break
        draws = _parse_page(html)
        if not draws:
            print(f"  page {page}: no rows; stopping")
            break
        new_on_page = [d for d in draws if d.draw not in existing]
        collected.extend(new_on_page)
        print(f"  page {page}: {len(draws)} rows, {len(new_on_page)} new")
        if incremental and len(new_on_page) < len(draws):
            stopped_early = True
            break
        if len(draws) < 50:
            break
        time.sleep(delay)

    collected.sort(key=lambda d: d.date)
    written = _write_csv(collected, existing)
    print(
        f"Scrape complete: wrote {written} new rows to {DATA_PATH}"
        + (" (incremental stop)" if stopped_early else "")
    )
    return written


if __name__ == "__main__":
    scrape()

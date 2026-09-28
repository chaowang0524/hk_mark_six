"""Hong Kong Mark Six historical draw scraper.

Source: https://www.pilio.idv.tw/ltohk/list.asp
Each page returns up to 20 draws in a single HTML table (id="ltotable"),
oldest-first when requested with orderby=old. The site does not expose a
draw ID, so we derive one ourselves: sequential per calendar year, starting
at 001, matching the existing CSV convention (e.g. "26/090").
"""

from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.pilio.idv.tw/ltohk/list.asp"
DATA_PATH = Path("data/raw_draws.csv")
CSV_HEADER = ["draw", "date", "n1", "n2", "n3", "n4", "n5", "n6", "extra"]
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
# "08/1326(四)" -> month 08, day 13, year 2026 (weekday label is informational).
DATE_RE = re.compile(r"^(\d{2})/(\d{2})(\d{2})\(.+\)$")


@dataclass
class Draw:
    date: str
    numbers: list[int]
    extra: int
    draw: str = field(default="")

    def as_row(self) -> list[str]:
        n = self.numbers + [0] * (6 - len(self.numbers))
        return [self.draw, self.date, *map(str, n[:6]), str(self.extra)]


def _fetch_page(page: int, session: requests.Session) -> str:
    resp = session.get(
        BASE_URL, params={"indexpage": page, "orderby": "old"}, timeout=20
    )
    resp.raise_for_status()
    resp.encoding = "utf-8"
    return resp.text


def _max_page(html: str) -> int | None:
    soup = BeautifulSoup(html, "html.parser")
    select = soup.find("select", attrs={"name": "indexpage"})
    if select is None:
        return None
    values = [
        int(opt["value"])
        for opt in select.find_all("option")
        if opt.get("value", "").isdigit()
    ]
    return max(values) if values else None


def _parse_page(html: str) -> list[Draw]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="ltotable")
    if table is None:
        return []
    draws: list[Draw] = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) != 3:
            continue
        date_raw = tds[0].get_text(strip=True)
        numbers_raw = tds[1].get_text(strip=True)
        extra_raw = tds[2].get_text(strip=True)
        m = DATE_RE.match(date_raw)
        if not m:
            continue
        month, day, yy = (int(x) for x in m.groups())
        try:
            numbers = [
                int(x) for x in numbers_raw.replace("\xa0", "").split(",") if x.strip()
            ]
            extra = int(extra_raw)
        except ValueError:
            continue
        if len(numbers) != 6:
            continue
        date = f"20{yy:02d}-{month:02d}-{day:02d}"
        draws.append(Draw(date=date, numbers=numbers, extra=extra))
    return draws


def _load_existing() -> tuple[set[str], dict[str, int]]:
    """Returns (known draw dates, highest sequence number seen per 2-digit year)."""
    existing_dates: set[str] = set()
    last_num_by_year: dict[str, int] = {}
    if not DATA_PATH.exists():
        return existing_dates, last_num_by_year
    with DATA_PATH.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_dates.add(row["date"])
            yy, num = row["draw"].split("/")
            last_num_by_year[yy] = max(last_num_by_year.get(yy, 0), int(num))
    return existing_dates, last_num_by_year


def _find_resume_page(
    session: requests.Session, max_page: int, last_known_date: str
) -> int:
    """Binary search (pages are date-ascending) for the earliest page that
    could contain draws newer than last_known_date, with a one-page margin."""
    lo, hi = 1, max_page
    result = max_page
    while lo <= hi:
        mid = (lo + hi) // 2
        draws = _parse_page(_fetch_page(mid, session))
        if not draws:
            hi = mid - 1
            continue
        if draws[-1].date > last_known_date:
            result = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return max(result - 1, 1)


def _assign_ids_and_write(
    draws: list[Draw], last_num_by_year: dict[str, int]
) -> int:
    if not draws:
        return 0
    counters = dict(last_num_by_year)
    for d in draws:
        yy = d.date[2:4]
        counters[yy] = counters.get(yy, 0) + 1
        d.draw = f"{yy}/{counters[yy]:03d}"

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not DATA_PATH.exists()
    needs_leading_newline = (
        not write_header
        and DATA_PATH.stat().st_size > 0
        and not DATA_PATH.read_bytes().endswith(b"\n")
    )
    with DATA_PATH.open("a", newline="") as f:
        if needs_leading_newline:
            f.write("\n")
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADER)
        for d in draws:
            writer.writerow(d.as_row())
    return len(draws)


def scrape(delay: float = 0.4, incremental: bool = True) -> int:
    """Scrape historical draws. Returns number of new rows appended."""
    existing_dates, last_num_by_year = _load_existing() if incremental else (set(), {})
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    first_html = _fetch_page(1, session)
    max_page = _max_page(first_html) or 1

    if incremental and existing_dates:
        start_page = _find_resume_page(session, max_page, max(existing_dates))
    else:
        start_page = 1

    collected: list[Draw] = []
    for page in range(start_page, max_page + 1):
        html = first_html if page == 1 else _fetch_page(page, session)
        draws = _parse_page(html)
        if not draws:
            print(f"  page {page}/{max_page}: no rows; stopping")
            break
        new_on_page = [d for d in draws if d.date not in existing_dates]
        collected.extend(new_on_page)
        print(f"  page {page}/{max_page}: {len(draws)} rows, {len(new_on_page)} new")
        if page < max_page:
            time.sleep(delay)

    collected.sort(key=lambda d: d.date)
    written = _assign_ids_and_write(collected, last_num_by_year)
    print(f"Scrape complete: wrote {written} new rows to {DATA_PATH}")
    return written


if __name__ == "__main__":
    scrape()

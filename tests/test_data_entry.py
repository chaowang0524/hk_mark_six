"""Tests for manual draw entry/removal in data_entry.py."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import data_entry  # noqa: E402
import scraper  # noqa: E402


@pytest.fixture
def raw_path(tmp_path):
    path = tmp_path / "raw_draws.csv"
    with patch.object(scraper, "DATA_PATH", path):
        yield path


def test_add_draw_to_empty_file(raw_path):
    draw_id = data_entry.add_draw("2026-01-06", [6, 5, 4, 3, 2, 1], 7)
    assert draw_id == "26/001"
    df = data_entry.read_tail()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["draw"] == "26/001"
    assert [row[f"n{i}"] for i in range(1, 7)] == [1, 2, 3, 4, 5, 6]
    assert row["extra"] == 7


def test_add_draw_increments_within_year(raw_path):
    data_entry.add_draw("2026-01-06", [1, 2, 3, 4, 5, 6], 7)
    draw_id = data_entry.add_draw("2026-01-08", [10, 11, 12, 13, 14, 15], 16)
    assert draw_id == "26/002"


def test_add_draw_resets_sequence_on_new_year(raw_path):
    data_entry.add_draw("2026-12-30", [1, 2, 3, 4, 5, 6], 7)
    draw_id = data_entry.add_draw("2027-01-02", [10, 11, 12, 13, 14, 15], 16)
    assert draw_id == "27/001"


def test_add_draw_rejects_duplicate_main(raw_path):
    with pytest.raises(ValueError, match="6 different main numbers"):
        data_entry.add_draw("2026-01-06", [1, 1, 3, 4, 5, 6], 7)


def test_add_draw_rejects_out_of_range(raw_path):
    with pytest.raises(ValueError, match="between 1 and 49"):
        data_entry.add_draw("2026-01-06", [1, 2, 3, 4, 5, 50], 7)
    with pytest.raises(ValueError, match="between 1 and 49"):
        data_entry.add_draw("2026-01-06", [1, 2, 3, 4, 5, 6], 0)


def test_add_draw_rejects_extra_equal_to_main(raw_path):
    with pytest.raises(ValueError, match="different from the 6 main"):
        data_entry.add_draw("2026-01-06", [1, 2, 3, 4, 5, 6], 6)


def test_add_draw_rejects_non_increasing_date(raw_path):
    data_entry.add_draw("2026-01-06", [1, 2, 3, 4, 5, 6], 7)
    with pytest.raises(ValueError, match="must be after"):
        data_entry.add_draw("2026-01-06", [10, 11, 12, 13, 14, 15], 16)
    with pytest.raises(ValueError, match="must be after"):
        data_entry.add_draw("2026-01-05", [10, 11, 12, 13, 14, 15], 16)


def test_delete_last_removes_most_recent_row(raw_path):
    data_entry.add_draw("2026-01-06", [1, 2, 3, 4, 5, 6], 7)
    data_entry.add_draw("2026-01-08", [10, 11, 12, 13, 14, 15], 16)
    removed = data_entry.delete_last()
    assert removed["draw"] == "26/002"
    df = data_entry.read_tail()
    assert len(df) == 1
    assert df.iloc[0]["draw"] == "26/001"


def test_delete_last_on_missing_file_returns_none(raw_path):
    assert data_entry.delete_last() is None


def test_read_tail_limits_to_n(raw_path):
    for day in range(6, 13):
        data_entry.add_draw(f"2026-01-{day:02d}", [1, 2, 3, 4, 5, 6], 7)
    df = data_entry.read_tail(3)
    assert len(df) == 3
    assert df.iloc[-1]["draw"] == "26/007"

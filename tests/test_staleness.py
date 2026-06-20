"""Tests for _maybe_rebuild_features staleness logic in models.py."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _run(raw_exists, raw_mtime, features_exists, features_mtime):
    """Call _maybe_rebuild_features with mocked paths and return build_features call count."""
    import models

    raw = MagicMock(spec=Path)
    raw.exists.return_value = raw_exists
    raw.stat.return_value = MagicMock(st_mtime=raw_mtime)

    feat = MagicMock(spec=Path)
    feat.exists.return_value = features_exists
    feat.stat.return_value = MagicMock(st_mtime=features_mtime)

    with (
        patch.object(models, "RAW_PATH", raw),
        patch.object(models, "FEATURES_PATH", feat),
        patch.object(models, "build_features") as mock_build,
    ):
        models._maybe_rebuild_features()
        return mock_build.call_count


def test_rebuilds_when_features_missing():
    assert _run(raw_exists=True, raw_mtime=100, features_exists=False, features_mtime=0) == 1


def test_rebuilds_when_raw_is_newer():
    assert _run(raw_exists=True, raw_mtime=200, features_exists=True, features_mtime=100) == 1


def test_no_rebuild_when_features_is_current():
    assert _run(raw_exists=True, raw_mtime=100, features_exists=True, features_mtime=200) == 0


def test_no_rebuild_when_raw_missing_and_features_exist():
    assert _run(raw_exists=False, raw_mtime=0, features_exists=True, features_mtime=100) == 0

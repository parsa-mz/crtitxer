"""Tests for artefact locations.

Twenty-one modules each carried their own
``DATA = Path("/absolute/path/to/data")``. That is not a style problem: it means
the repository cannot be run anywhere else without editing twenty-one files, and a reader cannot
tell from any one of them whether the path is a shared convention or that script's own choice.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from critxer.core import paths


def test_the_data_root_comes_from_the_environment_when_set(monkeypatch):
    monkeypatch.setenv("CRITXER_DATA", "/tmp/somewhere/else")

    assert paths.data_root() == Path("/tmp/somewhere/else")


def test_there_is_a_default_so_the_common_case_needs_no_setup(monkeypatch):
    monkeypatch.delenv("CRITXER_DATA", raising=False)

    assert paths.data_root().is_absolute()


def test_subdirectories_are_named_rather_than_spelled_out_at_call_sites(monkeypatch):
    """The three run directories are a shared convention; a typo in one would silently split it."""
    monkeypatch.setenv("CRITXER_DATA", "/tmp/d")

    assert paths.run_dir("r4") == Path("/tmp/d/r4")


def test_an_unknown_run_directory_is_refused(monkeypatch):
    """A new cell type is a decision to record here, not a string to invent at a call site."""
    monkeypatch.setenv("CRITXER_DATA", "/tmp/d")

    with pytest.raises(ValueError, match="ladder"):
        paths.run_dir("laddr")


def test_artefacts_resolve_under_the_data_root(monkeypatch):
    monkeypatch.setenv("CRITXER_DATA", "/tmp/d")

    assert paths.artefact("allocation.json") == Path("/tmp/d/allocation.json")


def test_the_root_is_read_at_call_time_not_import_time(monkeypatch):
    """Module-level constants captured the path at import, so a test could not redirect it."""
    monkeypatch.setenv("CRITXER_DATA", "/tmp/first")
    first = paths.data_root()
    monkeypatch.setenv("CRITXER_DATA", "/tmp/second")

    assert paths.data_root() != first

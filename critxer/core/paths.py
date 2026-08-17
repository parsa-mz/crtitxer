"""Where artefacts live.

Generations are kept outside the repository -- the clean arm alone is 26k parsed audits -- so every
run reads and writes under one root, overridable with ``CRITXER_DATA``.

Paths are resolved **on call**, not bound at import. A module-level constant cannot be redirected by
a test, by a second machine, or by anyone reproducing the study somewhere the original absolute path
does not exist.
"""

from __future__ import annotations

import os
from pathlib import Path

# `data/` beside the repository root. A default rather than a requirement: it makes the common case
# need no setup, and CRITXER_DATA overrides it everywhere at once. Resolved to an absolute path so a
# command run from a subdirectory does not silently write to a second, empty artefact tree.
DEFAULT_DATA_ROOT = (Path(__file__).resolve().parents[2] / "data")

# Per-condition checkpoints, one JSON per (model, condition, wording). Named here rather than
# spelled at each call site: a typo would not fail, it would silently create a fourth directory and
# split a run in half.
RUN_DIRS = ("ladder", "r4", "detection")


def data_root() -> Path:
    """The artefact root, from ``CRITXER_DATA`` or the default."""
    return Path(os.environ.get("CRITXER_DATA") or DEFAULT_DATA_ROOT)


def run_dir(name: str) -> Path:
    """One of the three per-condition checkpoint directories."""
    if name not in RUN_DIRS:
        raise ValueError(f"unknown run directory {name!r}; expected one of {RUN_DIRS}")
    return data_root() / name


def artefact(*parts: str) -> Path:
    """A named file under the artefact root."""
    return data_root().joinpath(*parts)

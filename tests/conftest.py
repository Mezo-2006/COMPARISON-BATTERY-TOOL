"""Shared pytest fixtures for the Digital Twin Validation Tool test suite.

These fixtures make every test file independent of the on-disk
``test_twin.csv`` / ``test_ecu.csv`` files (which the developer might
edit) and let each test build tiny in-memory DataFrames with known
values for reproducible assertions.

The ``final_tool/`` directory is added to ``sys.path`` so the engines
can be imported without a package install.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Make the engines importable when pytest is run from any directory.
# ---------------------------------------------------------------------------
_FINAL_TOOL_DIR = Path(__file__).resolve().parent.parent
if str(_FINAL_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_FINAL_TOOL_DIR))

from data_loader import LoadResult
from alignment_engine import AlignedData, AlignedSignal


# ---------------------------------------------------------------------------
# On-disk fixture paths (used by tests that exercise real CSV loading)
# ---------------------------------------------------------------------------
@pytest.fixture
def twin_csv_path() -> str:
    return str(_FINAL_TOOL_DIR / "test_twin.csv")


@pytest.fixture
def ecu_csv_path() -> str:
    return str(_FINAL_TOOL_DIR / "test_ecu.csv")


# ---------------------------------------------------------------------------
# In-memory LoadResult builders
# ---------------------------------------------------------------------------
def _make_load_result(
    source_label: str,
    rows: dict,
    source_columns: list | None = None,
    warnings: list | None = None,
    axis_kind: str = "timestamp",
) -> LoadResult:
    """Build a LoadResult from a dict-of-columns ``{name: [values]}``.

    ``timestamp`` column is sorted ascending and all arrays must share the
    same length.  ``source_columns`` defaults to the canonical names; pass
    a list to mimic the original CSV headers (used by tests that verify
    UI-display fields).  ``axis_kind`` defaults to ``"timestamp"``; pass
    ``"sequence"`` to build a fixture that mimics a plain id-counter axis.
    """
    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    cols = list(df.columns)
    if source_columns is None:
        source_columns = list(cols)

    timestamp = df["timestamp"].to_numpy(dtype=np.float64)
    return LoadResult(
        df=df,
        source_label=source_label,
        row_count=len(df),
        columns_found=[c for c in cols if c not in ("timestamp", "id")],
        columns_missing=[],
        source_columns=source_columns,
        time_range=(float(timestamp[0]), float(timestamp[-1])),
        warnings=warnings or [],
        axis_kind=axis_kind,
    )


@pytest.fixture
def make_load_result():
    """Expose the in-memory LoadResult builder to tests."""
    return _make_load_result


@pytest.fixture
def twin_result_five(make_load_result):
    """5-signal twin LoadResult with a 0.1 s sample period."""
    return make_load_result(
        "twin",
        {
            "timestamp":   [0.0, 0.1, 0.2, 0.3, 0.4],
            "voltage":     [3.30, 3.31, 3.32, 3.33, 3.34],
            "current":     [1.20, 1.21, 1.22, 1.23, 1.24],
            "temperature": [25.0, 25.1, 25.2, 25.3, 25.4],
            "soc":         [80.0, 79.9, 79.8, 79.7, 79.6],
            "soh":         [95.0, 95.0, 95.0, 95.0, 95.0],
        },
    )


@pytest.fixture
def ecu_result_five(make_load_result):
    """5-signal ECU LoadResult offset by 0.05 s from the twin."""
    return make_load_result(
        "ecu",
        {
            "timestamp":   [0.05, 0.15, 0.25, 0.35, 0.45],
            "voltage":     [3.305, 3.315, 3.325, 3.335, 3.345],
            "current":     [1.205, 1.215, 1.225, 1.235, 1.245],
            "temperature": [25.05, 25.15, 25.25, 25.35, 25.45],
            "soc":         [79.95, 79.85, 79.75, 79.65, 79.55],
            "soh":         [95.0, 95.0, 95.0, 95.0, 95.0],
        },
    )


@pytest.fixture
def ecu_result_three(make_load_result):
    """3-signal ECU LoadResult (no SoC/SoH) — for partial-overlap tests."""
    return make_load_result(
        "ecu",
        {
            "timestamp":   [0.05, 0.15, 0.25, 0.35, 0.45],
            "voltage":     [3.305, 3.315, 3.325, 3.335, 3.345],
            "current":     [1.205, 1.215, 1.225, 1.235, 1.245],
            "temperature": [25.05, 25.15, 25.25, 25.35, 25.45],
        },
    )


# ---------------------------------------------------------------------------
# In-memory AlignedData builder
# ---------------------------------------------------------------------------
@pytest.fixture
def make_aligned_data():
    """Factory fixture that builds an AlignedData from raw arrays.

    Usage in tests::

        aligned = make_aligned_data(
            timestamps=[0.0, 0.1, 0.2],
            signals={
                "voltage": ([3.3, 3.4, 3.5],  # twin
                           [3.31, 3.41, 3.0],  # ecu (last sample off)
                           ),
            },
        )
    """
    def _build(
        timestamps: list,
        signals: dict,
        alignment_method: str = "nearest",
        warnings: list | None = None,
        n_matched_override: int | None = None,
    ) -> AlignedData:
        ts = np.asarray(timestamps, dtype=np.float64)
        sig_dict = {}
        for name, (twin_vals, ecu_vals) in signals.items():
            twin = np.asarray(twin_vals, dtype=np.float64)
            ecu = np.asarray(ecu_vals, dtype=np.float64)
            sig_dict[name] = AlignedSignal(
                name=name,
                twin_values=twin,
                ecu_values=ecu,
                errors=ecu - twin,
            )
        # Compute n_matched (pairs valid in every signal)
        mask = np.ones(len(ts), dtype=bool)
        for s in sig_dict.values():
            mask &= ~np.isnan(s.twin_values)
            mask &= ~np.isnan(s.ecu_values)
        n_total = len(ts)
        n_matched = int(mask.sum()) if n_matched_override is None else n_matched_override
        return AlignedData(
            timestamps=ts,
            signals=sig_dict,
            n_total=n_total,
            n_matched=n_matched,
            max_delta_t=0.05,
            alignment_method=alignment_method,
            warnings=warnings or [],
        )
    return _build
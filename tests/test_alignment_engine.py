"""Tests for ``alignment_engine.py`` — twin ↔ ECU alignment."""

from __future__ import annotations

import numpy as np
import pytest

from alignment_engine import (
    AlignedData,
    AlignmentError,
    align,
    ALIGN_NEAREST,
    ALIGN_INTERPOLATE,
)


# ---------------------------------------------------------------------------
# Happy path: 5-signal fixtures
# ---------------------------------------------------------------------------
def test_nearest_all_five_signals_present(twin_result_five, ecu_result_five):
    aligned = align(twin_result_five, ecu_result_five, ALIGN_NEAREST)

    assert aligned.alignment_method == "nearest"
    assert aligned.n_total == 5
    # All 5 signals aligned.
    assert set(aligned.signal_names) == {
        "voltage", "current", "temperature", "soc", "soh"
    }
    # Reference timeline is the ECU's.
    np.testing.assert_allclose(
        aligned.timestamps,
        ecu_result_five.df["timestamp"].to_numpy(),
    )


def test_nearest_max_delta_t_within_half_period(twin_result_five, ecu_result_five):
    aligned = align(twin_result_five, ecu_result_five, ALIGN_NEAREST)
    # ECU offset is 0.05 s, twin period is 0.1 s → max delta ≤ 0.05.
    assert aligned.max_delta_t == pytest.approx(0.05, abs=1e-9)


def test_nearest_errors_are_half_step(twin_result_five, ecu_result_five):
    """The fixtures are linear ramps offset by half a sample period.

    Nearest-match error is exactly the half-step difference (0.005 for
    V/I, 0.05 for T, 0.05 for SoC, 0 for SoH which is constant).
    """
    aligned = align(twin_result_five, ecu_result_five, ALIGN_NEAREST)
    assert np.max(np.abs(aligned.signals["voltage"].errors)) == pytest.approx(0.005)
    assert np.max(np.abs(aligned.signals["current"].errors)) == pytest.approx(0.005)
    assert np.max(np.abs(aligned.signals["temperature"].errors)) == pytest.approx(0.05)
    assert np.max(np.abs(aligned.signals["soc"].errors)) == pytest.approx(0.05)
    assert np.max(np.abs(aligned.signals["soh"].errors)) == pytest.approx(0.0)


def test_interpolate_linear_twin_gives_zero_error(twin_result_five, ecu_result_five):
    """The twin is linear, so interpolation onto the ECU timestamps is exact."""
    aligned = align(twin_result_five, ecu_result_five, ALIGN_INTERPOLATE)

    assert aligned.alignment_method == "interpolate"
    # Last ECU sample (t=0.45) is outside the twin's range (0.0–0.4) → NaN.
    assert aligned.n_matched == 4
    for name in ("voltage", "current", "temperature", "soc"):
        errs = aligned.signals[name].errors
        # Exclude the last (NaN) sample.
        finite = errs[np.isfinite(errs)]
        assert np.allclose(finite, 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# ECU is the reference timeline (never dropped)
# ---------------------------------------------------------------------------
def test_ecu_timestamps_are_reference(twin_result_five, ecu_result_five):
    aligned = align(twin_result_five, ecu_result_five, ALIGN_NEAREST)
    np.testing.assert_allclose(
        aligned.timestamps,
        ecu_result_five.df["timestamp"].to_numpy(),
    )


# ---------------------------------------------------------------------------
# Partial-overlap: ECU has only 3 signals (no SoC/SoH)
# ---------------------------------------------------------------------------
def test_partial_overlap_only_common_signals_aligned(twin_result_five,
                                                       ecu_result_three):
    aligned = align(twin_result_five, ecu_result_three, ALIGN_NEAREST)

    assert set(aligned.signal_names) == {"voltage", "current", "temperature"}
    # SoC / SoH are silently skipped (not in ECU).
    assert "soc" not in aligned.signals
    assert "soh" not in aligned.signals


# ---------------------------------------------------------------------------
# Errors: no common signals / no time overlap
# ---------------------------------------------------------------------------
def test_no_common_signals_raises(make_load_result):
    twin = make_load_result(
        "twin",
        {"timestamp": [0.0, 0.1],
         "voltage": [3.3, 3.4], "current": [1.2, 1.3],
         "temperature": [25.0, 26.0], "soc": [80.0, 79.0],
         "soh": [95.0, 95.0]},
    )
    # ECU has none of those — only a column the alias system normally
    # wouldn't know about, but here we deliberately drop canonicals.
    ecu = make_load_result(
        "ecu",
        {"timestamp": [0.05, 0.15],
         "voltage": [3.31, 3.41]},  # only one
    )
    # Build a deliberately-mismatched ECU by overwriting columns_found.
    # Simpler: use only timestamps + a non-canonical column.
    import pandas as pd
    df = pd.DataFrame({"timestamp": [0.05, 0.15],
                       "pressure": [0.1, 0.2]})
    from data_loader import LoadResult
    ecu = LoadResult(
        df=df, source_label="ecu", row_count=2,
        columns_found=[], columns_missing=[],
        source_columns=["timestamp", "pressure"],
        time_range=(0.05, 0.15), warnings=[],
    )
    with pytest.raises(AlignmentError, match="No common signal"):
        align(twin, ecu, ALIGN_NEAREST)


def test_no_time_range_overlap_raises(make_load_result):
    twin = make_load_result(
        "twin",
        {"timestamp": [0.0, 0.1, 0.2],
         "voltage": [3.3, 3.4, 3.5],
         "current": [1.2, 1.3, 1.4],
         "temperature": [25.0, 26.0, 27.0],
         "soc": [80.0, 79.0, 78.0],
         "soh": [95.0, 95.0, 95.0]},
    )
    ecu = make_load_result(
        "ecu",
        {"timestamp": [10.0, 10.1, 10.2],
         "voltage": [3.31, 3.41, 3.51],
         "current": [1.21, 1.31, 1.41],
         "temperature": [25.1, 26.1, 27.1],
         "soc": [79.9, 78.9, 77.9],
         "soh": [95.0, 95.0, 95.0]},
    )
    with pytest.raises(AlignmentError, match="No time-range overlap"):
        align(twin, ecu, ALIGN_NEAREST)


# ---------------------------------------------------------------------------
# Unknown method falls back to nearest with a warning
# ---------------------------------------------------------------------------
def test_unknown_method_falls_back_to_nearest(twin_result_five,
                                                ecu_result_five):
    aligned = align(twin_result_five, ecu_result_five, method="bogus")
    assert aligned.alignment_method == "nearest"
    assert any("Unknown alignment method" in w for w in aligned.warnings)


# ---------------------------------------------------------------------------
# Nearest: large gaps → NaN-marked
# ---------------------------------------------------------------------------
def test_nearest_outside_twin_range_marked_nan(make_load_result):
    twin = make_load_result(
        "twin",
        {"timestamp": [0.0, 0.1, 0.2],
         "voltage": [3.3, 3.4, 3.5],
         "current": [1.2, 1.3, 1.4],
         "temperature": [25.0, 26.0, 27.0],
         "soc": [80.0, 79.0, 78.0],
         "soh": [95.0, 95.0, 95.0]},
    )
    ecu = make_load_result(
        "ecu",
        {"timestamp": [0.05, 0.15, 5.0],   # last is far outside
         "voltage": [3.305, 3.405, 9.0],
         "current": [1.205, 1.305, 9.0],
         "temperature": [25.05, 26.05, 9.0],
         "soc": [79.95, 78.95, 9.0],
         "soh": [95.0, 95.0, 95.0]},
    )
    aligned = align(twin, ecu, ALIGN_NEAREST)
    # The last ECU sample (t=5.0) should have NaN twin values.
    for sig in aligned.signals.values():
        assert np.isnan(sig.twin_values[-1])
    assert aligned.n_matched < aligned.n_total
    assert any("could not be matched" in w for w in aligned.warnings)


# ---------------------------------------------------------------------------
# Interpolate: out-of-range samples → NaN
# ---------------------------------------------------------------------------
def test_interpolate_out_of_range_marked_nan(make_load_result):
    twin = make_load_result(
        "twin",
        {"timestamp": [0.0, 0.1, 0.2],
         "voltage": [3.3, 3.4, 3.5],
         "current": [1.2, 1.3, 1.4],
         "temperature": [25.0, 26.0, 27.0],
         "soc": [80.0, 79.0, 78.0],
         "soh": [95.0, 95.0, 95.0]},
    )
    ecu = make_load_result(
        "ecu",
        {"timestamp": [-0.5, 0.05, 0.5],   # first and last are outside
         "voltage": [9.0, 3.305, 9.0],
         "current": [9.0, 1.205, 9.0],
         "temperature": [9.0, 25.05, 9.0],
         "soc": [9.0, 79.95, 9.0],
         "soh": [95.0, 95.0, 95.0]},
    )
    aligned = align(twin, ecu, ALIGN_INTERPOLATE)
    # Only the middle sample (t=0.05) is within the twin range.
    assert aligned.n_matched == 1
    for sig in aligned.signals.values():
        assert np.isnan(sig.twin_values[0])
        assert np.isnan(sig.twin_values[2])
        assert not np.isnan(sig.twin_values[1])
"""Tests for ``statistics_engine.py`` — metrics + worst mismatches."""

from __future__ import annotations

import numpy as np
import pytest

from statistics_engine import (
    DEFAULT_TOLERANCES,
    DEFAULT_WORST_N,
    SignalStat,
    StatisticsResult,
    WorstMismatch,
    compute,
)
from alignment_engine import ALIGN_NEAREST, ALIGN_INTERPOLATE, align


# ---------------------------------------------------------------------------
# Default tolerances, 5 signals
# ---------------------------------------------------------------------------
def test_compute_returns_all_six_summary_cards(twin_result_five, ecu_result_five):
    aligned = align(twin_result_five, ecu_result_five, ALIGN_NEAREST)
    stats = compute(aligned)

    assert stats.total_samples == 5
    assert stats.matched_samples == 5
    assert isinstance(stats.mean_abs_error, float)
    assert isinstance(stats.max_error, float)
    assert isinstance(stats.rmse, float)
    assert isinstance(stats.match_pct, float)
    assert 0.0 <= stats.match_pct <= 100.0


def test_default_tolerances_all_within(twin_result_five, ecu_result_five):
    """With the (loose) defaults, every sample is within tolerance → 100 %."""
    aligned = align(twin_result_five, ecu_result_five, ALIGN_NEAREST)
    stats = compute(aligned)
    assert stats.match_pct == pytest.approx(100.0)
    for s in stats.signal_stats.values():
        assert s.match_pct == pytest.approx(100.0)


def test_per_signal_stats_present_for_all_five(twin_result_five, ecu_result_five):
    aligned = align(twin_result_five, ecu_result_five, ALIGN_NEAREST)
    stats = compute(aligned)
    assert set(stats.signal_stats.keys()) == {
        "voltage", "current", "temperature", "soc", "soh",
    }
    for name, s in stats.signal_stats.items():
        assert s.name == name
        assert s.n_valid == 5
        assert s.n_within_tolerance == 5
        assert s.tolerance_pct == DEFAULT_TOLERANCES[name]


def test_soH_zero_error(twin_result_five, ecu_result_five):
    """SoH is constant in the fixtures → all errors zero."""
    aligned = align(twin_result_five, ecu_result_five, ALIGN_NEAREST)
    stats = compute(aligned)
    soh = stats.signal_stats["soh"]
    assert soh.mae == pytest.approx(0.0)
    assert soh.rmse == pytest.approx(0.0)
    assert soh.max_error == pytest.approx(0.0)
    assert soh.mean_error == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tolerance model
# ---------------------------------------------------------------------------
def test_tight_tolerances_drop_match_pct(twin_result_five, ecu_result_five):
    aligned = align(twin_result_five, ecu_result_five, ALIGN_NEAREST)
    tight = {k: 0.001 for k in
             ("voltage", "current", "temperature", "soc", "soh")}
    stats = compute(aligned, tolerances=tight)
    # SoH has zero error → still 100 %; everything else drops.
    assert stats.signal_stats["soh"].match_pct == pytest.approx(100.0)
    for name in ("voltage", "current", "temperature", "soc"):
        assert stats.signal_stats[name].match_pct == pytest.approx(0.0)
    # Overall match % = 0 (a pair needs EVERY signal within tol).
    assert stats.match_pct == pytest.approx(0.0)


def test_custom_tolerances_persist_in_result(twin_result_five, ecu_result_five):
    aligned = align(twin_result_five, ecu_result_five, ALIGN_NEAREST)
    tols = {"voltage": 1.5, "current": 3.0, "temperature": 1.0,
            "soc": 0.5, "soh": 0.5}
    stats = compute(aligned, tolerances=tols)
    assert stats.tolerances["voltage"] == 1.5
    assert stats.tolerances["current"] == 3.0
    assert stats.tolerances["temperature"] == 1.0


def test_unknown_signal_uses_default_with_warning(make_aligned_data):
    aligned = make_aligned_data(
        timestamps=[0.0, 0.1],
        signals={"voltage": ([3.3, 3.4], [3.31, 3.41])},
    )
    # Voltage has a default tolerance; no warning expected.
    stats = compute(aligned)
    assert stats.tolerances["voltage"] == DEFAULT_TOLERANCES["voltage"]
    assert not stats.warnings  # voltage is a known signal


# ---------------------------------------------------------------------------
# Worst-N mismatches
# ---------------------------------------------------------------------------
def test_worst_mismatches_limited_to_worst_n(twin_result_five, ecu_result_five):
    aligned = align(twin_result_five, ecu_result_five, ALIGN_NEAREST)
    stats = compute(aligned, worst_n=3)
    assert len(stats.worst_mismatches) == 3


def test_worst_mismatches_sorted_by_abs_error_descending(twin_result_five,
                                                           ecu_result_five):
    aligned = align(twin_result_five, ecu_result_five, ALIGN_NEAREST)
    stats = compute(aligned, worst_n=10)
    abs_errs = [abs(w.error) for w in stats.worst_mismatches]
    assert abs_errs == sorted(abs_errs, reverse=True)


def test_worst_mismatches_default_is_twenty(twin_result_five, ecu_result_five):
    aligned = align(twin_result_five, ecu_result_five, ALIGN_NEAREST)
    stats = compute(aligned)
    assert DEFAULT_WORST_N == 20
    # 5 samples × 5 signals = 25 candidates, so capped at 20.
    assert len(stats.worst_mismatches) <= 20


def test_worst_mismatch_fields_populated(twin_result_five, ecu_result_five):
    aligned = align(twin_result_five, ecu_result_five, ALIGN_NEAREST)
    stats = compute(aligned, worst_n=1)
    w = stats.worst_mismatches[0]
    assert isinstance(w.index, int)
    assert isinstance(w.timestamp, float)
    assert isinstance(w.signal, str)
    assert isinstance(w.twin_value, float)
    assert isinstance(w.ecu_value, float)
    assert isinstance(w.error, float)
    assert isinstance(w.within_tolerance, bool)


# ---------------------------------------------------------------------------
# Partial overlap: only 3 signals
# ---------------------------------------------------------------------------
def test_partial_overlap_stats_only_for_common(twin_result_five,
                                                 ecu_result_three):
    aligned = align(twin_result_five, ecu_result_three, ALIGN_NEAREST)
    stats = compute(aligned)
    assert set(stats.signal_stats.keys()) == {
        "voltage", "current", "temperature",
    }
    # No SoC/SoH tolerance entries.
    assert "soc" not in stats.tolerances
    assert "soh" not in stats.tolerances


# ---------------------------------------------------------------------------
# Interpolate path: 4 of 5 samples matched
# ---------------------------------------------------------------------------
def test_interpolate_stats_reflect_dropped_sample(twin_result_five,
                                                    ecu_result_five):
    aligned = align(twin_result_five, ecu_result_five, ALIGN_INTERPOLATE)
    stats = compute(aligned)
    assert stats.total_samples == 5
    assert stats.matched_samples == 4  # last ECU sample out of twin range
    for s in stats.signal_stats.values():
        assert s.n_valid == 4


# ---------------------------------------------------------------------------
# No valid pairs
# ---------------------------------------------------------------------------
def test_zero_valid_pairs_returns_zero_stats(make_aligned_data):
    aligned = make_aligned_data(
        timestamps=[0.0, 0.1],
        signals={"voltage": ([np.nan, np.nan], [3.3, 3.4])},
    )
    stats = compute(aligned)
    s = stats.signal_stats["voltage"]
    assert s.n_valid == 0
    assert s.mae == 0.0
    assert s.rmse == 0.0
    assert s.match_pct == 0.0
    assert stats.worst_mismatches == []
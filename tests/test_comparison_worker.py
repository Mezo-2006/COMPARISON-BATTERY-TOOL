"""Tests for ``comparison_worker.py`` — QThread orchestrator.

These tests instantiate a ``QApplication`` (in offscreen mode) so Qt's
signal/slot machinery works.  The worker is exercised via its signals
(``run_signal`` / ``rerun_signal``), and the ``finished`` /
``error_occurred`` signals are captured into lists for assertion.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Force the headless Qt platform before QApplication is constructed.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtWidgets

from alignment_engine import ALIGN_NEAREST, ALIGN_INTERPOLATE
from comparison_worker import ComparisonWorker
from statistics_engine import DEFAULT_WORST_N


# ---------------------------------------------------------------------------
# Session-scoped QApplication (one per test session, not per test)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    yield app


# ---------------------------------------------------------------------------
# Helpers: capture worker emissions
# ---------------------------------------------------------------------------
def _wire(worker):
    captured = {
        "progress": [],
        "finished": [],
        "errors": [],
    }
    worker.progress.connect(captured["progress"].append)
    worker.finished.connect(lambda a, s: captured["finished"].append((a, s)))
    worker.error_occurred.connect(captured["errors"].append)
    return captured


def _pump_until(qapp, predicate, max_iter=400, delay_ms=20):
    """Pump the Qt event loop until ``predicate()`` is truthy or timeout."""
    for _ in range(max_iter):
        qapp.processEvents()
        if predicate():
            return True
        QtCore.QThread.msleep(delay_ms)
    return False


# ---------------------------------------------------------------------------
# Full run via run_signal (with thread)
# ---------------------------------------------------------------------------
def test_full_run_emits_finished_with_aligned_and_stats(qapp, tmp_path,
                                                          twin_csv_path,
                                                          ecu_csv_path):
    worker = ComparisonWorker(
        twin_path=twin_csv_path,
        ecu_path=ecu_csv_path,
        alignment_method=ALIGN_NEAREST,
        worst_n=DEFAULT_WORST_N,
    )
    captured = _wire(worker)

    thread = QtCore.QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run_signal.emit)

    thread.start()
    try:
        ok = _pump_until(
            qapp,
            lambda: len(captured["finished"]) > 0 or len(captured["errors"]) > 0,
        )
        assert ok, "worker did not finish"
    finally:
        thread.quit()
        thread.wait(3000)

    assert not captured["errors"], f"unexpected error: {captured['errors']}"
    assert len(captured["finished"]) == 1
    aligned, stats = captured["finished"][0]
    assert set(aligned.signal_names) == {
        "voltage", "current", "temperature", "soc", "soh",
    }
    assert stats.total_samples == 10


def test_full_run_progress_is_monotonic(qapp, twin_csv_path, ecu_csv_path):
    worker = ComparisonWorker(twin_csv_path, ecu_csv_path)
    captured = _wire(worker)

    thread = QtCore.QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run_signal.emit)
    thread.start()
    try:
        _pump_until(qapp,
                     lambda: len(captured["finished"]) > 0
                              or len(captured["errors"]) > 0)
    finally:
        thread.quit()
        thread.wait(3000)

    # Progress should be a non-decreasing sequence ending at 100.
    assert captured["progress"][-1] == 100
    assert captured["progress"] == sorted(captured["progress"])


def test_full_run_caches_load_results(qapp, twin_csv_path, ecu_csv_path):
    worker = ComparisonWorker(twin_csv_path, ecu_csv_path)
    captured = _wire(worker)
    thread = QtCore.QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run_signal.emit)
    thread.start()
    try:
        _pump_until(qapp,
                     lambda: len(captured["finished"]) > 0
                              or len(captured["errors"]) > 0)
    finally:
        thread.quit()
        thread.wait(3000)

    assert worker.has_cached_load
    assert worker.twin_result is not None
    assert worker.ecu_result is not None
    assert worker.twin_result.row_count == 10
    assert worker.ecu_result.row_count == 10


# ---------------------------------------------------------------------------
# Error path: bad twin path
# ---------------------------------------------------------------------------
def test_bad_path_emits_error_occurred(qapp, tmp_path, ecu_csv_path):
    worker = ComparisonWorker(
        twin_path="/no/such/file.csv",
        ecu_path=ecu_csv_path,
    )
    captured = _wire(worker)

    thread = QtCore.QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run_signal.emit)
    thread.start()
    try:
        _pump_until(qapp,
                     lambda: len(captured["errors"]) > 0
                              or len(captured["finished"]) > 0)
    finally:
        thread.quit()
        thread.wait(3000)

    assert len(captured["errors"]) == 1
    assert "Comparison failed" in captured["errors"][0]
    assert not worker.has_cached_load


# ---------------------------------------------------------------------------
# Rerun: cached LoadResults, different method + tolerances
# ---------------------------------------------------------------------------
def test_rerun_uses_cached_load_results(qapp, twin_csv_path, ecu_csv_path):
    worker = ComparisonWorker(twin_csv_path, ecu_csv_path,
                              alignment_method=ALIGN_NEAREST)
    captured = _wire(worker)

    thread = QtCore.QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run_signal.emit)
    thread.start()

    # Wait for the initial run.
    _pump_until(qapp,
                 lambda: len(captured["finished"]) > 0
                          or len(captured["errors"]) > 0)
    assert not captured["errors"]
    aligned_first, _ = captured["finished"][0]
    assert aligned_first.alignment_method == "nearest"

    # Re-run with interpolate + tight tolerances.
    captured["finished"].clear()
    captured["errors"].clear()
    worker.rerun_signal.emit(
        ALIGN_INTERPOLATE,
        {"voltage": 0.01, "current": 0.01, "temperature": 0.01,
         "soc": 0.01, "soh": 0.01},
        5,
    )
    _pump_until(qapp,
                 lambda: len(captured["finished"]) > 0
                          or len(captured["errors"]) > 0)

    thread.quit()
    thread.wait(3000)

    assert not captured["errors"]
    aligned_second, stats_second = captured["finished"][0]
    assert aligned_second.alignment_method == "interpolate"
    # Cached results still around (not re-loaded from disk).
    assert worker.has_cached_load
    # Interpolate drops the last ECU sample (out of twin range).
    assert aligned_second.n_matched == 9
    # Linear twin + interpolation → errors ≈ 0 → 100 % match even at
    # tight tolerances.  Verify the tolerances actually propagated.
    assert stats_second.tolerances["voltage"] == 0.01


# ---------------------------------------------------------------------------
# Rerun without a prior run: error path
# ---------------------------------------------------------------------------
def test_rerun_without_prior_run_emits_error(qapp, tmp_path,
                                                twin_csv_path, ecu_csv_path):
    worker = ComparisonWorker(twin_csv_path, ecu_csv_path)
    captured = _wire(worker)

    thread = QtCore.QThread()
    worker.moveToThread(thread)
    thread.start()
    # Don't emit run_signal — jump straight to rerun_signal.
    worker.rerun_signal.emit(ALIGN_NEAREST, None, DEFAULT_WORST_N)
    _pump_until(qapp,
                 lambda: len(captured["errors"]) > 0
                          or len(captured["finished"]) > 0)
    thread.quit()
    thread.wait(3000)

    assert len(captured["errors"]) == 1
    assert "Cannot re-run" in captured["errors"][0]
    assert "files have not been loaded yet" in captured["errors"][0]


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------
def test_has_cached_load_false_before_run(qapp, twin_csv_path, ecu_csv_path):
    worker = ComparisonWorker(twin_csv_path, ecu_csv_path)
    assert not worker.has_cached_load
    assert worker.twin_result is None
    assert worker.ecu_result is None
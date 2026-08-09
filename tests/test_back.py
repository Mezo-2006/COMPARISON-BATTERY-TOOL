"""Smoke / integration tests for ``back.py`` MainWindow.

Drives the GUI via the same ``on_start_comparison`` /
``on_apply_rerun`` slots a user would trigger, then asserts that the
Preview / Statistics / Graphs / Export wiring populates the right
widgets.  Everything runs under the offscreen Qt platform so it can
execute headlessly on CI.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtWidgets

from back import MainWindow, _MAX_PREVIEW_ROWS
from export_engine import FORMAT_DATA_CSV, export_data


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    yield app


@pytest.fixture
def window(qapp, twin_csv_path, ecu_csv_path):
    win = MainWindow()
    # Simulate the user picking both CSVs via the browse dialogues.
    win.twin_path = twin_csv_path
    win.ecu_path = ecu_csv_path
    win._refresh_startButton_state()
    yield win
    win.close()


def _pump(qapp, predicate, max_iter=200, delay_ms=15):
    for _ in range(max_iter):
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(delay_ms / 1000)
    return False


# ---------------------------------------------------------------------------
# Tab 1: initial state + Start Comparison
# ---------------------------------------------------------------------------
def test_initial_state_start_disabled(qapp):
    win = MainWindow()
    assert not win.btnStartComparison.isEnabled()
    win.close()


def test_browse_enables_start(window):
    assert window.btnStartComparison.isEnabled()


# ---------------------------------------------------------------------------
# Full run populates every results tab
# ---------------------------------------------------------------------------
def test_full_run_populates_aligned_and_stats(window, qapp):
    window.on_start_comparison()
    ok = _pump(qapp, lambda: window.aligned_data is not None)
    assert ok
    qapp.processEvents()  # let the finished-handler populate UI

    assert set(window.aligned_data.signal_names) == {
        "voltage", "current", "temperature", "soc", "soh",
    }
    assert window.stats_result is not None
    assert window.stats_result.total_samples == 10


def test_preview_tables_populated(window, qapp):
    window.on_start_comparison()
    _pump(qapp, lambda: window.aligned_data is not None)
    qapp.processEvents()

    assert window.tableWidgetTwinPreview.rowCount() > 0
    assert window.tableWidgetEcuPreview.rowCount() > 0
    # 6 canonical columns: timestamp + 5 signals.
    assert window.tableWidgetTwinPreview.columnCount() == 6
    assert window.tableWidgetEcuPreview.columnCount() == 6


def test_statistics_cards_show_values(window, qapp):
    window.on_start_comparison()
    _pump(qapp, lambda: window.aligned_data is not None)
    qapp.processEvents()

    assert window.labelTotalSamplesValue.text() != "--"
    assert window.labelMatchedSamplesValue.text() != "--"
    assert window.labelMAEValue.text() != "--"
    assert window.labelMaxErrorValue.text() != "--"
    assert window.labelRMSEValue.text() != "--"
    assert window.labelMatchPercentageValue.text() != "--"


def test_per_signal_table_has_five_rows(window, qapp):
    window.on_start_comparison()
    _pump(qapp, lambda: window.aligned_data is not None)
    qapp.processEvents()
    assert window.tableWidgetPerSignalStats.rowCount() == 5


def test_worst_mismatches_table_populated(window, qapp):
    window.on_start_comparison()
    _pump(qapp, lambda: window.aligned_data is not None)
    qapp.processEvents()
    assert window.tableWidgetWorstMismatches.rowCount() > 0
    assert window.tableWidgetWorstMismatches.rowCount() <= 20


def test_graphs_tab_drawn(window, qapp):
    window.on_start_comparison()
    _pump(qapp, lambda: window.aligned_data is not None)
    qapp.processEvents()
    no = len(window.plotWidgetOverlay.getPlotItem().listDataItems())
    ne = len(window.plotWidgetError.getPlotItem().listDataItems())
    assert no == 10  # 5 signals * (twin + ECU), all ticked by default
    assert ne == 5   # all five signals


# ---------------------------------------------------------------------------
# Checkbox toggle re-renders plots without re-running the worker
# ---------------------------------------------------------------------------
def test_checkbox_toggle_updates_plots_only(window, qapp):
    window.on_start_comparison()
    _pump(qapp, lambda: window.aligned_data is not None)
    qapp.processEvents()

    window.checkBoxSignalVoltage.setChecked(False)
    qapp.processEvents()
    no = len(window.plotWidgetOverlay.getPlotItem().listDataItems())
    ne = len(window.plotWidgetError.getPlotItem().listDataItems())
    # Voltage drops out of both plots (8 = 4 remaining signals * 2).
    assert no == 8
    assert ne == 4

    window.checkBoxSignalVoltage.setChecked(True)
    qapp.processEvents()
    assert len(window.plotWidgetOverlay.getPlotItem().listDataItems()) == 10


# ---------------------------------------------------------------------------
# "Select only one" collapses the signal-selection row to a single signal
# ---------------------------------------------------------------------------
def test_select_only_one_forces_single_signal(window, qapp):
    window.on_start_comparison()
    _pump(qapp, lambda: window.aligned_data is not None)
    qapp.processEvents()

    window.checkBoxSignalSelectOnly.setChecked(True)
    qapp.processEvents()
    # Arming it collapses the (all-five-ticked) default down to Voltage.
    assert window.checkBoxSignalVoltage.isChecked()
    for cb in (window.checkBoxSignalCurrent, window.checkBoxSignalTemperature,
               window.checkBoxSignalSoc, window.checkBoxSignalSoh):
        assert not cb.isChecked()
    assert len(window.plotWidgetOverlay.getPlotItem().listDataItems()) == 2

    # Ticking a different signal while armed swaps the selection instead
    # of adding to it.
    window.checkBoxSignalSoh.setChecked(True)
    qapp.processEvents()
    assert not window.checkBoxSignalVoltage.isChecked()
    assert window.checkBoxSignalSoh.isChecked()
    assert len(window.plotWidgetOverlay.getPlotItem().listDataItems()) == 2

    # Disarming it doesn't change the current selection, just frees it up.
    window.checkBoxSignalSelectOnly.setChecked(False)
    qapp.processEvents()
    window.checkBoxSignalSoc.setChecked(True)
    qapp.processEvents()
    assert window.checkBoxSignalSoh.isChecked()
    assert window.checkBoxSignalSoc.isChecked()
    assert len(window.plotWidgetOverlay.getPlotItem().listDataItems()) == 4


# ---------------------------------------------------------------------------
# Apply & Re-run with a different alignment method
# ---------------------------------------------------------------------------
def test_apply_rerun_changes_alignment_method(window, qapp):
    window.on_start_comparison()
    _pump(qapp, lambda: window.aligned_data is not None)
    qapp.processEvents()
    assert window.aligned_data.alignment_method == "nearest"

    window.comboBoxAlignmentMethod.setCurrentIndex(1)  # interpolate
    window.on_apply_rerun()
    _pump(qapp,
          lambda: window.aligned_data is not None
                   and window.aligned_data.alignment_method == "interpolate")
    qapp.processEvents()
    assert window.aligned_data.alignment_method == "interpolate"
    # Interpolate drops the last sample (out of twin range).
    assert window.aligned_data.n_matched == 9


# ---------------------------------------------------------------------------
# Export end-to-end through the back.py logging helper
# ---------------------------------------------------------------------------
def test_export_logs_to_text_edit(window, qapp, tmp_path):
    window.on_start_comparison()
    _pump(qapp, lambda: window.aligned_data is not None)
    qapp.processEvents()

    out = tmp_path / "out.csv"
    result = export_data(window.aligned_data, FORMAT_DATA_CSV, str(out))
    window._log_export_result(result, kind="table")

    log_text = window.textEditExportLog.toPlainText()
    assert "Exported table (csv)" in log_text
    assert result.success
    assert out.exists()


# ---------------------------------------------------------------------------
# closeEvent doesn't hang
# ---------------------------------------------------------------------------
def test_close_event_does_not_hang(window, qapp):
    window.on_start_comparison()
    _pump(qapp, lambda: window.aligned_data is not None)
    qapp.processEvents()
    window.close()
    # Process events briefly to confirm no hang.
    for _ in range(20):
        qapp.processEvents()
        time.sleep(0.005)


# ---------------------------------------------------------------------------
# Tolerance spinbox precision + Apply & Re-run uses the edited value
# ---------------------------------------------------------------------------
def test_tolerance_spinbox_accepts_four_decimals(qapp):
    """Regression: the spinbox used to default to 2 decimals (so a 3rd
    decimal was rejected and the value silently stayed at 2.00). After
    setting decimals=4 in f.ui the spinbox must accept a 4-decimal value.
    """
    win = MainWindow()
    try:
        assert win.doubleSpinBoxVoltageTolerance.decimals() == 4
        win.doubleSpinBoxVoltageTolerance.setValue(0.1234)
        assert win.doubleSpinBoxVoltageTolerance.value() == 0.1234
        # Minimum is now 0 (was 0.1) so a tiny tolerance is accepted.
        win.doubleSpinBoxVoltageTolerance.setValue(0.0001)
        assert win.doubleSpinBoxVoltageTolerance.value() == 0.0001
    finally:
        win.close()


def test_apply_rerun_uses_edited_tolerance(window, qapp):
    """Regression: editing the voltage tolerance then Apply & Re-run must
    produce a statistics result whose voltage ``tolerance_pct`` equals the
    edited value (not the 2.00 default).
    """
    window.on_start_comparison()
    _pump(qapp, lambda: window.aligned_data is not None)
    qapp.processEvents()

    new_tol = 5.5
    window.doubleSpinBoxVoltageTolerance.setValue(new_tol)
    window.on_apply_rerun()
    _pump(qapp, lambda: window.stats_result is not None)
    qapp.processEvents()

    volt_stat = window.stats_result.signal_stats["voltage"]
    assert volt_stat.tolerance_pct == new_tol
    # The spinbox itself must still hold the edited value (not revert).
    assert window.doubleSpinBoxVoltageTolerance.value() == new_tol
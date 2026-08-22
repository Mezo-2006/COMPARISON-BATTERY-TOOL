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
from event_detector import EVENT_TIMEOUT


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
    for plot in (window.plotWidgetVoltage, window.plotWidgetCurrent,
                 window.plotWidgetTemperature, window.plotWidgetSoc,
                 window.plotWidgetSoh):
        assert len(plot.getPlotItem().listDataItems()) == 2  # twin + ECU
    ne = len(window.plotWidgetError.getPlotItem().listDataItems())
    assert ne == 5   # all five signals ticked by default


# ---------------------------------------------------------------------------
# Checkbox toggle re-renders the error graph only -- overlay plots are
# unconditional and don't react to the error-graph selection.
# ---------------------------------------------------------------------------
def test_checkbox_toggle_updates_error_graph_only(window, qapp):
    window.on_start_comparison()
    _pump(qapp, lambda: window.aligned_data is not None)
    qapp.processEvents()

    window.checkBoxSignalVoltage.setChecked(False)
    qapp.processEvents()
    # Voltage's own overlay plot is untouched by the error-graph checkbox.
    assert len(window.plotWidgetVoltage.getPlotItem().listDataItems()) == 2
    ne = len(window.plotWidgetError.getPlotItem().listDataItems())
    assert ne == 4  # voltage dropped from the error graph

    window.checkBoxSignalVoltage.setChecked(True)
    qapp.processEvents()
    assert len(window.plotWidgetError.getPlotItem().listDataItems()) == 5


# ---------------------------------------------------------------------------
# Live tab: "Freeze View" and "Take Snapshot" are separate controls now
# (used to be one combined "Freeze & Take Snapshot" button).
# ---------------------------------------------------------------------------
def test_freeze_view_pauses_graphs_unfreeze_resumes(
    window, qapp, make_aligned_data,
):
    aligned1 = make_aligned_data(
        timestamps=[0.0, 10.0, 100.0],
        signals={"voltage": ([3.3, 3.3, 3.3], [3.3, 3.3, 3.3])},
    )
    window._on_live_aligned(aligned1)
    qapp.processEvents()
    (_v_min, v_max), _ = window.plotWidgetVoltage.getPlotItem().viewRange()
    assert v_max == pytest.approx(100.0)

    window.btnLiveFreeze.setChecked(True)
    qapp.processEvents()
    assert window.btnLiveFreeze.text() == "Resume Live View"

    # A later tick's new data must NOT slide the frozen view.
    aligned2 = make_aligned_data(
        timestamps=[0.0, 10.0, 200.0],
        signals={"voltage": ([3.3, 3.3, 3.3], [3.3, 3.3, 3.3])},
    )
    window._on_live_aligned(aligned2)
    qapp.processEvents()
    (_v_min2, v_max2), _ = window.plotWidgetVoltage.getPlotItem().viewRange()
    assert v_max2 == pytest.approx(100.0)

    # Un-freezing snaps straight back to the latest cached data.
    window.btnLiveFreeze.setChecked(False)
    qapp.processEvents()
    (_v_min3, v_max3), _ = window.plotWidgetVoltage.getPlotItem().viewRange()
    assert v_max3 == pytest.approx(200.0)
    assert window.btnLiveFreeze.text() == "Freeze View"


def test_take_snapshot_captures_export_data_independent_of_freeze(
    window, qapp,
):
    import json

    def payload(obj):
        return json.dumps(obj).encode("utf-8")

    window.live_controller.reset.emit()
    # A huge interval so no tick fires mid-ingest -- this test only
    # cares about the buffer, not the periodic re-align.
    window.live_controller.start.emit(100_000, "bms/twin/#", "bms/actual/#")
    qapp.processEvents()
    base = 1_700_000_000_000.0
    for i in range(5):
        window.live_controller.ingest.emit(
            "bms/twin", payload({"t": base + i * 100, "v": 3.30 + i * 0.01}), 0.0,
        )
        window.live_controller.ingest.emit(
            "bms/actual", payload({"t": base + i * 100, "v": 3.31 + i * 0.01}), 0.0,
        )
    # Cross-thread ingest -- pump until the controller's own QThread has
    # actually processed the queued signals, or the buffer is still
    # empty by the time on_live_snapshot below reads it.
    ok = _pump(qapp, lambda: window.live_controller.buffer.twin_count == 5)
    assert ok
    assert window.aligned_data is None  # nothing ticked/snapshotted yet

    # Freezing first must not block Snapshot from capturing data.
    window.btnLiveFreeze.setChecked(True)
    qapp.processEvents()
    window.on_live_snapshot()
    qapp.processEvents()

    assert window.aligned_data is not None
    assert window.stats_result is not None
    assert window.aligned_data.n_total == 5


def test_reconnect_clears_freeze_view_state(window, qapp, monkeypatch):
    """A fresh MQTT connect must not carry Freeze View over from a
    previous session -- exercises the real ``_mqtt_connect`` path (via
    the same ``_FakeMqttWorker`` stand-in the reconnect-race test uses)
    rather than re-deriving its reset logic by hand.
    """
    import back as back_module

    monkeypatch.setattr(back_module, "MqttWorker", _FakeMqttWorker)
    window.checkBoxLiveAutoRefresh.setChecked(True)

    window.btnLiveFreeze.setChecked(True)
    qapp.processEvents()
    assert window.btnLiveFreeze.isChecked()
    window.plot_manager.set_live_follow(False)

    window.btnLiveConnect.setChecked(True)  # -> on_live_connect_toggled -> _mqtt_connect
    qapp.processEvents()

    assert not window.btnLiveFreeze.isChecked()
    assert window.btnLiveFreeze.text() == "Freeze View"
    assert all(window.plot_manager._live_follow.values())


# ---------------------------------------------------------------------------
# Live event detection must not mistake millisecond-scale aligned
# timestamps for wall-clock seconds (regression for the EVENT_TIMEOUT
# flood described in context.md's "Known gaps" -- see
# back.py._run_live_event_detection).
# ---------------------------------------------------------------------------
def test_live_event_detection_no_false_timeout_on_ms_timestamps(
    window, make_aligned_data,
):
    # Two consecutive "ticks" 500 ms apart on the live (Unix-epoch-ms)
    # timeline. Under the old bug (sample timestamp fed straight into
    # ``now``), the second tick's per-row elapsed would read as ~500 ms
    # of raw delta against a 1.5-*second* expected-interval threshold --
    # i.e. it looked like it was hundreds of times overdue, so
    # EVENT_TIMEOUT fired on essentially every tick.
    aligned1 = make_aligned_data(
        timestamps=[1_700_000_000_000.0],
        signals={"voltage": ([3.30], [3.30])},
    )
    window._run_live_event_detection(aligned1)

    aligned2 = make_aligned_data(
        timestamps=[1_700_000_000_000.0, 1_700_000_000_500.0],
        signals={"voltage": ([3.30, 3.31], [3.30, 3.31])},
    )
    window._run_live_event_detection(aligned2)

    events = [a.event for a in window.event_window.log.alerts]
    assert EVENT_TIMEOUT not in events


# ---------------------------------------------------------------------------
# Live MQTT: a stale worker's delayed "disconnected" signal must not kill
# a session the user has already reconnected (regression for "disconnect
# then connect, graphs no longer appear" -- see
# back.py._teardown_mqtt_thread).
# ---------------------------------------------------------------------------
class _FakeMqttWorker(QtCore.QObject):
    """Signal-compatible stand-in for MqttWorker with no real networking.

    Slots are plain unconnected signals -- the test drives ``connected`` /
    ``disconnected`` by hand to control timing precisely, instead of
    racing real paho callbacks.
    """

    connected = QtCore.pyqtSignal()
    disconnected = QtCore.pyqtSignal(int)
    message_received = QtCore.pyqtSignal(str, bytes, float)
    error_occurred = QtCore.pyqtSignal(str)
    connect_broker = QtCore.pyqtSignal(str, int, str, str, list)
    publish_broker = QtCore.pyqtSignal(str, object, int)
    disconnect_broker = QtCore.pyqtSignal()


def test_reconnect_survives_stale_disconnect_signal(window, qapp, monkeypatch):
    import back as back_module

    created_workers = []
    real_init = _FakeMqttWorker.__init__

    def _tracking_init(self):
        real_init(self)
        created_workers.append(self)

    monkeypatch.setattr(_FakeMqttWorker, "__init__", _tracking_init)
    monkeypatch.setattr(back_module, "MqttWorker", _FakeMqttWorker)

    window.checkBoxLiveAutoRefresh.setChecked(True)
    window.spinBoxLiveInterval.setValue(50)

    # Connect once.
    window.btnLiveConnect.setChecked(True)
    qapp.processEvents()
    worker_a = created_workers[-1]
    assert window.live_controller._timer is not None

    # User disconnects...
    window.btnLiveConnect.setChecked(False)
    qapp.processEvents()

    # ...then immediately reconnects, before worker A's async
    # "disconnected" (queued from paho's own network thread in real life)
    # has actually been delivered to the GUI thread.
    window.btnLiveConnect.setChecked(True)
    qapp.processEvents()
    worker_b = created_workers[-1]
    assert worker_b is not worker_a
    assert window.live_controller._timer is not None

    # Worker A's stale disconnect callback finally arrives.
    worker_a.disconnected.emit(0)
    qapp.processEvents()

    # The new session must be unaffected: still connected, timer still
    # armed. Before the fix, this stale signal was still wired to
    # ``_on_mqtt_disconnected``, which unchecked the button and stopped
    # the live controller's re-align timer -- freezing the graphs.
    assert window.btnLiveConnect.isChecked()
    assert window.live_controller._timer is not None


# ---------------------------------------------------------------------------
# "Select only one" collapses the error-graph selection to a single signal
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
    assert len(window.plotWidgetError.getPlotItem().listDataItems()) == 1
    # Overlay plots don't care about the error-graph selection.
    assert len(window.plotWidgetSoh.getPlotItem().listDataItems()) == 2

    # Ticking a different signal while armed swaps the selection instead
    # of adding to it.
    window.checkBoxSignalSoh.setChecked(True)
    qapp.processEvents()
    assert not window.checkBoxSignalVoltage.isChecked()
    assert window.checkBoxSignalSoh.isChecked()
    assert len(window.plotWidgetError.getPlotItem().listDataItems()) == 1

    # Disarming it doesn't change the current selection, just frees it up.
    window.checkBoxSignalSelectOnly.setChecked(False)
    qapp.processEvents()
    window.checkBoxSignalSoc.setChecked(True)
    qapp.processEvents()
    assert window.checkBoxSignalSoh.isChecked()
    assert window.checkBoxSignalSoc.isChecked()
    assert len(window.plotWidgetError.getPlotItem().listDataItems()) == 2


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
    # ``stats_result`` is already non-None from the run above -- pump for
    # the *new* value specifically, not just "not None", or the predicate
    # can be satisfied by the stale pre-rerun result before the
    # cross-thread rerun has actually finished (this was an intermittent
    # flake in the full suite; deterministic in isolation only by luck).
    ok = _pump(
        qapp,
        lambda: (window.stats_result is not None
                 and window.stats_result.signal_stats["voltage"].tolerance_pct
                 == new_tol),
    )
    assert ok
    qapp.processEvents()

    volt_stat = window.stats_result.signal_stats["voltage"]
    assert volt_stat.tolerance_pct == new_tol
    # The spinbox itself must still hold the edited value (not revert).
    assert window.doubleSpinBoxVoltageTolerance.value() == new_tol


# ---------------------------------------------------------------------------
# SoC / SoH now have their own Config-tab tolerance + threshold-warning
# controls, matching Voltage / Current / Temperature.
# ---------------------------------------------------------------------------
def test_soc_soh_tolerance_spinboxes_feed_config_settings(qapp):
    win = MainWindow()
    try:
        assert win.doubleSpinBoxSocTolerance.decimals() == 4
        assert win.doubleSpinBoxSohTolerance.decimals() == 4
        win.doubleSpinBoxSocTolerance.setValue(3.25)
        win.doubleSpinBoxSohTolerance.setValue(4.5)
        _method, tolerances = win._read_config_settings()
        assert tolerances["soc"] == 3.25
        assert tolerances["soh"] == 4.5
    finally:
        win.close()


def test_apply_rerun_uses_edited_soc_tolerance(window, qapp):
    window.on_start_comparison()
    _pump(qapp, lambda: window.aligned_data is not None)
    qapp.processEvents()

    new_tol = 6.25
    window.doubleSpinBoxSocTolerance.setValue(new_tol)
    window.on_apply_rerun()
    # ``stats_result`` is already non-None from the initial run above, so
    # pump for the *new* value specifically -- otherwise the predicate
    # can be satisfied by the stale pre-rerun result before the
    # cross-thread rerun has actually finished (matches the same
    # pump-predicate shape as ``test_apply_rerun_uses_edited_tolerance``
    # above, made robust here rather than relying on timing luck).
    ok = _pump(
        qapp,
        lambda: (window.stats_result is not None
                 and window.stats_result.signal_stats["soc"].tolerance_pct
                 == new_tol),
    )
    assert ok
    qapp.processEvents()

    soc_stat = window.stats_result.signal_stats["soc"]
    assert soc_stat.tolerance_pct == new_tol


def test_soc_soh_threshold_checkbox_enables_spinbox(qapp):
    win = MainWindow()
    try:
        assert not win.doubleSpinBoxSocThreshold.isEnabled()
        win.checkBoxEnableSocThreshold.setChecked(True)
        assert win.doubleSpinBoxSocThreshold.isEnabled()
        assert not win.doubleSpinBoxSohThreshold.isEnabled()
        win.checkBoxEnableSohThreshold.setChecked(True)
        assert win.doubleSpinBoxSohThreshold.isEnabled()

        thresholds = win._read_threshold_settings()
        assert thresholds["soc"][0] is True
        assert thresholds["soh"][0] is True
    finally:
        win.close()


def test_soc_threshold_warning_drawn_on_soc_plot(window, qapp):
    window.on_start_comparison()
    _pump(qapp, lambda: window.aligned_data is not None)
    qapp.processEvents()

    window.checkBoxEnableSocThreshold.setChecked(True)
    window.doubleSpinBoxSocThreshold.setValue(0.0)  # every SoC value "exceeds" 0
    qapp.processEvents()

    assert window.labelSocWarning.text() != ""
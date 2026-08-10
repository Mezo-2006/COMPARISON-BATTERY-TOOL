"""Main window for the Digital Twin Validation Tool.

This module is the **only** file that imports Qt and touches the UI
widgets directly.  It owns the :class:`ComparisonWorker` (a worker-object
moved to a ``QThread``), the :class:`PlotManager`, and optionally the
:class:`MqttWorker` for live mode.

Architecture
------------

The window follows the same pattern as Task 1's ``tool/back.py`` and
Task 3's ``tool_mbd_params/back.py``: it mixes in ``Ui_MainWindow`` (the
PyQt5-generated recipe) via multiple inheritance and calls
``setupUi(self)`` in ``__init__``.  All backend logic lives in the
pure-Python engines (``data_loader``, ``alignment_engine``,
``statistics_engine``, ``export_engine``) so they can be unit-tested in
isolation.  This file is pure wiring — every handler either delegates to
an engine or updates a widget.

State machine
-------------

::

    Idle ──browse twin──► TwinLoaded ──browse ECU──► BothLoaded
          ──browse ECU──► BothLoaded (twin optional in the UI but the
          comparison requires both, so the Start button only enables
          once both paths are set)

    BothLoaded ──Start Comparison──► Comparing ──worker finished──► ResultsReady
                                          ──worker error────► BothLoaded

    ResultsReady ──Apply & Re-run──► Comparing ──finished──► ResultsReady
                ──new files────────► Idle / TwinLoaded / BothLoaded

The "Apply & Re-run" path on the Config tab re-uses the
``ComparisonWorker`` instance with the already-cached ``LoadResult``s,
so changing tolerances or the alignment method is fast (no file re-I/O).
"""

from __future__ import annotations

import math
import sys
from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QTableWidgetItem

from f import Ui_MainWindow

from data_loader import LoadResult
from alignment_engine import (
    AlignedData,
    AlignmentError,
    ALIGN_NEAREST,
    ALIGN_INTERPOLATE,
)
from statistics_engine import StatisticsResult, DEFAULT_WORST_N
from comparison_worker import ComparisonWorker
from plot_manager import PlotManager
from export_engine import (
    export_data,
    export_report,
    ExportResult,
    FORMAT_DATA_CSV,
    FORMAT_DATA_EXCEL,
    FORMAT_REPORT_PDF,
    FORMAT_REPORT_HTML,
)
from mqtt_worker import MqttWorker
from live_controller import LiveController
from event_detector import EventDetector
from event_window import EventWindow


# ---------------------------------------------------------------------------
# Helpers: QTableWidget population
# ---------------------------------------------------------------------------
_MAX_PREVIEW_ROWS = 100
_FLOAT_FMT = "{:.4f}"

# Live-mode Graphs tab: width, in milliseconds (the live pipeline's wire
# timestamp unit -- see synthetic_data_generator's "Timestamp convention"
# section), of the sliding X-range window kept on screen. Passed to
# PlotManager.update(..., live_window_ms=...) only from the live re-align
# tick (_on_live_aligned) -- offline runs and live Freeze/Snapshot show the
# full accumulated range instead, matching a one-shot comparison rather
# than a scrolling stream. See PlotManager.update and _apply_live_window
# for why this must be pinned every tick rather than left to pyqtgraph's
# default auto-range: without it, a live plot's view either "compresses"
# (auto-fits the whole, ever-growing buffer instead of scrolling) or can
# get stuck on a stale manual pan/zoom range that no longer contains the
# fresh data after a disconnect + reconnect (the buffer is reset on
# reconnect), which reads as "the graphs stopped appearing".
_LIVE_PLOT_WINDOW_MS = 60_000.0

# Same idea, but for a "sequence" axis (a plain sample-id counter — see
# synthetic_data_generator's id_mode="sequence"): a millisecond-scale
# window is meaningless once the axis isn't time at all, so live mode
# uses this instead whenever ``AlignedData.axis_kind == "sequence"``
# (last N ids, rather than last N ms).
_LIVE_PLOT_WINDOW_SEQUENCE = 200.0


def _populate_preview_table(
    table: QtWidgets.QTableWidget,
    load_result: LoadResult,
    max_rows: int = _MAX_PREVIEW_ROWS,
) -> None:
    """Fill a ``QTableWidget`` with the first ``max_rows`` of a LoadResult.

    Columns come from ``load_result.df.columns`` (canonical names).
    Numeric values are formatted to 4 decimals; NaN renders as an empty
    cell so the user doesn't see "nan" everywhere.
    """
    df = load_result.df
    cols = list(df.columns)
    n_rows = min(len(df), max_rows)

    table.setRowCount(n_rows)
    table.setColumnCount(len(cols))
    table.setHorizontalHeaderLabels(cols)

    for r in range(n_rows):
        for c, col in enumerate(cols):
            val = df.iloc[r][col]
            if isinstance(val, float) and math.isnan(val):
                cell = ""
            elif isinstance(val, float):
                cell = _FLOAT_FMT.format(val)
            else:
                cell = str(val)
            table.setItem(r, c, QTableWidgetItem(cell))

    table.horizontalHeader().setSectionResizeMode(
        QtWidgets.QHeaderView.Stretch
    )


def _populate_per_signal_table(
    table: QtWidgets.QTableWidget,
    stats: StatisticsResult,
) -> None:
    headers = ["Signal", "MAE", "RMSE", "Max Error", "Bias",
               "Match %", "Tol %"]
    rows = list(stats.signal_stats.values())
    table.setRowCount(len(rows))
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)

    for r, s in enumerate(rows):
        cells = [
            s.name,
            f"{s.mae:.6f}",
            f"{s.rmse:.6f}",
            f"{s.max_error:.6f}",
            f"{s.mean_error:+.6f}",
            f"{s.match_pct:.2f}",
            f"{s.tolerance_pct:.2f}",
        ]
        for c, txt in enumerate(cells):
            table.setItem(r, c, QTableWidgetItem(txt))

    table.horizontalHeader().setSectionResizeMode(
        QtWidgets.QHeaderView.Stretch
    )


def _populate_worst_mismatches_table(
    table: QtWidgets.QTableWidget,
    stats: StatisticsResult,
) -> None:
    headers = ["Timestamp", "Signal", "Twin", "ECU", "Error",
               "Within Tol?"]
    rows = stats.worst_mismatches
    table.setRowCount(len(rows))
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)

    for r, w in enumerate(rows):
        cells = [
            f"{w.timestamp:.3f}",
            w.signal,
            f"{w.twin_value:.4f}",
            f"{w.ecu_value:.4f}",
            f"{w.error:+.4f}",
            "yes" if w.within_tolerance else "no",
        ]
        for c, txt in enumerate(cells):
            table.setItem(r, c, QTableWidgetItem(txt))

    table.horizontalHeader().setSectionResizeMode(
        QtWidgets.QHeaderView.Stretch
    )


def _aligned_row_to_dicts(aligned: AlignedData, i: int) -> tuple:
    """One aligned row -> ``(real_data, digital_twin_data)`` dicts.

    ECU is the "real" side (event_detector's convention matches
    alignment_engine's: ECU is ground truth). A signal is only included
    when its value at this row isn't NaN, so an unmatched sample reads
    as "missing" to ``EventDetector`` rather than a bogus NaN comparison.
    """
    ts = float(aligned.timestamps[i])
    real: dict = {"timestamp": ts}
    twin: dict = {"timestamp": ts}
    for name, sig in aligned.signals.items():
        ecu_v = float(sig.ecu_values[i])
        twin_v = float(sig.twin_values[i])
        if not math.isnan(ecu_v):
            real[name] = ecu_v
        if not math.isnan(twin_v):
            twin[name] = twin_v
    return real, twin


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    """Top-level window that owns the comparison pipeline and UI wiring."""

    # --- Config-tab combo → alignment-method constant mapping ---
    _ALIGNMENT_BY_INDEX = (ALIGN_NEAREST, ALIGN_INTERPOLATE)
    # --- Report-tab combo → export format constant mapping ---
    _DATA_FORMAT_BY_INDEX = (FORMAT_DATA_CSV, FORMAT_DATA_EXCEL)
    _REPORT_FORMAT_BY_INDEX = (FORMAT_REPORT_PDF, FORMAT_REPORT_HTML)

    def __init__(self) -> None:
        super().__init__()
        self.setupUi(self)

        # --- File state ----------------------------------------------------
        self.twin_path: Optional[str] = None
        self.ecu_path: Optional[str] = None

        # --- Results cache (from worker.finished) --------------------------
        self.aligned_data: Optional[AlignedData] = None
        self.stats_result: Optional[StatisticsResult] = None

        # --- Worker thread plumbing ---------------------------------------
        self.comparison_thread: Optional[QtCore.QThread] = None
        self.comparison_worker: Optional[ComparisonWorker] = None

        # --- Plot manager -------------------------------------------------
        self.plot_manager = PlotManager(
            self.plotWidgetVoltage,
            self.plotWidgetCurrent,
            self.plotWidgetTemperature,
            self.plotWidgetSoc,
            self.plotWidgetSoh,
            self.plotWidgetError,
            warning_labels={
                "voltage": self.labelVoltageWarning,
                "current": self.labelCurrentWarning,
            },
        )
        # Signal checkboxes on the Graphs tab, in a fixed order used by
        # both the "select only one" enforcement and _enabled_signals_set.
        self._signal_checkboxes = (
            self.checkBoxSignalVoltage,
            self.checkBoxSignalCurrent,
            self.checkBoxSignalTemperature,
            self.checkBoxSignalSoc,
            self.checkBoxSignalSoh,
        )
        self._signal_checkbox_names = {
            self.checkBoxSignalVoltage: "voltage",
            self.checkBoxSignalCurrent: "current",
            self.checkBoxSignalTemperature: "temperature",
            self.checkBoxSignalSoc: "soc",
            self.checkBoxSignalSoh: "soh",
        }

        # --- MQTT worker (Live tab) ---------------------------------------
        self.mqtt_thread: Optional[QtCore.QThread] = None
        self.mqtt_worker: Optional[MqttWorker] = None

        # --- Live controller (Live tab) -----------------------------------
        # Owns the LiveBuffer + the periodic re-align timer; runs on its
        # own QThread so heavy compute (align + stats + plots data prep)
        # never blocks the GUI thread. Like the comparison worker, this
        # is a long-lived worker — created once at startup, started/
        # stopped on connect/disconnect.
        self.live_thread: Optional[QtCore.QThread] = None
        self.live_controller: Optional[LiveController] = None

        # --- Event Detection (Tab 8) ---------------------------------------
        # EventWindow is a plain QWidget (not part of the pyuic5-generated
        # f.py) — the .ui file only reserves an empty tab + layout for it,
        # so it's embedded here rather than hand-authored in Designer XML.
        # Two detector instances because the two data sources need
        # different lifetimes: the offline one is thrown away and rebuilt
        # on every full run (a "run" is a self-contained batch), while the
        # live one must persist across ticks so spike/freeze/oscillation
        # history isn't lost between re-align timer firings.
        self.event_window = EventWindow()
        self.tabEvents.layout().addWidget(self.event_window)
        self.event_detector = EventDetector()
        self.live_event_detector = EventDetector()
        self._live_event_last_ts: float = float("-inf")

        # --- Wire everything ---------------------------------------------
        self._connect_signals()
        self._setup_live_controller()

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------
    def _connect_signals(self) -> None:
        # Tab 1: file browse + start
        self.btnBrowseTwinCsv.clicked.connect(self.on_browse_twin)
        self.btnBrowseEcu.clicked.connect(self.on_browse_ecu)
        self.btnStartComparison.clicked.connect(self.on_start_comparison)

        # Tab 5: Config re-run
        self.btnApplyRerun.clicked.connect(self.on_apply_rerun)

        # Tab 4: Graphs — signal-selection checkboxes + "select only one"
        for cb in self._signal_checkboxes:
            cb.toggled.connect(self.on_signal_selection_changed)
        self.checkBoxSignalSelectOnly.toggled.connect(
            self.on_select_only_one_toggled
        )

        # Tab 5: Threshold warnings — enable checkbox toggles its spinbox
        # and both re-render the plots (no realignment needed).
        self.checkBoxEnableVoltageThreshold.toggled.connect(
            self.doubleSpinBoxVoltageThreshold.setEnabled
        )
        self.checkBoxEnableCurrentThreshold.toggled.connect(
            self.doubleSpinBoxCurrentThreshold.setEnabled
        )
        for w in (self.checkBoxEnableVoltageThreshold,
                  self.checkBoxEnableCurrentThreshold):
            w.toggled.connect(self.on_signal_checkbox_changed)
        for w in (self.doubleSpinBoxVoltageThreshold,
                  self.doubleSpinBoxCurrentThreshold):
            w.valueChanged.connect(self.on_signal_checkbox_changed)

        # Tab 6: Export
        self.btnExportData.clicked.connect(self.on_export_data)
        self.btnExportReport.clicked.connect(self.on_export_report)

        # Tab 7: Live MQTT
        self.btnLiveConnect.toggled.connect(self.on_live_connect_toggled)
        self.btnLiveSnapshot.clicked.connect(self.on_live_snapshot)
        self.checkBoxLiveAutoRefresh.toggled.connect(
            self.on_live_auto_refresh_toggled
        )
        self.spinBoxLiveInterval.valueChanged.connect(
            self.on_live_interval_changed
        )

    def _setup_live_controller(self) -> None:
        """Create the long-lived LiveController + QThread once at startup.

        The controller stays alive across connect/disconnect cycles so
        its accumulated buffers aren't lost on a brief network blip
        (we only clear buffers on an explicit start). Moving the
        controller to its own thread means every ``ingest`` emission
        and every timer tick runs off the GUI thread.
        """
        self.live_thread = QtCore.QThread(self)
        self.live_controller = LiveController()
        self.live_controller.moveToThread(self.live_thread)

        # Forward controller results to the *same* populate methods the
        # offline path uses — no new widgets, no duplicate logic.
        self.live_controller.aligned_ready.connect(self._on_live_aligned)
        self.live_controller.stats_ready.connect(self._on_live_stats)
        self.live_controller.sample_count_changed.connect(
            self._on_live_sample_count
        )
        self.live_controller.error_occurred.connect(self._on_live_error)

        self.live_thread.finished.connect(self.live_thread.deleteLater)
        self.live_thread.start()

    # ------------------------------------------------------------------
    # Tab 1 — file browse
    # ------------------------------------------------------------------
    def on_browse_twin(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Digital Twin CSV", "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return
        self.twin_path = path
        self.lineEditTwinCsvPath.setText(path)
        self.labelTwinCsvInfo.setText("File selected.  Click Start Comparison "
                                       "after both files are chosen.")
        self._refresh_startButton_state()

    def on_browse_ecu(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ECU CSV", "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return
        self.ecu_path = path
        self.lineEditEcuPath.setText(path)
        self.labelEcuInfo.setText("File selected.  Click Start Comparison "
                                    "after both files are chosen.")
        self._refresh_startButton_state()

    def _refresh_startButton_state(self) -> None:
        self.btnStartComparison.setEnabled(
            self.twin_path is not None and self.ecu_path is not None
        )

    # ------------------------------------------------------------------
    # Tab 1 — start comparison (full run)
    # ------------------------------------------------------------------
    def on_start_comparison(self) -> None:
        """Create a fresh worker, move it to a QThread, and start."""
        if self.twin_path is None or self.ecu_path is None:
            QMessageBox.warning(self, "Missing files",
                                 "Please browse for both CSV files first.")
            return

        # Disable Start + Apply & Re-run while running.
        self.btnStartComparison.setEnabled(False)
        self.btnApplyRerun.setEnabled(False)
        self.progressBarComparison.setValue(0)

        # Tear down any previous worker/thread so we don't leak threads.
        self._teardown_comparison_thread()

        # Read current Config-tab settings so the first run uses them.
        method, tolerances = self._read_config_settings()

        self.comparison_thread = QtCore.QThread(self)
        self.comparison_worker = ComparisonWorker(
            twin_path=self.twin_path,
            ecu_path=self.ecu_path,
            alignment_method=method,
            tolerances=tolerances,
            worst_n=DEFAULT_WORST_N,
        )
        self.comparison_worker.moveToThread(self.comparison_thread)

        # Wire signals from worker → GUI slots.  Note: we deliberately do
        # NOT wire ``finished → thread.quit`` or ``finished → deleteLater``
        # here because the worker holds the cached LoadResults needed for
        # the Config tab's ``Apply & Re-run``.  The thread's event loop
        # stays alive so a later ``rerun`` slot can be queued onto it;
        # cleanup happens explicitly in ``_teardown_comparison_thread``
        # (called on close and on the next full run).
        self.comparison_worker.progress.connect(
            self.progressBarComparison.setValue
        )
        self.comparison_worker.finished.connect(self.on_comparison_finished)
        self.comparison_worker.error_occurred.connect(
            self.on_comparison_error
        )
        # Thread lifecycle: started → emit run_signal (which is
        # self-connected to the run slot on the worker thread).  We
        # don't wire finished → quit here because the worker + thread
        # must stay alive for the Config tab's Apply & Re-run.
        self.comparison_thread.started.connect(
            self.comparison_worker.run_signal.emit
        )

        self.comparison_thread.start()

    def _teardown_comparison_thread(self) -> None:
        """Quit + wait on any existing comparison thread before replacing it."""
        if self.comparison_thread is not None:
            self.comparison_thread.quit()
            self.comparison_thread.wait(3000)
            self.comparison_thread = None
            self.comparison_worker = None

    # ------------------------------------------------------------------
    # Worker slots (run on the GUI thread — queued delivery)
    # ------------------------------------------------------------------
    def on_comparison_finished(
        self,
        aligned: AlignedData,
        stats: StatisticsResult,
    ) -> None:
        """Populate Preview / Statistics / Graphs from the worker's output."""
        self.aligned_data = aligned
        self.stats_result = stats

        # --- Warnings (status bar) --------------------------------------
        all_warnings = aligned.warnings + stats.warnings
        if all_warnings:
            self.statusBar().showMessage(
                "Warnings: " + " | ".join(all_warnings[:3])
                + (" …" if len(all_warnings) > 3 else ""),
                10000,
            )

        # --- Preview tab -------------------------------------------------
        if self.comparison_worker is not None:
            twin_lr = self.comparison_worker.twin_result
            ecu_lr = self.comparison_worker.ecu_result
            if twin_lr is not None:
                _populate_preview_table(self.tableWidgetTwinPreview, twin_lr)
                self.labelTwinPreviewInfo.setText(
                    f"{twin_lr.row_count} rows × "
                    f"{len(twin_lr.columns_found)} cols  "
                    f"({twin_lr.time_range[0]:.2f}–{twin_lr.time_range[1]:.2f} s)"
                )
            if ecu_lr is not None:
                _populate_preview_table(self.tableWidgetEcuPreview, ecu_lr)
                self.labelEcuPreviewInfo.setText(
                    f"{ecu_lr.row_count} rows × "
                    f"{len(ecu_lr.columns_found)} cols  "
                    f"({ecu_lr.time_range[0]:.2f}–{ecu_lr.time_range[1]:.2f} s)"
                )

        # --- Statistics tab ---------------------------------------------
        self._populate_statistics_tab(stats)

        # --- Graphs tab -------------------------------------------------
        self.plot_manager.update(aligned, self._enabled_signals_set(),
                                 self._read_threshold_settings())

        # --- Events tab ---------------------------------------------------
        self._run_event_detection(aligned)

        # --- Re-enable buttons -----------------------------------------
        self.btnStartComparison.setEnabled(True)
        self.btnApplyRerun.setEnabled(self.comparison_worker is not None
                                      and self.comparison_worker.has_cached_load)
        # Jump to Statistics tab so the user sees results immediately.
        self.tabWidgetMain.setCurrentWidget(self.tabStatistics)

    def on_comparison_error(self, msg: str) -> None:
        self.btnStartComparison.setEnabled(
            self.twin_path is not None and self.ecu_path is not None
        )
        self.btnApplyRerun.setEnabled(False)
        QMessageBox.critical(self, "Comparison failed", msg)

    # ------------------------------------------------------------------
    # Tab 3 — Statistics population
    # ------------------------------------------------------------------
    def _populate_statistics_tab(self, stats: StatisticsResult) -> None:
        self.labelTotalSamplesValue.setText(str(stats.total_samples))
        self.labelMatchedSamplesValue.setText(str(stats.matched_samples))
        self.labelMAEValue.setText(f"{stats.mean_abs_error:.6f}")
        self.labelMaxErrorValue.setText(f"{stats.max_error:.6f}")
        self.labelRMSEValue.setText(f"{stats.rmse:.6f}")
        self.labelMatchPercentageValue.setText(f"{stats.match_pct:.2f} %")
        _populate_per_signal_table(self.tableWidgetPerSignalStats, stats)
        _populate_worst_mismatches_table(self.tableWidgetWorstMismatches, stats)

    # ------------------------------------------------------------------
    # Tab 8 — Event Detection
    # ------------------------------------------------------------------
    def _run_event_detection(self, aligned: AlignedData) -> None:
        """Full offline pass: fresh detector, replay the whole aligned table.

        Runs on Start Comparison *and* Apply & Re-run (both funnel through
        ``on_comparison_finished``) — a full run is a self-contained batch,
        so the event log is rebuilt from scratch each time rather than
        appended to, matching the Statistics/Graphs tabs' behaviour.
        Each row's own timestamp drives the detector's ``now`` so
        Timeout/Communication-Loss are judged against the recorded
        sample cadence, not how fast this loop happens to execute.
        """
        self.event_detector = EventDetector()
        self.event_window.clear()
        alerts = []
        for i in range(aligned.n_total):
            real, twin = _aligned_row_to_dicts(aligned, i)
            alerts += self.event_detector.update(
                real, twin, now=float(aligned.timestamps[i])
            )
        self.event_window.add_alerts(alerts)

    def _run_live_event_detection(self, aligned: AlignedData) -> None:
        """Incremental pass: only rows newer than the last one we've seen.

        ``aligned`` is the *whole* current snapshot on every tick (the
        live buffer is a ring buffer, so row indices aren't stable across
        ticks once old rows get trimmed) — cursor by timestamp instead of
        index, and keep the same EventDetector alive across ticks so
        spike/freeze/oscillation history spans the whole session.
        """
        alerts = []
        for i in range(aligned.n_total):
            ts = float(aligned.timestamps[i])
            if ts <= self._live_event_last_ts:
                continue
            real, twin = _aligned_row_to_dicts(aligned, i)
            alerts += self.live_event_detector.update(real, twin, now=ts)
            self._live_event_last_ts = ts
        self.event_window.add_alerts(alerts)

    # ------------------------------------------------------------------
    # Tab 5 — Config: Apply & Re-run + checkbox re-render
    # ------------------------------------------------------------------
    def on_apply_rerun(self) -> None:
        """Re-invoke the worker's ``rerun`` slot with new settings.

        The worker + thread are kept alive across runs (see
        ``on_start_comparison``), so a re-run is just a queued slot
        invocation — no new thread, no file re-parse.  The worker's
        ``rerun`` slot executes on the still-running worker thread.
        """
        if (self.comparison_worker is None
                or not self.comparison_worker.has_cached_load):
            QMessageBox.warning(
                self, "Nothing to re-run",
                "Run a full comparison first (Load & Extract tab).",
            )
            return

        method, tolerances = self._read_config_settings()
        self.btnApplyRerun.setEnabled(False)
        self.progressBarComparison.setValue(0)

        # Worker is already on a running QThread (we don't quit it on
        # finish).  Emit the rerun_signal — the worker's self-connected
        # slot picks it up on the worker thread's event loop (queued
        # connection).  Using a signal rather than a direct method call
        # is what makes the slot run on the worker thread (a direct
        # Python call would block the GUI thread).
        self.comparison_worker.rerun_signal.emit(
            method, tolerances, DEFAULT_WORST_N
        )

    def on_signal_checkbox_changed(self) -> None:
        """Re-render only the plots — no alignment recomputation needed."""
        if self.aligned_data is None:
            return
        self.plot_manager.update(self.aligned_data,
                                 self._enabled_signals_set(),
                                 self._read_threshold_settings())

    def on_signal_selection_changed(self, checked: bool) -> None:
        """An error-graph signal checkbox changed.

        Only affects ``plotWidgetError`` — the five overlay plots (V/I/T/
        SoC/SoH) always draw unconditionally. If "select only one" is
        armed, ticking a signal forces every other signal checkbox off
        (radio-button behaviour built on plain ``QCheckBox`` widgets, so
        the "Signals to Include in Error Graph" row can stay a single
        flat layout rather than switching widget types).
        """
        sender = self.sender()
        if (checked and self.checkBoxSignalSelectOnly.isChecked()
                and sender in self._signal_checkbox_names):
            self._force_single_signal_selection(sender)
        self.on_signal_checkbox_changed()

    def on_select_only_one_toggled(self, checked: bool) -> None:
        """Arming "select only one" immediately collapses the error graph
        to a single signal.

        Keeps whichever signal was already checked (the first one, in
        Voltage/Current/Temperature/SoC/SoH order, if more than one was
        ticked); if none was ticked, defaults to Voltage so the error
        graph isn't left empty. The five overlay plots are unaffected —
        they never gate on this selection.
        """
        if checked:
            already_checked = [cb for cb in self._signal_checkboxes
                                if cb.isChecked()]
            keep = already_checked[0] if already_checked else self._signal_checkboxes[0]
            self._force_single_signal_selection(keep)
        self.on_signal_checkbox_changed()

    def _force_single_signal_selection(self, keep: QtWidgets.QCheckBox) -> None:
        """Uncheck every signal checkbox except ``keep`` (checking it if needed).

        Uses ``blockSignals`` while flipping the *other* checkboxes so
        this doesn't recursively re-enter ``on_signal_selection_changed``
        for each one — the caller re-renders the plots itself once,
        after the selection has settled.
        """
        for cb in self._signal_checkboxes:
            if cb is keep or not cb.isChecked():
                continue
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
        if not keep.isChecked():
            keep.blockSignals(True)
            keep.setChecked(True)
            keep.blockSignals(False)

    def _read_threshold_settings(self) -> dict:
        """Read the Config tab's threshold-warning widgets.

        Returns ``{"voltage": (enabled, value), "current": (enabled,
        value)}`` for :meth:`PlotManager.update`.
        """
        return {
            "voltage": (self.checkBoxEnableVoltageThreshold.isChecked(),
                        float(self.doubleSpinBoxVoltageThreshold.value())),
            "current": (self.checkBoxEnableCurrentThreshold.isChecked(),
                        float(self.doubleSpinBoxCurrentThreshold.value())),
        }

    def _read_config_settings(self) -> tuple:
        """Read the Config tab's widgets and return (method, tolerances)."""
        idx = self.comboBoxAlignmentMethod.currentIndex()
        method = self._ALIGNMENT_BY_INDEX[idx]

        tolerances = {
            "voltage":     float(self.doubleSpinBoxVoltageTolerance.value()),
            "current":     float(self.doubleSpinBoxCurrentTolerance.value()),
            "temperature": float(self.doubleSpinBoxTemperatureTolerance.value()),
            # SoC / SoH use statistics_engine.DEFAULT_TOLERANCES (no UI
            # spinboxes for them) — omit them from the dict and the
            # engine will fall back.
        }
        return method, tolerances

    def _enabled_signals_set(self) -> set:
        """Signals ticked on the error graph's own selection row.

        Only drives ``plotWidgetError`` — the five overlay plots (V/I/T/
        SoC/SoH) always draw unconditionally and never consult this set.
        """
        return {name for cb, name in self._signal_checkbox_names.items()
                if cb.isChecked()}

    # ------------------------------------------------------------------
    # Tab 6 — Export
    # ------------------------------------------------------------------
    def on_export_data(self) -> None:
        if self.aligned_data is None:
            QMessageBox.warning(self, "Nothing to export",
                                 "Run a comparison first.")
            return

        fmt_idx = self.comboBoxDataFormat.currentIndex()
        fmt = self._DATA_FORMAT_BY_INDEX[fmt_idx]
        ext = "csv" if fmt == FORMAT_DATA_CSV else "xlsx"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export comparison table", "",
            f"{ext.upper()} Files (*.{ext});;All Files (*)",
        )
        if not path:
            return

        result = export_data(self.aligned_data, fmt, path)
        self._log_export_result(result, kind="table")

    def on_export_report(self) -> None:
        if self.aligned_data is None or self.stats_result is None:
            QMessageBox.warning(self, "Nothing to export",
                                 "Run a comparison first.")
            return

        fmt_idx = self.comboBoxReportFormat.currentIndex()
        fmt = self._REPORT_FORMAT_BY_INDEX[fmt_idx]
        ext = "pdf" if fmt == FORMAT_REPORT_PDF else "html"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export comparison report", "",
            f"{ext.upper()} Files (*.{ext});;All Files (*)",
        )
        if not path:
            return

        result = export_report(
            self.aligned_data, self.stats_result, fmt, path,
        )
        self._log_export_result(result, kind="report")

    def _log_export_result(self, result: ExportResult, kind: str = "") -> None:
        """Append a line to textEditExportLog and show errors as a dialog."""
        if result.success:
            self.textEditExportLog.append(result.message)
        else:
            err = result.error or "Unknown error"
            line = (f"[FAILED] {kind} export to {result.output_path}: {err}")
            self.textEditExportLog.append(line)
            QMessageBox.critical(self, "Export failed", err)

    # ------------------------------------------------------------------
    # Tab 7 — Live (MQTT)
    # ------------------------------------------------------------------
    def on_live_connect_toggled(self, checked: bool) -> None:
        if checked:
            self._mqtt_connect()
        else:
            self._mqtt_disconnect()

    def _mqtt_connect(self) -> None:
        host = self.lineEditMqttHost.text().strip() or "localhost"
        port = int(self.spinBoxMqttPort.value())
        user = self.lineEditMqttUser.text().strip() or None
        pwd = self.lineEditMqttPass.text() or None
        qos = self.comboBoxMqttQos.currentIndex()
        twin_topic = self.lineEditMqttTwinTopic.text().strip() or "bms/twin/#"
        ecu_topic = self.lineEditMqttEcuTopic.text().strip() or "bms/actual/#"

        topics = [(twin_topic, qos), (ecu_topic, qos)]

        self.btnLiveConnect.setText("Disconnect Live Stream")
        self.labelLiveStatus.setText("Status: Connecting…")

        # Tear down any previous MQTT worker/thread.
        self._teardown_mqtt_thread()

        self.mqtt_thread = QtCore.QThread(self)
        self.mqtt_worker = MqttWorker()
        self.mqtt_worker.moveToThread(self.mqtt_thread)

        self.mqtt_worker.connected.connect(self._on_mqtt_connected)
        self.mqtt_worker.disconnected.connect(self._on_mqtt_disconnected)
        self.mqtt_worker.error_occurred.connect(self._on_mqtt_error)
        # Forward every incoming packet to the live controller instead
        # of the status-bar stub. The controller (on its own thread)
        # parses + accumulates; this thread just relays raw bytes.
        self.mqtt_worker.message_received.connect(
            self._on_mqtt_message_to_controller
        )

        # Cleanup on thread finish.
        self.mqtt_thread.finished.connect(self.mqtt_thread.deleteLater)

        self.mqtt_thread.start()
        # Invoke the connect slot on the worker thread (queued).
        self.mqtt_worker.connect_broker.emit(host, port, user, pwd, topics)

        # Start the live controller's re-align timer with the GUI's
        # current interval + topic roots. The controller runs on its
        # own thread (created in _setup_live_controller) and parses
        # every packet against these roots.
        interval_ms = int(self.spinBoxLiveInterval.value())
        auto = self.checkBoxLiveAutoRefresh.isChecked()
        self.labelModeIndicator.setText(
            "Mode: Live (MQTT) — streaming. Preview/Statistics/Graphs "
            "update on the refresh interval."
        )
        if auto and self.live_controller is not None:
            # Fresh connect → drop any stale samples from a previous
            # session, then arm the re-align timer with the current
            # interval + topic roots.
            self.live_controller.reset.emit()
            self.live_controller.start.emit(
                interval_ms, twin_topic, ecu_topic,
            )
            # Same reset for the Events tab: a new session means a new
            # event history, not a continuation of the last one.
            self.live_event_detector = EventDetector()
            self._live_event_last_ts = float("-inf")
            self.event_window.clear()

    def _mqtt_disconnect(self) -> None:
        if self.mqtt_worker is not None:
            self.mqtt_worker.disconnect_broker.emit()
        # Stop the re-align timer (buffers stay intact until reconnect).
        if self.live_controller is not None:
            self.live_controller.stop.emit()
        self.labelLiveStatus.setText("Status: Disconnecting…")
        self.btnLiveConnect.setText("Connect Live Stream")
        self.labelModeIndicator.setText(
            "Mode: Offline (CSV) — load both files, then Start Comparison. "
            "Use Tab 7 (Live) to stream over MQTT instead."
        )

    def _teardown_mqtt_thread(self) -> None:
        if self.mqtt_thread is not None:
            self.mqtt_thread.quit()
            self.mqtt_thread.wait(3000)
            self.mqtt_thread = None
            self.mqtt_worker = None

    # --- MQTT slots (delivered on the GUI thread) ---
    def _on_mqtt_connected(self) -> None:
        self.labelLiveStatus.setText("Status: Connected")
        self.btnLiveConnect.setText("Disconnect Live Stream")

    def _on_mqtt_disconnected(self, rc: int) -> None:
        suffix = "" if rc == 0 else f" (rc={rc})"
        self.labelLiveStatus.setText(f"Status: Disconnected{suffix}")
        self.btnLiveConnect.setText("Connect Live Stream")
        self.btnLiveConnect.setChecked(False)

    def _on_mqtt_error(self, msg: str) -> None:
        self.labelLiveStatus.setText(f"Status: Error — {msg}")
        self.btnLiveConnect.setChecked(False)
        self.btnLiveConnect.setText("Connect Live Stream")
        QMessageBox.critical(self, "MQTT error", msg)

    def _on_mqtt_message_to_controller(
        self, topic: str, payload: bytes, ts: float,
    ) -> None:
        """Forward every raw MQTT packet to the live controller.

        The controller lives on its own QThread; ``ingest`` is a Qt
        signal, so this emission is a queued connection and the parse +
        accumulate work happens off the GUI thread. We also show a
        short-lived status-bar blip so the user can see traffic.
        """
        if self.live_controller is not None:
            self.live_controller.ingest.emit(topic, payload, ts)
        self.statusBar().showMessage(
            f"MQTT <- {topic}: {len(payload)} B", 2000,
        )

    # --- Live controller result slots (delivered on the GUI thread) ---
    # These reuse the *same* populate methods the offline path uses, so
    # the Preview / Statistics / Graphs tabs look identical whether the
    # data came from a CSV or an MQTT stream.
    def _on_live_aligned(self, aligned: AlignedData) -> None:
        """New aligned snapshot from the live controller.

        Populates the Preview tab (both twin + ECU tables, reading the
        controller's buffer for raw rows) and refreshes the plots.
        We do *not* auto-jump tabs in live mode — the user may be
        watching any tab and tab-hopping every 500 ms would be jarring.
        """
        self.aligned_data = aligned
        # Refresh the preview tables from the controller's buffers so
        # the user sees the live rows that produced this aligned data.
        if self.live_controller is not None:
            twin_lr, ecu_lr = self.live_controller.buffer.build_results()
            if twin_lr is not None:
                _populate_preview_table(self.tableWidgetTwinPreview, twin_lr)
                self.labelTwinPreviewInfo.setText(
                    f"{twin_lr.row_count} rows x "
                    f"{len(twin_lr.columns_found)} cols  "
                    f"({twin_lr.time_range[0]:.2f}-"
                    f"{twin_lr.time_range[1]:.2f} s)"
                )
            if ecu_lr is not None:
                _populate_preview_table(self.tableWidgetEcuPreview, ecu_lr)
                self.labelEcuPreviewInfo.setText(
                    f"{ecu_lr.row_count} rows x "
                    f"{len(ecu_lr.columns_found)} cols  "
                    f"({ecu_lr.time_range[0]:.2f}-"
                    f"{ecu_lr.time_range[1]:.2f} s)"
                )
        # Re-render plots with the currently-enabled signal set. Pin the
        # sliding live window so the view shifts with new data instead of
        # compressing to fit the whole growing buffer (see
        # _LIVE_PLOT_WINDOW_MS / _LIVE_PLOT_WINDOW_SEQUENCE). The window
        # width itself depends on what the axis actually is.
        window = (_LIVE_PLOT_WINDOW_SEQUENCE if aligned.axis_kind == "sequence"
                  else _LIVE_PLOT_WINDOW_MS)
        self.plot_manager.update(aligned, self._enabled_signals_set(),
                                 self._read_threshold_settings(),
                                 live_window_ms=window)
        # Feed only the newly-arrived rows through the (persistent) live
        # event detector.
        self._run_live_event_detection(aligned)

    def _on_live_stats(self, stats: StatisticsResult) -> None:
        """New statistics snapshot — populate the Statistics tab."""
        self.stats_result = stats
        self._populate_statistics_tab(stats)

    def _on_live_sample_count(self, n_twin: int, n_ecu: int) -> None:
        """Update the Live-tab status line with current buffer sizes."""
        self.labelLiveStatus.setText(
            f"Status: Connected — twin {n_twin} / ECU {n_ecu} samples"
        )

    def _on_live_error(self, msg: str) -> None:
        """Non-fatal live error (bad packet, transient align failure)."""
        self.statusBar().showMessage(f"Live: {msg}", 5000)

    # --- Live-tab UI widget handlers ---
    def on_live_auto_refresh_toggled(self, checked: bool) -> None:
        """Start/stop the re-align timer without touching the MQTT socket.

        When the user unchecks Auto-refresh we stop the timer (plots
        freeze at the last snapshot); re-checking restarts it with the
        current interval spinbox value. Only meaningful while connected.
        """
        if self.live_controller is None:
            return
        if checked and self.mqtt_worker is not None:
            interval_ms = int(self.spinBoxLiveInterval.value())
            twin_topic = (self.lineEditMqttTwinTopic.text().strip()
                          or "bms/twin/#")
            ecu_topic = (self.lineEditMqttEcuTopic.text().strip()
                         or "bms/actual/#")
            self.live_controller.start.emit(
                interval_ms, twin_topic, ecu_topic,
            )
        else:
            self.live_controller.stop.emit()

    def on_live_interval_changed(self) -> None:
        """Apply the new interval live — restart the timer if it's running.

        We can't ``setInterval`` on a timer that lives on another thread
        safely from here, so we re-emit ``start`` with the new value;
        the controller's ``_on_start`` rebuilds the timer (cheap).
        """
        if (self.live_controller is None
                or not self.checkBoxLiveAutoRefresh.isChecked()
                or self.mqtt_worker is None):
            return
        interval_ms = int(self.spinBoxLiveInterval.value())
        twin_topic = self.lineEditMqttTwinTopic.text().strip() or "bms/twin/#"
        ecu_topic = self.lineEditMqttEcuTopic.text().strip() or "bms/actual/#"
        self.live_controller.start.emit(interval_ms, twin_topic, ecu_topic)

    def on_live_snapshot(self) -> None:
        """Freeze: build a one-off aligned snapshot and stash it as the
        current ``self.aligned_data`` / ``self.stats_result`` so the
        Export tab acts on the frozen state.

        This makes the live stream "capture-able": the user can hit
        Freeze, switch to Export, and save the current comparison.
        """
        if self.live_controller is None:
            QMessageBox.information(
                self, "No live data",
                "Connect a live stream first, then take a snapshot.",
            )
            return
        twin_lr, ecu_lr = self.live_controller.buffer.build_results()
        if twin_lr is None or ecu_lr is None:
            QMessageBox.warning(
                self, "Not enough data",
                "Need samples from both twin and ECU streams to snapshot.",
            )
            return
        try:
            from alignment_engine import align as _align
            from statistics_engine import compute as _compute
            aligned = _align(twin_lr, ecu_lr, method="nearest")
            stats = _compute(aligned, tolerances=self._read_config_settings()[1])
        except Exception as exc:
            QMessageBox.critical(self, "Snapshot failed", str(exc))
            return
        self.aligned_data = aligned
        self.stats_result = stats
        self._populate_statistics_tab(stats)
        self.plot_manager.update(aligned, self._enabled_signals_set(),
                                 self._read_threshold_settings())
        self.statusBar().showMessage(
            "Live snapshot taken — Export tab now acts on this snapshot.", 6000,
        )

    # ------------------------------------------------------------------
    # Window close: tear down threads so we don't hang on exit
    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:
        self._teardown_comparison_thread()
        if self.mqtt_worker is not None:
            self.mqtt_worker.disconnect_broker.emit()
        self._teardown_mqtt_thread()
        # Tear down the long-lived live controller + its thread.
        if self.live_controller is not None:
            self.live_controller.stop.emit()
        if self.live_thread is not None:
            self.live_thread.quit()
            self.live_thread.wait(3000)
            self.live_thread = None
            self.live_controller = None
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
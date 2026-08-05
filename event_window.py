"""GUI Event Window for the Event Detection module (Task 14).

``EventWindow`` is a plain ``QWidget`` — not wired into ``f.py`` (the
pyuic5-generated main window: hand-editing it is a footgun since
regenerating from ``f.ui`` would wipe any changes). It is self-contained
so ``back.py`` can drop it into a new tab later with a couple of lines,
and it can also be run standalone for a live demo (see ``__main__``
below), which replays synthetic real/twin dict pairs through
``EventDetector`` on a timer so every rule in the spec gets exercised.

Layout
------
A table (Timestamp / Signal / Event / Severity / Real Value / Digital
Twin / Difference / Threshold) fed by :meth:`EventWindow.add_alerts`,
plus Export CSV / Excel / PDF buttons wired to ``event_export.py``.
"""

from __future__ import annotations

import sys
from typing import List

from PyQt5 import QtCore, QtGui, QtWidgets

from event_detector import Alert, EventDetector, EventLog, SEVERITY_HIGH, SEVERITY_MEDIUM
from event_export import (
    EventExportError, FORMAT_CSV, FORMAT_EXCEL, FORMAT_PDF, export_event_log,
)

_COLUMNS = [
    "Timestamp", "Signal", "Event", "Severity",
    "Real Value", "Digital Twin", "Difference", "Threshold",
]

_SEVERITY_COLOR = {
    SEVERITY_HIGH: QtGui.QColor("#5c1f1f"),
    SEVERITY_MEDIUM: QtGui.QColor("#5c4a1f"),
}


class EventWindow(QtWidgets.QWidget):
    """Dedicated Event Window: table of alerts + export buttons."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Event Detection")
        self.log = EventLog()
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        self.table = QtWidgets.QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        layout.addWidget(self.table)

        self.summary_label = QtWidgets.QLabel("0 events")
        layout.addWidget(self.summary_label)

        btn_row = QtWidgets.QHBoxLayout()
        self.export_csv_btn = QtWidgets.QPushButton("Export CSV…")
        self.export_excel_btn = QtWidgets.QPushButton("Export Excel…")
        self.export_pdf_btn = QtWidgets.QPushButton("Export PDF…")
        self.clear_btn = QtWidgets.QPushButton("Clear")
        for b in (self.export_csv_btn, self.export_excel_btn,
                  self.export_pdf_btn, self.clear_btn):
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        self.export_csv_btn.clicked.connect(lambda: self._export(FORMAT_CSV, "CSV Files (*.csv)", ".csv"))
        self.export_excel_btn.clicked.connect(lambda: self._export(FORMAT_EXCEL, "Excel Files (*.xlsx)", ".xlsx"))
        self.export_pdf_btn.clicked.connect(lambda: self._export(FORMAT_PDF, "PDF Files (*.pdf)", ".pdf"))
        self.clear_btn.clicked.connect(self._on_clear)

    # ------------------------------------------------------------------
    # Feed alerts in (Task 14: display detected alerts)
    # ------------------------------------------------------------------
    def add_alerts(self, alerts: List[Alert]) -> None:
        if not alerts:
            return
        self.log.add_many(alerts)
        for alert in alerts:
            self._append_row(alert)
        self.summary_label.setText(f"{len(self.log)} event(s)")

    def _append_row(self, alert: Alert) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        values = [
            f"{alert.timestamp:.3f}",
            alert.signal,
            alert.event,
            alert.severity,
            "" if alert.real_value is None else f"{alert.real_value:.4f}",
            "" if alert.digital_twin_value is None else f"{alert.digital_twin_value:.4f}",
            "" if alert.difference is None else f"{alert.difference:+.4f}",
            "" if alert.threshold is None else f"{alert.threshold:.4f}",
        ]
        color = _SEVERITY_COLOR.get(alert.severity)
        for col, text in enumerate(values):
            item = QtWidgets.QTableWidgetItem(text)
            if color is not None:
                item.setBackground(color)
            self.table.setItem(row, col, item)
        self.table.scrollToBottom()

    # ------------------------------------------------------------------
    # Export (Task 15)
    # ------------------------------------------------------------------
    def _export(self, fmt: str, file_filter: str, default_ext: str) -> None:
        if len(self.log) == 0:
            QtWidgets.QMessageBox.information(self, "Nothing to export", "No events logged yet.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Event Log", f"event_log{default_ext}", file_filter,
        )
        if not path:
            return
        try:
            result = export_event_log(self.log, fmt, path)
        except EventExportError as exc:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(exc))
            return
        if not result.success:
            QtWidgets.QMessageBox.critical(self, "Export failed", result.error or "Unknown error")
            return
        QtWidgets.QMessageBox.information(self, "Export complete", result.message)

    def clear(self) -> None:
        """Drop every logged alert and empty the table (public — callable
        from ``back.py`` on a fresh comparison run / live reconnect, not
        just from the Clear button)."""
        self.log.clear()
        self.table.setRowCount(0)
        self.summary_label.setText("0 events")

    def _on_clear(self) -> None:
        self.clear()


# ---------------------------------------------------------------------------
# Standalone demo: feeds synthetic real/twin dict pairs through
# EventDetector on a timer so every rule in the spec fires at least once.
# ---------------------------------------------------------------------------
def _demo_stream():
    """Yield (real_data, digital_twin_data) pairs that exercise every rule."""
    t = 0.0
    voltage = 3.72
    yield {"timestamp": t, "voltage": voltage, "current": 5.3, "temperature": 26.1, "soc": 79.8}, \
          {"timestamp": t, "voltage": 3.70, "current": 5.2, "temperature": 26.0, "soc": 80.1}

    # A run of frozen values (Value Freeze).
    for _ in range(6):
        t += 0.1
        yield {"timestamp": t, "voltage": voltage, "current": 5.3, "temperature": 26.1, "soc": 79.8}, \
              {"timestamp": t, "voltage": 3.70, "current": 5.2, "temperature": 26.0, "soc": 80.1}

    # A spike (Spike + Threshold Exceeded).
    t += 0.1
    yield {"timestamp": t, "voltage": 4.18, "current": 5.3, "temperature": 26.1, "soc": 79.8}, \
          {"timestamp": t, "voltage": 3.70, "current": 5.2, "temperature": 26.0, "soc": 80.1}

    # Out-of-range current + a sudden drop next tick.
    t += 0.1
    yield {"timestamp": t, "voltage": 3.72, "current": 40.0, "temperature": 26.1, "soc": 79.8}, \
          {"timestamp": t, "voltage": 3.70, "current": 5.2, "temperature": 26.0, "soc": 80.1}
    t += 0.1
    yield {"timestamp": t, "voltage": 3.72, "current": 2.0, "temperature": 26.1, "soc": 79.8}, \
          {"timestamp": t, "voltage": 3.70, "current": 5.2, "temperature": 26.0, "soc": 80.1}

    # An oscillating signal.
    for v in (3.70, 3.80, 3.70, 3.80, 3.70):
        t += 0.1
        yield {"timestamp": t, "voltage": v, "current": 5.3, "temperature": 26.1, "soc": 79.8}, \
              {"timestamp": t, "voltage": 3.70, "current": 5.2, "temperature": 26.0, "soc": 80.1}

    # A permanent jump in SoC (Unexpected Jump).
    for soc in (45.0, 45.2, 44.9, 45.1):
        t += 0.1
        yield {"timestamp": t, "voltage": 3.72, "current": 5.3, "temperature": 26.1, "soc": soc}, \
              {"timestamp": t, "voltage": 3.70, "current": 5.2, "temperature": 26.0, "soc": 80.1}

    # A missing signal + a stalled twin timestamp (Communication Loss).
    while True:
        t += 0.1
        real = {"timestamp": t, "voltage": 3.72, "current": 5.3, "temperature": 26.1, "soc": 79.8}
        twin = {"timestamp": 15.25, "voltage": 3.70, "current": 5.2, "temperature": 26.0}
        yield real, twin


class _DemoRunner(QtCore.QObject):
    """Drives the demo stream into an EventDetector on a QTimer."""

    def __init__(self, window: EventWindow):
        super().__init__(window)
        self._window = window
        self._detector = EventDetector()
        self._stream = _demo_stream()
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(600)

    def _tick(self) -> None:
        try:
            real, twin = next(self._stream)
        except StopIteration:
            self._timer.stop()
            return
        alerts = self._detector.update(real, twin)
        self._window.add_alerts(alerts)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = EventWindow()
    window.resize(900, 560)
    window.show()
    runner = _DemoRunner(window)  # noqa: F841 — keeps the timer alive
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

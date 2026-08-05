"""GUI front-end for the synthetic twin/ECU MQTT publisher experiment.

Wraps ``synthetic_mqtt_publisher.py``'s connect -> publish-on-timer loop
(same ``MqttWorker`` + ``SyntheticBatteryGenerator`` pair) behind a PyQt5
form instead of CLI flags, so connection/generator settings can be tweaked
and the run started/stopped without retyping a command line.

Run it::

    python trial.py
"""

from __future__ import annotations

import sys

from PyQt5 import QtCore, QtWidgets

from mqtt_worker import MqttWorker
from synthetic_data_generator import GeneratorConfig, SyntheticBatteryGenerator, batch_to_json_bytes


class TrialWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Synthetic MQTT Publisher — Trial GUI")

        self._thread: QtCore.QThread | None = None
        self._worker: MqttWorker | None = None
        self._gen: SyntheticBatteryGenerator | None = None
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._publish_next)
        self._batches_sent = 0

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        form = QtWidgets.QFormLayout()

        self.host_edit = QtWidgets.QLineEdit("localhost")
        self.port_spin = QtWidgets.QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(1883)
        self.username_edit = QtWidgets.QLineEdit()
        self.password_edit = QtWidgets.QLineEdit()
        self.password_edit.setEchoMode(QtWidgets.QLineEdit.Password)

        self.twin_topic_edit = QtWidgets.QLineEdit("bms/twin/data")
        self.ecu_topic_edit = QtWidgets.QLineEdit("bms/actual/data")
        self.qos_combo = QtWidgets.QComboBox()
        self.qos_combo.addItems(["0", "1", "2"])
        self.qos_combo.setCurrentIndex(1)

        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(42)
        self.dt_spin = QtWidgets.QDoubleSpinBox()
        self.dt_spin.setRange(0.001, 10.0)
        self.dt_spin.setDecimals(3)
        self.dt_spin.setValue(0.1)
        self.batch_size_spin = QtWidgets.QSpinBox()
        self.batch_size_spin.setRange(1, 10_000)
        self.batch_size_spin.setValue(10)
        self.interval_spin = QtWidgets.QSpinBox()
        self.interval_spin.setRange(10, 60_000)
        self.interval_spin.setValue(1000)
        self.count_spin = QtWidgets.QSpinBox()
        self.count_spin.setRange(0, 1_000_000)
        self.count_spin.setValue(0)
        self.count_spin.setSpecialValueText("infinite")

        form.addRow("Host", self.host_edit)
        form.addRow("Port", self.port_spin)
        form.addRow("Username", self.username_edit)
        form.addRow("Password", self.password_edit)
        form.addRow("Twin topic", self.twin_topic_edit)
        form.addRow("ECU topic", self.ecu_topic_edit)
        form.addRow("QoS", self.qos_combo)
        form.addRow("Seed", self.seed_spin)
        form.addRow("dt (s)", self.dt_spin)
        form.addRow("Batch size", self.batch_size_spin)
        form.addRow("Interval (ms)", self.interval_spin)
        form.addRow("Count (0 = forever)", self.count_spin)

        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)

        self.status_label = QtWidgets.QLabel("Idle")
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(btn_row)
        layout.addWidget(self.status_label)
        layout.addWidget(self.log_view)

    def _log(self, msg: str) -> None:
        self.log_view.appendPlainText(msg)

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------
    def _on_start(self) -> None:
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Connecting...")
        self._batches_sent = 0

        self._gen = SyntheticBatteryGenerator(
            GeneratorConfig(seed=self.seed_spin.value(), dt=self.dt_spin.value())
        )

        self._thread = QtCore.QThread()
        self._worker = MqttWorker()
        self._worker.moveToThread(self._thread)
        self._worker.connected.connect(self._on_connected)
        self._worker.disconnected.connect(self._on_disconnected)
        self._worker.error_occurred.connect(self._on_error)
        self._thread.start()

        self._worker.connect_broker.emit(
            self.host_edit.text(),
            self.port_spin.value(),
            self.username_edit.text(),
            self.password_edit.text(),
            [],
        )

    def _on_stop(self) -> None:
        self._timer.stop()
        self.status_label.setText("Disconnecting...")
        if self._worker is not None:
            self._worker.disconnect_broker.emit()

    def _teardown_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Worker signal handlers
    # ------------------------------------------------------------------
    def _on_connected(self) -> None:
        self.status_label.setText("Connected — publishing")
        self._log(
            f"connected to {self.host_edit.text()}:{self.port_spin.value()}  "
            f"seed={self.seed_spin.value()}"
        )
        self._timer.setInterval(self.interval_spin.value())
        self._timer.start()

    def _on_disconnected(self, rc: int) -> None:
        self._log(f"disconnected (rc={rc})")
        self.status_label.setText("Idle")
        self._teardown_thread()

    def _on_error(self, msg: str) -> None:
        self._log(f"ERROR: {msg}")
        self.status_label.setText("Error")
        self._timer.stop()
        self._teardown_thread()

    def _publish_next(self) -> None:
        count = self.count_spin.value()
        if count > 0 and self._batches_sent >= count:
            self._timer.stop()
            self._log(f"sent {self._batches_sent} batch(es); disconnecting...")
            self._worker.disconnect_broker.emit()
            return

        twin_payload, ecu_payload = self._gen.batch(self.batch_size_spin.value())
        twin_bytes = batch_to_json_bytes(twin_payload)
        ecu_bytes = batch_to_json_bytes(ecu_payload)
        qos = int(self.qos_combo.currentText())

        self._worker.publish_broker.emit(self.twin_topic_edit.text(), twin_bytes, qos)
        self._worker.publish_broker.emit(self.ecu_topic_edit.text(), ecu_bytes, qos)
        self._batches_sent += 1
        self._log(
            f"batch {self._batches_sent}: "
            f"{self.twin_topic_edit.text()} ({len(twin_bytes)} B), "
            f"{self.ecu_topic_edit.text()} ({len(ecu_bytes)} B)"
        )

    def closeEvent(self, event: QtCore.QEvent) -> None:
        self._timer.stop()
        if self._worker is not None:
            self._worker.disconnect_broker.emit()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
        event.accept()


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    win = TrialWindow()
    win.resize(480, 640)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

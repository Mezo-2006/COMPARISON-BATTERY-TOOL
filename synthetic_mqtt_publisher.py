"""Publish seeded synthetic twin/ECU battery telemetry to an MQTT broker.

Test-only tool for exercising the Live tab end to end (including against
a broker running on a *different* machine on the LAN) without needing
real Simulink / ECU hardware. Uses the same ``MqttWorker`` + worker-object
pattern as ``mqtt_worker_sender.py`` so it behaves like the real
publishers the tool expects, but sources its payloads from
``synthetic_data_generator.SyntheticBatteryGenerator`` instead of a fixed
ramp.

Run it (after ``pip install -r requirements.txt``)::

    python synthetic_mqtt_publisher.py --host 192.168.1.23 --seed 42

Point ``--host`` at whatever machine is running the broker (Mosquitto or
otherwise); ``localhost`` works for a broker on this same machine. Then,
in the tool's Live tab, set the same host/port and the default topic
roots (``bms/twin/#`` / ``bms/actual/#``) and hit Connect — the
Statistics tab should start filling in as batches arrive.
"""

from __future__ import annotations

import argparse
import sys

from PyQt5 import QtCore

from mqtt_worker import MqttWorker
from synthetic_data_generator import GeneratorConfig, SyntheticBatteryGenerator, batch_to_json_bytes


class PublisherApp(QtCore.QObject):
    """Connect -> publish a twin+ecu batch on a timer -> repeat until --count."""

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self._args = args
        self._gen = SyntheticBatteryGenerator(GeneratorConfig(seed=args.seed, dt=args.dt))
        self._batches_sent = 0

        self._thread = QtCore.QThread()
        self._worker = MqttWorker()
        self._worker.moveToThread(self._thread)

        self._worker.connected.connect(self._on_connected)
        self._worker.disconnected.connect(self._on_disconnected)
        self._worker.error_occurred.connect(self._on_error)
        self._thread.finished.connect(self._thread.deleteLater)

        self._timer = QtCore.QTimer()
        self._timer.setInterval(args.interval_ms)
        self._timer.timeout.connect(self._publish_next)

    def start(self) -> None:
        self._thread.start()
        self._worker.connect_broker.emit(
            self._args.host, self._args.port,
            self._args.username, self._args.password,
            [],  # a publisher doesn't need to subscribe to anything
        )

    def _on_connected(self) -> None:
        a = self._args
        print(f"[publisher] connected to {a.host}:{a.port}  seed={a.seed}  "
              f"topics=({a.twin_topic}, {a.ecu_topic})")
        self._timer.start()

    def _on_disconnected(self, rc: int) -> None:
        print(f"[publisher] disconnected  (rc={rc})")
        self._thread.quit()
        self._thread.wait()
        QtCore.QCoreApplication.quit()

    def _on_error(self, msg: str) -> None:
        print(f"[publisher] ERROR: {msg}", file=sys.stderr)
        self._thread.quit()
        self._thread.wait()
        QtCore.QCoreApplication.quit()

    def _publish_next(self) -> None:
        a = self._args
        if a.count > 0 and self._batches_sent >= a.count:
            self._timer.stop()
            print(f"[publisher] sent {self._batches_sent} batch(es); disconnecting...")
            self._worker.disconnect_broker.emit()
            return

        twin_payload, ecu_payload = self._gen.batch(a.batch_size)
        twin_bytes = batch_to_json_bytes(twin_payload)
        ecu_bytes = batch_to_json_bytes(ecu_payload)

        self._worker.publish_broker.emit(a.twin_topic, twin_bytes, a.qos)
        self._worker.publish_broker.emit(a.ecu_topic, ecu_bytes, a.qos)
        self._batches_sent += 1
        print(f"[publisher] batch {self._batches_sent}: "
              f"{a.twin_topic} ({len(twin_bytes)} B), "
              f"{a.ecu_topic} ({len(ecu_bytes)} B)")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="localhost", help="broker host/IP (the other PC's LAN IP for a remote broker)")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--username", default="")
    p.add_argument("--password", default="")
    p.add_argument("--twin-topic", default="bms/twin/data", help="must fall under the tool's twin root, default bms/twin/#")
    p.add_argument("--ecu-topic", default="bms/actual/data", help="must fall under the tool's ecu root, default bms/actual/#")
    p.add_argument("--qos", type=int, default=1, choices=(0, 1, 2))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dt", type=float, default=0.1, help="seconds of simulated time per sample")
    p.add_argument("--batch-size", type=int, default=10, help="samples per MQTT message")
    p.add_argument("--interval-ms", type=int, default=1000, help="wall-clock ms between published batches")
    p.add_argument("--count", type=int, default=0, help="number of batches to send; 0 = run forever (Ctrl+C to stop)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    app = QtCore.QCoreApplication(sys.argv)
    publisher = PublisherApp(args)
    publisher.start()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

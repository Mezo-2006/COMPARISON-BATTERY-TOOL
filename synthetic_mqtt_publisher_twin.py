"""Publish seeded synthetic Digital Twin battery telemetry to an MQTT broker.

Twin-only half of the split synthetic publisher: run this alongside
``synthetic_mqtt_publisher_ecu.py`` (as two separate processes, optionally
on two different machines) to exercise the Live tab with the twin and
real-cell/ECU streams coming from genuinely independent publishers,
instead of one process publishing both — closer to how the real Simulink
(twin) and NXP MPC5744P (ECU) sources will actually be wired up.

Test-only tool for exercising the Live tab end to end (including against
a broker running on a *different* machine on the LAN) without needing
real Simulink hardware. Uses the same ``MqttWorker`` + worker-object
pattern as ``mqtt_worker_sender.py`` so it behaves like a real publisher,
but sources its payload from ``synthetic_data_generator.SyntheticBatteryGenerator``.

Run it (after ``pip install -r requirements.txt``), together with
``synthetic_mqtt_publisher_ecu.py`` using the **same** ``--seed``::

    python synthetic_mqtt_publisher_twin.py --host 192.168.1.23 --seed 42
    python synthetic_mqtt_publisher_ecu.py  --host 192.168.1.23 --seed 42

Point ``--host`` at whatever machine is running the broker (Mosquitto or
otherwise); ``localhost`` works for a broker on this same machine. Then,
in the tool's Live tab, set the same host/port and the default topic
roots (``bms/twin/#`` / ``bms/actual/#``) and hit Connect — the
Statistics tab should start filling in as batches arrive from both.

Staying correlated across two processes
----------------------------------------
Each of the two scripts runs its own ``SyntheticBatteryGenerator``, which
internally computes *both* the twin and ecu values every step (that's
what keeps ``|ecu - twin|`` realistically small — see
``synthetic_data_generator``'s module docstring) — this script simply
only *publishes* the twin half and discards the ecu half it also
computed. Two processes given the **same** ``--seed`` / ``--dt`` /
``--batch-size`` run byte-identical internal trajectories, so the twin
stream published here and the ecu stream published by
``synthetic_mqtt_publisher_ecu.py`` are the same correlated pair a single
combined process would have produced.

Real-world clock skew between the two streams comes from two places:
whatever gap there is between actually starting the two processes (pass
the same ``--start-time``, a Unix epoch **milliseconds** value — see
``synthetic_data_generator``'s "Timestamp convention" section — to pin
both to the identical anchor and eliminate that gap), and the built-in
``--ecu-time-offset`` (default 50 ms), which models the small, realistic
clock skew a real twin/ECU pair would also have.
"""

from __future__ import annotations

import argparse
import sys

from PyQt5 import QtCore

from mqtt_worker import MqttWorker
from synthetic_data_generator import GeneratorConfig, SyntheticBatteryGenerator, batch_to_json_bytes


class TwinPublisherApp(QtCore.QObject):
    """Connect -> publish a twin-only batch on a timer -> repeat until --count."""

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self._args = args
        self._gen = SyntheticBatteryGenerator(GeneratorConfig(
            seed=args.seed, dt=args.dt, ecu_time_offset=args.ecu_time_offset,
            start_time=args.start_time,
        ))
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
        print(f"[twin publisher] connected to {a.host}:{a.port}  seed={a.seed}  "
              f"topic={a.twin_topic}")
        self._timer.start()

    def _on_disconnected(self, rc: int) -> None:
        print(f"[twin publisher] disconnected  (rc={rc})")
        self._thread.quit()
        self._thread.wait()
        QtCore.QCoreApplication.quit()

    def _on_error(self, msg: str) -> None:
        print(f"[twin publisher] ERROR: {msg}", file=sys.stderr)
        self._thread.quit()
        self._thread.wait()
        QtCore.QCoreApplication.quit()

    def _publish_next(self) -> None:
        a = self._args
        if a.count > 0 and self._batches_sent >= a.count:
            self._timer.stop()
            print(f"[twin publisher] sent {self._batches_sent} batch(es); disconnecting...")
            self._worker.disconnect_broker.emit()
            return

        # Both twin and ecu are computed every step (they share one RNG-
        # driven trajectory) so this stream stays correlated with whatever
        # synthetic_mqtt_publisher_ecu.py is publishing under the same
        # seed -- only the twin half is actually sent here.
        twin_payload, _ecu_payload = self._gen.batch(a.batch_size)
        twin_bytes = batch_to_json_bytes(twin_payload)

        self._worker.publish_broker.emit(a.twin_topic, twin_bytes, a.qos)
        self._batches_sent += 1
        print(f"[twin publisher] batch {self._batches_sent}: "
              f"{a.twin_topic} ({len(twin_bytes)} B)")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="localhost", help="broker host/IP (the other PC's LAN IP for a remote broker)")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--username", default="")
    p.add_argument("--password", default="")
    p.add_argument("--twin-topic", default="bms/twin/data", help="must fall under the tool's twin root, default bms/twin/#")
    p.add_argument("--qos", type=int, default=1, choices=(0, 1, 2))
    p.add_argument("--seed", type=int, default=42, help="MUST match synthetic_mqtt_publisher_ecu.py's --seed to stay correlated")
    p.add_argument("--dt", type=float, default=0.1, help="seconds of simulated time per sample")
    p.add_argument("--ecu-time-offset", type=float, default=0.05, help="seconds; must match the ecu script's value")
    p.add_argument("--start-time", type=float, default=None,
                    help="Unix epoch MILLISECONDS to anchor the synthetic clock to; "
                         "pass the same value to synthetic_mqtt_publisher_ecu.py to "
                         "eliminate process-launch clock skew between the two streams. "
                         "Default: real wall clock (time.time()*1000) at startup.")
    p.add_argument("--batch-size", type=int, default=10, help="samples per MQTT message")
    p.add_argument("--interval-ms", type=int, default=1000, help="wall-clock ms between published batches")
    p.add_argument("--count", type=int, default=0, help="number of batches to send; 0 = run forever (Ctrl+C to stop)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    app = QtCore.QCoreApplication(sys.argv)
    publisher = TwinPublisherApp(args)
    publisher.start()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

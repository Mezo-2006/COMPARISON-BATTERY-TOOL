"""Standalone MQTT *sender* demo using the generic ``MqttWorker``.

Run it (after ``pip install paho-mqtt PyQt5``) against a local Mosquitto
broker::

    python3 mqtt_worker_sender.py

It connects to the broker, publishes ``MESSAGE_COUNT`` JSON messages on
``TOPIC`` at one-per-:data:`MESSAGE_INTERVAL_MS` milliseconds, then
disconnects and quits. No GUI — it uses a ``QCoreApplication`` so the
same Qt signal/slot machinery exercised by the real tool is exercised
here too.

Tweak the values in the configuration block below to point at another
broker / topic.
"""

import json
import sys
import time

from PyQt5 import QtCore

from mqtt_worker import MqttWorker


# ──────────────────────────────────────────────────────────────────────
#  Configuration — edit these values to match your broker / topic.
# ──────────────────────────────────────────────────────────────────────
HOST = "10.212.239.59"
PORT = 1883
USERNAME = ""            # leave empty for anonymous broker access
PASSWORD = ""
QOS = 1

TOPIC = "bms/actual/voltage"          # topic this sender publishes to
MESSAGE_COUNT = 5                     # how many messages to send
MESSAGE_INTERVAL_MS = 1000            # gap between messages (Qt timer)

CLIENT_LABEL = "sender"               # baked into the demo payload so the
                                       # receiver can tell senders apart


def build_payload(seq):
    """Construct one demo JSON payload."""
    return json.dumps({
        "source": CLIENT_LABEL,
        "seq": seq,
        "voltage": 3.30 + 0.01 * seq,   # a gentle ramp so you can see it move
        "current": 1.20,
        "temperature": 25.0,
        "timestamp": time.time(),
    }).encode("utf-8")


class SenderApp(QtCore.QObject):
    """Tiny controller: connect → publish on a timer → quit."""

    def __init__(self):
        super().__init__()

        self._seq = 0

        # ── Worker on its own QThread (the worker-object pattern) ──────
        self._thread = QtCore.QThread()
        self._worker = MqttWorker()
        self._worker.moveToThread(self._thread)

        # Wire the worker's signals to our slots.
        self._worker.connected.connect(self._on_connected)
        self._worker.disconnected.connect(self._on_disconnected)
        self._worker.error_occurred.connect(self._on_error)

        # Clean up the thread when the worker is done.
        self._thread.finished.connect(self._thread.deleteLater)

        # Periodic publishing timer — fires on the *caller* (main) thread,
        # each tick emits ``publish_broker`` which is delivered queued to
        # the worker thread.
        self._timer = QtCore.QTimer()
        self._timer.setInterval(MESSAGE_INTERVAL_MS)
        self._timer.timeout.connect(self._publish_next)

    # ── Lifecycle ───────────────────────────────────────────────────
    def start(self):
        self._thread.start()
        self._worker.connect_broker.emit(
            HOST, PORT, USERNAME, PASSWORD,
            [],          # a sender doesn't subscribe to anything;
                         # per-topic QoS would live in these (filter, qos) pairs
        )

    # ── Worker-signal slots ─────────────────────────────────────────
    def _on_connected(self):
        print(f"[sender] connected to {HOST}:{PORT}  (QoS {QOS})")
        self._timer.start()

    def _on_disconnected(self, rc):
        print(f"[sender] disconnected  (rc={rc})")
        # Tear the worker's QThread down *before* the app exits so Qt
        # never destroys a still-running thread (which aborts the process).
        self._thread.quit()
        self._thread.wait()
        QtCore.QCoreApplication.quit()

    def _on_error(self, msg):
        print(f"[sender] ERROR: {msg}", file=sys.stderr)
        self._thread.quit()
        self._thread.wait()
        QtCore.QCoreApplication.quit()

    # ── Timer slot ───────────────────────────────────────────────────
    def _publish_next(self):
        if self._seq >= MESSAGE_COUNT:
            self._timer.stop()
            print(f"[sender] published {MESSAGE_COUNT} messages; disconnecting…")
            self._worker.disconnect_broker.emit()
            return

        payload = build_payload(self._seq)
        self._worker.publish_broker.emit(TOPIC, payload, QOS)
        print(f"[sender] → {TOPIC}  seq={self._seq}  {len(payload)} B")
        self._seq += 1


def main():
    app = QtCore.QCoreApplication(sys.argv)
    sender = SenderApp()
    sender.start()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
"""Standalone MQTT *receiver* demo using the generic ``MqttWorker``.

Run it (after ``pip install paho-mqtt PyQt5``) against a local Mosquitto
broker::

    python3 mqtt_worker_receiver.py

It connects to the broker, subscribes to ``TOPICS``, prints every
matched message, and exits after ``TIMEOUT_MS`` of inactivity (or when
the user presses Ctrl+C). No GUI — it uses a ``QCoreApplication`` so the
same Qt signal/slot machinery exercised by the real tool is exercised
here too.

This file also shows the *consumer-side* routing the worker no longer
does itself: if you subscribe to more than one topic root you can tell
them apart by inspecting the topic string in the ``message_received``
slot — see ``_classify`` below.

Tweak the values in the configuration block below to match your broker.
"""

import sys

from PyQt5 import QtCore

from mqtt_worker import MqttWorker


# ──────────────────────────────────────────────────────────────────────
#  Configuration — edit these values to match your broker / topics.
# ──────────────────────────────────────────────────────────────────────
HOST = "172.20.10.2"
PORT = 1883
USERNAME = ""            # leave empty for anonymous broker access
PASSWORD = ""
QOS = 1

# A list of (topic_filter, qos) pairs the worker should subscribe to.
# Use MQTT wildcard filters ("+"/single-level, "#" /multi-level) per MQTT-3.1.
TOPICS = [
    ("bms/actual/data", QOS),
    ("bms/twin/data", QOS),
]

# Idle-exit watchdog: if no message arrives for this long, quit. Set to 0
# to listen forever (Ctrl+C to exit).
TIMEOUT_MS = 1_000_000


class ReceiverApp(QtCore.QObject):
    """Tiny controller: connect → print messages → idle-exit."""

    def __init__(self):
        super().__init__()

        # ── Worker on its own QThread (the worker-object pattern) ──────
        self._thread = QtCore.QThread()
        self._worker = MqttWorker()
        self._worker.moveToThread(self._thread)

        # Wire the worker's signals to our slots.
        self._worker.connected.connect(self._on_connected)
        self._worker.disconnected.connect(self._on_disconnected)
        self._worker.message_received.connect(self._on_message)
        self._worker.error_occurred.connect(self._on_error)

        # Clean up the thread when the worker is done.
        self._thread.finished.connect(self._thread.deleteLater)

        # Idle-exit watchdog. Restarted on every inbound message so the
        # app only quits once traffic truly dries up.
        self._watchdog = QtCore.QTimer()
        self._watchdog.setInterval(TIMEOUT_MS) if TIMEOUT_MS > 0 else None
        if TIMEOUT_MS > 0:
            self._watchdog.setSingleShot(True)
            self._watchdog.timeout.connect(self._on_timeout)

    # ── Lifecycle ───────────────────────────────────────────────────
    def start(self):
        self._thread.start()
        self._worker.connect_broker.emit(
            HOST, PORT, USERNAME, PASSWORD, TOPICS,
        )

    # ── Worker-signal slots ─────────────────────────────────────────
    def _on_connected(self):
        print(f"[receiver] connected to {HOST}:{PORT}  (QoS {QOS})")
        print(f"[receiver] subscribed to: {', '.join(t for t, _ in TOPICS)}")
        if TIMEOUT_MS > 0:
            self._watchdog.start()

    def _on_disconnected(self, rc):
        print(f"[receiver] disconnected  (rc={rc})")
        # Tear the worker's QThread down *before* the app exits so Qt
        # never destroys a still-running thread (which aborts the process).
        self._thread.quit()
        self._thread.wait()
        QtCore.QCoreApplication.quit()

    def _on_error(self, msg):
        print(f"[receiver] ERROR: {msg}", file=sys.stderr)
        self._thread.quit()
        self._thread.wait()
        QtCore.QCoreApplication.quit()

    def _on_message(self, topic, payload, ts):
        # Restart the idle watchdog on every inbound message.
        if TIMEOUT_MS > 0:
            self._watchdog.start()

        source = self._classify(topic)
        body = payload.decode("utf-8", errors="replace")
        print(f"[receiver] ← [{source}] {topic}  ({len(payload)} B)  {body}")

    # ── Watchdog ────────────────────────────────────────────────────
    def _on_timeout(self):
        print(f"[receiver] no messages for {TIMEOUT_MS} ms — exiting.")
        self._worker.disconnect_broker.emit()

    # ── Consumer-side routing (kept OUT of the generic worker) ──────
    @staticmethod
    def _classify(topic):
        """Tag a topic with a stream label.

        The generic worker emits ``(topic, payload, ts)`` without any
        project-specific routing. This is where the consumer decides
        which stream a message belongs to — prefix-match against each
        subscribed root. For the real tool this lives in
        ``MessageParser``; the example just prints the tag.
        """
        for root, _ in TOPICS:
            prefix = root[:-2] if root.endswith("/#") else root
            if topic == prefix or topic.startswith(prefix + "/"):
                return root
        return "?"


def main():
    app = QtCore.QCoreApplication(sys.argv)
    receiver = ReceiverApp()
    receiver.start()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
"""Generic MQTT worker for the BrightSkies tooling stack.

This module contains *only* the broker plumbing: connect, subscribe,
publish, disconnect. It knows nothing about the project's two-stream
(actual / digital-twin) routing, the JSON payload schema, thresholds, or
anything else — that lives in the consumer. Keeping the network layer
generic means the same ``MqttWorker`` works for the Final Comparison
Tool's dual-stream subscriber, a one-off publisher demo, or any other
MQTT client the project needs.

Threading model
---------------
Worker-object + ``moveToThread`` (same pattern as ``ComparisonWorker`` in
the Task 1 Excel tool and ``ParameterExtractionWorker`` in the Task 3
extractor). ``MqttWorker`` is a plain ``QObject``; the ``QThread``
created by the caller only owns the thread's lifecycle.

Importantly, the worker uses paho's ``loop_start()`` — *not*
``loop_forever()`` — so paho's network loop runs on paho's own
background thread while the worker's Qt event loop keeps spinning. That
is what makes later slot calls (``publish_broker``, ``disconnect_broker``)
reachable: Qt queued connections only fire when the receiving thread
runs its event loop, and ``loop_forever`` would have blocked it forever.

All paho callbacks run on paho's network thread; emitting a Qt signal
from them is thread-safe and delivers the payload to the GUI thread via
Qt's queued-connection mechanism.
"""

import time

from PyQt5 import QtCore

import paho.mqtt.client as mqtt


class MqttWorker(QtCore.QObject):
    """Background MQTT client — generic connect / subscribe / publish.

    Lifecycle (driven from the caller, e.g. ``back.py``)::

        thread   = QThread()
        worker   = MqttWorker()
        worker.moveToThread(thread)
        wire worker.connected / .disconnected / .message_received /
             .error_occurred to your slots
        thread.start()
        worker.connect_broker.emit(host, port, user, pwd, topics)
        # …later…
        worker.publish_broker.emit(topic, payload, qos)
        worker.disconnect_broker.emit()
        thread.quit(); thread.wait()

    Signals (emitted on paho's network thread, delivered queued):

    - ``connected()``                       — CONNECT succeeded.
    - ``disconnected(int rc)``              — connection lost (rc=0 is clean).
    - ``message_received(str topic, bytes payload, float ts)``
    - ``error_occurred(str)``                — async error that isn't a clean disconnect.

    Slots (invoked via ``emit`` from the caller, run on the worker thread):

    - ``connect_broker(str host, int port, str user, str pwd, list topics)``
      where ``topics`` is a list of ``(topic_filter, qos)`` pairs to
      subscribe to after CONNECT.
    - ``publish_broker(str topic, bytes or str payload, int qos)``
    - ``disconnect_broker()``
    """

    # --- Signals ----------------------------------------------------------
    connected = QtCore.pyqtSignal()
    disconnected = QtCore.pyqtSignal(int)
    message_received = QtCore.pyqtSignal(str, bytes, float)
    error_occurred = QtCore.pyqtSignal(str)

    # --- Slots (emitted by the caller) -----------------------------------
    connect_broker = QtCore.pyqtSignal(str, int, str, str, list)
    publish_broker = QtCore.pyqtSignal(str, object, int)
    disconnect_broker = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()

        self._client: mqtt.Client = None
        self._topics = []  # the (topic_filter, qos) pairs to subscribe to

        # Route the slot signals to our handlers here so the worker is
        # self-contained — callers only need to moveToThread + start.
        self.connect_broker.connect(self._do_connect)
        self.publish_broker.connect(self._do_publish)
        self.disconnect_broker.connect(self._do_disconnect)

    # ------------------------------------------------------------------
    # paho callbacks — each maps a paho event to a Qt signal.
    # These run on paho's network thread, never on the GUI thread.
    # ------------------------------------------------------------------

    def _on_connect(self, client, userdata, flags, rc):
        if rc != 0:
            self.error_occurred.emit(
                f"MQTT connection refused by broker (rc={rc})."
            )
            return

        # Subscribe only after a successful CONNECT. ``self._topics`` is
        # the list of (topic_filter, qos) pairs the caller passed to
        # ``connect_broker``. paho accepts ``None`` for an empty subscribe
        # list — guard against that so an accidental ``[]`` doesn't crash.
        if self._topics:
            client.subscribe(self._topics)

        self.connected.emit()

    def _on_disconnect(self, client, userdata, rc):
        # rc == 0 → clean (we called ``disconnect()`` ourselves).
        # rc != 0 → unexpected loss; paho's loop threads will retry.
        self.disconnected.emit(rc)

    def _on_message(self, client, userdata, msg):
        # Forward the raw bytes plus a fallback wall-clock timestamp.
        # If the payload carries its own device-side timestamp, the
        # consumer may override it after parsing.
        self.message_received.emit(msg.topic, msg.payload, time.time())

    # ------------------------------------------------------------------
    # Slot handlers (run on the worker thread).
    # ------------------------------------------------------------------

    def _do_connect(self, host, port, user, pwd, topics):
        self._topics = list(topics) if topics else []

        # ``clean_session=True`` keeps things simple: the broker drops
        # queued messages from any previous session at (re)connect. Flip
        # to ``False`` + a fixed ``client_id`` for store-and-forward
        # across tool restarts.
        self._client = mqtt.Client(
            protocol=mqtt.MQTTv311,
            clean_session=True,
        )

        # Credentials are optional. Only set them when a username was
        # supplied so we don't try to authenticate to an anonymous broker.
        if user:
            self._client.username_pw_set(user, pwd)

        # Wire paho callbacks → Qt-signal emitters.
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

        # Blocking connect; a brief delay here is fine because this slot
        # already runs off the GUI thread.
        try:
            self._client.connect(host, port, keepalive=60)
        except Exception as exc:
            self.error_occurred.emit(f"MQTT connect failed: {exc}")
            return

        # ``loop_start()`` puts paho's network loop on paho's own
        # background thread and returns immediately, so the worker
        # thread's Qt event loop keeps spinning — that's what makes
        # ``publish_broker`` / ``disconnect_broker`` slots reachable.
        self._client.loop_start()

    def _do_publish(self, topic, payload, qos):
        if self._client is None:
            self.error_occurred.emit(
                "publish called before a successful connect_broker."
            )
            return
        try:
            self._client.publish(topic, payload, qos=qos)
        except Exception as exc:
            self.error_occurred.emit(f"MQTT publish failed: {exc}")

    def _do_disconnect(self):
        if self._client is None:
            return
        try:
            # Order matters: disconnect() sends the DISCONNECT packet and
            # signals paho's network loop to exit; loop_stop() then joins
            # that thread. Reversing the order can drop the DISCONNECT
            # packet because the loop is already stopped.
            self._client.disconnect()       # send DISCONNECT (best-effort)
            self._client.loop_stop()        # join paho's network thread
        except Exception as exc:
            # Socket already closed / never opened — don't crash on cleanup.
            self.error_occurred.emit(f"MQTT disconnect error: {exc}")
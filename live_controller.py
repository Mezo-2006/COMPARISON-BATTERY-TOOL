"""Controller that bridges the MQTT worker and the comparison engine in live mode.

Role in the architecture
------------------------
Live mode has three threads:

  1. **paho's network thread** — owned by :class:`mqtt_worker.MqttWorker`.
     This is where raw MQTT packets are read off the socket. The worker
     emits ``message_received(topic, payload, ts)`` per packet.
  2. **the controller's own QThread** — *this* module. It owns a
     :class:`live_accumulator.LiveBuffer` and a ``QTimer``. Every
     incoming message is forwarded here (via the ``ingest`` signal),
     parsed + accumulated off the GUI thread, then a periodic timer
     rebuilds the alignment + statistics and emits *results* back to
     the GUI thread for plot / table updates.
  3. **the GUI thread** — owns the widgets. It only ever receives
     ``QGraphics``-safe result objects (``AlignedData``, ``StatisticsResult``)
     and lightweight scalars (sample counts, status strings); it never
     touches a raw payload or runs a heavy compute.

This separation keeps the GUI responsive even when the stream is
saturating the broker at hundreds of Hz: parsing, accumulation, and
alignment/stats all run on the controller's thread, leaving the GUI
thread free to repaint.

Signals / slots
---------------
**Slots** (emitted by ``back.py``, execute on the controller's thread):

  - ``ingest(str topic, bytes payload, float ts)`` — one packet from
    the MQTT worker. Parse → accumulate. Does *not* trigger a re-align;
    that's the timer's job (see Update trigger decision below).
  - ``start(int interval_ms, str twin_root, str ecu_root)`` — turn the
    re-align timer on with the given cadence.
  - ``stop()`` — turn the timer off and clear the buffers.

**Signals** (emitted on the controller's thread, delivered queued to GUI):

  - ``stats_ready(StatisticsResult)`` — fresh statistics; route to the
    Statistics tab populate method.
  - ``aligned_ready(AlignedData)`` — fresh aligned data; route to the
    Preview table + PlotManager.
  - ``sample_count_changed(int, int)`` — ``(n_twin, n_ecu)`` for the
    Live-tab status line.
  - ``error_occurred(str)`` — non-fatal live error (bad packet, align
    failure on a sparse snapshot). The GUI shows it in the status bar;
    the stream keeps going.

Update trigger
--------------
A single ``QTimer`` inside the controller fires every ``interval_ms``
milliseconds (default 500 ms, configurable in the Live tab). On each
tick:

  1. Build two :class:`data_loader.LoadResult` snapshots from the
     buffers (skips the tick if either is empty).
  2. Call :func:`alignment_engine.align` (nearest method, matches the
     offline default).
  3. Call :func:`statistics_engine.compute` (default tolerances —
     ``back.py`` re-runs ``compute`` with the user's tolerances before
     populating the Statistics tab so live results honour the Config tab
     just like offline).
  4. Emit ``aligned_ready`` and ``stats_ready``.

The timer runs on the controller's own thread (we create it with
``QTimer(self)`` after ``moveToThread``), so the tick's compute does
not stall the GUI thread's event loop.
"""

from __future__ import annotations

from typing import Optional

from PyQt5 import QtCore

from live_accumulator import LiveBuffer
from live_schema import LiveSchemaError, parse_message


class LiveController(QtCore.QObject):
    """Qt glue between MQTT ingress and the comparison pipeline.

    Lifecycle (driven from ``back.py``)::

        thread     = QThread()
        controller = LiveController()
        controller.moveToThread(thread)
        wire controller.aligned_ready / .stats_ready /
             .sample_count_changed / .error_occurred to your slots
        thread.start()
        controller.start.emit(interval_ms, twin_root, ecu_root)
        # feed packets in:
        controller.ingest.emit(topic, payload, ts)
        # …later…
        controller.stop.emit()
        thread.quit(); thread.wait()

    The controller owns a :class:`LiveBuffer` (pure Python) plus a
    ``QTimer`` created on the controller's own thread. The timer is
    started/stopped via the ``start`` / ``stop`` slot signals so that
    the ``QTimer`` lives on the same thread its timeout fires on (a
    cross-thread ``QTimer`` is a well-known Qt footgun — ``QTimer``
    objects belong to the thread that created them).
    """

    # --- Signals (emitted here, delivered to GUI thread queued) ----------
    aligned_ready = QtCore.pyqtSignal(object)          # AlignedData
    stats_ready = QtCore.pyqtSignal(object)            # StatisticsResult
    sample_count_changed = QtCore.pyqtSignal(int, int)  # (n_twin, n_ecu)
    error_occurred = QtCore.pyqtSignal(str)

    # --- Slots (emitted by back.py, execute on this thread) --------------
    ingest = QtCore.pyqtSignal(str, bytes, float)
    start = QtCore.pyqtSignal(int, str, str)            # (ms, twin_root, ecu_root)
    stop = QtCore.pyqtSignal()
    reset = QtCore.pyqtSignal()                          # clear buffers

    def __init__(self, tolerance_overrides: Optional[dict] = None):
        super().__init__()

        self.buffer = LiveBuffer()
        self._tolerances = dict(tolerance_overrides) if tolerance_overrides else {}
        self._twin_root = ""
        self._ecu_root = ""
        self._timer: Optional[QtCore.QTimer] = None

        # Route slot signals to handlers here so the controller is
        # self-contained — the caller only needs moveToThread + start.
        self.ingest.connect(self._on_ingest)
        self.start.connect(self._on_start)
        self.stop.connect(self._on_stop)
        self.reset.connect(self._on_reset)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_ingest(self, topic: str, payload: bytes, ts: float) -> None:
        """Parse one MQTT packet and feed it to the accumulator."""
        try:
            samples = parse_message(
                topic, payload, self._twin_root, self._ecu_root,
            )
        except LiveSchemaError as exc:
            # Bad packet — log and continue; the stream keeps going.
            self.error_occurred.emit(f"Bad live packet on {topic!r}: {exc}")
            return
        if not samples:
            return
        self.buffer.add_samples(samples)
        # Cheaper than a full rebuild — just the counts for the status bar.
        n_twin, n_ecu = self.buffer.sample_counts()
        self.sample_count_changed.emit(n_twin, n_ecu)

    def _on_start(self, interval_ms: int, twin_root: str, ecu_root: str) -> None:
        """Remember the topic roots for routing and start the tick timer.

        Does **not** clear the buffer — that's ``reset``'s job. This
        separation matters because ``back.py`` re-emits ``start`` when
        the user nudges the refresh-interval spinbox mid-stream, and
        that must not wipe the accumulated data.
        """
        self._twin_root = twin_root
        self._ecu_root = ecu_root

        # (Re)create the timer on *this* thread so its timeout fires here.
        if self._timer is not None:
            self._timer.stop()
            self._timer.timeout.disconnect()
        self._timer = QtCore.QTimer(self)
        self._timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._timer.timeout.connect(self._tick)
        self._timer.start(max(int(interval_ms), 50))

    def _on_stop(self) -> None:
        """Stop the tick timer. Buffers stay intact until ``reset``."""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _on_reset(self) -> None:
        """Clear both buffers — used by ``back.py`` on a fresh connect."""
        self.buffer.clear()

    # ------------------------------------------------------------------
    # Periodic re-align + re-stat
    # ------------------------------------------------------------------
    def _tick(self) -> None:
        """Snapshot buffers → align → statistics → emit results.

        Imports are local so the module can be loaded in a test
        environment that doesn't have the engines (though they're pure
        Python and always present). Keeping them local also avoids
        import cycles: the engines import ``data_loader``, which is
        already imported by ``live_accumulator``.
        """
        from alignment_engine import align, AlignmentError
        from statistics_engine import compute, StatisticsResult

        twin_result, ecu_result = self.buffer.build_results()
        if twin_result is None or ecu_result is None:
            return  # not enough data yet — wait for more packets

        try:
            aligned = align(twin_result, ecu_result, method="nearest")
        except AlignmentError as exc:
            self.error_occurred.emit(f"Alignment failed: {exc}")
            return

        # Always emit the aligned data — the Preview tab needs it even
        # before statistics are computed.
        self.aligned_ready.emit(aligned)

        try:
            stats = compute(aligned, tolerances=self._tolerances or None)
        except Exception as exc:
            self.error_occurred.emit(f"Statistics failed: {exc}")
            return
        self.stats_ready.emit(stats)

    # ------------------------------------------------------------------
    # Config — the GUI updates tolerances when the user changes them
    # ------------------------------------------------------------------
    def set_tolerances(self, tolerances: dict) -> None:
        """Update the tolerance dict used by the next ``compute`` call.

        Called from the GUI thread when the user edits the Config tab.
        Reading a dict is not atomic in CPython but the GIL makes a
        pointer-sized read safe, and the next tick sees the new dict.
        """
        self._tolerances = dict(tolerances) if tolerances else {}
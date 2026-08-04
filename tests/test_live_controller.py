"""Tests for ``live_controller.py`` — the Qt glue between MQTT and the
comparison engine.

These tests instantiate a ``QApplication`` (offscreen) so Qt's
signal/slot machinery works. The controller is exercised via its
``ingest`` and ``start`` slot signals, and the ``aligned_ready`` /
``stats_ready`` / ``sample_count_changed`` / ``error_occurred`` signals
are captured into lists for assertion.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

# Force the headless Qt platform before QApplication is constructed.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtWidgets

from live_controller import LiveController


# ---------------------------------------------------------------------------
# Session-scoped QApplication
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    yield app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _wire(controller):
    """Capture every signal the controller emits into parallel lists."""
    captured = {
        "aligned": [],
        "stats": [],
        "counts": [],
        "errors": [],
    }
    controller.aligned_ready.connect(captured["aligned"].append)
    controller.stats_ready.connect(captured["stats"].append)
    controller.sample_count_changed.connect(
        lambda n_t, n_e: captured["counts"].append((n_t, n_e)),
    )
    controller.error_occurred.connect(captured["errors"].append)
    return captured


def _pump(qapp, predicate, max_iter=400, delay_ms=20):
    """Pump the Qt event loop until ``predicate()`` is truthy or timeout."""
    for _ in range(max_iter):
        qapp.processEvents()
        if predicate():
            return True
        QtCore.QThread.msleep(delay_ms)
    return False


def _payload(obj) -> bytes:
    return json.dumps(obj).encode("utf-8")


TWIN_ROOT = "bms/twin/#"
ECU_ROOT = "bms/actual/#"


# ---------------------------------------------------------------------------
# Threaded controller fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def controller_on_thread(qapp):
    controller = LiveController()
    thread = QtCore.QThread()
    controller.moveToThread(thread)
    thread.start()
    yield controller, thread
    controller.stop.emit()
    thread.quit()
    thread.wait(3000)


# ---------------------------------------------------------------------------
# Ingest -> sample count
# ---------------------------------------------------------------------------
class TestIngest:
    def test_single_twin_packet_emits_count(
            self, controller_on_thread, qapp,
    ):
        controller, thread = controller_on_thread
        captured = _wire(controller)
        # Arm roots with a huge interval so no tick fires during the
        # ingest-only test (start no longer clears the buffer).
        controller.start.emit(100_000, TWIN_ROOT, ECU_ROOT)
        _ = _pump(qapp, lambda: False, max_iter=2, delay_ms=5)

        controller.ingest.emit(
            "bms/twin", _payload({"t": 0.0, "v": 3.3}), 0.0,
        )
        ok = _pump(qapp, lambda: len(captured["counts"]) > 0)
        assert ok
        n_twin, n_ecu = captured["counts"][-1]
        assert n_twin == 1 and n_ecu == 0

    def test_batch_packet_accumulates_many_rows(
            self, controller_on_thread, qapp,
    ):
        controller, thread = controller_on_thread
        captured = _wire(controller)
        controller.start.emit(100_000, TWIN_ROOT, ECU_ROOT)
        _ = _pump(qapp, lambda: False, max_iter=2, delay_ms=5)
        controller.ingest.emit(
            "bms/actual",
            _payload({"samples": [
                {"t": 0.0, "v": 3.3},
                {"t": 0.1, "v": 3.4},
                {"t": 0.2, "v": 3.5},
            ]}),
            0.0,
        )
        ok = _pump(qapp, lambda: len(captured["counts"]) > 0)
        assert ok
        n_twin, n_ecu = captured["counts"][-1]
        assert n_ecu == 3

    def test_bad_packet_emits_error_not_count(
            self, controller_on_thread, qapp,
    ):
        controller, thread = controller_on_thread
        captured = _wire(controller)
        controller.start.emit(100_000, TWIN_ROOT, ECU_ROOT)
        _ = _pump(qapp, lambda: False, max_iter=2, delay_ms=5)
        controller.ingest.emit("bms/twin", b"not json", 0.0)
        ok = _pump(qapp, lambda: len(captured["errors"]) > 0)
        assert ok
        assert captured["counts"] == []


# ---------------------------------------------------------------------------
# Start -> tick -> aligned_ready + stats_ready
# ---------------------------------------------------------------------------
class TestTick:
    def test_emits_aligned_and_stats_when_both_streams_present(
            self, controller_on_thread, qapp,
    ):
        controller, thread = controller_on_thread
        captured = _wire(controller)

        # Start the tick timer FIRST (sets roots + arms timer, no clear),
        # then feed both streams enough samples for alignment to succeed.
        controller.start.emit(50, TWIN_ROOT, ECU_ROOT)
        _ = _pump(qapp, lambda: False, max_iter=2, delay_ms=5)
        for i in range(3):
            controller.ingest.emit(
                "bms/twin", _payload({"t": float(i), "v": 3.3}), 0.0,
            )
            controller.ingest.emit(
                "bms/actual", _payload({"t": float(i), "v": 3.31}), 0.0,
            )
        ok = _pump(qapp, lambda: len(captured["counts"]) > 0)
        assert ok
        ok = _pump(
            qapp,
            lambda: len(captured["aligned"]) > 0 and len(captured["stats"]) > 0,
            max_iter=200, delay_ms=25,
        )
        assert ok, (
            f"no aligned/stats emitted. aligned={len(captured['aligned'])}"
            f" stats={len(captured['stats'])} errors={captured['errors']}"
        )
        assert captured["errors"] == []
        aligned = captured["aligned"][-1]
        stats = captured["stats"][-1]
        assert "voltage" in aligned.signal_names
        assert stats.total_samples == 3

    def test_tick_skips_when_one_side_empty(
            self, controller_on_thread, qapp,
    ):
        controller, thread = controller_on_thread
        captured = _wire(controller)
        controller.start.emit(50, TWIN_ROOT, ECU_ROOT)
        _ = _pump(qapp, lambda: False, max_iter=2, delay_ms=5)
        controller.ingest.emit(
            "bms/twin", _payload({"t": 0.0, "v": 3.3}), 0.0,
        )
        ok = _pump(qapp, lambda: len(captured["counts"]) > 0)
        assert ok
        # Pump for a while — no aligned/stats should fire (ecu empty).
        _ = _pump(qapp, lambda: False, max_iter=10, delay_ms=25)
        assert captured["aligned"] == []
        assert captured["stats"] == []

    def test_start_does_not_clear_buffer(
            self, controller_on_thread, qapp,
    ):
        # ``start`` only sets roots + arms the timer; it must not wipe
        # accumulated data (interval changes mid-stream call start again).
        controller, thread = controller_on_thread
        captured = _wire(controller)
        controller.start.emit(100_000, TWIN_ROOT, ECU_ROOT)
        _ = _pump(qapp, lambda: False, max_iter=2, delay_ms=5)
        controller.ingest.emit(
            "bms/twin", _payload({"t": 0.0, "v": 3.3}), 0.0,
        )
        ok = _pump(qapp, lambda: len(captured["counts"]) > 0)
        assert ok
        assert controller.buffer.twin_count == 1
        # Re-start (e.g. interval changed) — buffer survives.
        controller.start.emit(100_000, TWIN_ROOT, ECU_ROOT)
        _ = _pump(qapp, lambda: False, max_iter=4, delay_ms=5)
        assert controller.buffer.twin_count == 1

    def test_reset_clears_buffer(
            self, controller_on_thread, qapp,
    ):
        controller, thread = controller_on_thread
        captured = _wire(controller)
        controller.start.emit(100_000, TWIN_ROOT, ECU_ROOT)
        _ = _pump(qapp, lambda: False, max_iter=2, delay_ms=5)
        controller.ingest.emit(
            "bms/twin", _payload({"t": 0.0, "v": 3.3}), 0.0,
        )
        ok = _pump(qapp, lambda: len(captured["counts"]) > 0)
        assert ok
        assert controller.buffer.twin_count == 1
        # reset signal clears both buffers (fresh-connect semantics).
        controller.reset.emit()
        _ = _pump(qapp, lambda: False, max_iter=4, delay_ms=5)
        assert controller.buffer.twin_count == 0


# ---------------------------------------------------------------------------
# set_tolerances
# ---------------------------------------------------------------------------
class TestSetTolerances:
    def test_tolerances_applied_to_compute(
            self, controller_on_thread, qapp,
    ):
        controller, thread = controller_on_thread
        captured = _wire(controller)
        controller.start.emit(50, TWIN_ROOT, ECU_ROOT)
        _ = _pump(qapp, lambda: False, max_iter=2, delay_ms=5)
        # Feed twin=3.3, ecu=3.31 (error 0.01 ~ 0.3 %).  A tight
        # tolerance (0.1 %) yields 0 % match; a loose tolerance (5 %)
        # yields 100 % match. Observe the tight-tolerance run here.
        for i in range(5):
            controller.ingest.emit(
                "bms/twin", _payload({"t": float(i), "v": 3.3}), 0.0,
            )
            controller.ingest.emit(
                "bms/actual", _payload({"t": float(i), "v": 3.31}), 0.0,
            )
        ok = _pump(qapp, lambda: len(captured["counts"]) > 0)
        assert ok

        controller.set_tolerances({"voltage": 0.1})
        ok = _pump(
            qapp,
            lambda: len(captured["stats"]) > 0,
            max_iter=200, delay_ms=25,
        )
        assert ok, f"errors={captured['errors']}"
        # 0.1 % tolerance on a ~0.3 % error → 0 % match.
        assert captured["stats"][-1].match_pct == pytest.approx(0.0)
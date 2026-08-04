"""Comparison worker for the Digital Twin Validation Tool.

Orchestrates the *offline* comparison pipeline on a background thread so
the GUI stays responsive while CSV files are loaded, aligned, and turned
into statistics.

The worker follows the **worker-object + ``moveToThread``** pattern used
by Task 1's ``ComparisonWorker`` and Task 3's ``ParameterExtractionWorker``
(see ``tool/back.py`` and ``tool_mbd_params/back.py``): a ``QObject``
whose ``run`` / ``rerun`` slots are invoked on a dedicated ``QThread``.
It never touches UI widgets directly — it only emits ``progress``,
``finished``, and ``error_occurred`` signals back to the GUI thread.

Pipeline
--------

Three pure-Python backend modules are chained in order:

1. **DataLoader** (``data_loader.load_csv``) — read + normalise each CSV.
2. **AlignmentEngine** (``alignment_engine.align``) — align twin onto ECU.
3. **StatisticsEngine** (``statistics_engine.compute``) — compute metrics.

Two entry points are exposed as Qt slots:

- **``run``** — the full pipeline (used the first time files are loaded).
  Loading the two CSVs is the slow part, so the ``LoadResult`` objects
  are cached on the worker after a successful run.
- **``rerun``** — skips ``load_csv`` and re-inokes only
  ``AlignmentEngine`` + ``StatisticsEngine``.  This is what the Config
  tab's **Apply & Re-run** button uses when the user changes tolerances
  or the alignment method without picking new files.

Progress reporting
------------------

``progress`` emits a percentage in ``[0, 100]``.  The mapping is:

- Full run (``run``):
  ``0`` → start, ``10`` → twin loaded, ``30`` → ECU loaded,
  ``60`` → aligned, ``90`` → stats computed, ``100`` → done.
- Re-run (``rerun``):
  ``0`` → start, ``40`` → aligned, ``80`` → stats computed, ``100`` → done.

The granularity is coarse (per-stage, not per-row) because each backend
stage is a single vectorised numpy/pandas call; per-row progress would
add overhead without user-visible benefit.
"""

from __future__ import annotations

from typing import Dict, Optional

from PyQt5 import QtCore

from data_loader import LoadResult, load_csv
from alignment_engine import (
    AlignedData,
    AlignmentError,
    align,
    ALIGN_NEAREST,
)
from statistics_engine import (
    DEFAULT_WORST_N,
    StatisticsResult,
    compute,
)


class ComparisonWorker(QtCore.QObject):
    """Worker-object that runs the offline comparison pipeline on a QThread."""

    # --- Signals (all emitted from the worker thread, delivered queued to
    #     the GUI thread by Qt's cross-thread signal/slot mechanism) ---
    progress = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(object, object)   # AlignedData, StatisticsResult
    error_occurred = QtCore.pyqtSignal(str)

    # --- Invocation channels (signals emitted from the GUI thread, queued
    #     to the worker thread).  Direct method calls on a worker QObject
    #     bypass Qt's event queue and execute on the caller's thread; using
    #     a self-connected signal ensures the slot runs on the worker
    #     thread.  Same pattern as ``MqttWorker.connect_broker``.
    run_signal = QtCore.pyqtSignal()
    rerun_signal = QtCore.pyqtSignal(str, object, int)

    def __init__(
        self,
        twin_path: str,
        ecu_path: str,
        alignment_method: str = ALIGN_NEAREST,
        tolerances: Optional[Dict[str, float]] = None,
        worst_n: int = DEFAULT_WORST_N,
    ) -> None:
        super().__init__()
        self.twin_path = twin_path
        self.ecu_path = ecu_path
        self.alignment_method = alignment_method
        self.tolerances = tolerances
        self.worst_n = worst_n

        # CachedLoadResults from the last successful ``run``.  ``rerun``
        # uses these to skip the slow file-I/O stage.
        self._twin_result: Optional[LoadResult] = None
        self._ecu_result: Optional[LoadResult] = None

        # Self-connect the invocation channels to the slots.  Once the
        # worker is moved to its QThread, emitting these signals from the
        # GUI thread causes the slots to run on the worker thread (queued
        # connection, picked up by the worker thread's event loop).
        self.run_signal.connect(self.run)
        self.rerun_signal.connect(self.rerun)

    # ------------------------------------------------------------------
    # Full pipeline (first run)
    # ------------------------------------------------------------------
    @QtCore.pyqtSlot()
    def run(self) -> None:
        """Run the full pipeline: load → align → compute.

        Everything in this method executes on the background thread.  It
        must never touch UI widgets directly — only ``progress``,
        ``finished``, and ``error_occurred`` cross back to the main
        thread.  Any exception is caught and surfaced as
        ``error_occurred`` so the GUI shows a message box instead of
        the worker thread dying silently.
        """
        try:
            self.progress.emit(0)

            # --- 1. Load twin CSV ---------------------------------------
            twin_result = load_csv(self.twin_path, "twin")
            self._twin_result = twin_result
            self.progress.emit(10)

            # --- 2. Load ECU CSV ---------------------------------------
            ecu_result = load_csv(self.ecu_path, "ecu")
            self._ecu_result = ecu_result
            self.progress.emit(30)

            # --- 3. Align ----------------------------------------------
            aligned = align(twin_result, ecu_result, self.alignment_method)
            self.progress.emit(60)

            # --- 4. Compute statistics ---------------------------------
            stats = compute(aligned, self.tolerances, self.worst_n)
            self.progress.emit(90)

            # --- 5. Done ------------------------------------------------
            self.progress.emit(100)
            self.finished.emit(aligned, stats)

        except (AlignmentError, Exception) as exc:
            self.error_occurred.emit(
                f"Comparison failed:\n{exc}"
            )

    # ------------------------------------------------------------------
    # Re-run (cached load; new alignment / tolerance settings)
    # ------------------------------------------------------------------
    @QtCore.pyqtSlot(str, object, int)
    def rerun(
        self,
        alignment_method: str,
        tolerances: Optional[Dict[str, float]],
        worst_n: int,
    ) -> None:
        """Re-run alignment + statistics on the cached LoadResults.

        Used by the Config tab's **Apply & Re-run** button.  Requires a
        prior successful ``run`` (i.e. ``_twin_result`` and
        ``_ecu_result`` are populated).  If they aren't, emits
        ``error_occurred`` instead of crashing.

        Parameters
        ----------
        alignment_method
            New alignment strategy (``ALIGN_NEAREST`` or
            ``ALIGN_INTERPOLATE``).
        tolerances
            New per-signal tolerance map (or ``None`` for defaults).
        worst_n
            New worst-N cap for the mismatches table.
        """
        if self._twin_result is None or self._ecu_result is None:
            self.error_occurred.emit(
                "Cannot re-run: files have not been loaded yet. "
                "Run a full comparison first."
            )
            return

        try:
            self.progress.emit(0)

            # Update persisted settings so a later full re-run (if any)
            # picks them up too.
            self.alignment_method = alignment_method
            self.tolerances = tolerances
            self.worst_n = worst_n

            aligned = align(
                self._twin_result, self._ecu_result, alignment_method
            )
            self.progress.emit(40)

            stats = compute(aligned, tolerances, worst_n)
            self.progress.emit(80)

            self.progress.emit(100)
            self.finished.emit(aligned, stats)

        except (AlignmentError, Exception) as exc:
            self.error_occurred.emit(
                f"Re-run failed:\n{exc}"
            )

    # ------------------------------------------------------------------
    # Accessors (for back.py to read cached state)
    # ------------------------------------------------------------------
    @property
    def has_cached_load(self) -> bool:
        """True if both LoadResults are cached and a re-run is possible."""
        return self._twin_result is not None and self._ecu_result is not None

    @property
    def twin_result(self) -> Optional[LoadResult]:
        return self._twin_result

    @property
    def ecu_result(self) -> Optional[LoadResult]:
        return self._ecu_result
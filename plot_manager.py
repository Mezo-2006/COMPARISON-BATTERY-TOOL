"""Plot manager for the Digital Twin Validation Tool.

Wraps the six promoted ``pyqtgraph.PlotWidget`` instances on the Graphs
tab and redraws them from an :class:`AlignedData`:

- ``plotWidgetVoltage``     — twin voltage vs ECU voltage (two overlaid curves)
- ``plotWidgetCurrent``     — twin current vs ECU current
- ``plotWidgetTemperature`` — twin temperature vs ECU temperature
- ``plotWidgetSoc``         — twin SoC vs ECU SoC
- ``plotWidgetSoh``         — twin SoH vs ECU SoH
- ``plotWidgetError``       — error over time (ECU − twin), one curve per
  signal ticked on the "Signals to Include in Error Graph" row

The five overlay plots (V/I/T/SoC/SoH) are **unconditional** — each draws
its twin-vs-ECU pair whenever that signal is present in the aligned data,
with no checkbox gating. Only the error plot has a selection control
(``checkBoxSignalVoltage`` / ``Current`` / ``Temperature`` / ``Soc`` /
``Soh``, plus ``checkBoxSignalSelectOnly`` for radio-button-style single
selection), positioned directly above it on the Graphs tab — so the
question "which signal(s)?" only ever applies to the error graph, not to
whether a signal's own overlay plot shows up.

Unlike the three pure-Python engines, this module **imports Qt and
pyqtgraph** — it directly manipulates the plot widgets. It is instantiated
by ``back.py`` after ``ComparisonWorker.finished`` delivers an
``AlignedData`` and re-invoked whenever the error-graph's signal-selection
checkboxes change.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Set, Tuple

import numpy as np
import pyqtgraph as pg
from pyqtgraph import PlotWidget
from PyQt5.QtWidgets import QLabel

from alignment_engine import AlignedData, AlignedSignal

# Smoother curves — cheap for the data volumes this tool handles.
pg.setConfigOptions(antialias=True)


# ---------------------------------------------------------------------------
# Colour palette for the error plot (one curve per signal)
# ---------------------------------------------------------------------------
# pyqtgraph accepts colours as (R, G, B) tuples, hex strings, or names.  We
# use a fixed mapping so the same signal always gets the same colour across
# runs, with a cycle fallback for any unexpected signal name.
_SIGNAL_ORDER = ["voltage", "current", "temperature", "soc", "soh"]

_ERROR_COLOURS: dict = {
    "voltage":     "#e41a1c",   # red
    "current":     "#377eb8",   # blue
    "temperature": "#4daf4a",   # green
    "soc":         "#984ea3",   # purple
    "soh":         "#ff7f00",   # orange
}
_ERROR_COLOUR_CYCLE = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
                       "#ff7f00", "#a65628", "#f781bf", "#999999"]

# Twin vs ECU overlay colours (twin faint, ECU bold).  Same for every overlay
# plot so the user learns a single visual convention.
_TWIN_COLOUR = (100, 149, 237)   # cornflower blue — the model is the "reference"
_ECU_COLOUR  = (20, 20, 20)      # near-black — ground truth

# Threshold-warning styling: dashed limit line, a bold red "flood" for the
# samples that cross it, and a translucent fill so the overrun is
# unmistakable at a glance.
_THRESHOLD_LINE_COLOUR = (255, 140, 0)   # amber
_EXCEEDANCE_COLOUR     = (211, 47, 47)   # red
_EXCEEDANCE_FILL       = (211, 47, 47, 60)

# Bottom-axis label, keyed by ``AlignedData.axis_kind`` — "sequence" for a
# plain sample counter (no time units at all), "timestamp" for real time
# (offline: whatever unit the file uses; live: Unix ms).
_AXIS_LABELS = {
    "sequence": "Sample ID",
    "timestamp": "Time (offline: file units; live: Unix ms)",
}

# (enabled, value) per signal, as read from the Config tab.
ThresholdSpec = Tuple[bool, float]


class PlotManager:
    """Redraws the five overlay plots + one error plot from aligned data."""

    def __init__(
        self,
        plot_voltage: PlotWidget,
        plot_current: PlotWidget,
        plot_temperature: PlotWidget,
        plot_soc: PlotWidget,
        plot_soh: PlotWidget,
        plot_error: PlotWidget,
        warning_labels: Optional[Dict[str, QLabel]] = None,
    ) -> None:
        self.plot_voltage = plot_voltage
        self.plot_current = plot_current
        self.plot_temperature = plot_temperature
        self.plot_soc = plot_soc
        self.plot_soh = plot_soh
        self.plot_error = plot_error
        self._overlay_plots: Dict[str, PlotWidget] = {
            "voltage": plot_voltage,
            "current": plot_current,
            "temperature": plot_temperature,
            "soc": plot_soc,
            "soh": plot_soh,
        }
        # Optional {"voltage": QLabel, "current": QLabel, ...} banners that
        # get a red "exceeded" message when a threshold is crossed, and are
        # cleared otherwise.  ``back.py`` wires these to the Graphs tab;
        # tests that don't care about warnings simply omit them.
        self.warning_labels = warning_labels or {}

        # Per-plot "should the live sliding window keep pinning this
        # plot's X range" flag -- see ``_apply_live_window`` /
        # ``resume_live_follow``. Starts True (follow live data) for
        # every plot; a user drag/wheel gesture on a plot flips its own
        # entry to False so that plot stops getting overridden every
        # tick, without touching the others.
        self._live_follow: Dict[PlotWidget, bool] = {
            p: True for p in self._all_plots()
        }
        for plot in self._all_plots():
            # ``PlotItem.sigRangeChangedManually`` (not the ViewBox's own
            # copy) is what we want here: pyqtgraph forwards the
            # ViewBox's signal onto it for a real drag/wheel gesture,
            # *and* ``PlotItem.autoBtnClicked`` re-emits the same signal
            # when the user clicks the "A" auto-range button -- so one
            # connection catches both ways a user can wrestle control of
            # the view back. See ``_on_user_range_change``.
            plot.plotItem.sigRangeChangedManually.connect(
                lambda _mask, p=plot: self._on_user_range_change(p)
            )

        # Apply a consistent look to every plot.
        for plot in self._all_plots():
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.addLegend(offset=(10, 10))
            plot.setBackground("w")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(
        self,
        aligned: AlignedData,
        enabled_signals: Optional[Iterable[str]] = None,
        thresholds: Optional[Dict[str, ThresholdSpec]] = None,
        live_window_ms: Optional[float] = None,
    ) -> None:
        """Redraw every plot from ``aligned``.

        Parameters
        ----------
        aligned
            Output of :func:`alignment_engine.align`.
        enabled_signals
            Canonical signal names ticked on the error graph's own
            selection row. Only affects ``plotWidgetError`` — the five
            overlay plots (V/I/T/SoC/SoH) always draw whenever their
            signal is present in ``aligned``, regardless of this set. If
            ``None``, every signal present in ``aligned`` is enabled on
            the error graph too.
        thresholds
            Optional ``{"voltage": (enabled, value), "current": (enabled,
            value)}`` from the Config tab's threshold-warning controls.
            When a signal's threshold is enabled, samples above ``value``
            are highlighted on its own overlay plot and the matching
            warning label (if any) is set to a red summary; otherwise the
            label is cleared.
        live_window_ms
            When given (live mode only), each plot is pinned to a
            sliding ``[latest_valid_t - live_window_ms, latest_valid_t]``
            X range instead of auto-fitting every accumulated sample.
            Despite the name, this is in whatever unit
            ``aligned.axis_kind`` is — Unix epoch milliseconds for
            ``"timestamp"``, a sample count for ``"sequence"`` —
            ``back.py`` picks the right one before calling in. See
            ``back.py``'s ``_LIVE_PLOT_WINDOW_MS`` /
            ``_LIVE_PLOT_WINDOW_SEQUENCE`` for why. ``None`` (the offline
            default) leaves pyqtgraph's normal auto-range behaviour in
            place so a finished run's full time range is visible. See
            ``_apply_live_window`` for what "pinned" means per-plot and
            when a plot is left alone instead.
        """
        thresholds = thresholds or {}
        if enabled_signals is None:
            enabled: Set[str] = set(aligned.signal_names)
        else:
            enabled = set(enabled_signals)

        for name in _SIGNAL_ORDER:
            self._plot_overlay(self._overlay_plots[name], name, aligned,
                              thresholds.get(name))
        self._plot_errors(aligned, enabled)

        if live_window_ms is not None and aligned.timestamps.size > 0:
            self._apply_live_window(aligned, enabled, live_window_ms)

    def clear_all(self) -> None:
        """Clear every plot.  Called by ``back.py`` when new files are loaded."""
        for plot in self._all_plots():
            plot.clear()
            self._clear_legend(plot)

    def set_live_follow(self, follow: bool) -> None:
        """Set every plot's live-follow state at once.

        ``follow=False`` is what the Live tab's "Freeze View" toggle
        uses: it pauses the sliding window on every plot until
        explicitly resumed, unlike a bare one-off re-render (which the
        *next* live tick would just redraw over anyway, since
        ``_on_live_aligned`` calls :meth:`update` with
        ``live_window_ms`` set on every tick regardless). Curves keep
        updating with new samples either way — this only pauses the
        X-range pin, not the redraw.
        """
        for plot in self._all_plots():
            self._live_follow[plot] = follow

    def resume_live_follow(self) -> None:
        """Re-arm the sliding live window on every plot.

        Call this whenever a live session (re)starts fresh — ``back.py``
        calls it right alongside ``LiveController.reset`` on a new MQTT
        connect. A user who dragged/zoomed/auto-ranged a plot (or froze
        the view — see :meth:`set_live_follow`) during the *previous*
        session gets a clean slate: the new session's data is what
        they'll see tracked again, rather than being stuck on a stale
        manual range from before (the original bug this window-pinning
        code was built to fix — see context.md §5.7). A thin alias over
        ``set_live_follow(True)`` kept because call sites already spell
        it this way for the reconnect case.
        """
        self.set_live_follow(True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _all_plots(self):
        return (self.plot_voltage, self.plot_current, self.plot_temperature,
                self.plot_soc, self.plot_soh, self.plot_error)

    @staticmethod
    def _clear_legend(plot: PlotWidget) -> None:
        """Clear a plot's legend if one exists.

        pyqtgraph 0.14 dropped the convenience ``getLegend()`` accessor;
        the legend is now stored on ``PlotItem`` and may be ``None`` if
        ``addLegend()`` was never called.  Clearing it on re-plot keeps
        the legend from accumulating stale entries across updates.
        """
        legend = getattr(plot.plotItem, "legend", None)
        if legend is not None:
            legend.clear()

    def _on_user_range_change(self, plot: PlotWidget) -> None:
        """A user manually took control of ``plot`` — stop auto-pinning it.

        Connected to that plot's ``PlotItem.sigRangeChangedManually``,
        which pyqtgraph emits for a direct mouse gesture (pan,
        scroll-zoom, right-drag-zoom) *and* for a click on pyqtgraph's
        own "A" auto-range button (``PlotItem.autoBtnClicked`` re-emits
        the same signal) — never for our own programmatic ``setXRange``
        calls below, so this can't see its own pin as a "user" change.
        Only this one plot's follow flag is cleared; the other five keep
        tracking live data normally. Cleared back to following on the
        next ``resume_live_follow()`` (a fresh live session), not on a
        timer — the user asked to look around, and should get to keep
        looking until they reconnect.
        """
        self._live_follow[plot] = False

    def _apply_live_window(
        self,
        aligned: AlignedData,
        enabled: Set[str],
        window_ms: float,
    ) -> None:
        """Pin each plot's X range to its own latest ``window_ms`` of data.

        Two things this fixes over a single shared ``[t_max -
        window_ms, t_max]`` window applied to every plot identically:

        1. **A plot going blank while others keep updating.** Signals
           don't necessarily update at the same cadence (e.g. SoH
           refreshing far less often than voltage). Pinning every plot
           to the *reference* (ECU) timeline's latest timestamp means a
           sparser signal's own last *valid* sample can fall outside
           that shared window even though it's the most recent data
           that signal actually has — the plot then shows an empty
           range while its neighbours look fine. Using each overlay
           plot's own last non-NaN twin/ECU pair as that plot's
           ``t_max`` keeps its window anchored to data it actually has.
        2. **The view snapping back the instant a user tries to look
           around.** Forcing ``setXRange`` unconditionally on every tick
           (the previous behaviour) fought any user interaction — a
           manual pan/zoom, or clicking pyqtgraph's own "auto-range"
           button, both got overridden within one tick, which read as
           "it moves on its own and reverts immediately." ``self.
           _live_follow`` gates the forced pin per plot now (cleared by
           a user drag/zoom/auto-btn click -- see
           ``_on_user_range_change``) -- that plot alone stops being
           pinned; the rest keep following normally. It only clears back
           to following on ``resume_live_follow()`` (a fresh live
           session), which is what actually fixes the original "stuck
           stale range after reconnect" bug this pinning was built for
           -- reconnecting is the right place to reset the view, not
           every single tick regardless of what the user is doing.

        The error plot doesn't get a single "the" signal's own valid
        range (it overlays whichever signals are ticked in ``enabled``)
        -- its ``t_max`` is the latest valid sample across just those
        enabled signals, falling back to the aligned reference timeline
        if none of them have any valid samples in the current snapshot.
        """
        ts = aligned.timestamps
        global_t_max = float(np.nanmax(ts))

        for name in _SIGNAL_ORDER:
            plot = self._overlay_plots[name]
            t_max = global_t_max
            if name in aligned.signals:
                sig = aligned.signals[name]
                valid = (~np.isnan(sig.twin_values)) & (~np.isnan(sig.ecu_values))
                if valid.any():
                    t_max = float(np.nanmax(ts[valid]))
            self._pin_plot(plot, t_max, window_ms)

        err_t_max = None
        for name in _SIGNAL_ORDER:
            if name not in aligned.signals or name not in enabled:
                continue
            sig = aligned.signals[name]
            valid = (~np.isnan(sig.twin_values)) & (~np.isnan(sig.ecu_values))
            if valid.any():
                candidate = float(np.nanmax(ts[valid]))
                err_t_max = candidate if err_t_max is None else max(err_t_max, candidate)
        self._pin_plot(self.plot_error,
                       global_t_max if err_t_max is None else err_t_max,
                       window_ms)

    def _pin_plot(self, plot: PlotWidget, t_max: float, window_ms: float) -> None:
        """Set ``plot``'s X range to ``[t_max - window_ms, t_max]`` unless
        the user currently has manual control of it (``self._live_follow``,
        see ``_apply_live_window`` / ``_on_user_range_change``)."""
        if not self._live_follow.get(plot, True):
            return
        t_min = t_max - max(window_ms, 0.0)
        plot.setXRange(t_min, t_max, padding=0)

    def _plot_overlay(
        self,
        plot: PlotWidget,
        signal_name: str,
        aligned: AlignedData,
        threshold: Optional[ThresholdSpec] = None,
    ) -> None:
        """Plot twin vs ECU for one signal on its dedicated PlotWidget.

        Unconditional: draws whenever ``signal_name`` is present in
        ``aligned`` — there is no per-signal checkbox for the overlay
        plots (only the error graph has a selection row).
        """
        plot.clear()
        self._clear_legend(plot)
        label = self.warning_labels.get(signal_name)

        # If the signal isn't in this run (e.g. 3-signal ECU CSV), leave
        # the plot empty.
        if signal_name not in aligned.signals:
            plot.setLabel("left", signal_name.title(), units="")
            if label is not None:
                label.setText("")
            return

        sig: AlignedSignal = aligned.signals[signal_name]
        ts = aligned.timestamps

        # Mask out NaN pairs so the curves don't draw spurious vertical jumps.
        valid = (~np.isnan(sig.twin_values)) & (~np.isnan(sig.ecu_values))
        t_win = ts[valid]
        twin_v = sig.twin_values[valid]
        ecu_v = sig.ecu_values[valid]

        plot.plot(t_win, twin_v, pen=pg.mkPen(_TWIN_COLOUR, width=1.5),
                  name="Digital Twin")
        plot.plot(t_win, ecu_v, pen=pg.mkPen(_ECU_COLOUR, width=2.0),
                  name="ECU")

        plot.setLabel("left", signal_name.title())
        plot.setLabel("bottom", _AXIS_LABELS.get(aligned.axis_kind, _AXIS_LABELS["timestamp"]))

        self._apply_threshold(plot, label, signal_name, t_win, ecu_v,
                              threshold)

    def _apply_threshold(
        self,
        plot: PlotWidget,
        label: Optional[QLabel],
        signal_name: str,
        t_win: np.ndarray,
        ecu_v: np.ndarray,
        threshold: Optional[ThresholdSpec],
    ) -> None:
        """Draw the threshold line + red flood-fill for samples over it.

        Compares the ECU (ground-truth) trace against the limit, since
        that's the measurement a real threshold breach would come from.
        When any sample exceeds the limit: a dashed amber line marks it,
        the offending stretch of curve is redrawn in bold red with a
        translucent fill down to the limit, and ``label`` gets a plain-
        English summary of the worst breach. With nothing to warn about
        (disabled, no data, or never exceeded) the label is cleared.
        """
        if label is not None:
            label.setText("")
        if threshold is None or ecu_v.size == 0:
            return
        enabled, limit = threshold
        if not enabled:
            return

        plot.addItem(pg.InfiniteLine(
            pos=limit, angle=0, movable=False,
            pen=pg.mkPen(_THRESHOLD_LINE_COLOUR, width=1.5,
                        style=Qt_PenStyle_Dash()),
            label=f"limit {limit:g}",
            labelOpts={"color": _THRESHOLD_LINE_COLOUR, "position": 0.95},
        ))

        exceeded = ecu_v > limit
        if not exceeded.any():
            return

        flood = np.where(exceeded, ecu_v, np.nan)
        plot.plot(t_win, flood, pen=pg.mkPen(_EXCEEDANCE_COLOUR, width=2.5),
                  fillLevel=limit, brush=_EXCEEDANCE_FILL,
                  name="Over limit")

        if label is not None:
            worst_idx = int(np.nanargmax(np.where(exceeded, ecu_v, -np.inf)))
            label.setText(
                f"⚠ {signal_name.title()} exceeded {limit:g} "
                f"(peak {ecu_v[worst_idx]:.2f} at t={t_win[worst_idx]:.2f}s)"
            )

    def _plot_errors(
        self,
        aligned: AlignedData,
        enabled: Set[str],
    ) -> None:
        """Plot per-signal error (ECU − twin) over time on the error widget.

        The only plot gated by ``enabled`` — the "Signals to Include in
        Error Graph" checkboxes above it on the Graphs tab.
        """
        plot = self.plot_error
        plot.clear()
        self._clear_legend(plot)

        colour_idx = 0
        for name in _SIGNAL_ORDER:
            if name not in aligned.signals or name not in enabled:
                continue
            sig = aligned.signals[name]

            ts = aligned.timestamps
            err = sig.errors
            valid = (~np.isnan(sig.twin_values)) & (~np.isnan(sig.ecu_values))

            colour = _ERROR_COLOURS.get(
                name, _ERROR_COLOUR_CYCLE[colour_idx % len(_ERROR_COLOUR_CYCLE)]
            )
            colour_idx += 1

            plot.plot(ts[valid], err[valid],
                      pen=pg.mkPen(colour, width=1.5),
                      name=name.title())

        # Zero reference line so drift direction is obvious.
        plot.addItem(pg.InfiniteLine(pos=0.0, angle=0,
                                      pen=pg.mkPen((200, 200, 200),
                                                   style=Qt_PenStyle_Dash(),
                                                   width=1.0)))
        plot.setLabel("left", "Error (ECU − Twin)")
        plot.setLabel("bottom", _AXIS_LABELS.get(aligned.axis_kind, _AXIS_LABELS["timestamp"]))


# ---------------------------------------------------------------------------
# Small helper: Qt::DashLine without importing QtCore at module top
# ---------------------------------------------------------------------------
def Qt_PenStyle_Dash():
    """Return ``QtCore.Qt.DashLine`` lazily (keeps the import local)."""
    from PyQt5 import QtCore
    return QtCore.Qt.DashLine

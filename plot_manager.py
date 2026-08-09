"""Plot manager for the Digital Twin Validation Tool.

Wraps the two promoted ``pyqtgraph.PlotWidget`` instances on the Graphs
tab and redraws them from an :class:`AlignedData`:

- ``plotWidgetOverlay`` — actual (ECU) vs Digital Twin, one pair of
  curves (twin dashed, ECU solid, both in the signal's colour) per
  signal the user has ticked on the Graphs tab's signal-selection row.
  Any subset of the five signals (voltage, current, temperature, SoC,
  SoH) may be overlaid together.
- ``plotWidgetError``   — error over time (ECU minus twin), one curve
  per ticked signal — the same set that's on the overlay plot, just the
  single error quantity instead of the twin/ECU pair.

Which signals are enabled is controlled entirely by the Graphs tab's own
checkboxes (``checkBoxSignalVoltage`` / ``Current`` / ``Temperature`` /
``Soc`` / ``Soh``, plus ``checkBoxSignalSelectOnly`` which constrains the
above to a single choice) — ``back.py`` reads them into a plain
``set[str]`` and passes it to :meth:`PlotManager.update`. Unlike the
previous four-plot layout, SoC/SoH are no longer force-included: what you
tick is what you get on both plots.

Unlike the three pure-Python engines, this module **imports Qt and
pyqtgraph** — it directly manipulates the plot widgets. It is instantiated
by ``back.py`` after ``ComparisonWorker.finished`` delivers an
``AlignedData`` and re-invoked whenever the Graphs tab's signal-selection
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
# Colour palette — one colour per signal, shared by the overlay plot (twin
# dashed / ECU solid, same colour) and the error plot (one solid curve).
# Fixed mapping so a signal keeps the same colour across runs and between
# the two plots, with a cycle fallback for any unexpected signal name.
# ---------------------------------------------------------------------------
_SIGNAL_ORDER = ["voltage", "current", "temperature", "soc", "soh"]

_SIGNAL_COLOURS: dict = {
    "voltage":     "#e41a1c",   # red
    "current":     "#377eb8",   # blue
    "temperature": "#4daf4a",   # green
    "soc":         "#984ea3",   # purple
    "soh":         "#ff7f00",   # orange
}
_SIGNAL_COLOUR_CYCLE = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
                        "#ff7f00", "#a65628", "#f781bf", "#999999"]

# Threshold-warning styling: dashed limit line, a bold red "flood" for the
# samples that cross it, and a translucent fill so the overrun is
# unmistakable at a glance.
_THRESHOLD_LINE_COLOUR = (255, 140, 0)   # amber
_EXCEEDANCE_COLOUR     = (211, 47, 47)   # red
_EXCEEDANCE_FILL       = (211, 47, 47, 60)

# (enabled, value) per signal, as read from the Config tab.
ThresholdSpec = Tuple[bool, float]


class PlotManager:
    """Redraws the two Graphs-tab plots (overlay + error) from aligned data."""

    def __init__(
        self,
        plot_overlay: PlotWidget,
        plot_error: PlotWidget,
        warning_labels: Optional[Dict[str, QLabel]] = None,
    ) -> None:
        self.plot_overlay = plot_overlay
        self.plot_error = plot_error
        # Optional {"voltage": QLabel, "current": QLabel, ...} banners that
        # get a red "exceeded" message when a threshold is crossed, and are
        # cleared otherwise. ``back.py`` wires these to the Graphs tab;
        # tests that don't care about warnings simply omit them.
        self.warning_labels = warning_labels or {}

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
        """Redraw both plots from ``aligned``.

        Parameters
        ----------
        aligned
            Output of :func:`alignment_engine.align`.
        enabled_signals
            Canonical signal names the user has ticked on the Graphs tab's
            signal-selection row. If ``None``, all signals present in
            ``aligned`` are enabled. Unlike the previous design, SoC/SoH
            are **not** force-included — only what's ticked is drawn.
        thresholds
            Optional ``{"voltage": (enabled, value), "current": (enabled,
            value)}`` from the Config tab's threshold-warning controls.
            When a signal's threshold is enabled and that signal is
            currently plotted, samples above ``value`` are highlighted on
            the overlay plot and the matching warning label (if any) is
            set to a red summary; otherwise the label is cleared.
        live_window_ms
            When given (live mode only), both plots are pinned to a
            sliding ``[latest_t - live_window_ms, latest_t]`` X range
            (in the live pipeline's wire timestamp unit, Unix epoch
            milliseconds) instead of auto-fitting every accumulated
            sample. This keeps a live stream's view *shifting* forward at
            a constant width as new data arrives rather than
            *compressing* the whole growing history into the same view,
            and it also guarantees the view snaps back onto the fresh
            data after a reconnect (auto-range can otherwise get stuck on
            a stale manual zoom/pan from before a disconnect). ``None``
            (the offline default) leaves pyqtgraph's normal auto-range
            behaviour in place so a finished run's full time range is
            visible.
        """
        thresholds = thresholds or {}
        if enabled_signals is None:
            enabled: Set[str] = set(aligned.signal_names)
        else:
            enabled = set(enabled_signals)

        self._plot_overlay(aligned, enabled, thresholds)
        self._plot_errors(aligned, enabled)

        if live_window_ms is not None and aligned.timestamps.size > 0:
            t_max = float(np.nanmax(aligned.timestamps))
            self._apply_live_window(t_max, live_window_ms)

    def clear_all(self) -> None:
        """Clear every plot. Called by ``back.py`` when new files are loaded."""
        for plot in self._all_plots():
            plot.clear()
            self._clear_legend(plot)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _all_plots(self):
        return (self.plot_overlay, self.plot_error)

    @staticmethod
    def _clear_legend(plot: PlotWidget) -> None:
        """Clear a plot's legend if one exists.

        pyqtgraph 0.14 dropped the convenience ``getLegend()`` accessor;
        the legend is now stored on ``PlotItem`` and may be ``None`` if
        ``addLegend()`` was never called. Clearing it on re-plot keeps
        the legend from accumulating stale entries across updates.
        """
        legend = getattr(plot.plotItem, "legend", None)
        if legend is not None:
            legend.clear()

    def _apply_live_window(self, t_max: float, window_ms: float) -> None:
        """Pin both plots' X range to the latest ``window_ms`` milliseconds.

        Called unconditionally on every live tick (not just once) so the
        view always tracks the newest data — this also overrides any
        manual pan/zoom the user made before, which is what prevents a
        stream from appearing to "go blank" after a disconnect/reconnect
        (a stale manual range that no longer contains the fresh, reset
        buffer's timestamps).
        """
        t_min = t_max - max(window_ms, 0.0)
        for plot in self._all_plots():
            plot.setXRange(t_min, t_max, padding=0)

    def _plot_overlay(
        self,
        aligned: AlignedData,
        enabled: Set[str],
        thresholds: Dict[str, ThresholdSpec],
    ) -> None:
        """Plot twin vs ECU for every enabled signal on the shared overlay plot."""
        plot = self.plot_overlay
        plot.clear()
        self._clear_legend(plot)

        for label_widget in self.warning_labels.values():
            label_widget.setText("")

        colour_idx = 0
        any_plotted = False
        for name in _SIGNAL_ORDER:
            if name not in aligned.signals or name not in enabled:
                continue

            sig: AlignedSignal = aligned.signals[name]
            ts = aligned.timestamps

            # Mask out NaN pairs so the curves don't draw spurious vertical
            # jumps.
            valid = (~np.isnan(sig.twin_values)) & (~np.isnan(sig.ecu_values))
            t_win = ts[valid]
            twin_v = sig.twin_values[valid]
            ecu_v = sig.ecu_values[valid]

            colour = _SIGNAL_COLOURS.get(
                name, _SIGNAL_COLOUR_CYCLE[colour_idx % len(_SIGNAL_COLOUR_CYCLE)]
            )
            colour_idx += 1
            title = name.title()

            plot.plot(t_win, twin_v,
                      pen=pg.mkPen(colour, width=1.5, style=Qt_PenStyle_Dash()),
                      name=f"{title} (Twin)")
            plot.plot(t_win, ecu_v, pen=pg.mkPen(colour, width=2.0),
                      name=f"{title} (ECU)")
            any_plotted = True

            self._apply_threshold(plot, self.warning_labels.get(name), name,
                                  colour, t_win, ecu_v, thresholds.get(name))

        plot.setLabel("left", "Value")
        plot.setLabel("bottom", "Time (offline: file units; live: Unix ms)")
        if not any_plotted:
            plot.setLabel("left", "Value (no signal selected)")

    def _apply_threshold(
        self,
        plot: PlotWidget,
        label: Optional[QLabel],
        signal_name: str,
        colour,
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
        (disabled, no data, or never exceeded) the label is left cleared
        (already reset for every signal at the top of ``_plot_overlay``).
        """
        if threshold is None or ecu_v.size == 0:
            return
        enabled, limit = threshold
        if not enabled:
            return

        plot.addItem(pg.InfiniteLine(
            pos=limit, angle=0, movable=False,
            pen=pg.mkPen(_THRESHOLD_LINE_COLOUR, width=1.5,
                        style=Qt_PenStyle_Dash()),
            label=f"{signal_name.title()} limit {limit:g}",
            labelOpts={"color": _THRESHOLD_LINE_COLOUR, "position": 0.95},
        ))

        exceeded = ecu_v > limit
        if not exceeded.any():
            return

        flood = np.where(exceeded, ecu_v, np.nan)
        plot.plot(t_win, flood, pen=pg.mkPen(_EXCEEDANCE_COLOUR, width=2.5),
                  fillLevel=limit, brush=_EXCEEDANCE_FILL,
                  name=f"{signal_name.title()} over limit")

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
        """Plot per-signal error (ECU − twin) over time on the error widget."""
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

            colour = _SIGNAL_COLOURS.get(
                name, _SIGNAL_COLOUR_CYCLE[colour_idx % len(_SIGNAL_COLOUR_CYCLE)]
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
        plot.setLabel("bottom", "Time (offline: file units; live: Unix ms)")


# ---------------------------------------------------------------------------
# Small helper: Qt::DashLine without importing QtCore at module top
# ---------------------------------------------------------------------------
def Qt_PenStyle_Dash():
    """Return ``QtCore.Qt.DashLine`` lazily (keeps the import local)."""
    from PyQt5 import QtCore
    return QtCore.Qt.DashLine

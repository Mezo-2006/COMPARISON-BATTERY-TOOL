"""Plot manager for the Digital Twin Validation Tool.

Wraps the four promoted ``pyqtgraph.PlotWidget`` instances on the Graphs
tab and redraws them from an :class:`AlignedData`:

- ``plotWidgetVoltage``     — twin voltage vs ECU voltage (two overlaid curves)
- ``plotWidgetCurrent``     — twin current vs ECU current
- ``plotWidgetTemperature`` — twin temperature vs ECU temperature
- ``plotWidgetError``       — error over time (ECU − twin), one curve per enabled signal

Signal-inclusion is controlled by the Config tab's checkboxes
(``checkBoxIncludeVoltage`` / ``Current`` / ``Temperature``).  SoC and
SoH have no dedicated overlay widget and no checkbox — they always
appear on the error plot when present in the aligned data, so the user
can still see their drift even though they aren't on the V/I/T tabs.

Unlike the three pure-Python engines, this module **imports Qt and
pyqtgraph** — it directly manipulates the plot widgets.  It is instantiated
by ``back.py`` after ``ComparisonWorker.finished`` delivers an
``AlignedData`` and re-invoked whenever the Config tab's enabled-signal
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
# runs, with a cycle fallback for any unexpected signal names.
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

# (enabled, value) per signal, as read from the Config tab.
ThresholdSpec = Tuple[bool, float]


class PlotManager:
    """Redraws the four Graphs-tab plots from aligned data."""

    def __init__(
        self,
        plot_voltage: PlotWidget,
        plot_current: PlotWidget,
        plot_temperature: PlotWidget,
        plot_error: PlotWidget,
        warning_labels: Optional[Dict[str, QLabel]] = None,
    ) -> None:
        self.plot_voltage = plot_voltage
        self.plot_current = plot_current
        self.plot_temperature = plot_temperature
        self.plot_error = plot_error
        # Optional {"voltage": QLabel, "current": QLabel, ...} banners that
        # get a red "exceeded" message when a threshold is crossed, and are
        # cleared otherwise.  ``back.py`` wires these to the Graphs tab;
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
    ) -> None:
        """Redraw all four plots from ``aligned``.

        Parameters
        ----------
        aligned
            Output of :func:`alignment_engine.align`.
        enabled_signals
            Canonical signal names that the user has ticked on the Config
            tab.  If ``None``, all signals present in ``aligned`` are
            enabled.  SoC and SoH are always shown on the error plot when
            present (they have no checkbox), regardless of this set.
        thresholds
            Optional ``{"voltage": (enabled, value), "current": (enabled,
            value)}`` from the Config tab's threshold-warning controls.
            When a signal's threshold is enabled, samples above ``value``
            are highlighted on its overlay plot and the matching warning
            label (if any) is set to a red summary; otherwise the label
            is cleared.
        """
        thresholds = thresholds or {}
        if enabled_signals is None:
            enabled: Set[str] = set(aligned.signal_names)
        else:
            enabled = set(enabled_signals)
            # SoC/SoH have no checkbox — always include them on the error
            # plot when present in the data.
            enabled |= {"soc", "soh"} & set(aligned.signal_names)

        self._plot_overlay(self.plot_voltage, "voltage", aligned, enabled,
                          thresholds.get("voltage"))
        self._plot_overlay(self.plot_current, "current", aligned, enabled,
                          thresholds.get("current"))
        self._plot_overlay(self.plot_temperature, "temperature",
                           aligned, enabled, thresholds.get("temperature"))
        self._plot_errors(aligned, enabled)

    def clear_all(self) -> None:
        """Clear every plot.  Called by ``back.py`` when new files are loaded."""
        for plot in self._all_plots():
            plot.clear()
            self._clear_legend(plot)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _all_plots(self):
        return (self.plot_voltage, self.plot_current,
                self.plot_temperature, self.plot_error)

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

    def _plot_overlay(
        self,
        plot: PlotWidget,
        signal_name: str,
        aligned: AlignedData,
        enabled: Set[str],
        threshold: Optional[ThresholdSpec] = None,
    ) -> None:
        """Plot twin vs ECU for one signal on its dedicated PlotWidget."""
        plot.clear()
        self._clear_legend(plot)
        label = self.warning_labels.get(signal_name)

        # If the signal isn't in this run (e.g. 3-signal ECU CSV) or the
        # user has unticked its checkbox, leave the plot empty.
        if signal_name not in aligned.signals or signal_name not in enabled:
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
        plot.setLabel("bottom", "Time", units="s")

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
        """Plot per-signal error (ECU − twin) over time on the error widget."""
        plot = self.plot_error
        plot.clear()
        self._clear_legend(plot)

        colour_idx = 0
        for name, sig in aligned.signals.items():
            if name not in enabled:
                continue

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
        plot.setLabel("bottom", "Time", units="s")


# ---------------------------------------------------------------------------
# Small helper: Qt::DashLine without importing QtCore at module top
# ---------------------------------------------------------------------------
def Qt_PenStyle_Dash():
    """Return ``QtCore.Qt.DashLine`` lazily (keeps the import local)."""
    from PyQt5 import QtCore
    return QtCore.Qt.DashLine
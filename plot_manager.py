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

from typing import Iterable, Optional, Set

import numpy as np
import pyqtgraph as pg
from pyqtgraph import PlotWidget

from alignment_engine import AlignedData, AlignedSignal


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
_TWIN_COLOUR = (180, 180, 180)   # grey — the model is the "reference"
_ECU_COLOUR  = (0, 0, 0)         # black — ground truth


class PlotManager:
    """Redraws the four Graphs-tab plots from aligned data."""

    def __init__(
        self,
        plot_voltage: PlotWidget,
        plot_current: PlotWidget,
        plot_temperature: PlotWidget,
        plot_error: PlotWidget,
    ) -> None:
        self.plot_voltage = plot_voltage
        self.plot_current = plot_current
        self.plot_temperature = plot_temperature
        self.plot_error = plot_error

        # Apply a consistent look to every plot.
        for plot in self._all_plots():
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.addLegend(offset=(10, 10))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(
        self,
        aligned: AlignedData,
        enabled_signals: Optional[Iterable[str]] = None,
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
        """
        if enabled_signals is None:
            enabled: Set[str] = set(aligned.signal_names)
        else:
            enabled = set(enabled_signals)
            # SoC/SoH have no checkbox — always include them on the error
            # plot when present in the data.
            enabled |= {"soc", "soh"} & set(aligned.signal_names)

        self._plot_overlay(self.plot_voltage, "voltage", aligned, enabled)
        self._plot_overlay(self.plot_current, "current", aligned, enabled)
        self._plot_overlay(self.plot_temperature, "temperature",
                           aligned, enabled)
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
    ) -> None:
        """Plot twin vs ECU for one signal on its dedicated PlotWidget."""
        plot.clear()
        self._clear_legend(plot)

        # If the signal isn't in this run (e.g. 3-signal ECU CSV) or the
        # user has unticked its checkbox, leave the plot empty.
        if signal_name not in aligned.signals or signal_name not in enabled:
            plot.setLabel("left", signal_name.title(), units="")
            return

        sig: AlignedSignal = aligned.signals[signal_name]
        ts = aligned.timestamps

        # Mask out NaN pairs so the curves don't draw spurious vertical jumps.
        valid = (~np.isnan(sig.twin_values)) & (~np.isnan(sig.ecu_values))
        t_win = ts[valid]
        twin_v = sig.twin_values[valid]
        ecu_v = sig.ecu_values[valid]

        plot.plot(t_win, twin_v, pen=pg.mkPen(_TWIN_COLOUR, width=1.0),
                  name="Digital Twin")
        plot.plot(t_win, ecu_v, pen=pg.mkPen(_ECU_COLOUR, width=2.0),
                  name="ECU")

        plot.setLabel("left", signal_name.title())
        plot.setLabel("bottom", "Time", units="s")

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
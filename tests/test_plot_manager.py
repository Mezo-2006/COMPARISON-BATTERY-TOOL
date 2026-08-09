"""Tests for ``plot_manager.py`` — the five overlay plots + one error plot.

These tests need a ``QApplication`` so the promoted ``PlotWidget``
instances from ``f.py`` can be constructed. They count ``listDataItems``
to verify that the right number of curves were drawn (and no others).
"""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets

from alignment_engine import ALIGN_NEAREST, align
from f import Ui_MainWindow
from plot_manager import PlotManager


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    yield app


@pytest.fixture
def main_window(qapp):
    win = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(win)
    yield ui
    win.close()


def _curves(plot_widget):
    return len(plot_widget.getPlotItem().listDataItems())


def _make_pm(main_window, **kwargs):
    return PlotManager(
        main_window.plotWidgetVoltage,
        main_window.plotWidgetCurrent,
        main_window.plotWidgetTemperature,
        main_window.plotWidgetSoc,
        main_window.plotWidgetSoh,
        main_window.plotWidgetError,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Overlay plots are unconditional -- always drawn regardless of the error
# graph's selection.
# ---------------------------------------------------------------------------
def test_update_draws_all_five_overlay_plots(main_window, twin_result_five, ecu_result_five):
    aligned = align(twin_result_five, ecu_result_five, ALIGN_NEAREST)
    pm = _make_pm(main_window)
    pm.update(aligned)

    for plot in (main_window.plotWidgetVoltage, main_window.plotWidgetCurrent,
                 main_window.plotWidgetTemperature, main_window.plotWidgetSoc,
                 main_window.plotWidgetSoh):
        assert _curves(plot) == 2  # twin + ECU
    assert _curves(main_window.plotWidgetError) == 5  # all five signals


def test_overlay_plots_ignore_error_graph_selection(main_window, twin_result_five, ecu_result_five):
    """Unticking a signal on the error-graph row must not touch its overlay plot."""
    aligned = align(twin_result_five, ecu_result_five, ALIGN_NEAREST)
    pm = _make_pm(main_window)
    pm.update(aligned, enabled_signals=set())  # nothing ticked for the error graph

    for plot in (main_window.plotWidgetVoltage, main_window.plotWidgetCurrent,
                 main_window.plotWidgetTemperature, main_window.plotWidgetSoc,
                 main_window.plotWidgetSoh):
        assert _curves(plot) == 2  # still drawn -- overlay plots are unconditional
    assert _curves(main_window.plotWidgetError) == 0  # error graph respects the selection


# ---------------------------------------------------------------------------
# Error graph selection
# ---------------------------------------------------------------------------
def test_enabled_subset_only_voltage_on_error_graph(main_window, twin_result_five, ecu_result_five):
    aligned = align(twin_result_five, ecu_result_five, ALIGN_NEAREST)
    pm = _make_pm(main_window)
    pm.update(aligned, enabled_signals={"voltage"})

    assert _curves(main_window.plotWidgetError) == 1
    assert _curves(main_window.plotWidgetVoltage) == 2


# ---------------------------------------------------------------------------
# clear_all empties every plot
# ---------------------------------------------------------------------------
def test_clear_all_empties_every_plot(main_window, twin_result_five, ecu_result_five):
    aligned = align(twin_result_five, ecu_result_five, ALIGN_NEAREST)
    pm = _make_pm(main_window)
    pm.update(aligned)
    pm.clear_all()

    for plot in (main_window.plotWidgetVoltage, main_window.plotWidgetCurrent,
                 main_window.plotWidgetTemperature, main_window.plotWidgetSoc,
                 main_window.plotWidgetSoh, main_window.plotWidgetError):
        assert _curves(plot) == 0


# ---------------------------------------------------------------------------
# Partial 3-signal ECU: only V/I/T have data; SoC/SoH overlay plots stay empty
# ---------------------------------------------------------------------------
def test_partial_three_signal_ecu(main_window, twin_result_five, ecu_result_three):
    aligned = align(twin_result_five, ecu_result_three, ALIGN_NEAREST)
    pm = _make_pm(main_window)
    pm.update(aligned)

    assert _curves(main_window.plotWidgetVoltage) == 2
    assert _curves(main_window.plotWidgetCurrent) == 2
    assert _curves(main_window.plotWidgetTemperature) == 2
    assert _curves(main_window.plotWidgetSoc) == 0
    assert _curves(main_window.plotWidgetSoh) == 0
    assert _curves(main_window.plotWidgetError) == 3  # V/I/T only


# ---------------------------------------------------------------------------
# Threshold warnings (on the signal's own overlay plot)
# ---------------------------------------------------------------------------
def test_threshold_exceeded_adds_flood_curve_and_warning_label(
    main_window, make_aligned_data,
):
    aligned = make_aligned_data(
        timestamps=[0.0, 0.1, 0.2, 0.3],
        signals={"voltage": ([3.30, 3.31, 3.40, 3.50],
                             [3.30, 3.31, 3.42, 3.55])},
    )
    label = QtWidgets.QLabel()
    pm = _make_pm(main_window, warning_labels={"voltage": label})
    pm.update(aligned, thresholds={"voltage": (True, 3.4)})

    # twin + ECU + the red "over limit" flood curve.
    assert _curves(main_window.plotWidgetVoltage) == 3
    assert "3.4" in label.text()
    assert "0.3" in label.text()  # worst breach is the last sample


def test_threshold_disabled_draws_no_flood_curve(main_window, make_aligned_data):
    aligned = make_aligned_data(
        timestamps=[0.0, 0.1],
        signals={"voltage": ([3.30, 3.31], [3.30, 3.55])},
    )
    label = QtWidgets.QLabel()
    pm = _make_pm(main_window, warning_labels={"voltage": label})
    pm.update(aligned, thresholds={"voltage": (False, 3.4)})

    assert _curves(main_window.plotWidgetVoltage) == 2  # no flood curve
    assert label.text() == ""


def test_threshold_never_exceeded_keeps_label_empty(main_window, make_aligned_data):
    aligned = make_aligned_data(
        timestamps=[0.0, 0.1],
        signals={"voltage": ([3.30, 3.31], [3.30, 3.31])},
    )
    label = QtWidgets.QLabel()
    pm = _make_pm(main_window, warning_labels={"voltage": label})
    pm.update(aligned, thresholds={"voltage": (True, 4.2)})

    assert _curves(main_window.plotWidgetVoltage) == 2  # no breach, no flood
    assert label.text() == ""


def test_threshold_unaffected_by_error_graph_selection(main_window, make_aligned_data):
    """Threshold highlighting lives on the overlay plot, so it doesn't
    care whether voltage is ticked for the error graph."""
    aligned = make_aligned_data(
        timestamps=[0.0, 0.1],
        signals={"voltage": ([3.30, 3.31], [3.30, 3.55])},
    )
    label = QtWidgets.QLabel()
    pm = _make_pm(main_window, warning_labels={"voltage": label})
    pm.update(aligned, enabled_signals=set(), thresholds={"voltage": (True, 3.4)})

    assert label.text() != ""
    assert _curves(main_window.plotWidgetVoltage) == 3
    assert _curves(main_window.plotWidgetError) == 0


# ---------------------------------------------------------------------------
# Live sliding window
# ---------------------------------------------------------------------------
def test_live_window_pins_x_range_to_latest_samples(main_window, make_aligned_data):
    aligned = make_aligned_data(
        timestamps=[0.0, 10.0, 100.0],
        signals={"voltage": ([3.3, 3.3, 3.3], [3.3, 3.3, 3.3])},
    )
    pm = _make_pm(main_window)
    pm.update(aligned, live_window_ms=30.0)

    (x_min, x_max), _ = main_window.plotWidgetVoltage.getPlotItem().viewRange()
    assert x_max == pytest.approx(100.0)
    assert x_min == pytest.approx(70.0)
    # Applied to the error plot too, not just the overlay plots.
    (ex_min, ex_max), _ = main_window.plotWidgetError.getPlotItem().viewRange()
    assert ex_max == pytest.approx(100.0)
    assert ex_min == pytest.approx(70.0)


def test_no_live_window_leaves_default_autorange(main_window, make_aligned_data):
    """Offline calls (live_window_ms=None) shouldn't pin a live sliding window."""
    aligned = make_aligned_data(
        timestamps=[0.0, 10.0, 100.0],
        signals={"voltage": ([3.3, 3.3, 3.3], [3.3, 3.3, 3.3])},
    )
    pm = _make_pm(main_window)
    pm.update(aligned)  # live_window_ms defaults to None

    (x_min, x_max), _ = main_window.plotWidgetVoltage.getPlotItem().viewRange()
    # Not pinned to a 30s-wide slice at the end of the series -- whatever
    # pyqtgraph's own auto-range picked is fine, we just didn't override it.
    assert not (x_min == pytest.approx(70.0) and x_max == pytest.approx(100.0))

"""Tests for ``plot_manager.py`` — the four pyqtgraph PlotWidgets.

These tests need a ``QApplication`` so the promoted ``PlotWidget``
instances from ``f.py`` can be constructed.  They count ``listDataItems``
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


# ---------------------------------------------------------------------------
# Update with all defaults
# ---------------------------------------------------------------------------
def test_update_all_signals_draws_two_curves_per_overlay(main_window,
                                                            twin_result_five,
                                                            ecu_result_five):
    aligned = align(twin_result_five, ecu_result_five, ALIGN_NEAREST)
    pm = PlotManager(
        main_window.plotWidgetVoltage,
        main_window.plotWidgetCurrent,
        main_window.plotWidgetTemperature,
        main_window.plotWidgetError,
    )
    pm.update(aligned)

    assert _curves(main_window.plotWidgetVoltage) == 2      # twin + ECU
    assert _curves(main_window.plotWidgetCurrent) == 2
    assert _curves(main_window.plotWidgetTemperature) == 2
    # Error plot: 5 signals (V/I/T/SoC/SoH — SoC/SoH always on error).
    assert _curves(main_window.plotWidgetError) == 5


# ---------------------------------------------------------------------------
# Checkbox subset: only voltage is enabled
# ---------------------------------------------------------------------------
def test_enabled_subset_only_voltage_keeps_soc_soh_on_error(main_window,
                                                              twin_result_five,
                                                              ecu_result_five):
    aligned = align(twin_result_five, ecu_result_five, ALIGN_NEAREST)
    pm = PlotManager(
        main_window.plotWidgetVoltage,
        main_window.plotWidgetCurrent,
        main_window.plotWidgetTemperature,
        main_window.plotWidgetError,
    )
    pm.update(aligned, enabled_signals={"voltage"})

    assert _curves(main_window.plotWidgetVoltage) == 2  # voltage ticked
    assert _curves(main_window.plotWidgetCurrent) == 0  # unticked
    assert _curves(main_window.plotWidgetTemperature) == 0
    # Error plot: V (ticked) + SoC + SoH (always on) = 3 curves.
    assert _curves(main_window.plotWidgetError) == 3


# ---------------------------------------------------------------------------
# clear_all empties every plot
# ---------------------------------------------------------------------------
def test_clear_all_empties_every_plot(main_window,
                                        twin_result_five,
                                        ecu_result_five):
    aligned = align(twin_result_five, ecu_result_five, ALIGN_NEAREST)
    pm = PlotManager(
        main_window.plotWidgetVoltage,
        main_window.plotWidgetCurrent,
        main_window.plotWidgetTemperature,
        main_window.plotWidgetError,
    )
    pm.update(aligned)
    pm.clear_all()

    assert _curves(main_window.plotWidgetVoltage) == 0
    assert _curves(main_window.plotWidgetCurrent) == 0
    assert _curves(main_window.plotWidgetTemperature) == 0
    assert _curves(main_window.plotWidgetError) == 0


# ---------------------------------------------------------------------------
# Partial 3-signal ECU: error plot gets only V/I/T
# ---------------------------------------------------------------------------
def test_partial_three_signal_ecu_only_vit_on_error(main_window,
                                                       twin_result_five,
                                                       ecu_result_three):
    aligned = align(twin_result_five, ecu_result_three, ALIGN_NEAREST)
    pm = PlotManager(
        main_window.plotWidgetVoltage,
        main_window.plotWidgetCurrent,
        main_window.plotWidgetTemperature,
        main_window.plotWidgetError,
    )
    pm.update(aligned)
    assert _curves(main_window.plotWidgetError) == 3  # V/I/T only


# ---------------------------------------------------------------------------
# Threshold warnings
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
    pm = PlotManager(
        main_window.plotWidgetVoltage,
        main_window.plotWidgetCurrent,
        main_window.plotWidgetTemperature,
        main_window.plotWidgetError,
        warning_labels={"voltage": label},
    )
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
    pm = PlotManager(
        main_window.plotWidgetVoltage,
        main_window.plotWidgetCurrent,
        main_window.plotWidgetTemperature,
        main_window.plotWidgetError,
        warning_labels={"voltage": label},
    )
    pm.update(aligned, thresholds={"voltage": (False, 3.4)})

    assert _curves(main_window.plotWidgetVoltage) == 2  # no flood curve
    assert label.text() == ""


def test_threshold_never_exceeded_keeps_label_empty(main_window, make_aligned_data):
    aligned = make_aligned_data(
        timestamps=[0.0, 0.1],
        signals={"voltage": ([3.30, 3.31], [3.30, 3.31])},
    )
    label = QtWidgets.QLabel()
    pm = PlotManager(
        main_window.plotWidgetVoltage,
        main_window.plotWidgetCurrent,
        main_window.plotWidgetTemperature,
        main_window.plotWidgetError,
        warning_labels={"voltage": label},
    )
    pm.update(aligned, thresholds={"voltage": (True, 4.2)})

    assert _curves(main_window.plotWidgetVoltage) == 2  # no breach, no flood
    assert label.text() == ""
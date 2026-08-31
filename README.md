# BrightSkies Digital Twin Validation Tool

A PyQt5 desktop application that compares **Digital Twin** (Simulink) output
against **ECU** (NXP MPC5744P) output for a battery monitoring / digital-twin
system, across five signals: **voltage, current, temperature, SoC, SoH**.

## Features

- **Offline mode** — load two CSV files (Digital Twin + ECU), run
  alignment and statistics, then inspect data previews, per-signal stats,
  overlay/error graphs, and export results.
- **Live mode** — stream both sources over MQTT into the same comparison
  engine in real time.
- **Event detection** — stateful anomaly detection (spike, freeze,
  oscillation, jump, timeout, comm-loss, threshold, out-of-range) with an
  alert table and CSV/Excel/PDF export.
- **Export** — results and event logs can be exported to CSV, Excel, PDF,
  or HTML.

## Architecture

- Backend engines (`data_loader.py`, `alignment_engine.py`,
  `statistics_engine.py`, `export_engine.py`, `live_schema.py`,
  `live_accumulator.py`, `event_detector.py`, `event_export.py`) are pure
  Python — no Qt dependency — so they're unit-testable from the CLI.
- Qt-facing code lives in `back.py` (MainWindow), `comparison_worker.py`,
  `plot_manager.py` (pyqtgraph), `mqtt_worker.py`, `live_controller.py`,
  and `event_window.py`.
- `f.ui` is the Qt Designer source of truth for the UI; `f.py` is
  generated from it via `pyuic5` and should never be hand-edited.

See [context.md](context.md) for the full architecture, module inventory,
and UI conventions.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Running

```bash
python back.py
```

## Testing

```bash
pytest
```

## Requirements

- Python 3.x
- PyQt5, pyqtgraph, pandas, numpy, paho-mqtt, reportlab, openpyxl

See [requirements.txt](requirements.txt) for the full pinned dependency list.

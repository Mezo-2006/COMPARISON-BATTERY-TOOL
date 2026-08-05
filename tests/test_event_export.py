"""Tests for ``event_export.py`` — CSV/Excel/PDF export of the event log."""

from __future__ import annotations

import pandas as pd
import pytest

from event_detector import Alert, EventLog
from event_export import (
    EventExportError,
    FORMAT_CSV,
    FORMAT_EXCEL,
    FORMAT_PDF,
    export_event_log,
)


@pytest.fixture
def sample_log() -> EventLog:
    log = EventLog()
    log.add(Alert(
        timestamp=15.25, signal="voltage", event="Spike", severity="High",
        real_value=4.18, digital_twin_value=3.70, difference=0.48,
        threshold=0.10, source="real",
    ))
    log.add(Alert(
        timestamp=15.30, signal="soc", event="Unexpected Jump", severity="High",
        real_value=45.0, digital_twin_value=80.0, difference=-35.0,
        threshold=15.0, source="real",
    ))
    return log


def test_csv_export_writes_task15_columns(tmp_path, sample_log):
    out = tmp_path / "events.csv"
    result = export_event_log(sample_log, FORMAT_CSV, str(out))
    assert result.success
    assert result.n_rows == 2
    assert out.exists()

    df = pd.read_csv(out)
    assert list(df.columns) == [
        "Signal", "Real Value", "Digital Twin Value", "Difference",
        "Threshold", "Event Type", "Severity", "Timestamp",
    ]
    assert len(df) == 2
    assert df.iloc[0]["Event Type"] == "Spike"


def test_excel_export_round_trips(tmp_path, sample_log):
    out = tmp_path / "events.xlsx"
    result = export_event_log(sample_log, FORMAT_EXCEL, str(out))
    assert result.success
    assert out.exists()
    df = pd.read_excel(out)
    assert len(df) == 2


def test_pdf_export_writes_file(tmp_path, sample_log):
    out = tmp_path / "events.pdf"
    result = export_event_log(sample_log, FORMAT_PDF, str(out))
    assert result.success
    assert out.exists()
    assert out.stat().st_size > 0


def test_unknown_format_raises():
    log = EventLog()
    with pytest.raises(EventExportError):
        export_event_log(log, "yaml", "out.yaml")


def test_export_empty_log_produces_header_only(tmp_path):
    out = tmp_path / "empty.csv"
    result = export_event_log(EventLog(), FORMAT_CSV, str(out))
    assert result.success
    assert result.n_rows == 0
    df = pd.read_csv(out)
    assert len(df) == 0


def test_log_line_is_timestamped(tmp_path, sample_log):
    out = tmp_path / "events.csv"
    result = export_event_log(sample_log, FORMAT_CSV, str(out))
    assert result.message.startswith("[")
    assert "events.csv" in result.message

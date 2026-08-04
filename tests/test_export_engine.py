"""Tests for ``export_engine.py`` — CSV/Excel/PDF/HTML writers."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alignment_engine import ALIGN_NEAREST, align
from statistics_engine import compute
from export_engine import (
    ExportError,
    export_data,
    export_report,
    FORMAT_DATA_CSV,
    FORMAT_DATA_EXCEL,
    FORMAT_REPORT_PDF,
    FORMAT_REPORT_HTML,
)


# ---------------------------------------------------------------------------
# Fixtures: build aligned + stats once per test module
# ---------------------------------------------------------------------------
@pytest.fixture
def aligned_and_stats(twin_result_five, ecu_result_five):
    aligned = align(twin_result_five, ecu_result_five, ALIGN_NEAREST)
    stats = compute(aligned)
    return aligned, stats


# ---------------------------------------------------------------------------
# CSV data export
# ---------------------------------------------------------------------------
def test_csv_export_writes_all_signal_columns(tmp_path, aligned_and_stats):
    aligned, _ = aligned_and_stats
    out = tmp_path / "data.csv"
    result = export_data(aligned, FORMAT_DATA_CSV, str(out))

    assert result.success
    assert result.n_rows == aligned.n_total
    assert out.exists()

    loaded = pd.read_csv(out)
    assert "timestamp" in loaded.columns
    assert "voltage_twin" in loaded.columns
    assert "voltage_ecu" in loaded.columns
    assert "voltage_error" in loaded.columns
    assert "soh_twin" in loaded.columns  # 5-signal support
    assert len(loaded) == aligned.n_total


def test_csv_export_log_line_is_timestamped(tmp_path, aligned_and_stats):
    aligned, _ = aligned_and_stats
    out = tmp_path / "data.csv"
    result = export_data(aligned, FORMAT_DATA_CSV, str(out))
    assert result.message.startswith("[")
    assert "Exported table (csv)" in result.message
    assert str(out) in result.message


# ---------------------------------------------------------------------------
# Excel data export
# ---------------------------------------------------------------------------
def test_excel_export_roundtrip(tmp_path, aligned_and_stats):
    aligned, _ = aligned_and_stats
    out = tmp_path / "data.xlsx"
    result = export_data(aligned, FORMAT_DATA_EXCEL, str(out))

    assert result.success
    assert out.exists() and out.stat().st_size > 0

    loaded = pd.read_excel(out, engine="openpyxl")
    assert len(loaded) == aligned.n_total
    assert "voltage_twin" in loaded.columns


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------
def test_html_report_contains_all_sections(tmp_path, aligned_and_stats):
    aligned, stats = aligned_and_stats
    out = tmp_path / "report.html"
    result = export_report(aligned, stats, FORMAT_REPORT_HTML, str(out),
                            title="Test Report")
    assert result.success
    content = out.read_text()
    assert "<h1>Test Report</h1>" in content
    assert "Summary" in content
    assert "Per-Signal Breakdown" in content
    assert "Worst Mismatches" in content


def test_html_report_contains_all_five_signals(tmp_path, aligned_and_stats):
    aligned, stats = aligned_and_stats
    out = tmp_path / "report.html"
    export_report(aligned, stats, FORMAT_REPORT_HTML, str(out))
    content = out.read_text()
    for sig in ("voltage", "current", "temperature", "soc", "soh"):
        assert sig in content


def test_html_report_records_alignment_method(tmp_path, aligned_and_stats):
    aligned, stats = aligned_and_stats
    out = tmp_path / "report.html"
    export_report(aligned, stats, FORMAT_REPORT_HTML, str(out))
    content = out.read_text()
    assert f"Alignment method: <code>{aligned.alignment_method}</code>" in content


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------
def test_pdf_report_has_valid_pdf_magic(tmp_path, aligned_and_stats):
    aligned, stats = aligned_and_stats
    out = tmp_path / "report.pdf"
    result = export_report(aligned, stats, FORMAT_REPORT_PDF, str(out))
    assert result.success
    assert out.read_bytes()[:5] == b"%PDF-"


def test_pdf_report_size_nonzero(tmp_path, aligned_and_stats):
    aligned, stats = aligned_and_stats
    out = tmp_path / "report.pdf"
    export_report(aligned, stats, FORMAT_REPORT_PDF, str(out))
    assert out.stat().st_size > 500  # reportlab shell + 3 tables ~2-5 KB


# ---------------------------------------------------------------------------
# Format-constant errors
# ---------------------------------------------------------------------------
def test_unknown_data_format_raises(tmp_path, aligned_and_stats):
    aligned, _ = aligned_and_stats
    with pytest.raises(ExportError, match="Unknown data format"):
        export_data(aligned, "json", str(tmp_path / "x.json"))


def test_unknown_report_format_raises(tmp_path, aligned_and_stats):
    aligned, stats = aligned_and_stats
    with pytest.raises(ExportError, match="Unknown report format"):
        export_report(aligned, stats, "docx", str(tmp_path / "x.docx"))


# ---------------------------------------------------------------------------
# Partial overlap: 3-signal ECU
# ---------------------------------------------------------------------------
def test_partial_html_report_omits_soc_soh(tmp_path,
                                            twin_result_five,
                                            ecu_result_three):
    aligned = align(twin_result_five, ecu_result_three, ALIGN_NEAREST)
    stats = compute(aligned)
    out = tmp_path / "partial.html"
    export_report(aligned, stats, FORMAT_REPORT_HTML, str(out))
    content = out.read_text()
    for sig in ("voltage", "current", "temperature"):
        assert sig in content
    assert "soc" not in content
    assert "soh" not in content


def test_partial_csv_export_has_only_common_signal_columns(tmp_path,
                                                             twin_result_five,
                                                             ecu_result_three):
    aligned = align(twin_result_five, ecu_result_three, ALIGN_NEAREST)
    out = tmp_path / "partial.csv"
    export_data(aligned, FORMAT_DATA_CSV, str(out))
    loaded = pd.read_csv(out)
    assert "voltage_twin" in loaded.columns
    assert "soh_twin" not in loaded.columns  # SoH not in this run


# ---------------------------------------------------------------------------
# Worst-N respected in report content
# ---------------------------------------------------------------------------
def test_html_report_respects_worst_n(tmp_path,
                                       twin_result_five,
                                       ecu_result_five):
    aligned = align(twin_result_five, ecu_result_five, ALIGN_NEAREST)
    stats = compute(aligned, worst_n=2)
    out = tmp_path / "worst2.html"
    export_report(aligned, stats, FORMAT_REPORT_HTML, str(out))
    content = out.read_text()
    assert "top 2" in content
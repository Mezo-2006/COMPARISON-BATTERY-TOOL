"""Tests for ``data_loader.py`` — CSV loader + alias resolver."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_loader import LoadResult, DataLoaderError, load_csv


# ---------------------------------------------------------------------------
# Happy path: the on-disk fixtures
# ---------------------------------------------------------------------------
def test_load_twin_fixture(twin_csv_path):
    result = load_csv(twin_csv_path, "twin")
    assert result.source_label == "twin"
    assert result.row_count == 10
    assert "timestamp" in result.df.columns
    # All 5 canonical signals present.
    for signal in ("voltage", "current", "temperature", "soc", "soh"):
        assert signal in result.columns_found, f"missing {signal}"
        assert result.columns_missing == []


def test_load_ecu_fixture_with_aliases(ecu_csv_path):
    """ECU fixture uses ``time_s/v/i/temp_C`` headers — must resolve to canonicals."""
    result = load_csv(ecu_csv_path, "ecu")
    assert result.row_count == 10
    for signal in ("voltage", "current", "temperature", "soc", "soh"):
        assert signal in result.columns_found
    # Source headers preserved for the UI's preview info label.
    assert "V" in result.source_columns
    assert "time_s" in result.source_columns


# ---------------------------------------------------------------------------
# Alias resolution: case / whitespace insensitivity
# ---------------------------------------------------------------------------
def test_alias_case_insensitive(tmp_path):
    csv = tmp_path / "Case.csv"
    csv.write_text("Timestamp,Voltage,Current,Temperature,SOC,SOH\n"
                   "0.0,3.3,1.2,25.0,80.0,95.0\n")
    result = load_csv(str(csv), "twin")
    assert set(["voltage", "current", "temperature", "soc", "soh"]).issubset(
        result.columns_found
    )


def test_alias_whitespace_in_header(tmp_path):
    csv = tmp_path / "ws.csv"
    csv.write_text(" time , v , i , temp , soc , soh \n"
                   "0.0,3.3,1.2,25.0,80.0,95.0\n")
    result = load_csv(str(csv), "twin")
    assert "voltage" in result.columns_found
    assert "timestamp" in result.df.columns


def test_alias_pack_voltage_and_vbat(tmp_path):
    csv = tmp_path / "pack.csv"
    csv.write_text("time,pack_voltage,ibat,cell_temp,state_of_charge,state_of_health\n"
                   "0.0,3.3,1.2,25.0,80.0,95.0\n")
    result = load_csv(str(csv), "twin")
    assert "voltage" in result.columns_found
    assert "current" in result.columns_found
    assert "temperature" in result.columns_found
    assert "soc" in result.columns_found
    assert "soh" in result.columns_found


# ---------------------------------------------------------------------------
# The T-ambiguity rule (see data_loader.py docstring)
# ---------------------------------------------------------------------------
def test_t_claimed_by_timestamp_when_v_present(tmp_path):
    """`t` → timestamp if a voltage column exists (twin layout)."""
    csv = tmp_path / "twin.csv"
    csv.write_text("t,V,I,Temperature,SOC,SOH\n"
                   "0.0,3.3,1.2,25.0,80.0,95.0\n")
    result = load_csv(str(csv), "twin")
    assert "timestamp" in result.df.columns
    assert "voltage" in result.columns_found
    # ``T`` (capitalised) is the temperature column here, not the lowercase ``t``.
    assert "temperature" in result.columns_found


def test_t_treated_as_temperature_when_no_other_timestamp(tmp_path):
    """`t` → temperature when there is no voltage column AND no other
    timestamp candidate matched in the alias loop.

    Per the docstring's T-ambiguity rule, a CSV with only ``t`` as a
    time-like column and no voltage is interpreted as ``t`` = temperature,
    which then leaves the CSV without a timestamp and raises
    :class:`DataLoaderError`.  This test verifies that the special rule
    fires (no crash inside the resolver) and the missing-timestamp fatal
    path surfaces cleanly.
    """
    csv = tmp_path / "temp_only.csv"
    csv.write_text("t,current,soc,soh\n"
                   "25.0,1.2,80.0,95.0\n")
    with pytest.raises(DataLoaderError):
        load_csv(str(csv), "twin")


# ---------------------------------------------------------------------------
# Convenience properties
# ---------------------------------------------------------------------------
def test_has_signal_properties(tmp_path):
    csv = tmp_path / "x.csv"
    csv.write_text("time,V,I,temp,soc,soh\n0.0,3.3,1.2,25.0,80.0,95.0\n")
    r = load_csv(str(csv), "twin")
    assert r.has_timestamp and r.has_voltage and r.has_current
    assert r.has_temperature and r.has_soc and r.has_soh


def test_missing_signal_reported_not_fatal(tmp_path):
    csv = tmp_path / "partial.csv"
    csv.write_text("timestamp,voltage,current,temperature\n"
                   "0.0,3.3,1.2,25.0\n0.1,3.4,1.3,26.0\n")
    r = load_csv(str(csv), "twin")
    assert "soc" in r.columns_missing and "soh" in r.columns_missing
    assert not r.has_soc and not r.has_soh
    # Loading does not raise — partial CSVs are useful.
    assert r.row_count == 2


# ---------------------------------------------------------------------------
# Timestamp normalisation: numeric vs ISO datetime strings
# ---------------------------------------------------------------------------
def test_iso_datetime_timestamps(tmp_path):
    csv = tmp_path / "iso.csv"
    csv.write_text("timestamp,voltage,current,temperature,soc,soh\n"
                   "2024-01-01 00:00:00,3.3,1.2,25.0,80.0,95.0\n"
                   "2024-01-01 00:00:01,3.4,1.3,26.0,79.0,95.0\n")
    r = load_csv(str(csv), "twin")
    # Converted to POSIX seconds; difference between rows is ~1.0 s.
    ts = r.df["timestamp"].to_numpy()
    assert ts[1] - ts[0] == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Data cleaning: NaN / duplicates / all-NaN rows
# ---------------------------------------------------------------------------
def test_non_numeric_value_coerced_to_nan_with_warning(tmp_path):
    csv = tmp_path / "nan.csv"
    csv.write_text("timestamp,voltage,current,temperature,soc,soh\n"
                   "0.0,3.3,1.2,25.0,80.0,95.0\n"
                   "0.1, oops ,1.3,26.0,79.0,95.0\n")
    r = load_csv(str(csv), "twin")
    assert r.row_count == 2  # row kept, voltage coerced to NaN
    assert any("non-numeric" in w.lower() or "nan" in w.lower()
               for w in r.warnings)
    assert np.isnan(r.df.loc[1, "voltage"])


def test_duplicate_timestamps_keep_first_with_warning(tmp_path):
    csv = tmp_path / "dup.csv"
    csv.write_text("timestamp,voltage,current,temperature,soc,soh\n"
                   "0.0,3.3,1.2,25.0,80.0,95.0\n"
                   "0.0,9.9,9.9,9.9,9.9,9.9\n"   # duplicate timestamp
                   "0.1,3.4,1.3,26.0,79.0,95.0\n")
    r = load_csv(str(csv), "twin")
    assert r.row_count == 2
    assert any("duplicate" in w.lower() for w in r.warnings)
    # First occurrence wins — 3.3, not 9.9.
    assert r.df.loc[0, "voltage"] == pytest.approx(3.3)


def test_all_nan_row_dropped(tmp_path):
    csv = tmp_path / "allnan.csv"
    csv.write_text("timestamp,voltage,current,temperature,soc,soh\n"
                   "0.0,3.3,1.2,25.0,80.0,95.0\n"
                   "0.1,,,,,\n"
                   "0.2,3.4,1.3,26.0,79.0,95.0\n")
    r = load_csv(str(csv), "twin")
    assert r.row_count == 2  # all-NaN middle row dropped


def test_rows_sorted_by_timestamp(tmp_path):
    csv = tmp_path / "unsorted.csv"
    csv.write_text("timestamp,voltage,current,temperature,soc,soh\n"
                   "0.2,3.4,1.3,26.0,79.0,95.0\n"
                   "0.0,3.3,1.2,25.0,80.0,95.0\n"
                   "0.1,3.31,1.21,25.1,79.9,95.0\n")
    r = load_csv(str(csv), "twin")
    ts = r.df["timestamp"].to_numpy()
    assert np.all(np.diff(ts) > 0), "timestamps should be sorted ascending"


# ---------------------------------------------------------------------------
# axis_kind: "id" header -> sequence, everything else -> timestamp
# ---------------------------------------------------------------------------
def test_id_header_sets_sequence_axis_kind(tmp_path):
    csv = tmp_path / "seq.csv"
    csv.write_text("id,voltage,current,temperature,soc,soh\n"
                   "0,3.3,1.2,25.0,80.0,95.0\n"
                   "1,3.31,1.21,25.1,79.9,95.0\n")
    r = load_csv(str(csv), "twin")
    assert r.axis_kind == "sequence"
    assert "timestamp" in r.df.columns  # still stored internally as "timestamp"
    assert list(r.df["timestamp"]) == [0.0, 1.0]


def test_timestamp_header_sets_timestamp_axis_kind(tmp_path):
    csv = tmp_path / "ts.csv"
    csv.write_text("timestamp,voltage,current,temperature,soc,soh\n"
                   "0.0,3.3,1.2,25.0,80.0,95.0\n")
    r = load_csv(str(csv), "twin")
    assert r.axis_kind == "timestamp"


def test_t_claimed_as_timestamp_is_not_sequence(tmp_path):
    """`t` (claimed as timestamp via the T-ambiguity rule) is real time,
    not a sequence counter -- only a header literally named `id` is."""
    csv = tmp_path / "t.csv"
    csv.write_text("t,V,I,Temperature,SOC,SOH\n"
                   "0.0,3.3,1.2,25.0,80.0,95.0\n")
    r = load_csv(str(csv), "twin")
    assert r.axis_kind == "timestamp"


def test_default_load_result_axis_kind_is_timestamp(make_load_result):
    r = make_load_result("twin", {"timestamp": [0.0], "voltage": [3.3]})
    assert r.axis_kind == "timestamp"


# ---------------------------------------------------------------------------
# Fatal errors
# ---------------------------------------------------------------------------
def test_file_not_found_raises():
    with pytest.raises(DataLoaderError):
        load_csv("/no/such/file.csv", "twin")


def test_empty_file_raises(tmp_path):
    csv = tmp_path / "empty.csv"
    csv.write_text("")
    with pytest.raises(DataLoaderError):
        load_csv(str(csv), "twin")


def test_header_only_raises(tmp_path):
    csv = tmp_path / "header.csv"
    csv.write_text("timestamp,voltage,current,temperature,soc,soh\n")
    with pytest.raises(DataLoaderError):
        load_csv(str(csv), "twin")


def test_no_timestamp_column_raises(tmp_path):
    csv = tmp_path / "nots.csv"
    csv.write_text("voltage,current,temperature,soc,soh\n"
                   "3.3,1.2,25.0,80.0,95.0\n")
    with pytest.raises(DataLoaderError):
        load_csv(str(csv), "twin")


def test_no_rows_after_cleaning_raises(tmp_path):
    csv = tmp_path / "alljunk.csv"
    csv.write_text("timestamp,voltage,current,temperature,soc,soh\n"
                   "0.0,3.3,1.2,25.0,80.0,95.0\n"
                   "0.1,,,,,\n")
    # Edit: make ALL rows all-NaN so none survive after cleaning.
    csv.write_text("timestamp,voltage,current,temperature,soc,soh\n"
                   "0.0,,,,,\n"
                   "0.1,,,,,\n")
    with pytest.raises(DataLoaderError):
        load_csv(str(csv), "twin")
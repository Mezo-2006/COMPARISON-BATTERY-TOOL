"""Data loader for the Digital Twin Validation Tool.

Reads a CSV file produced either by the Digital Twin (Simulink model) or by
the ECU (NXP MPC5744P), normalises the column names, validates the
structure, sorts by timestamp, and returns a ``LoadResult`` containing the
clean DataFrame plus metadata the UI needs (row count, column names, time
range, warnings).

This module is **pure Python** — no Qt imports — so it can be unit-tested
in isolation and reused by the live (MQTT) path later, where incoming
samples are accumulated into a DataFrame and fed through the same
``LoadResult`` contract.

CSV schema
----------
Both CSV files are expected to have a timestamp column and at least one
signal column (voltage, current, temperature, SoC, SoH).  Column names are
matched case-insensitively against a set of aliases so that the twin's CSV
(``Time, Voltage, Current, Temperature, SoC, SoH``) and the ECU's CSV
(``timestamp_s, v, i, temp_C, soc, soh``) are both recognised::

    Alias group   Accepted header (case-insensitive, whitespace-trimmed)
    -----------   -------------------------------------------------------
    timestamp     timestamp, time, t, ts, time_s, timestamp_s, id
    voltage       voltage, v, vbat, pack_voltage, voltage_v, batt_voltage
    current       current, i, ibat, pack_current, current_a, batt_current
    temperature   temperature, temp, t (only if voltage group didn't claim it),
                  temp_c, temperature_c, cell_temp
    soc           soc, state_of_charge, charge, charge_percent, soc_percent,
                  soc_pct
    soh           soh, state_of_health, health, health_percent, soh_percent,
                  soh_pct

Ambiguity rule: the single-letter ``t`` is claimed by **timestamp** if a
``v`` column also exists (the common twin layout); otherwise it is treated
as temperature.  This avoids a hard crash on the rare CSV that uses ``t``
for temperature without a separate timestamp column.


Timestamp normalisation
-----------------------
- Numeric values are treated as seconds (relative or epoch — the alignment
  engine only cares about *relative* differences, not the absolute epoch).
- ISO-8601 datetime strings (e.g. ``2025-07-15T10:30:00Z``) are parsed
  with ``pd.to_datetime`` and converted to POSIX seconds.
- Integer values are kept as-is (seconds).

After normalisation the ``timestamp`` column is float64 seconds and the
DataFrame is sorted ascending by it.

Sequence-id axis
-----------------
A header literally named ``id`` is accepted as the timestamp column (see
the alias table above) but means something different: it's a plain
integer sample counter, not a time value (this is what
``synthetic_data_generator``'s ``id_mode="sequence"`` emits). The loader
still stores these values in the same ``timestamp`` column internally
(alignment/statistics only need a sortable, interpolatable numeric axis
and don't care what it represents) but records which kind it was on
``LoadResult.axis_kind`` (``"timestamp"`` or ``"sequence"``) so the
Graphs tab can label the X axis correctly either way.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Column-alias lookup
# ---------------------------------------------------------------------------
# Maps canonical name → ordered list of accepted header names (lower-cased,
# whitespace-stripped).  The first match wins, so put more-specific aliases
# before less-specific ones.
_ALIAS_MAP: dict[str, list[str]] = {
    "timestamp":   ["timestamp", "time", "ts", "time_s", "timestamp_s", "id"],
    "voltage":    ["voltage", "v", "vbat", "pack_voltage", "voltage_v",
                   "batt_voltage"],
    "current":    ["current", "i", "ibat", "pack_current", "current_a",
                   "batt_current"],
    "temperature": ["temperature", "temp", "temp_c", "temperature_c",
                    "cell_temp"],
    "soc":        ["soc", "state_of_charge", "charge", "charge_percent",
                   "soc_percent", "soc_pct"],
    "soh":        ["soh", "state_of_health", "health", "health_percent",
                   "soh_percent", "soh_pct"],
}

# Special rule for single-letter ``t``: claim for timestamp if ``v`` exists
# too (the common twin layout), otherwise treat as temperature.  Handled
# in ``_resolve_columns``.

# Canonical column ordering, reused by ``build_load_result`` and the live
# accumulator so column lists stay consistent across the codebase.
_CANONICAL_COLUMNS: list[str] = [
    "timestamp", "voltage", "current", "temperature", "soc", "soh",
]
# Canonical signal columns (everything except timestamp) — used to compute
# ``columns_missing`` for a ``LoadResult``.
_CANONICAL_SIGNALS: list[str] = [
    "voltage", "current", "temperature", "soc", "soh",
]


# ---------------------------------------------------------------------------
# Result / error types
# ---------------------------------------------------------------------------
@dataclass
class LoadResult:
    """Everything the rest of the pipeline needs from a loaded CSV."""

    df: pd.DataFrame
    source_label: str                          # "twin" or "ecu"
    row_count: int
    columns_found: List[str]                  # canonical names present
    columns_missing: List[str]                # canonical names absent
    source_columns: List[str]                 # original CSV header names
    time_range: Tuple[float, float]           # (t_min, t_max) in seconds
    warnings: List[str] = field(default_factory=list)
    # What the "timestamp" column actually *means*: "timestamp" for a real
    # time axis (seconds or Unix ms), "sequence" when the source column was
    # literally named "id" (a plain sample counter, no time semantics).
    # Determined once, from which alias matched during column resolution --
    # see ``_resolve_columns`` -- and carried through alignment so the
    # Graphs tab can label/scale the X axis correctly for either.
    axis_kind: str = "timestamp"

    @property
    def has_timestamp(self) -> bool:
        return "timestamp" in self.columns_found

    @property
    def has_voltage(self) -> bool:
        return "voltage" in self.columns_found

    @property
    def has_current(self) -> bool:
        return "current" in self.columns_found

    @property
    def has_temperature(self) -> bool:
        return "temperature" in self.columns_found

    @property
    def has_soc(self) -> bool:
        return "soc" in self.columns_found

    @property
    def has_soh(self) -> bool:
        return "soh" in self.columns_found


class DataLoaderError(Exception):
    """Fatal error during CSV loading (file missing, no timestamp, parse error)."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_csv(path: str, source_label: str) -> LoadResult:
    """Load and validate one CSV file.

    Parameters
    ----------
    path
        Filesystem path to the CSV file.
    source_label
        ``"twin"`` or ``"ecu"`` — baked into the ``LoadResult`` so
        downstream code can route without re-deriving the source.

    Returns
    -------
    LoadResult
        Clean DataFrame (columns = canonical names, sorted by timestamp)
        plus metadata.

    Raises
    ------
    DataLoaderError
        If the file cannot be read, has no timestamp column, or is empty.
    """
    # --- File existence ----------------------------------------------------
    if not os.path.isfile(path):
        raise DataLoaderError(f"File not found: {path}")

    # --- Read CSV ----------------------------------------------------------
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise DataLoaderError(f"CSV file is empty: {path}")
    except Exception as exc:
        raise DataLoaderError(f"Failed to parse CSV ({path}): {exc}")

    if df.empty:
        raise DataLoaderError(f"CSV has headers but no data rows: {path}")

    original_columns = list(df.columns)

    # --- Resolve column names ----------------------------------------------
    canonical_map, warnings = _resolve_columns(original_columns)

    if "timestamp" not in canonical_map.values():
        raise DataLoaderError(
            f"No timestamp column found in {path}.\n"
            f"Headers present: {original_columns}\n"
            f"Accepted aliases: {_ALIAS_MAP['timestamp']}"
        )

    # Which original header won the "timestamp" slot? If it was literally
    # "id", this run's primary axis is a sequence counter, not real time
    # (see synthetic_data_generator's id_mode) -- record that now, before
    # the rename below erases the original header name.
    ts_original = next(
        orig for orig, canon in canonical_map.items() if canon == "timestamp"
    )
    axis_kind = "sequence" if ts_original.strip().lower() == "id" else "timestamp"

    # Rename to canonical names, keep only recognised columns
    rename_dict = {orig: canon for orig, canon in canonical_map.items()}
    df = df.rename(columns=rename_dict)

    # Keep only the canonical columns we recognised + nothing else
    canonical_present = [c for c in
                         ["timestamp", "voltage", "current", "temperature",
                          "soc", "soh"]
                         if c in df.columns]
    df = df[canonical_present].copy()

    # --- Normalise timestamp to float seconds -----------------------------
    df = _normalise_timestamp(df, warnings)

    # --- Coerce signal columns to float ------------------------------------
    for col in ["voltage", "current", "temperature", "soc", "soh"]:
        if col in df.columns:
            n_nan_before = int(df[col].isna().sum())
            df[col] = pd.to_numeric(df[col], errors="coerce")
            n_nan_after = int(df[col].isna().sum())
            n_coerced = n_nan_after - n_nan_before
            if n_coerced:
                warnings.append(
                    f"{source_label}: {n_coerced} non-numeric value(s) in "
                    f"'{col}' coerced to NaN."
                )

    # --- Sort by timestamp -------------------------------------------------
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --- Duplicate-timestamp detection -------------------------------------
    n_dupes = int(df["timestamp"].duplicated().sum())
    if n_dupes:
        warnings.append(
            f"{source_label}: {n_dupes} duplicate timestamp(s) found "
            f"(kept first occurrence, later rows dropped)."
        )
        df = df.drop_duplicates(subset="timestamp", keep="first") \
               .reset_index(drop=True)

    # --- Drop rows where all signal columns are NaN (trailing blanks) ------
    signal_cols = [c for c in
                   ["voltage", "current", "temperature", "soc", "soh"]
                   if c in df.columns]
    if signal_cols:
        all_nan_mask = df[signal_cols].isna().all(axis=1)
        n_all_nan = int(all_nan_mask.sum())
        if n_all_nan:
            warnings.append(
                f"{source_label}: {n_all_nan} row(s) with all-NaN signals "
                f"dropped."
            )
            df = df[~all_nan_mask].reset_index(drop=True)

    if df.empty:
        raise DataLoaderError(
            f"No valid data rows remain after cleaning: {path}"
        )

    # --- Assemble result ---------------------------------------------------
    return build_load_result(
        df=df,
        source_label=source_label,
        source_columns=original_columns,
        warnings=warnings,
        axis_kind=axis_kind,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def build_load_result(
    df: pd.DataFrame,
    source_label: str,
    source_columns: List[str],
    warnings: Optional[List[str]] = None,
    axis_kind: str = "timestamp",
) -> LoadResult:
    """Assemble a ``LoadResult`` from an already-clean DataFrame.

    This is the shared tail end of every load path — the offline CSV
    loader and the live (MQTT) accumulator both produce a DataFrame with
    canonical column names, a float64 ``timestamp`` column sorted
    ascending, deduplicated rows, and coerced signal columns, then hand
    it here to compute the metadata the rest of the pipeline needs
    (``columns_found`` / ``columns_missing`` / ``time_range`` /
    ``row_count``).

    The DataFrame is **not** modified — callers keep a reference so the
    produced ``LoadResult`` stays in sync with whatever buffer they own.

    Parameters
    ----------
    df
        Clean DataFrame, canonical column names, sorted by ``timestamp``.
    source_label
        ``"twin"`` or ``"ecu"`` — baked into the result for routing.
    source_columns
        The *original* column names as seen on the wire / in the CSV
        header (before alias resolution). Stored for diagnostics.
    warnings
        Optional list of warning strings collected during cleaning.
        Defaults to a fresh empty list.
    axis_kind
        ``"timestamp"`` (default) for a real time axis, or ``"sequence"``
        when the primary column is a plain sample counter (the live
        accumulator passes this through from ``LiveSample.axis_kind``).

    Returns
    -------
    LoadResult
    """
    columns_found = [c for c in _CANONICAL_COLUMNS if c in df.columns]
    columns_missing = [c for c in _CANONICAL_SIGNALS if c not in df.columns]

    if df.empty:
        t_min, t_max = 0.0, 0.0
    else:
        t_min = float(df["timestamp"].iloc[0])
        t_max = float(df["timestamp"].iloc[-1])

    return LoadResult(
        df=df,
        source_label=source_label,
        row_count=len(df),
        columns_found=columns_found,
        columns_missing=columns_missing,
        source_columns=source_columns,
        time_range=(t_min, t_max),
        warnings=list(warnings) if warnings else [],
        axis_kind=axis_kind,
    )
def _resolve_columns(original: list[str]) -> tuple[dict[str, str], list[str]]:
    """Map original header → canonical name.

    Returns ``(canonical_map, warnings)`` where ``canonical_map`` is
    ``{original_name: canonical_name}``.
    """
    canonical_map: dict[str, str] = {}
    used_canonical: set[str] = set()
    warnings: list[str] = []

    # Build a lower-cased lookup: lower_name → original_name
    lower_to_original = {}
    for col in original:
        key = col.strip().lower()
        if key not in lower_to_original:
            lower_to_original[key] = col
        else:
            warnings.append(f"Duplicate header (case-variant) ignored: {col}")

    # For each canonical group, find the first matching alias.
    # "id" is handled as a timestamp fallback below so it doesn't steal
    # timestamp from payloads that also carry a proper time key like "t".
    for canon, aliases in _ALIAS_MAP.items():
        for alias in aliases:
            if canon == "timestamp" and alias == "id":
                continue
            if alias in lower_to_original and canon not in used_canonical:
                canonical_map[lower_to_original[alias]] = canon
                used_canonical.add(canon)
                break

    # Special rule for single-letter ``t``. Note the "temperature already
    # used" check only guards the *temperature* branch below (to avoid
    # mapping two original columns onto the same canonical "temperature"
    # slot when an unambiguous column like "temp" already claimed it) --
    # it must NOT gate the whole rule, or an unrelated "Temperature"
    # column (its own unambiguous alias, nothing to do with "t") would
    # incorrectly block "t" from becoming timestamp when "v" is present.
    if "t" in lower_to_original and "timestamp" not in used_canonical:
        # ``t`` + ``v`` present → treat ``t`` as timestamp (common twin
        # layout), otherwise treat ``t`` as temperature.
        if "v" in lower_to_original:
            canonical_map[lower_to_original["t"]] = "timestamp"
            used_canonical.add("timestamp")
        elif "temperature" not in used_canonical:
            canonical_map[lower_to_original["t"]] = "temperature"
            used_canonical.add("temperature")

    # Timestamp fallback: if no explicit timestamp alias was found, accept
    # "id" as the timestamp source.
    if "timestamp" not in used_canonical and "id" in lower_to_original:
        canonical_map[lower_to_original["id"]] = "timestamp"
        used_canonical.add("timestamp")

    return canonical_map, warnings


def _normalise_timestamp(df: pd.DataFrame,
                         warnings: list[str]) -> pd.DataFrame:
    """Convert the ``timestamp`` column to float64 seconds.

    - If already numeric, keep as-is (type-cast to float).
    - If string / object, attempt ``pd.to_datetime`` → POSIX seconds.
    """
    col = df["timestamp"]

    if pd.api.types.is_numeric_dtype(col):
        df["timestamp"] = col.astype(np.float64)
    else:
        # Try datetime parsing
        try:
            dt = pd.to_datetime(col, errors="coerce")
            n_failed = int(dt.isna().sum())
            if n_failed and n_failed == len(dt):
                # All failed — maybe it's a numeric column read as string
                raise DataLoaderError(
                    "Could not parse timestamp column as datetime or numeric."
                )
            if n_failed:
                warnings.append(
                    f"{n_failed} unparseable timestamp value(s) "
                    f"dropped."
                )
                df = df[~dt.isna()].reset_index(drop=True)
                dt = dt[~dt.isna()]
            df["timestamp"] = dt.astype(np.int64).values / 1e9
        except DataLoaderError:
            raise
        except Exception as exc:
            raise DataLoaderError(f"Failed to parse timestamp column: {exc}")

    return df
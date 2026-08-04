"""Statistics engine for the Digital Twin Validation Tool.

Given an :class:`AlignedData` (output of :mod:`alignment_engine`) and a
per-signal tolerance percentage, compute every metric shown on the
Statistics tab:

- Six **summary cards** (total samples, matched samples, mean absolute
  error, max error, RMSE, overall match %).
- A **per-signal breakdown** (one row per signal: MAE, RMSE, max error,
  mean error / bias, match %).
- A **worst-N mismatches** list (top sample pairs by absolute error).

Like the DataLoader and AlignmentEngine, this module is **pure Python**
(imports only ``numpy``, stdlib, and ``alignment_engine.AlignedData``) so
it can be unit-tested in isolation and reused by the live-MQTT path.

Tolerance model
---------------
A sample pair for a signal is considered **within tolerance** when::

    abs(ecu_value - twin_value) <= (tol_pct / 100) * abs(ecu_value)

i.e. the absolute error is within ``tol_pct`` percent of the ECU (ground
truth) value.  An absolute floor (``_ABS_FLOOR`` per signal) avoids a
divide-by-near-zero situation for signals that can legitimately read 0
(e.g. current at rest).  When the ECU value's magnitude is below the
floor, the floor itself is used as the reference magnitude.

The overall **summary Match %** is the fraction of *matched* (non-NaN)
sample pairs whose **every** enabled signal is within tolerance — i.e.
a pair counts as a match only when all signals agree simultaneously.
Each per-signal row has its own Match % computed independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from alignment_engine import AlignedData, AlignedSignal


# ---------------------------------------------------------------------------
# Default per-signal tolerance (%) and absolute floor (in signal units)
# ---------------------------------------------------------------------------
# These defaults are used only when the caller does not supply a tolerance
# for a signal (e.g. the UI's Config tab only exposes spinboxes for the
# three physical signals; SoC / SoH fall back to these defaults).
DEFAULT_TOLERANCES: Dict[str, float] = {
    "voltage":     1.0,   # ±1 % of reading
    "current":     2.0,   # ±2 % of reading
    "temperature": 2.0,   # ±2 % of reading
    "soc":         1.0,   # ±1 % of reading
    "soh":         1.0,   # ±1 % of reading
}

# Absolute floor (in signal units) below which the tolerance is measured
# against the floor rather than the reading.  Prevents a near-zero reading
# from producing an unrealistically tight tolerance band.
_ABS_FLOOR: Dict[str, float] = {
    "voltage":     0.5,   # V
    "current":     0.5,   # A
    "temperature": 1.0,   # °C
    "soc":         1.0,   # %
    "soh":         1.0,   # %
}

# Default number of worst-mismatch rows surfaced to the UI.
DEFAULT_WORST_N = 20


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class SignalStat:
    """Per-signal statistics for the breakdown table."""

    name: str
    mae: float                 # mean absolute error over valid pairs
    rmse: float                # root-mean-square error
    max_error: float           # largest |error|
    mean_error: float          # signed mean error (bias: ecu - twin)
    n_valid: int               # pairs where neither side is NaN
    n_within_tolerance: int    # valid pairs whose |error| ≤ tolerance
    match_pct: float           # n_within_tolerance / n_valid × 100
    tolerance_pct: float       # the tolerance used for this signal


@dataclass
class WorstMismatch:
    """One row of the worst-mismatches table."""

    index: int                 # row index in the aligned arrays
    timestamp: float           # ECU timestamp (seconds)
    signal: str                # canonical signal name
    twin_value: float
    ecu_value: float
    error: float               # ecu - twin (signed)
    within_tolerance: bool


@dataclass
class StatisticsResult:
    """Output of :func:`compute` — everything the Statistics tab needs."""

    # --- Six summary cards ---
    total_samples: int
    matched_samples: int       # non-NaN pairs (all signals valid)
    mean_abs_error: float      # mean of |ecu - twin| over all valid pairs
    max_error: float           # max |error| across all signals / samples
    rmse: float                # root-mean-square error over all valid pairs
    match_pct: float           # within-tolerance pairs / matched × 100

    # --- Detailed tables ---
    signal_stats: Dict[str, SignalStat] = field(default_factory=dict)
    worst_mismatches: List[WorstMismatch] = field(default_factory=list)

    # --- Bookkeeping ---
    tolerances: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def signal_names(self) -> List[str]:
        return list(self.signal_stats.keys())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute(
    aligned: AlignedData,
    tolerances: Optional[Dict[str, float]] = None,
    worst_n: int = DEFAULT_WORST_N,
) -> StatisticsResult:
    """Compute all statistics for the Statistics tab.

    Parameters
    ----------
    aligned
        Output of :func:`alignment_engine.align`.
    tolerances
        Mapping of canonical signal name → tolerance percentage.  Any
        signal not present falls back to :data:`DEFAULT_TOLERANCES`.
        Values are interpreted as percentages (e.g. ``1.5`` means 1.5 %).
    worst_n
        Maximum number of rows to include in the worst-mismatches list.
        Defaults to 20.

    Returns
    -------
    StatisticsResult
        Summary cards, per-signal breakdown, and worst mismatches.
    """
    warnings: List[str] = []
    tols = _resolve_tolerances(aligned, tolerances, warnings)

    # --- Per-signal statistics ---------------------------------------------
    signal_stats: Dict[str, SignalStat] = {}
    all_abs_errors: List[np.ndarray] = []   # for global MAE / RMSE / max
    per_signal_within: Dict[str, np.ndarray] = {}  # bool mask per signal

    for name, sig in aligned.signals.items():
        stat, within_mask, abs_err = _signal_stats(sig, tols.get(name))
        signal_stats[name] = stat
        all_abs_errors.append(abs_err)
        per_signal_within[name] = within_mask

    # --- Global summary cards ---------------------------------------------
    total_samples = aligned.n_total
    matched_samples = aligned.n_matched

    if all_abs_errors:
        stacked = np.concatenate(all_abs_errors)
        mean_abs_error = float(np.mean(stacked)) if stacked.size else 0.0
        max_error = float(np.max(stacked)) if stacked.size else 0.0
        rmse = float(np.sqrt(np.mean(stacked ** 2))) if stacked.size else 0.0
    else:
        mean_abs_error = 0.0
        max_error = 0.0
        rmse = 0.0

    # Overall Match %: a matched pair counts as "within tolerance" only
    # when *every* signal's error is within that signal's tolerance.
    if matched_samples > 0 and per_signal_within:
        overall_within = np.ones(aligned.n_total, dtype=bool)
        # Only consider pairs that are matched (valid in all signals).
        valid_mask = _matched_mask(aligned)
        for name, within in per_signal_within.items():
            overall_within &= within
        n_overall_within = int((overall_within & valid_mask).sum())
        match_pct = n_overall_within / matched_samples * 100.0
    else:
        match_pct = 0.0

    # --- Worst-N mismatches ------------------------------------------------
    worst_mismatches = _worst_mismatches(
        aligned, per_signal_within, worst_n
    )

    return StatisticsResult(
        total_samples=total_samples,
        matched_samples=matched_samples,
        mean_abs_error=mean_abs_error,
        max_error=max_error,
        rmse=rmse,
        match_pct=match_pct,
        signal_stats=signal_stats,
        worst_mismatches=worst_mismatches,
        tolerances=dict(tols),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _resolve_tolerances(
    aligned: AlignedData,
    tolerances: Optional[Dict[str, float]],
    warnings: List[str],
) -> Dict[str, float]:
    """Merge caller-supplied tolerances with defaults for active signals."""
    tols: Dict[str, float] = {}
    caller = tolerances or {}
    for name in aligned.signal_names:
        if name in caller:
            tols[name] = float(caller[name])
        elif name in DEFAULT_TOLERANCES:
            tols[name] = DEFAULT_TOLERANCES[name]
        else:
            # Unknown signal — pick a conservative 5 % default.
            tols[name] = 5.0
            warnings.append(
                f"No default tolerance for signal '{name}'; using 5.0 %."
            )
    return tols


def _signal_stats(
    sig: AlignedSignal,
    tol_pct: float,
) -> tuple:
    """Compute one SignalStat plus the within-tolerance mask and abs errors.

    Returns ``(SignalStat, within_mask, abs_errors)``.
    """
    twin = sig.twin_values
    ecu = sig.ecu_values
    valid = (~np.isnan(twin)) & (~np.isnan(ecu))
    n_valid = int(valid.sum())

    if n_valid == 0:
        stat = SignalStat(
            name=sig.name,
            mae=0.0,
            rmse=0.0,
            max_error=0.0,
            mean_error=0.0,
            n_valid=0,
            n_within_tolerance=0,
            match_pct=0.0,
            tolerance_pct=tol_pct if tol_pct is not None else 0.0,
        )
        return stat, np.zeros(len(twin), dtype=bool), np.array([])

    err = sig.errors[valid]
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    max_error = float(np.max(abs_err))
    mean_error = float(np.mean(err))

    # Within-tolerance mask over the full array (NaN-safe).
    within_mask = _within_tolerance_mask(ecu, sig.errors, tol_pct, sig.name)

    n_within = int((within_mask & valid).sum())
    match_pct = (n_within / n_valid * 100.0) if n_valid else 0.0

    stat = SignalStat(
        name=sig.name,
        mae=mae,
        rmse=rmse,
        max_error=max_error,
        mean_error=mean_error,
        n_valid=n_valid,
        n_within_tolerance=n_within,
        match_pct=match_pct,
        tolerance_pct=tol_pct if tol_pct is not None else 0.0,
    )
    return stat, within_mask, abs_err


def _within_tolerance_mask(
    ecu_values: np.ndarray,
    errors: np.ndarray,
    tol_pct: Optional[float],
    signal_name: str,
) -> np.ndarray:
    """Boolean mask: True where |error| ≤ tol % of |ecu| (with floor)."""
    if tol_pct is None:
        tol_pct = DEFAULT_TOLERANCES.get(signal_name, 5.0)
    floor = _ABS_FLOOR.get(signal_name, 1.0)
    tol_abs = (tol_pct / 100.0) * np.maximum(np.abs(ecu_values), floor)
    return np.abs(errors) <= tol_abs


def _matched_mask(aligned: AlignedData) -> np.ndarray:
    """Boolean mask of samples valid across all signals."""
    mask = np.ones(aligned.n_total, dtype=bool)
    for sig in aligned.signals.values():
        mask &= (~np.isnan(sig.twin_values)) & (~np.isnan(sig.ecu_values))
    return mask


def _worst_mismatches(
    aligned: AlignedData,
    per_signal_within: Dict[str, np.ndarray],
    worst_n: int,
) -> List[WorstMismatch]:
    """Collect the top-``worst_n`` sample pairs by absolute error."""
    if worst_n <= 0 or not aligned.signals:
        return []

    candidates: list = []  # (abs_error, index, signal_name)
    for name, sig in aligned.signals.items():
        twin = sig.twin_values
        ecu = sig.ecu_values
        valid = (~np.isnan(twin)) & (~np.isnan(ecu))
        abs_err = np.abs(sig.errors)
        idxs = np.where(valid)[0]
        for i in idxs:
            candidates.append((float(abs_err[i]), int(i), name))

    # Sort by absolute error descending; take top worst_n.
    candidates.sort(key=lambda c: c[0], reverse=True)
    top = candidates[:worst_n]

    out: List[WorstMismatch] = []
    for abs_e, idx, name in top:
        sig = aligned.signals[name]
        out.append(WorstMismatch(
            index=idx,
            timestamp=float(aligned.timestamps[idx]),
            signal=name,
            twin_value=float(sig.twin_values[idx]),
            ecu_value=float(sig.ecu_values[idx]),
            error=float(sig.errors[idx]),
            within_tolerance=bool(per_signal_within[name][idx]),
        ))
    return out
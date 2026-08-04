"""Alignment engine for the Digital Twin Validation Tool.

Takes two ``LoadResult`` DataFrames (twin + ECU) and aligns them by
timestamp so that each output row represents a matched sample pair.  The
engine handles the fact that the twin and the ECU almost certainly sample
at different instants — even if both run at 10 Hz, their clocks are not
synchronised, so the ECU's ``t = 0.05 s`` falls between the twin's
``t = 0.0 s`` and ``t = 0.1 s``.

Two alignment strategies are supported, selected from the Config tab's
``comboBoxAlignmentMethod``:

1. **Nearest Timestamp** — for each ECU sample, find the twin sample with
   the closest timestamp.  Simple and predictable; may introduce up to
   half a sample period of jitter.
2. **Linear Interpolation** — interpolate the twin's signals onto the
   ECU's timestamps.  Smoother but assumes the twin is roughly linear
   between samples (fine for slow signals, questionable for fast
   transients).

Design note: the ECU is the *ground truth* (real hardware), so the ECU's
timestamps define the common reference timeline.  The twin is aligned
*onto* the ECU.  This means an ECU sample is never dropped — every ECU
row gets a matched twin value (nearest or interpolated).

This module is **pure Python** — no Qt imports — so it can be unit-tested
in isolation with small DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_loader import LoadResult


# ---------------------------------------------------------------------------
# Strategy constants
# ---------------------------------------------------------------------------
# String constants used by both this module and ``back.py`` when reading the
# ``comboBoxAlignmentMethod`` combo box.  Using constants instead of magic
# strings prevents typos that would silently fall through to the default.
ALIGN_NEAREST = "nearest"
ALIGN_INTERPOLATE = "interpolate"

# Signals that the StatisticsEngine and PlotManager know how to handle.
# Only signals present in *both* LoadResults are aligned; the rest are
# silently skipped (the StatisticsEngine reports them as missing).
# The five signals are: voltage, current, temperature, SoC (state of
# charge), SoH (state of health).
_SIGNAL_COLUMNS = ["voltage", "current", "temperature", "soc", "soh"]

# If the largest timestamp gap between an ECU sample and its matched twin
# sample exceeds this threshold (in seconds), a warning is emitted so the
# user knows the alignment may be unreliable.  Half a second is a
# conservative default for 10 Hz sampling.
_MAX_DELTA_T_WARN_S = 0.5


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class AlignedSignal:
    """One signal's worth of aligned twin / ECU arrays."""

    name: str                             # canonical name: "voltage", etc.
    twin_values: np.ndarray              # float64, same length as timestamps
    ecu_values: np.ndarray              # float64, same length as timestamps
    errors: np.ndarray                  # ecu_values - twin_values

    @property
    def n_samples(self) -> int:
        return len(self.errors)

    @property
    def n_valid(self) -> int:
        """Number of sample pairs where neither side is NaN."""
        return int((~np.isnan(self.twin_values) & ~np.isnan(self.ecu_values)).sum())


@dataclass
class AlignedData:
    """Output of the alignment engine — everything downstream needs."""

    timestamps: np.ndarray                    # common timeline (ECU's),
                                              # float64 seconds, ascending
    signals: Dict[str, AlignedSignal]         # keyed by canonical name
    n_total: int                              # len(timestamps)
    n_matched: int                            # samples where all signals
                                              # have valid pairs
    max_delta_t: float                        # largest |ecu_t - matched_twin_t|
    alignment_method: str                     # "nearest" or "interpolate"
    warnings: List[str] = field(default_factory=list)

    @property
    def signal_names(self) -> List[str]:
        return list(self.signals.keys())


class AlignmentError(Exception):
    """Fatal error during alignment (no overlapping signals, no overlap)."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def align(
    twin_result: LoadResult,
    ecu_result: LoadResult,
    method: str = ALIGN_NEAREST,
) -> AlignedData:
    """Align twin data onto the ECU's timestamps.

    Parameters
    ----------
    twin_result, ecu_result
        ``LoadResult`` objects produced by ``data_loader.load_csv``.
    method
        ``"nearest"`` or ``"interpolate"``.  Any other value falls back to
        ``"nearest"`` with a warning.

    Returns
    -------
    AlignedData
        Common timeline (ECU timestamps), per-signal aligned arrays,
        metadata, and warnings.

    Raises
    ------
    AlignmentError
        If the two DataFrames have no overlapping signal columns or no
        overlapping time range.
    """
    twin_df = twin_result.df
    ecu_df = ecu_result.df
    warnings: List[str] = []

    # --- Validate method ---------------------------------------------------
    if method not in (ALIGN_NEAREST, ALIGN_INTERPOLATE):
        warnings.append(
            f"Unknown alignment method '{method}'; falling back to "
            f"'{ALIGN_NEAREST}'."
        )
        method = ALIGN_NEAREST

    # --- Determine which signals to align ----------------------------------
    common_signals = [
        col for col in _SIGNAL_COLUMNS
        if col in twin_df.columns and col in ecu_df.columns
    ]
    if not common_signals:
        raise AlignmentError(
            "No common signal columns between twin and ECU.\n"
            f"Twin columns:   {list(twin_df.columns)}\n"
            f"ECU columns:    {list(ecu_df.columns)}"
        )

    # --- Extract timestamp arrays -----------------------------------------
    twin_ts = twin_df["timestamp"].to_numpy(dtype=np.float64)
    ecu_ts = ecu_df["timestamp"].to_numpy(dtype=np.float64)

    # --- Check time-range overlap ------------------------------------------
    # If the twin's time range and the ECU's time range don't overlap at
    # all, alignment is meaningless.  This can happen if the two CSVs were
    # recorded at completely different times.
    overlap_start = max(twin_ts[0], ecu_ts[0])
    overlap_end = min(twin_ts[-1], ecu_ts[-1])
    if overlap_end < overlap_start:
        raise AlignmentError(
            "No time-range overlap between twin and ECU.\n"
            f"Twin range: {twin_ts[0]:.3f} – {twin_ts[-1]:.3f} s\n"
            f"ECU range:  {ecu_ts[0]:.3f} – {ecu_ts[-1]:.3f} s"
        )

    # --- Align each signal -------------------------------------------------
    # The ECU timestamps are the reference timeline.  For each ECU
    # timestamp we find the corresponding twin value (nearest or
    # interpolated).  ECU samples outside the twin's time range get NaN
    # for the twin side (nearest) or are clipped (interpolation) — we
    # mark them so the StatisticsEngine can skip them.
    signals: Dict[str, AlignedSignal] = {}
    min_valid_idx = 0  # first ECU index within twin time range
    max_delta_t = 0.0

    for signal_name in common_signals:
        twin_vals = twin_df[signal_name].to_numpy(dtype=np.float64)
        ecu_vals = ecu_df[signal_name].to_numpy(dtype=np.float64)

        if method == ALIGN_NEAREST:
            aligned_twin, deltas = _align_nearest(twin_ts, twin_vals, ecu_ts)
        else:
            aligned_twin, deltas = _align_interpolate(twin_ts, twin_vals,
                                                     ecu_ts)

        # Track the largest timestamp gap across all signals (the gap is
        # the same for every signal, but computing it here keeps the
        # per-signal loop self-contained).
        if deltas.size > 0:
            max_delta_t = max(max_delta_t, float(np.max(deltas)))

        errors = ecu_vals - aligned_twin
        signals[signal_name] = AlignedSignal(
            name=signal_name,
            twin_values=aligned_twin,
            ecu_values=ecu_vals,
            errors=errors,
        )

    # --- Compute n_matched (samples where all signals are valid) ----------
    # A sample is "matched" if, for every aligned signal, *both* the twin
    # and ECU values are non-NaN.  This is the denominator for Match %.
    n_total = len(ecu_ts)
    if signals:
        valid_mask = np.ones(n_total, dtype=bool)
        for sig in signals.values():
            twin_valid = ~np.isnan(sig.twin_values)
            ecu_valid = ~np.isnan(sig.ecu_values)
            valid_mask &= twin_valid & ecu_valid
        n_matched = int(valid_mask.sum())
    else:
        n_matched = 0

    # --- Warnings ----------------------------------------------------------
    if max_delta_t > _MAX_DELTA_T_WARN_S:
        warnings.append(
            f"Largest timestamp gap between ECU and matched twin sample "
            f"is {max_delta_t:.3f} s (> {_MAX_DELTA_T_WARN_S} s threshold). "
            f"Alignment may be unreliable — consider a finer sample period."
        )

    n_outside = n_total - n_matched
    if n_outside > 0:
        reason = (
            "outside twin time range" if method == ALIGN_INTERPOLATE
            else "no valid twin sample"
        )
        warnings.append(
            f"{n_outside} ECU sample(s) could not be matched ({reason})."
        )

    return AlignedData(
        timestamps=ecu_ts,
        signals=signals,
        n_total=n_total,
        n_matched=n_matched,
        max_delta_t=max_delta_t,
        alignment_method=method,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Internal: nearest-timestamp alignment
# ---------------------------------------------------------------------------
def _align_nearest(
    twin_ts: np.ndarray,
    twin_vals: np.ndarray,
    ecu_ts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """For each ECU timestamp, find the closest twin sample.

    Returns ``(aligned_twin_values, deltas)`` where ``deltas[i]`` is
    ``abs(ecu_ts[i] - twin_ts[matched_index])``.

    ECU timestamps outside the twin's time range are matched to the
    nearest edge sample (the first or last twin sample), and the
    resulting large delta serves as a flag that the match is unreliable.
    The caller can detect this via ``deltas > threshold`` and the
    StatisticsEngine skips such pairs because we mark the twin value as
    NaN when the gap exceeds the twin's median sample period — but here
    we keep it simple and let the caller decide.
    """
    # ``np.searchsorted(twin_ts, ecu_ts, side='left')`` gives, for each
    # ECU timestamp, the insertion point that keeps twin_ts sorted.  The
    # nearest twin sample is either at that index or the one before it.
    indices = np.searchsorted(twin_ts, ecu_ts, side="left")

    # Clamp to valid twin indices [0, len-1] so ECU timestamps outside the
    # twin range are matched to the edge sample.
    indices_left = np.clip(indices - 1, 0, len(twin_ts) - 1)
    indices_right = np.clip(indices, 0, len(twin_ts) - 1)

    # Choose the closer of the two candidates for each ECU timestamp.
    delta_left = np.abs(ecu_ts - twin_ts[indices_left])
    delta_right = np.abs(ecu_ts - twin_ts[indices_right])
    choose_right = delta_right < delta_left
    nearest_indices = np.where(choose_right, indices_right, indices_left)
    deltas = np.where(choose_right, delta_right, delta_left)

    aligned_twin = twin_vals[nearest_indices].astype(np.float64)

    # Mark ECU samples that fall outside the twin's time range as NaN so
    # they are not counted as "matched" by the StatisticsEngine.  These
    # are samples where the nearest twin sample is at the edge and the
    # gap is larger than the twin's typical sample period.
    # Heuristic: if the gap exceeds 2× the twin's median sample period,
    # treat as unmatched.
    if len(twin_ts) > 1:
        twin_period = float(np.median(np.diff(twin_ts)))
    else:
        twin_period = np.inf
    outside_mask = deltas > (2.0 * twin_period)
    aligned_twin[outside_mask] = np.nan

    return aligned_twin, deltas


# ---------------------------------------------------------------------------
# Internal: linear-interpolation alignment
# ---------------------------------------------------------------------------
def _align_interpolate(
    twin_ts: np.ndarray,
    twin_vals: np.ndarray,
    ecu_ts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate twin values onto ECU timestamps.

    ``np.interp`` handles the mechanics: it linearly interpolates the
    twin signal at each ECU timestamp and *clips* ECU timestamps outside
    the twin's range to the edge values (flat extrapolation).  We mark
    out-of-range samples as NaN so they don't pollute the statistics.

    Returns ``(aligned_twin_values, deltas)`` where ``deltas[i]`` is the
    distance from ``ecu_ts[i]`` to the nearest twin timestamp (used only
    for the max-delta sanity check).
    """
    # Mask out NaNs in twin_vals — ``np.interp`` doesn't handle them; we
    # drop the NaN rows before interpolation.
    twin_valid = ~np.isnan(twin_vals)
    if not twin_valid.any():
        return np.full(len(ecu_ts), np.nan), np.full(len(ecu_ts), 0.0)

    clean_twin_ts = twin_ts[twin_valid]
    clean_twin_vals = twin_vals[twin_valid]

    aligned_twin = np.interp(ecu_ts, clean_twin_ts, clean_twin_vals)

    # Mark ECU samples outside the twin time range as NaN (np.interp
    # would otherwise flat-extrapolate, giving a false sense of match).
    in_range = (ecu_ts >= clean_twin_ts[0]) & (ecu_ts <= clean_twin_ts[-1])
    aligned_twin[~in_range] = np.nan

    # Deltas: distance to the nearest twin timestamp (for sanity check).
    indices = np.searchsorted(clean_twin_ts, ecu_ts, side="left")
    indices_left = np.clip(indices - 1, 0, len(clean_twin_ts) - 1)
    indices_right = np.clip(indices, 0, len(clean_twin_ts) - 1)
    delta_left = np.abs(ecu_ts - clean_twin_ts[indices_left])
    delta_right = np.abs(ecu_ts - clean_twin_ts[indices_right])
    deltas = np.minimum(delta_left, delta_right)

    return aligned_twin, deltas
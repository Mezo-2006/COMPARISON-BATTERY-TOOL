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

Design note (id-aware matching, live path): when both ``LoadResult``s
carry a publisher ``id`` column (the live MQTT path with QoS-1 dedup),
each ECU row is first paired to the twin row sharing its ``id``.  This
resolves a *delivery shift* between the two streams exactly — a twin
sample and an ECU sample produced for the same logical instant may
arrive with different wall-clock timestamps, but matching them by id
sidesteps the shift.  The matching id only has to land in the ring
buffer before ``max_age_ms`` / ``max_rows`` trims it (see
``live_accumulator``); the controller's periodic tick re-runs
``align()`` so a late id is picked up on the next tick.  ECU rows with
no matching twin id fall back to the nearest-timestamp match below.
Offline CSVs never carry an ``id`` column, so the id-aware branch is a
no-op there and the pure-timestamp behaviour is unchanged.

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
# sample exceeds this threshold, a warning is emitted so the user knows the
# alignment may be unreliable. Half a second (500 ms) is a conservative
# default for 10 Hz sampling.
#
# Unit caveat: this engine is unit-agnostic on paper (it just compares
# float timestamps), but this *specific* constant is tuned for the live
# pipeline's wire convention — Unix-epoch **milliseconds** (see
# ``synthetic_data_generator``'s "Timestamp convention" section and
# ``live_accumulator.DEFAULT_MAX_AGE_MS``). Offline CSV timestamps are
# whatever unit the file itself uses (this project's own fixtures are
# small, roughly-seconds-scale numbers), so this warning can under- or
# over-fire on a CSV run — it's cosmetic (appended to ``warnings``, never
# fatal), not a correctness issue for the alignment math itself. A future
# refactor could thread an explicit unit through ``align()`` if this
# becomes a real pain point.
_MAX_DELTA_T_WARN_MS = 500.0


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
    # "timestamp" (real time) or "sequence" (plain sample counter) — see
    # ``data_loader.LoadResult.axis_kind``. Taken from the ECU side (the
    # reference timeline this module aligns onto); a mismatch with the
    # twin side is reported in ``warnings`` rather than raised, since
    # alignment itself only needs a sortable numeric axis either way.
    axis_kind: str = "timestamp"

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

    # --- Id-aware matching (live path with shifted streams) ---------------
    # When both sides carry a publisher ``id`` (live MQTT, QoS-1), pair
    # twin and ECU rows by id first so a delivery shift between the two
    # streams is resolved exactly.  See ``_compute_match_plan`` and the
    # module docstring for the "late id arrives within the ring-buffer
    # window" rationale.  Offline CSVs never carry an ``id`` column, so
    # this stays ``None`` there and the pure-timestamp path is unchanged.
    match_plan = _compute_match_plan(twin_df, ecu_df)
    ids_available = match_plan is not None

    # --- Check time-range overlap ------------------------------------------
    # If the twin's and ECU's time ranges don't overlap at all, alignment
    # by timestamp is meaningless.  When ids are available, though, a
    # shifted stream can still be paired by id, so we defer the refusal
    # until we know whether by-id matching produced anything (see the
    # post-alignment check below).
    overlap_start = max(twin_ts[0], ecu_ts[0])
    overlap_end = min(twin_ts[-1], ecu_ts[-1])
    ranges_overlap = overlap_end >= overlap_start
    if not ranges_overlap and not ids_available:
        raise AlignmentError(
            "No time-range overlap between twin and ECU.\n"
            f"Twin range: {twin_ts[0]:.3f} – {twin_ts[-1]:.3f} s\n"
            f"ECU range:  {ecu_ts[0]:.3f} – {ecu_ts[-1]:.3f} s"
        )

    # --- Align each signal -------------------------------------------------
    # The ECU timestamps are the reference timeline.  For each ECU
    # timestamp we find the corresponding twin value (by id when a
    # match_plan entry exists, else nearest or interpolated).  ECU
    # samples with no id match and outside the twin's time range get
    # NaN for the twin side so the StatisticsEngine skips them.
    signals: Dict[str, AlignedSignal] = {}
    max_delta_t = 0.0

    for signal_name in common_signals:
        twin_vals = twin_df[signal_name].to_numpy(dtype=np.float64)
        ecu_vals = ecu_df[signal_name].to_numpy(dtype=np.float64)

        if method == ALIGN_NEAREST:
            aligned_twin, deltas, id_matched_mask = _align_nearest(
                twin_ts, twin_vals, ecu_ts, match_plan=match_plan,
            )
        else:
            aligned_twin, deltas, id_matched_mask = _align_interpolate(
                twin_ts, twin_vals, ecu_ts,
            )

        # Largest timestamp gap across fallback (non-id-matched) pairs only.
        # An id-matched pair's wall-clock gap is a delivery shift between
        # the two streams, not an alignment-quality signal, so it must not
        # trip the "alignment may be unreliable" gap warning below.  The
        # gap is identical across signals; computing it here keeps the
        # per-signal loop self-contained.
        if deltas.size > 0:
            fallback_deltas = deltas[~id_matched_mask]
            if fallback_deltas.size > 0:
                max_delta_t = max(max_delta_t, float(np.max(fallback_deltas)))

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

    # If ids were available, the upfront no-overlap raise was skipped; if
    # by-id matching then produced zero pairs AND the time ranges don't
    # overlap, nothing can be aligned — refuse now (mirrors the original
    # no-overlap contract for the genuinely-unalignable case).
    if ids_available and n_matched == 0 and not ranges_overlap:
        raise AlignmentError(
            "No matches by id and no time-range overlap between twin and "
            "ECU.\n"
            f"Twin range: {twin_ts[0]:.3f} – {twin_ts[-1]:.3f} s\n"
            f"ECU range:  {ecu_ts[0]:.3f} – {ecu_ts[-1]:.3f} s"
        )

    # --- Warnings ----------------------------------------------------------
    if max_delta_t > _MAX_DELTA_T_WARN_MS:
        warnings.append(
            f"Largest timestamp gap between ECU and matched twin sample "
            f"is {max_delta_t:.3f} (> {_MAX_DELTA_T_WARN_MS} threshold, "
            f"assuming millisecond-scale live timestamps). "
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

    # --- Axis kind -----------------------------------------------------
    # ECU is the reference timeline this module aligns onto, so its axis
    # kind wins. A twin/ECU mismatch (one's a real time axis, the other a
    # sequence counter) can't happen from a single generator run, but
    # guard it anyway rather than silently mislabel the Graphs tab.
    twin_axis_kind = getattr(twin_result, "axis_kind", "timestamp")
    ecu_axis_kind = getattr(ecu_result, "axis_kind", "timestamp")
    if twin_axis_kind != ecu_axis_kind:
        warnings.append(
            f"Twin and ECU disagree on axis kind (twin={twin_axis_kind!r}, "
            f"ecu={ecu_axis_kind!r}); using ECU's ({ecu_axis_kind!r})."
        )

    return AlignedData(
        timestamps=ecu_ts,
        signals=signals,
        n_total=n_total,
        n_matched=n_matched,
        max_delta_t=max_delta_t,
        alignment_method=method,
        warnings=warnings,
        axis_kind=ecu_axis_kind,
    )


# ---------------------------------------------------------------------------
# Internal: nearest-timestamp alignment
# ---------------------------------------------------------------------------
def _align_nearest(
    twin_ts: np.ndarray,
    twin_vals: np.ndarray,
    ecu_ts: np.ndarray,
    match_plan: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each ECU sample, find the matching twin sample.

    When ``match_plan`` is provided, ECU rows whose plan entry is ``>= 0``
    are paired directly to that twin row *by id* — these are the same
    logical sample arriving at a different wall-clock time, so the twin
    value is taken exactly and is never NaN'd.  ECU rows whose plan
    entry is ``-1`` (no matching twin id, or ids not in use at all) fall
    back to nearest-timestamp matching with the 2× median-period NaN
    heuristic.  When ``match_plan`` is ``None`` (offline CSV path, or a
    live stream that omits ids) every ECU row takes the fallback path —
    identical to the historical behaviour.

    Returns ``(aligned_twin_values, deltas, id_matched_mask)`` where
    ``deltas[i]`` is ``abs(ecu_ts[i] - twin_ts[matched_index])`` and
    ``id_matched_mask[i]`` is ``True`` iff row ``i`` was paired by id.
    The caller uses ``id_matched_mask`` to exclude id-matched wall-clock
    gaps from the alignment-quality ("gap too large") warning.
    """
    n = len(ecu_ts)
    aligned_twin = np.empty(n, dtype=np.float64)
    deltas = np.zeros(n, dtype=np.float64)
    id_matched_mask = np.zeros(n, dtype=bool)

    if match_plan is not None:
        id_rows = match_plan >= 0
        if id_rows.any():
            twin_idx = match_plan[id_rows]
            aligned_twin[id_rows] = twin_vals[twin_idx]
            deltas[id_rows] = np.abs(ecu_ts[id_rows] - twin_ts[twin_idx])
            id_matched_mask[id_rows] = True
        fallback_mask = ~id_rows
    else:
        fallback_mask = np.ones(n, dtype=bool)

    if fallback_mask.any():
        fb_ts = ecu_ts[fallback_mask]

        # ``np.searchsorted(twin_ts, ecu_ts, side='left')`` gives, for each
        # ECU timestamp, the insertion point that keeps twin_ts sorted.
        # The nearest twin sample is either at that index or the one before.
        indices = np.searchsorted(twin_ts, fb_ts, side="left")

        # Clamp to valid twin indices [0, len-1] so ECU timestamps outside
        # the twin range are matched to the edge sample.
        indices_left = np.clip(indices - 1, 0, len(twin_ts) - 1)
        indices_right = np.clip(indices, 0, len(twin_ts) - 1)

        # Choose the closer of the two candidates for each ECU timestamp.
        delta_left = np.abs(fb_ts - twin_ts[indices_left])
        delta_right = np.abs(fb_ts - twin_ts[indices_right])
        choose_right = delta_right < delta_left
        nearest_indices = np.where(choose_right, indices_right, indices_left)
        fb_deltas = np.where(choose_right, delta_right, delta_left)
        fb_aligned = twin_vals[nearest_indices].astype(np.float64)

        # Mark fallback ECU samples that fall outside the twin's time range
        # as NaN so they are not counted as "matched" by the
        # StatisticsEngine.  Heuristic: a gap exceeding 2× the twin's
        # median sample period is treated as unmatched.
        if len(twin_ts) > 1:
            twin_period = float(np.median(np.diff(twin_ts)))
        else:
            twin_period = np.inf
        outside_mask = fb_deltas > (2.0 * twin_period)
        fb_aligned[outside_mask] = np.nan

        aligned_twin[fallback_mask] = fb_aligned
        deltas[fallback_mask] = fb_deltas

    return aligned_twin, deltas, id_matched_mask


# ---------------------------------------------------------------------------
# Internal: linear-interpolation alignment
# ---------------------------------------------------------------------------
def _align_interpolate(
    twin_ts: np.ndarray,
    twin_vals: np.ndarray,
    ecu_ts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearly interpolate twin values onto ECU timestamps.

    ``np.interp`` handles the mechanics: it linearly interpolates the
    twin signal at each ECU timestamp and *clips* ECU timestamps outside
    the twin's range to the edge values (flat extrapolation).  We mark
    out-of-range samples as NaN so they don't pollute the statistics.

    Id-aware matching is not wired through interpolation: the live path
    always uses ``ALIGN_NEAREST`` (see ``live_controller._tick``), and
    offline CSVs never carry an ``id`` column, so interpolation never
    sees ids in practice.  The returned ``id_matched_mask`` is therefore
    all-False, which makes ``align()`` count every interpolated delta
    toward ``max_delta_t`` — identical to the historical behaviour.

    Returns ``(aligned_twin_values, deltas, id_matched_mask)`` where
    ``deltas[i]`` is the distance from ``ecu_ts[i]`` to the nearest twin
    timestamp (used only for the max-delta sanity check).
    """
    id_matched_mask = np.zeros(len(ecu_ts), dtype=bool)

    # Mask out NaNs in twin_vals — ``np.interp`` doesn't handle them; we
    # drop the NaN rows before interpolation.
    twin_valid = ~np.isnan(twin_vals)
    if not twin_valid.any():
        empty = np.full(len(ecu_ts), np.nan)
        return empty, np.full(len(ecu_ts), 0.0), id_matched_mask

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

    return aligned_twin, deltas, id_matched_mask


# ---------------------------------------------------------------------------
# Internal: id-keyed match plan (live path with shifted streams)
# ---------------------------------------------------------------------------
def _compute_match_plan(
    twin_df: pd.DataFrame,
    ecu_df: pd.DataFrame,
) -> Optional[np.ndarray]:
    """Map each ECU row to a twin row by shared publisher ``id``.

    Returns an int64 array of length ``len(ecu_df)`` where entry ``i`` is
    the twin row index whose ``id`` matches ECU row ``i``'s ``id``, or
    ``-1`` when ECU row ``i`` has no matching twin id (the caller falls
    back to nearest-timestamp matching for that row).

    Returns ``None`` when ids are not usable on both sides — either the
    ``id`` column is absent on one side, or neither side has at least
    one non-null id.  The offline CSV path never carries an ``id``
    column, so this returns ``None`` there and the pure-timestamp path
    stays in effect; a live stream that omits ids also gets ``None``.

    The first twin row seen for a given id wins (mirrors the buffer's
    ``keep="first"`` QoS-1 dedup, so a redelivered twin id is the same
    row the buffer kept).  Ids are stringified on both sides so an int
    ``42`` and a string ``"42"`` from different publishers can't miss
    each other on a pandas object dtype.
    """
    if "id" not in twin_df.columns or "id" not in ecu_df.columns:
        return None
    if not twin_df["id"].notna().any() or not ecu_df["id"].notna().any():
        return None

    twin_by_id: Dict[str, int] = {}
    for idx, val in enumerate(twin_df["id"].to_list()):
        if pd.isna(val):
            continue
        key = str(val)
        if key not in twin_by_id:
            twin_by_id[key] = idx

    plan = np.full(len(ecu_df), -1, dtype=np.int64)
    for i, val in enumerate(ecu_df["id"].to_list()):
        if pd.isna(val):
            continue
        j = twin_by_id.get(str(val))
        if j is not None:
            plan[i] = j
    return plan
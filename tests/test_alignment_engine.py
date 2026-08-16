"""Tests for ``alignment_engine.py`` — twin ↔ ECU alignment."""

from __future__ import annotations

import numpy as np
import pytest

from alignment_engine import (
    AlignedData,
    AlignmentError,
    align,
    ALIGN_NEAREST,
    ALIGN_INTERPOLATE,
)


# ---------------------------------------------------------------------------
# Happy path: 5-signal fixtures
# ---------------------------------------------------------------------------
def test_nearest_all_five_signals_present(twin_result_five, ecu_result_five):
    aligned = align(twin_result_five, ecu_result_five, ALIGN_NEAREST)

    assert aligned.alignment_method == "nearest"
    assert aligned.n_total == 5
    # All 5 signals aligned.
    assert set(aligned.signal_names) == {
        "voltage", "current", "temperature", "soc", "soh"
    }
    # Reference timeline is the ECU's.
    np.testing.assert_allclose(
        aligned.timestamps,
        ecu_result_five.df["timestamp"].to_numpy(),
    )


def test_nearest_max_delta_t_within_half_period(twin_result_five, ecu_result_five):
    aligned = align(twin_result_five, ecu_result_five, ALIGN_NEAREST)
    # ECU offset is 0.05 s, twin period is 0.1 s → max delta ≤ 0.05.
    assert aligned.max_delta_t == pytest.approx(0.05, abs=1e-9)


def test_nearest_errors_are_half_step(twin_result_five, ecu_result_five):
    """The fixtures are linear ramps offset by half a sample period.

    Nearest-match error is exactly the half-step difference (0.005 for
    V/I, 0.05 for T, 0.05 for SoC, 0 for SoH which is constant).
    """
    aligned = align(twin_result_five, ecu_result_five, ALIGN_NEAREST)
    assert np.max(np.abs(aligned.signals["voltage"].errors)) == pytest.approx(0.005)
    assert np.max(np.abs(aligned.signals["current"].errors)) == pytest.approx(0.005)
    assert np.max(np.abs(aligned.signals["temperature"].errors)) == pytest.approx(0.05)
    assert np.max(np.abs(aligned.signals["soc"].errors)) == pytest.approx(0.05)
    assert np.max(np.abs(aligned.signals["soh"].errors)) == pytest.approx(0.0)


def test_interpolate_linear_twin_gives_zero_error(twin_result_five, ecu_result_five):
    """The twin is linear, so interpolation onto the ECU timestamps is exact."""
    aligned = align(twin_result_five, ecu_result_five, ALIGN_INTERPOLATE)

    assert aligned.alignment_method == "interpolate"
    # Last ECU sample (t=0.45) is outside the twin's range (0.0–0.4) → NaN.
    assert aligned.n_matched == 4
    for name in ("voltage", "current", "temperature", "soc"):
        errs = aligned.signals[name].errors
        # Exclude the last (NaN) sample.
        finite = errs[np.isfinite(errs)]
        assert np.allclose(finite, 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# ECU is the reference timeline (never dropped)
# ---------------------------------------------------------------------------
def test_ecu_timestamps_are_reference(twin_result_five, ecu_result_five):
    aligned = align(twin_result_five, ecu_result_five, ALIGN_NEAREST)
    np.testing.assert_allclose(
        aligned.timestamps,
        ecu_result_five.df["timestamp"].to_numpy(),
    )


# ---------------------------------------------------------------------------
# Partial-overlap: ECU has only 3 signals (no SoC/SoH)
# ---------------------------------------------------------------------------
def test_partial_overlap_only_common_signals_aligned(twin_result_five,
                                                       ecu_result_three):
    aligned = align(twin_result_five, ecu_result_three, ALIGN_NEAREST)

    assert set(aligned.signal_names) == {"voltage", "current", "temperature"}
    # SoC / SoH are silently skipped (not in ECU).
    assert "soc" not in aligned.signals
    assert "soh" not in aligned.signals


# ---------------------------------------------------------------------------
# axis_kind: propagated from the ECU (reference-timeline) side
# ---------------------------------------------------------------------------
def test_axis_kind_defaults_timestamp(twin_result_five, ecu_result_five):
    aligned = align(twin_result_five, ecu_result_five, ALIGN_NEAREST)
    assert aligned.axis_kind == "timestamp"


def test_axis_kind_sequence_propagates(make_load_result):
    twin = make_load_result(
        "twin",
        {"timestamp": [0.0, 1.0], "voltage": [3.3, 3.4]},
        axis_kind="sequence",
    )
    ecu = make_load_result(
        "ecu",
        {"timestamp": [0.0, 1.0], "voltage": [3.31, 3.41]},
        axis_kind="sequence",
    )
    aligned = align(twin, ecu, ALIGN_NEAREST)
    assert aligned.axis_kind == "sequence"


def test_axis_kind_mismatch_warns_and_uses_ecu(make_load_result):
    twin = make_load_result(
        "twin",
        {"timestamp": [0.0, 1.0], "voltage": [3.3, 3.4]},
        axis_kind="sequence",
    )
    ecu = make_load_result(
        "ecu",
        {"timestamp": [0.0, 1.0], "voltage": [3.31, 3.41]},
        axis_kind="timestamp",
    )
    aligned = align(twin, ecu, ALIGN_NEAREST)
    assert aligned.axis_kind == "timestamp"  # ECU wins
    assert any("disagree on axis kind" in w for w in aligned.warnings)


# ---------------------------------------------------------------------------
# Errors: no common signals / no time overlap
# ---------------------------------------------------------------------------
def test_no_common_signals_raises(make_load_result):
    twin = make_load_result(
        "twin",
        {"timestamp": [0.0, 0.1],
         "voltage": [3.3, 3.4], "current": [1.2, 1.3],
         "temperature": [25.0, 26.0], "soc": [80.0, 79.0],
         "soh": [95.0, 95.0]},
    )
    # ECU has none of those — only a column the alias system normally
    # wouldn't know about, but here we deliberately drop canonicals.
    ecu = make_load_result(
        "ecu",
        {"timestamp": [0.05, 0.15],
         "voltage": [3.31, 3.41]},  # only one
    )
    # Build a deliberately-mismatched ECU by overwriting columns_found.
    # Simpler: use only timestamps + a non-canonical column.
    import pandas as pd
    df = pd.DataFrame({"timestamp": [0.05, 0.15],
                       "pressure": [0.1, 0.2]})
    from data_loader import LoadResult
    ecu = LoadResult(
        df=df, source_label="ecu", row_count=2,
        columns_found=[], columns_missing=[],
        source_columns=["timestamp", "pressure"],
        time_range=(0.05, 0.15), warnings=[],
    )
    with pytest.raises(AlignmentError, match="No common signal"):
        align(twin, ecu, ALIGN_NEAREST)


def test_no_time_range_overlap_raises(make_load_result):
    twin = make_load_result(
        "twin",
        {"timestamp": [0.0, 0.1, 0.2],
         "voltage": [3.3, 3.4, 3.5],
         "current": [1.2, 1.3, 1.4],
         "temperature": [25.0, 26.0, 27.0],
         "soc": [80.0, 79.0, 78.0],
         "soh": [95.0, 95.0, 95.0]},
    )
    ecu = make_load_result(
        "ecu",
        {"timestamp": [10.0, 10.1, 10.2],
         "voltage": [3.31, 3.41, 3.51],
         "current": [1.21, 1.31, 1.41],
         "temperature": [25.1, 26.1, 27.1],
         "soc": [79.9, 78.9, 77.9],
         "soh": [95.0, 95.0, 95.0]},
    )
    with pytest.raises(AlignmentError, match="No time-range overlap"):
        align(twin, ecu, ALIGN_NEAREST)


# ---------------------------------------------------------------------------
# Unknown method falls back to nearest with a warning
# ---------------------------------------------------------------------------
def test_unknown_method_falls_back_to_nearest(twin_result_five,
                                                ecu_result_five):
    aligned = align(twin_result_five, ecu_result_five, method="bogus")
    assert aligned.alignment_method == "nearest"
    assert any("Unknown alignment method" in w for w in aligned.warnings)


# ---------------------------------------------------------------------------
# Nearest: large gaps → NaN-marked
# ---------------------------------------------------------------------------
def test_nearest_outside_twin_range_marked_nan(make_load_result):
    twin = make_load_result(
        "twin",
        {"timestamp": [0.0, 0.1, 0.2],
         "voltage": [3.3, 3.4, 3.5],
         "current": [1.2, 1.3, 1.4],
         "temperature": [25.0, 26.0, 27.0],
         "soc": [80.0, 79.0, 78.0],
         "soh": [95.0, 95.0, 95.0]},
    )
    ecu = make_load_result(
        "ecu",
        {"timestamp": [0.05, 0.15, 5.0],   # last is far outside
         "voltage": [3.305, 3.405, 9.0],
         "current": [1.205, 1.305, 9.0],
         "temperature": [25.05, 26.05, 9.0],
         "soc": [79.95, 78.95, 9.0],
         "soh": [95.0, 95.0, 95.0]},
    )
    aligned = align(twin, ecu, ALIGN_NEAREST)
    # The last ECU sample (t=5.0) should have NaN twin values.
    for sig in aligned.signals.values():
        assert np.isnan(sig.twin_values[-1])
    assert aligned.n_matched < aligned.n_total
    assert any("could not be matched" in w for w in aligned.warnings)


# ---------------------------------------------------------------------------
# Interpolate: out-of-range samples → NaN
# ---------------------------------------------------------------------------
def test_interpolate_out_of_range_marked_nan(make_load_result):
    twin = make_load_result(
        "twin",
        {"timestamp": [0.0, 0.1, 0.2],
         "voltage": [3.3, 3.4, 3.5],
         "current": [1.2, 1.3, 1.4],
         "temperature": [25.0, 26.0, 27.0],
         "soc": [80.0, 79.0, 78.0],
         "soh": [95.0, 95.0, 95.0]},
    )
    ecu = make_load_result(
        "ecu",
        {"timestamp": [-0.5, 0.05, 0.5],   # first and last are outside
         "voltage": [9.0, 3.305, 9.0],
         "current": [9.0, 1.205, 9.0],
         "temperature": [9.0, 25.05, 9.0],
         "soc": [9.0, 79.95, 9.0],
         "soh": [95.0, 95.0, 95.0]},
    )
    aligned = align(twin, ecu, ALIGN_INTERPOLATE)
    # Only the middle sample (t=0.05) is within the twin range.
    assert aligned.n_matched == 1
    for sig in aligned.signals.values():
        assert np.isnan(sig.twin_values[0])
        assert np.isnan(sig.twin_values[2])
        assert not np.isnan(sig.twin_values[1])


# ---------------------------------------------------------------------------
# Id-aware matching: shifted live streams (the bug being fixed)
# ---------------------------------------------------------------------------
def test_align_by_id_resolves_shifted_streams_no_overlap(make_load_result):
    """Twin and ECU carry the same ids but arrive with a 600 ms delivery
    shift, so their timestamp ranges don't overlap at all.  Before the
    fix this raised ``No time-range overlap``; now the ids pair exactly,
    alignment succeeds, and the "unreliable gap" warning does NOT fire
    (the wall-clock shift between id-matched pairs is not an alignment-
    quality signal)."""
    twin = make_load_result(
        "twin",
        {"timestamp": [0.0, 100.0, 200.0],
         "voltage": [3.30, 3.40, 3.50],
         "id": ["1", "2", "3"]},
    )
    ecu = make_load_result(
        "ecu",
        {"timestamp": [600.0, 700.0, 800.0],
         "voltage": [3.31, 3.41, 3.51],
         "id": ["1", "2", "3"]},
    )
    aligned = align(twin, ecu, ALIGN_NEAREST)
    assert aligned.n_total == 3
    assert aligned.n_matched == 3
    # By-id pairing: ECU id "1" -> twin v=3.30 (error 0.01), NOT the
    # nearest-in-time twin sample (ts 200, v 3.50, which would give -0.19).
    np.testing.assert_allclose(
        aligned.signals["voltage"].twin_values, [3.30, 3.40, 3.50],
    )
    np.testing.assert_allclose(
        aligned.signals["voltage"].errors, [0.01, 0.01, 0.01],
    )
    assert not any("unreliable" in w for w in aligned.warnings)
    assert not any("could not be matched" in w for w in aligned.warnings)


def test_align_by_id_overrides_nearest_timestamp_when_ranges_overlap(
    make_load_result,
):
    """Even when the timestamp ranges DO overlap, ids must win: ECU ts 90
    is nearest twin ts 100 (v 3.40), but ECU id "a" maps to twin ts 0
    (v 3.30). Without id-aware matching the engine silently paired by
    time and got the wrong twin sample."""
    twin = make_load_result(
        "twin",
        {"timestamp": [0.0, 100.0, 200.0, 300.0],
         "voltage": [3.30, 3.40, 3.50, 3.60],
         "id": ["a", "b", "c", "d"]},
    )
    ecu = make_load_result(
        "ecu",
        {"timestamp": [90.0, 110.0],
         "voltage": [9.99, 9.99],
         "id": ["a", "c"]},
    )
    aligned = align(twin, ecu, ALIGN_NEAREST)
    # id "a" -> twin idx 0 (v 3.30 @ ts 0); id "c" -> twin idx 2 (v 3.50 @ ts 200).
    np.testing.assert_allclose(
        aligned.signals["voltage"].twin_values, [3.30, 3.50],
    )


def test_align_by_id_unmatched_ecu_id_marked_nan(make_load_result):
    """Some ECU ids match a twin row, some don't. The matched ones pair
    by id; the unmatched one (its twin id isn't in the buffer yet, or is
    genuinely missing) is left NaN — NOT timestamp-matched to a wrong
    twin — and counted as unmatched. A late twin id will pair it on a
    future tick (see ``test_align_unmatched_ecu_waits_for_twin_id``)."""
    twin = make_load_result(
        "twin",
        {"timestamp": [0.0, 100.0, 200.0],
         "voltage": [3.30, 3.40, 3.50],
         "id": ["1", "2", "3"]},
    )
    ecu = make_load_result(
        "ecu",
        {"timestamp": [600.0, 700.0, 800.0],
         "voltage": [3.31, 3.41, 9.99],
         "id": ["1", "2", "999"]},   # "999" has no twin counterpart
    )
    aligned = align(twin, ecu, ALIGN_NEAREST)
    assert aligned.n_matched == 2
    # First two pair by id with the half-step error.
    np.testing.assert_allclose(
        aligned.signals["voltage"].twin_values[:2], [3.30, 3.40],
    )
    # The unmatched row's twin value is NaN.
    assert np.isnan(aligned.signals["voltage"].twin_values[2])
    assert any("could not be matched" in w for w in aligned.warnings)


def test_align_by_id_no_match_and_no_overlap_still_raises(make_load_result):
    """Ids are present on both sides but none match, AND the timestamp
    ranges don't overlap — genuinely unalignable. The deferred refusal
    fires after the by-id attempt produces nothing."""
    twin = make_load_result(
        "twin",
        {"timestamp": [0.0, 100.0, 200.0],
         "voltage": [3.30, 3.40, 3.50],
         "id": ["1", "2", "3"]},
    )
    ecu = make_load_result(
        "ecu",
        {"timestamp": [600.0, 700.0, 800.0],
         "voltage": [3.31, 3.41, 3.51],
         "id": ["100", "101", "102"]},   # no id overlap with twin
    )
    with pytest.raises(AlignmentError, match="No matches by id"):
        align(twin, ecu, ALIGN_NEAREST)


def test_align_by_id_null_ecu_id_unmatched(make_load_result):
    """An ECU row whose own ``id`` is null while ids are otherwise in
    use has no id to pair by, so it is left unmatched (NaN) rather than
    timestamp-matched to a wrong twin. Matched rows still pair by id."""
    twin = make_load_result(
        "twin",
        {"timestamp": [0.0, 100.0],
         "voltage": [3.30, 3.40],
         "id": ["1", "2"]},
    )
    ecu = make_load_result(
        "ecu",
        {"timestamp": [105.0, 600.0],
         "voltage": [3.41, 9.99],
         "id": ["2", None]},   # first matches by id; second has no id
    )
    aligned = align(twin, ecu, ALIGN_NEAREST)
    # ECU id "2" (@ ts 105) -> twin idx 1 (v 3.40 @ ts 100), error 0.01.
    np.testing.assert_allclose(aligned.signals["voltage"].twin_values[0], 3.40)
    # ECU row with null id -> unmatched -> NaN.
    assert np.isnan(aligned.signals["voltage"].twin_values[1])
    assert aligned.n_matched == 1


def test_align_unmatched_ecu_waits_for_twin_id(make_load_result):
    """The regression guard for the user's bug: an ECU row whose id has
    no twin counterpart in the buffer yet must stay NaN (NOT get
    timestamp-matched to a wrong twin). When the twin id later arrives
    within the buffer window, the same ECU row pairs exactly — the
    controller's next ``_tick`` re-runs ``align()`` and resolves it.

    The ECU s3 sample is deliberately placed at ts=210 (right next to
    twin s2 @ ts=200) so that WITHOUT the fix the old timestamp
    fallback would match it to twin s2 (gap 10 < 2×100) and emit a
    WRONG pair; with the fix it stays NaN until twin s3 arrives."""
    # --- Phase A: twin knows s0,s1,s2; ECU has s0,s1,s3 (s3's twin
    # not arrived yet). ECU s3 must be NaN, not twin s2's value. ---
    twin_a = make_load_result(
        "twin",
        {"timestamp": [0.0, 100.0, 200.0],
         "voltage": [3.30, 3.40, 3.50],
         "id": ["s0", "s1", "s2"]},
    )
    ecu = make_load_result(
        "ecu",
        {"timestamp": [10.0, 110.0, 210.0],
         "voltage": [9.99, 9.99, 9.99],
         "id": ["s0", "s1", "s3"]},
    )
    aligned_a = align(twin_a, ecu, ALIGN_NEAREST)
    assert aligned_a.n_matched == 2
    np.testing.assert_allclose(
        aligned_a.signals["voltage"].twin_values[:2], [3.30, 3.40],
    )
    # The crucial guard: ECU s3 is NaN, NOT the nearest-in-time twin
    # value (3.50 from twin s2 @200) that the old fallback would give.
    assert np.isnan(aligned_a.signals["voltage"].twin_values[2])
    assert np.nan != 3.50  # sanity (the old wrong match would be 3.50)

    # --- Phase B: twin s3 arrives (still inside the buffer window).
    # The same ECU rows now align by id, s3 included. ---
    twin_b = make_load_result(
        "twin",
        {"timestamp": [0.0, 100.0, 200.0, 220.0],
         "voltage": [3.30, 3.40, 3.50, 3.60],
         "id": ["s0", "s1", "s2", "s3"]},
    )
    aligned_b = align(twin_b, ecu, ALIGN_NEAREST)
    assert aligned_b.n_matched == 3
    np.testing.assert_allclose(
        aligned_b.signals["voltage"].twin_values,
        [3.30, 3.40, 3.60],   # ECU s3 -> twin s3 (v 3.60), not twin s2
    )


def test_align_by_id_offline_no_id_column_unchanged(twin_result_five,
                                                      ecu_result_five):
    """The offline fixtures have no ``id`` column, so the id-aware branch
    must be a no-op: n_matched and max_delta_t are identical to the
    historical pure-timestamp result."""
    aligned = align(twin_result_five, ecu_result_five, ALIGN_NEAREST)
    assert aligned.n_matched == 5
    assert aligned.max_delta_t == pytest.approx(0.05, abs=1e-9)
    assert not any("unreliable" in w for w in aligned.warnings)
"""Unit tests for ``live_accumulator.py`` — the live ring buffer.

Pure Python: no Qt, no paho. Builds :class:`LiveSample` objects in
memory and feeds them into a :class:`LiveBuffer`, then asserts on the
buffer's row counts and the ``LoadResult`` snapshots it produces.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from live_accumulator import LiveBuffer, _SourceBuffer
from live_schema import LiveSample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _twin_sample(t: float, v: float = 3.3, sample_id=None) -> LiveSample:
    s = LiveSample(
        source_label="twin",
        columns={"timestamp": float(t), "voltage": float(v)},
    )
    s.id = str(sample_id) if sample_id is not None else None
    return s


def _ecu_sample(t: float, v: float = 3.31, sample_id=None) -> LiveSample:
    s = LiveSample(
        source_label="ecu",
        columns={"timestamp": float(t), "voltage": float(v)},
    )
    s.id = str(sample_id) if sample_id is not None else None
    return s


# ---------------------------------------------------------------------------
# Basic append
# ---------------------------------------------------------------------------
class TestAppend:
    def test_single_twin_sample_increments_count(self):
        buf = LiveBuffer()
        buf.add_sample(_twin_sample(0.0))
        assert buf.twin_count == 1
        assert buf.ecu_count == 0

    def test_unknown_source_label_is_ignored(self):
        buf = LiveBuffer()
        s = LiveSample("unknown", {"timestamp": 0.0})
        buf.add_sample(s)
        assert buf.sample_counts() == (0, 0)

    def test_add_samples_batch(self):
        buf = LiveBuffer()
        buf.add_samples([_twin_sample(0.0), _twin_sample(0.1)])
        assert buf.twin_count == 2


# ---------------------------------------------------------------------------
# Snapshot → LoadResult
# ---------------------------------------------------------------------------
class TestBuildResults:
    def test_empty_buffers_return_none_pair(self):
        buf = LiveBuffer()
        twin_lr, ecu_lr = buf.build_results()
        assert twin_lr is None
        assert ecu_lr is None

    def test_one_side_empty_returns_none_pair(self):
        buf = LiveBuffer()
        buf.add_sample(_twin_sample(0.0))
        twin_lr, ecu_lr = buf.build_results()
        # Only twin has data — ecu is empty, so the pair is (None, None).
        assert twin_lr is None
        assert ecu_lr is None

    def test_both_sides_full_returns_load_results(self):
        buf = LiveBuffer()
        buf.add_sample(_twin_sample(0.0))
        buf.add_sample(_ecu_sample(0.0))
        twin_lr, ecu_lr = buf.build_results()
        assert twin_lr is not None and ecu_lr is not None
        assert twin_lr.source_label == "twin"
        assert ecu_lr.source_label == "ecu"
        assert twin_lr.row_count == 1
        assert ecu_lr.row_count == 1
        assert twin_lr.columns_found == ["timestamp", "voltage"]
        assert "current" in twin_lr.columns_missing

    def test_snapshot_is_a_copy(self):
        # Build a result, then append more — the snapshot should be
        # unaffected (the DataFrame inside the LoadResult must not
        # mutate when the buffer grows).
        buf = LiveBuffer()
        buf.add_sample(_twin_sample(0.0))
        buf.add_sample(_ecu_sample(0.0))
        twin_lr, _ = buf.build_results()
        snapshot_rows = twin_lr.row_count
        buf.add_sample(_twin_sample(0.1))
        assert twin_lr.row_count == snapshot_rows  # unchanged

    def test_time_range_computed(self):
        buf = LiveBuffer()
        buf.add_sample(_twin_sample(5.0))
        buf.add_sample(_twin_sample(10.0))
        buf.add_sample(_ecu_sample(7.0))
        twin_lr, _ = buf.build_results()
        assert twin_lr.time_range == (5.0, 10.0)


# ---------------------------------------------------------------------------
# Ring-buffer trimming
# ---------------------------------------------------------------------------
class TestTrimming:
    def test_row_cap_drops_oldest(self):
        buf = LiveBuffer(max_rows=3, max_age_s=10_000)
        for i in range(5):
            buf.add_sample(_twin_sample(float(i)))
        assert buf.twin_count == 3
        # Oldest dropped → timestamps [2, 3, 4].
        twin_lr, _ = buf.build_results()
        # Need an ecu sample for the snapshot to return data; add one.
        buf.add_sample(_ecu_sample(2.0))
        twin_lr, _ = buf.build_results()
        assert list(twin_lr.df["timestamp"]) == [2.0, 3.0, 4.0]

    def test_age_cap_drops_old_rows(self):
        buf = LiveBuffer(max_rows=10_000, max_age_s=2.0)
        buf.add_sample(_twin_sample(0.0))
        buf.add_sample(_twin_sample(1.0))
        buf.add_sample(_twin_sample(10.0))  # newest — cutoff becomes 8.0
        buf.add_sample(_ecu_sample(10.0))   # ecu present so snapshot works
        twin_lr, _ = buf.build_results()
        # Rows at t=0.0 and t=1.0 are older than 10.0 - 2.0 = 8.0 → dropped.
        assert list(twin_lr.df["timestamp"]) == [10.0]

    def test_duplicate_timestamp_deduped(self):
        buf = LiveBuffer()
        buf.add_sample(_twin_sample(1.0, v=3.3))
        buf.add_sample(_twin_sample(1.0, v=3.4))  # same ts, later wins
        buf.add_sample(_ecu_sample(1.0))
        twin_lr, _ = buf.build_results()
        assert twin_lr.row_count == 1
        # keep="last" → the second voltage value survives.
        assert twin_lr.df["voltage"].iloc[0] == 3.4

    def test_out_of_order_sorted_ascending(self):
        buf = LiveBuffer()
        buf.add_sample(_twin_sample(2.0))
        buf.add_sample(_twin_sample(0.0))
        buf.add_sample(_twin_sample(1.0))
        buf.add_sample(_ecu_sample(0.0))
        twin_lr, _ = buf.build_results()
        assert list(twin_lr.df["timestamp"]) == [0.0, 1.0, 2.0]


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------
class TestClear:
    def test_clear_empties_both_buffers(self):
        buf = LiveBuffer()
        buf.add_sample(_twin_sample(0.0))
        buf.add_sample(_ecu_sample(0.0))
        assert buf.sample_counts() == (1, 1)
        buf.clear()
        assert buf.sample_counts() == (0, 0)


# ---------------------------------------------------------------------------
# Partial signals (NaN handling)
# ---------------------------------------------------------------------------
class TestPartialSignals:
    def test_missing_signal_absent_from_columns(self):
        # A signal never carried by any sample is not in the buffer's
        # columns at all (pandas concat only unions seen columns); it
        # therefore appears in columns_missing, not columns_found.
        buf = LiveBuffer()
        buf.add_sample(LiveSample(
            "twin", {"timestamp": 0.0, "voltage": 3.3},
        ))
        buf.add_sample(_ecu_sample(0.0))
        twin_lr, _ = buf.build_results()
        assert "current" not in twin_lr.df.columns
        assert "current" in twin_lr.columns_missing
        assert twin_lr.df["voltage"].iloc[0] == 3.3

    def test_later_sample_widens_columns_to_nan(self):
        # A signal introduced by a later sample widens the buffer;
        # earlier rows get NaN for it (pandas fills the union).
        buf = LiveBuffer()
        buf.add_sample(LiveSample("twin", {"timestamp": 0.0, "voltage": 3.3}))
        buf.add_sample(LiveSample("twin", {"timestamp": 0.1, "current": 1.2}))
        buf.add_sample(LiveSample("ecu", {"timestamp": 0.0, "voltage": 3.3}))
        twin_lr, _ = buf.build_results()
        assert "current" in twin_lr.df.columns
        assert math.isnan(twin_lr.df["current"].iloc[0])   # row 0 had no current
        assert twin_lr.df["current"].iloc[1] == 1.2        # row 1 did

    def test_full_signal_set_in_load_result_columns_found(self):
        buf = LiveBuffer()
        full = {"timestamp": 0.0, "voltage": 3.3, "current": 1.0,
                "temperature": 25.0, "soc": 80.0, "soh": 99.0}
        buf.add_sample(LiveSample("twin", full))
        buf.add_sample(_ecu_sample(0.0))
        twin_lr, _ = buf.build_results()
        assert set(twin_lr.columns_found) == {
            "timestamp", "voltage", "current", "temperature", "soc", "soh",
        }
        assert twin_lr.columns_missing == []


# ---------------------------------------------------------------------------
# QoS-1 deduplication by id
# ---------------------------------------------------------------------------
class TestQos1Dedup:
    def test_redelivered_id_dropped_keep_first(self):
        # QoS-1 at-least-once: the same id arrives twice. The buffer
        # must keep only the FIRST delivery and drop the redelivery.
        buf = LiveBuffer()
        buf.add_sample(_twin_sample(0.0, v=3.30, sample_id="s1"))
        buf.add_sample(_twin_sample(0.0, v=3.99, sample_id="s1"))  # dup
        buf.add_sample(_ecu_sample(0.0))
        twin_lr, _ = buf.build_results()
        assert twin_lr.row_count == 1
        # keep="first" → the original voltage survives, not the dup.
        assert twin_lr.df["voltage"].iloc[0] == 3.30

    def test_distinct_ids_both_kept(self):
        buf = LiveBuffer()
        buf.add_sample(_twin_sample(0.0, v=3.30, sample_id="s1"))
        buf.add_sample(_twin_sample(0.1, v=3.40, sample_id="s2"))
        buf.add_sample(_ecu_sample(0.0))
        twin_lr, _ = buf.build_results()
        assert twin_lr.row_count == 2

    def test_no_id_falls_back_to_timestamp_dedup(self):
        # Without an id, the buffer dedupes on timestamp (keep last).
        buf = LiveBuffer()
        buf.add_sample(_twin_sample(1.0, v=3.30))
        buf.add_sample(_twin_sample(1.0, v=3.40))  # same ts, no id
        buf.add_sample(_ecu_sample(1.0))
        twin_lr, _ = buf.build_results()
        assert twin_lr.row_count == 1
        assert twin_lr.df["voltage"].iloc[0] == 3.40  # keep="last"

    def test_id_column_not_in_snapshot(self):
        # The internal id column must not leak into the LoadResult
        # snapshot — downstream pipeline columns stay canonical.
        buf = LiveBuffer()
        buf.add_sample(_twin_sample(0.0, sample_id="s1"))
        buf.add_sample(_ecu_sample(0.0, sample_id="e1"))
        twin_lr, ecu_lr = buf.build_results()
        assert "id" not in twin_lr.df.columns
        assert "id" not in ecu_lr.df.columns

    def test_mixed_id_and_no_id_in_same_buffer(self):
        # A stream that starts without ids then switches to ids (or
        # vice-versa): rows with ids dedupe by id; rows without dedupe
        # by timestamp. Both coexist without one clobbering the other.
        buf = LiveBuffer()
        buf.add_sample(_twin_sample(0.0, v=3.0))                  # no id
        buf.add_sample(_twin_sample(0.0, v=3.1))                  # no id, dup ts → keep last
        buf.add_sample(_twin_sample(0.1, v=3.2, sample_id="x"))    # id
        buf.add_sample(_twin_sample(0.1, v=3.9, sample_id="x"))    # id dup → drop
        buf.add_sample(_ecu_sample(0.0))
        twin_lr, _ = buf.build_results()
        # 2 rows survive: the no-id row at t=0.0 (last wins, v=3.1)
        # and the id row at t=0.1 (first wins, v=3.2).
        assert twin_lr.row_count == 2
        assert list(twin_lr.df["voltage"]) == [3.1, 3.2]

    def test_int_id_stringified(self):
        # An int id from the wire is stored as a string so it can be
        # compared against str ids without dtype collisions.
        buf = LiveBuffer()
        buf.add_sample(_twin_sample(0.0, sample_id=1))
        buf.add_sample(_twin_sample(0.0, sample_id="1"))  # str "1" == int 1
        buf.add_sample(_ecu_sample(0.0))
        twin_lr, _ = buf.build_results()
        assert twin_lr.row_count == 1  # the str "1" dup is dropped
"""Append-only ring buffer that turns a live MQTT stream into ``LoadResult``s.

The live path reuses the *entire* offline comparison engine (alignment,
statistics, plots, export). The only thing it doesn't have is a CSV file
on disk — the samples arrive one at a time over MQTT. This module is the
adapter: it accumulates parsed :class:`live_schema.LiveSample` objects
into two growing :class:`pandas.DataFrame` s (one per source) and, on
demand, snapshots the current contents into a :class:`data_loader.LoadResult`
that the rest of the pipeline can consume exactly as if it had come from
``data_loader.load_csv``.

Design notes
------------
**Why a ring buffer, not unbounded append?**
A long live session (hours, 10 Hz) would otherwise grow without bound.
The buffer trims to two independent caps, either of which trips first:

  - ``max_rows``   — keep at most N most recent rows per source.
  - ``max_age_ms`` — drop rows whose timestamp is older than
    ``newest_t - max_age_ms``.

Defaults (10 000 rows, 600 000 ms = 10 minutes) bound a 10 Hz stream to
~17 minutes of history and ~80 KB per column — comfortable on any modern
machine. Both are configurable per-source at construction.

**Timestamp unit: milliseconds.** ``timestamp`` values flowing through
this buffer are real Unix epoch milliseconds (see
``synthetic_data_generator``'s "Timestamp convention" section) — that's
why ``max_age_ms`` is named/scaled in milliseconds rather than the
seconds ``max_age_s`` this module used before real-time-based live
testing existed. A real ECU/Simulink publisher must emit the same unit.

**Why two DataFrames, not one?**
The offline pipeline is built around two separate ``LoadResult``s
(twin + ECU); keeping two buffers parallels that and avoids a
``source`` column that would have to be filtered out before every
align call.

**Why ``float64`` and sorted?**
The alignment engine assumes both ``timestamp`` columns are float64 and
ascending. We enforce that here on every append (timestamps usually
arrive in order, but an out-of-order or duplicate packet would otherwise
trip the alignment engine's overlap check). Sort + drop-duplicates is a
cheap ``O(n log n)`` operation at the cadence the controller ticks.

**Pure Python.**
No Qt, no paho — the controller (``live_controller.py``) is the only
piece that touches Qt and MQTT. This keeps the buffer easy to unit-test
in isolation.

**QoS-1 deduplication + id-aware alignment.**
``live_schema.parse_message`` extracts an optional publisher-assigned
``id`` from each sample (keys ``id`` / ``seq`` / ``msg_id`` / ...). When
present, the buffer stores it in an internal ``id`` column and dedupes
on it with ``keep="first"`` so an at-least-once redelivery is dropped.
When no ``id`` is carried, the buffer falls back to timestamp-based
dedup (``keep="last"``). The ``id`` column is **retained** in the
snapshot returned by ``snapshot()`` so the alignment engine can pair
twin and ECU rows by id when one stream is delivered with a shift —
the matching id only has to arrive before the ring buffer drops it.
The offline CSV path never carries an ``id`` column, so the extra
column is a live-only addition that downstream code (preview table,
export) tolerates alongside the canonical columns.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from data_loader import build_load_result
from live_schema import LiveSample


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MAX_ROWS: int = 10_000
DEFAULT_MAX_AGE_MS: float = 600_000.0   # 10 minutes, in Unix-epoch milliseconds

# Canonical column ordering — kept in sync with data_loader.
_CANONICAL_COLUMNS: list[str] = [
    "timestamp", "voltage", "current", "temperature", "soc", "soh",
]
_SIGNAL_COLUMNS: list[str] = [
    "voltage", "current", "temperature", "soc", "soh",
]


# ---------------------------------------------------------------------------
# Per-source buffer
# ---------------------------------------------------------------------------
class _SourceBuffer:
    """Append-only buffer for one source (twin or ecu).

    Holds a single DataFrame, enforces the ring-buffer trim, and exposes
    a snapshot method. Kept private — the public API is
    :class:`LiveBuffer` which owns two of these.
    """

    def __init__(self, source_label: str, max_rows: int, max_age_ms: float):
        self.source_label = source_label
        self.max_rows = max_rows
        self.max_age_ms = max_age_ms
        # Empty DataFrame — columns are added on demand as the first
        # sample carrying each arrives (pandas concat unions columns).
        # This keeps ``build_load_result``'s ``columns_found`` honest:
        # it only reports signals that were actually *seen* in some
        # sample, not every canonical column padded with NaN.
        self._df = pd.DataFrame()
        # "timestamp" or "sequence" — set from each sample's own
        # LiveSample.axis_kind. A stream is one mode for its whole life,
        # so just remembering the latest write is correct.
        self.axis_kind = "timestamp"

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------
    def add(
        self,
        columns: dict,
        sample_id: Optional[str] = None,
        axis_kind: str = "timestamp",
    ) -> None:
        """Append one sample's columns (canonical name → value).

        Builds a one-row DataFrame from only the columns present in the
        sample, then concatenates. pandas fills missing columns with
        NaN on the union, so a later sample carrying a new signal
        transparently widens the buffer.  When ``sample_id`` is not
        ``None`` (QoS-1 dedup), an ``id`` column is added so ``_trim``
        can drop redeliveries.  ``axis_kind`` (from the sample's own
        ``LiveSample.axis_kind``) is remembered on the buffer so
        ``build_results()`` can pass it through to the ``LoadResult``.
        """
        row = {c: [v] for c, v in columns.items() if c in _CANONICAL_COLUMNS}
        if "timestamp" not in row:
            return  # defensive — live_schema should never let this through

        if sample_id is not None:
            row["id"] = [sample_id]

        self.axis_kind = axis_kind
        new_row = pd.DataFrame(row)
        self._df = pd.concat([self._df, new_row], ignore_index=True)
        self._trim()

    # ------------------------------------------------------------------
    # Trim
    # ------------------------------------------------------------------
    def _trim(self) -> None:
        """Apply the row-count and age caps, then sort + dedupe.

        Dedup strategy (QoS 1 at-least-once):
          - Rows with a non-null ``id``: drop duplicates on ``id``,
            **keep first** (the first delivery is the real one; a
            redelivery with the same id is dropped).
          - Rows without an ``id`` (``id`` column absent or NaN): fall
            back to timestamp dedup, **keep last** (live streams favour
            the newest value for a repeated timestamp).
        """
        df = self._df
        if df.empty or "timestamp" not in df.columns:
            return

        # --- Age cap -----------------------------------------------------
        ts = df["timestamp"]
        newest = float(ts.iloc[-1])
        if self.max_age_ms is not None and self.max_age_ms > 0:
            cutoff = newest - self.max_age_ms
            df = df[ts >= cutoff]

        # --- Row cap -----------------------------------------------------
        if len(df) > self.max_rows:
            df = df.iloc[-self.max_rows:]

        # --- Sort + dedupe -----------------------------------------------
        df = df.sort_values("timestamp", kind="mergesort")

        if "id" in df.columns and df["id"].notna().any():
            has_id = df["id"].notna()
            with_id = df[has_id].drop_duplicates(subset="id", keep="first")
            without_id = df[~has_id].drop_duplicates(
                subset="timestamp", keep="last",
            )
            df = pd.concat([with_id, without_id]) \
                   .sort_values("timestamp", kind="mergesort")
        else:
            df = df.drop_duplicates(subset="timestamp", keep="last")

        df = df.reset_index(drop=True)
        self._df = df

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------
    def snapshot(self) -> pd.DataFrame:
        """Return a *copy* of the current buffer as a clean DataFrame.

        The copy is important: the alignment engine may hold a reference
        to the DataFrame while the buffer keeps growing in the
        background. Without the copy, an in-place append to ``self._df``
        could mutate the snapshot the engine is reading. ``copy()`` is
        cheap on a 10 000-row frame (<1 ms) and runs at most a few times
        per second on the controller's tick.

        The internal ``id`` column (used for QoS-1 dedup) is **retained**
        so the alignment engine can pair twin and ECU rows by id when one
        stream is delivered with a shift relative to the other. On the
        offline CSV path no ``id`` column exists, so this is a live-only
        extra column that downstream code (preview table, export)
        tolerates alongside the canonical timestamp + signal columns.
        """
        return self._df.copy()

    @property
    def row_count(self) -> int:
        return len(self._df)

    @property
    def time_range(self):
        if self._df.empty:
            return (0.0, 0.0)
        return (
            float(self._df["timestamp"].iloc[0]),
            float(self._df["timestamp"].iloc[-1]),
        )


# ---------------------------------------------------------------------------
# Public API: LiveBuffer (two _SourceBuffers + a snapshot helper)
# ---------------------------------------------------------------------------
class LiveBuffer:
    """Two-source ring buffer that produces ``LoadResult`` snapshots.

    Owns one :class:`_SourceBuffer` per source label (``"twin"`` /
    ``"ecu"``). The live controller feeds :class:`LiveSample` objects
    in via :meth:`add_sample`; periodically it calls :meth:`build_results`
    on a tick to drive the alignment + statistics engines.
    """

    def __init__(
        self,
        max_rows: int = DEFAULT_MAX_ROWS,
        max_age_ms: float = DEFAULT_MAX_AGE_MS,
    ):
        self._buffers = {
            "twin": _SourceBuffer("twin", max_rows, max_age_ms),
            "ecu": _SourceBuffer("ecu", max_rows, max_age_ms),
        }

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------
    def add_sample(self, sample: LiveSample) -> None:
        """Route one parsed sample to its source buffer and append it."""
        buf = self._buffers.get(sample.source_label)
        if buf is None:
            # live_schema only ever produces "twin"/"ecu", but stay
            # defensive — a future source would otherwise silently drop.
            return
        buf.add(sample.columns, sample_id=sample.id, axis_kind=sample.axis_kind)

    def add_samples(self, samples: List[LiveSample]) -> None:
        """Convenience: feed a batch of samples in arrival order."""
        for s in samples:
            self.add_sample(s)

    # ------------------------------------------------------------------
    # Snapshot → LoadResult pair
    # ------------------------------------------------------------------
    def build_results(self) -> tuple:
        """Snapshot both buffers into ``(twin_result, ecu_result)``.

        Each ``LoadResult`` is built with :func:`data_loader.build_load_result`
        so its metadata (``columns_found`` / ``columns_missing`` /
        ``time_range`` / ``row_count``) matches what the offline path
        produces. Downstream code (alignment, statistics, plots) sees
        no difference between a live snapshot and a CSV load.

        Returns ``(None, None)`` if either buffer is empty — the
        controller uses that to skip the tick (no point aligning if we
        only have one stream's worth of data).
        """
        twin_df = self._buffers["twin"].snapshot()
        ecu_df = self._buffers["ecu"].snapshot()

        if twin_df.empty or ecu_df.empty:
            return None, None

        twin_result = build_load_result(
            df=twin_df,
            source_label="twin",
            source_columns=list(twin_df.columns),
            axis_kind=self._buffers["twin"].axis_kind,
        )
        ecu_result = build_load_result(
            df=ecu_df,
            source_label="ecu",
            source_columns=list(ecu_df.columns),
            axis_kind=self._buffers["ecu"].axis_kind,
        )
        return twin_result, ecu_result

    # ------------------------------------------------------------------
    # Diagnostics for the UI's status line
    # ------------------------------------------------------------------
    def sample_counts(self) -> tuple:
        """``(n_twin, n_ecu)`` current row counts for the status bar."""
        return (
            self._buffers["twin"].row_count,
            self._buffers["ecu"].row_count,
        )

    @property
    def twin_count(self) -> int:
        return self._buffers["twin"].row_count

    @property
    def ecu_count(self) -> int:
        return self._buffers["ecu"].row_count

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Drop every row from both buffers (used on disconnect)."""
        for buf in self._buffers.values():
            buf._df = pd.DataFrame()
            buf.axis_kind = "timestamp"
# Project Context — BrightSkies Digital Twin Validation Tool

Living document. Update when decisions change. Read before editing the
codebase.

---

## 1. Objective

A PyQt5 desktop tool that compares **Digital Twin** (Simulink) output
vs **ECU** (NXP MPC5744P) output for a battery monitoring / digital-twin
system. Two modes:

- **Offline** — load two CSV files, run alignment + statistics, show
  preview / stats / graphs / export.
- **Live** — stream both sources over MQTT into the same comparison
  engine in real time.

Five signals: **voltage, current, temperature, SoC, SoH**.

---

## 2. Architecture — hard rules

### 2.1 Pure-Python engines, Qt only in `back.py`

Backend engines (`data_loader`, `alignment_engine`,
`statistics_engine`, `export_engine`, `live_schema`,
`live_accumulator`) are **pure Python** — no Qt, no paho imports — so
they can be unit-tested from the CLI with pytest.

**Only `back.py`, `comparison_worker.py`, `plot_manager.py`,
`mqtt_worker.py`, and `live_controller.py` import Qt.** `PlotManager`
imports pyqtgraph (Qt-based, not pure Python) — that's expected.

If you add a new module that the engines depend on, it must be pure
Python too.

### 2.2 `f.ui` is the source of truth for the UI, NOT `f.py`

`f.py` is **pyuic5-generated**. Never hand-edit `f.py`. Edit `f.ui`
directly, then recompile:

```bash
source ../venv/bin/activate
pyuic5 f.ui -o f.py
```

Verify with `python -c "import f"` and run the test suite. A diff check
(`pyuic5 f.ui -o /tmp/check.py && diff /tmp/check.py f.py`) confirms no
hand-edits leaked in.

### 2.3 Long-lived worker threads, not per-run threads

`ComparisonWorker` and `LiveController` are **long-lived workers**:
created once, moved to a `QThread` with `moveToThread`, kept alive
across runs. We deliberately do **NOT** wire `finished → thread.quit /
deleteLater` (that pattern tears down on every run; we want the worker
to stay alive so cached `LoadResult`s / live buffers support re-runs
without re-I/O).

`MqttWorker` is also long-lived but torn down on disconnect (it owns a
network socket).

### 2.4 Self-connected slot signals for cross-thread invocation

To make a slot run on the worker's thread (not the GUI thread), emit a
**signal** that the worker self-connects to its own slot. A direct
Python method call would block the GUI thread. Pattern:

```python
self.run_signal = QtCore.pyqtSignal(...)        # declared on the worker
self.run_signal.connect(self.run)               # in __init__
# caller: self.worker.run_signal.emit(...)      # queued to worker thread
```

`ComparisonWorker` (run/rerun) and `LiveController` (ingest/start/
stop/reset) both use this pattern.

---

## 3. Module inventory

All files live in `final_tool/`.

| File | LOC | Role | Status |
|------|-----|------|--------|
| `f.ui` | — | Qt Designer UI source (7 tabs) | Done |
| `f.py` | ~800 | pyuic5-generated from `f.ui` | Done |
| `back.py` | ~720 | MainWindow — only Qt-touching wiring | Done |
| `data_loader.py` | ~360 | CSV loader + `build_load_result` helper | Done |
| `alignment_engine.py` | ~355 | nearest + interpolate alignment | Done |
| `statistics_engine.py` | ~352 | per-signal + global metrics | Done |
| `comparison_worker.py` | ~230 | QThread orchestrator (offline) | Done |
| `plot_manager.py` | ~330 | pyqtgraph 6-plot wrapper (5 overlays + error) | Done |
| `export_engine.py` | ~463 | CSV/Excel/PDF/HTML export | Done |
| `mqtt_worker.py` | ~188 | generic MQTT client | Done |
| `live_schema.py` | ~400 | JSON payload parser + topic routing + id | Done |
| `live_accumulator.py` | ~300 | ring buffer (max rows + max age) | Done |
| `live_controller.py` | ~200 | Qt glue: MQTT → comparison engine | Done |

Docs in `docs/` (per-module line-by-line explanations) and
`high_lvl_explaination.md` (architecture overview).

Tests in `tests/` — **237 pytest tests, all passing (~4s)**.

---

## 4. UI conventions

### 4.1 Tabs (7)

1. Load & Extract — file browse + Start Comparison
2. Preview — twin + ECU QTableWidgets
3. Statistics — 6 summary cards + per-signal + worst-mismatches tables
4. Graphs/Overlay — 6 pyqtgraph PlotWidgets: 5 unconditional per-signal
   overlays (V/I/T/SoC/SoH) + 1 selectable error plot
5. Config/Tolerance — 3 tolerance spinboxes + alignment combo + threshold warnings + Apply & Re-run
6. Report/Export — CSV/Excel data + PDF/HTML report
7. Live (MQTT) — broker config + topic roots + Auto-refresh + interval + Freeze/Snapshot

### 4.1a Graphs/Overlay tab — 5 unconditional overlays + 1 selectable error plot

Six `PlotWidget`s:

- `plotWidgetVoltage` / `Current` / `Temperature` / `Soc` / `Soh` — twin
  (cornflower blue) vs ECU (near-black) on its own dedicated plot,
  **unconditional**: each draws whenever that signal is present in the
  aligned data. There is no checkbox for these five — they always show.
- `plotWidgetError` — one error curve (ECU − twin) per signal ticked on
  its own selection row, positioned directly above the error plot (not
  above the whole tab, and not shared with the five overlay plots).

The "Signals to Include in Error Graph" row —
`checkBoxSignalVoltage` / `Current` / `Temperature` / `Soc` / `Soh` (all
five, independently), plus `checkBoxSignalSelectOnly` — only ever
affects `plotWidgetError`. `checkBoxSignalSelectOnly` forces radio-button
behaviour on top of the plain checkboxes — ticking one signal while it's
armed unchecks the rest, and arming it collapses whatever's currently
ticked down to one. `back.py`'s `_force_single_signal_selection`
implements this with `blockSignals` guards (see
`on_signal_selection_changed` / `on_select_only_one_toggled`).
`PlotManager._enabled_signals_set`/`update`'s `enabled_signals` parameter
is consulted **only** by `_plot_errors` — `_plot_overlay` (used for all
five individual plots) never looks at it.

Threshold-warning highlighting (dashed limit line + red flood-fill,
Config tab's `checkBoxEnableVoltageThreshold` / `Current`) is drawn on
the signal's own overlay plot — `PlotManager._apply_threshold` is called
once per signal (V/I/T/SoC/SoH) that has a configured threshold,
independent of the error graph's selection.

(This tab briefly went through a "2 shared plots" design — one combined
overlay for whatever's ticked, one combined error plot — before settling
back on 5 always-on individual overlays + 1 selectable error plot. If you
see references to `plotWidgetOverlay` anywhere, that's stale.)

### 4.1b Graphs tab: scrollable, not squeezed

Six full-size plots stacked in one tab don't fit any reasonable window
height without shrinking to unreadable slivers — so `tabGraphs` wraps its
`splitterGraphs` in a `QScrollArea` (`scrollAreaGraphs`, `widgetResizable
= true`), and every one of the six `PlotWidget`s has `minimumSize` height
`280` set in `f.ui`. The splitter still lets the user drag to give one
plot more room than another; once the *total* minimum height exceeds the
tab's visible area, the scroll area grows a vertical scrollbar instead of
pyqtgraph's plots being squashed below readability. Keep the 280px floor
(or raise it) on any new plot added here — dropping it is how this bug
comes back.

### 4.2 Tolerance spinboxes

- `decimals = 4`, `minimum = 0.0`, `maximum = 50.0`, `singleStep = 0.01`, default `2.0`.
- SoC / SoH have **no** tolerance spinboxes in the UI — they use
  `statistics_engine.DEFAULT_TOLERANCES`. `_read_config_settings()`
  deliberately omits them from the dict so the engine falls back.
- **Regression fixed:** previously the spinbox defaulted to 2 decimals
  and minimum 0.1, so a 3rd decimal was silently rejected and Apply &
  Re-run used 2.00. Do NOT remove `decimals=4` from `f.ui`.

### 4.3 Widget naming

`btnXxx` (buttons), `labelXxx` (labels), `lineEditXxx` (text inputs),
`spinBoxXxx` / `doubleSpinBoxXxx` (numeric), `checkBoxXxx`,
`comboBoxXxx`, `tableWidgetXxx`, `plotWidgetXxx`, `groupBoxXxx`,
`tabXxx`. Match this when adding widgets to `f.ui`.

---

## 5. Live mode — design decisions

### 5.1 Three-thread architecture

1. **paho's network thread** (owned by `MqttWorker`) — raw socket reads.
2. **LiveController's `QThread`** — parse + accumulate + align + stats
   (the `ingest` slot and the `QTimer`'s `_tick`).
3. **GUI thread** — receives `aligned_ready` / `stats_ready` /
   `sample_count_changed` and updates widgets only.

Separating the controller from paho means heavy compute (align + stats)
never blocks the network loop. Separating both from the GUI means the
GUI never stalls under a bursty stream.

### 5.2 Update trigger: periodic timer, not per-sample

A `QTimer` inside `LiveController` fires every `interval_ms` (default
500, user-configurable 50–10000). On each tick: snapshot buffers →
`align()` → `compute()` → emit results. Decoupling from the message
rate keeps the GUI stable.

The `QTimer` is created **inside** `_on_start` with `QTimer(self)` after
`moveToThread` — a cross-thread `QTimer` is a well-known Qt footgun
(timers belong to their creating thread).

### 5.3 `start` vs `reset` — critical separation

- `start(interval_ms, twin_root, ecu_root)` — sets roots + arms the
  timer. Does **NOT** clear the buffer.
- `reset()` — clears both buffers.

Why separated: `back.py` re-emits `start` when the user nudges the
interval spinbox mid-stream, and that must not wipe accumulated data.
Fresh connects emit `reset` then `start`.

### 5.4 Live results reuse offline populate methods

`LiveController.aligned_ready → _on_live_aligned` calls the **same**
`_populate_preview_table` + `plot_manager.update` that the offline
`on_comparison_finished` uses. `stats_ready → _on_live_stats` calls the
same `_populate_statistics_tab`. Live and offline are visual-identical —
no duplicate UI logic.

### 5.5 No auto tab-jump in live mode

Offline mode jumps to the Statistics tab on finish. Live mode does
**NOT** auto-jump tabs — at 500 ms cadence it would be jarring. The user
watches whatever tab they're on.

### 5.6 Freeze & Snapshot button

`on_live_snapshot` builds a one-off aligned+stats snapshot from the
current buffers and stashes it as `self.aligned_data` /
`self.stats_result` so the Export tab acts on the frozen state.

### 5.7 Sliding live plot window (not auto-range)

`PlotManager.update(..., live_window_ms=...)` — passed only from
`_on_live_aligned` (`back.py._LIVE_PLOT_WINDOW_MS`, default 60 000 ms) —
pins both Graphs-tab plots to `[latest_t - live_window_ms, latest_t]`
via `setXRange(..., padding=0)` **every tick**, instead of leaving
pyqtgraph's default auto-range in charge. Two bugs this fixes at once:

- **"Compresses instead of shifting."** Auto-range re-fits *every*
  accumulated sample on every redraw, so as the live buffer grows toward
  its cap (`live_accumulator.DEFAULT_MAX_AGE_MS`, 10 minutes) the whole
  history gets squeezed into the same view width instead of the view
  scrolling forward at a constant width.
- **"Graphs stop appearing after disconnect → reconnect."** pyqtgraph's
  auto-range gets permanently disabled the moment a user manually
  pans/zooms a plot, pinning it to that manual range. `reset` clears the
  buffer on reconnect, so a stale manual range from before the disconnect
  may no longer contain the fresh data at all — reading as "the graph
  went blank." Calling `setXRange` unconditionally every tick overrides
  any stuck manual range and snaps the view back onto live data.

Offline runs and Freeze/Snapshot pass `live_window_ms=None` (the
default) — they want the full accumulated range visible, not a scrolling
window.

---

## 6. Payload schema (live mode)

### 6.0 Timestamp unit — Unix epoch MILLISECONDS

**`t` / `timestamp` is a real Unix epoch timestamp in milliseconds** —
e.g. `1754748123456` for 2025-08-09T12:02:03.456Z (like JavaScript's
`Date.now()`). It's a `double` on the wire (JSON has no int64). This is
the single most important wire-format detail for anything that publishes
into the Live tab — see `synthetic_data_generator.py`'s "Timestamp
convention" module-docstring section for the full rationale, and the
copy-pasteable spec in this repo's PR/chat history for handing to a
publisher-side (Simulink/ECU) team.

Every part of the live pipeline that compares raw timestamp deltas is
scaled to this unit: `live_accumulator.DEFAULT_MAX_AGE_MS` (600 000 =
10 minutes, buffer aging window) and `alignment_engine._MAX_DELTA_T_WARN_MS`
(500 = half a second, nearest-match gap-warning threshold — cosmetic
only, not fatal). **A publisher that sends seconds instead of
milliseconds will not error, but will silently defeat the buffer's aging
window and spam the gap-warning** — see `alignment_engine.py`'s
`_MAX_DELTA_T_WARN_MS` comment for the (documented, accepted) offline/live
unit-scale caveat this creates for CSV-sourced runs, which stay in
whatever unit the CSV itself uses.

Do NOT confuse this with offline CSV timestamps, which are unaffected —
`data_loader`/`alignment_engine` are unit-agnostic and just use whatever
scale the CSV file already has (this project's own fixtures are small,
roughly-seconds-scale numbers).

### 6.1 JSON, two shapes

**Single sample:**
```json
{"t": 1754748123456, "v": 3.65, "i": -1.2, "temp": 28.1, "soc": 78.3, "soh": 99.0, "id": "s1"}
```

**Batch:**
```json
{"samples": [
    {"t": 1754748123400, "v": 3.65, "i": -1.2, "temp": 28.0, "soc": 78.2, "soh": 99.0, "id": "a"},
    {"t": 1754748123500, "v": 3.64, "i": -1.1, "temp": 28.1, "soc": 78.3, "soh": 99.0, "id": "b"}
]}
```

### 6.2 Alias resolution

Keys matched case-insensitively against `data_loader._ALIAS_MAP`
(`v`/`voltage`/`vbat` → `voltage`, etc.). The single-letter `t`
ambiguity rule: `t` → timestamp if a voltage alias was found, else →
temperature.

### 6.3 Deduplication id (QoS 1) + id-aware alignment

MQTT QoS 1 = **at-least-once** → broker may redeliver. Each sample may
carry an optional `id` (accepted keys: `id`, `seq`, `seq_id`, `msg_id`,
`message_id`, case-insensitive). The id is **stringified** so int `42`
and str `"42"` from different publishers don't collide on pandas dtype.

The accumulator dedupes by `id` with `keep="first"` (first delivery is
real; redelivery dropped). When no `id` is present, falls back to
timestamp dedup (`keep="last"`). The `id` column is **retained** in
`snapshot()` (no longer stripped) so the alignment engine can pair twin
and ECU rows by id when one stream is delivered with a shift relative
to the other — see §6.5. The `id` column is **not** a canonical signal,
so `build_load_result` keeps it out of `columns_found`/`columns_missing`;
downstream code (preview table, export) tolerates it alongside the
canonical timestamp + signal columns. (Offline CSVs never carry an
`id` column — see §6.6 — so this is a live-only addition.)

**Do NOT remove the id field or the stringification — QoS 1 redelivery
would double-count samples.**

### 6.4 Timestamp is mandatory

A sample without a timestamp raises `LiveSchemaError` (hard fail). A
non-numeric signal value becomes `NaN` (soft fail — the sample's
timestamp + other valid signals still get recorded).

### 6.5 Id-aware alignment — robustness to delivery shifts

When both twin and ECU snapshots carry an `id` column (the live path
with QoS-1 dedup, §6.3), `alignment_engine.align()` pairs each ECU row
to the twin row sharing its `id` **before** falling back to
nearest-timestamp matching. This resolves a *delivery shift* between
the two streams exactly: a twin sample and an ECU sample produced for
the same logical instant may arrive with different wall-clock
timestamps (one path lagging), but matching them by `id` sidesteps the
shift and the misleading "gap is bigger than threshold" warning it
used to produce. ECU rows whose `id` has no twin counterpart fall back
to the existing nearest-timestamp path (with the 2× twin-period NaN
heuristic for out-of-range samples).

The controller's periodic `_tick` (§5.2) re-runs `align()` every
`interval_ms` on a fresh snapshot, and the ring buffer (§7) retains up
to `max_age_ms` / `max_rows` of history — so a late `id` only has to
arrive within the buffer window to be paired on the next tick. There is
no per-sample "waiting" or re-queue machinery; the existing
tick-and-re-align cadence is the mechanism.

The "no time-range overlap" fatal `AlignmentError` is **gated on id
availability**: when ids are present it is skipped up front and
re-checked after alignment — if by-id matching produced zero pairs AND
the timestamp ranges don't overlap, alignment is refused with a new
"No matches by id and no time-range overlap" message. When ids are
absent (offline CSVs, or live streams that omit ids), the original
upfront no-overlap raise is unchanged.

Wall-clock deltas for **id-matched** pairs are excluded from
`AlignedData.max_delta_t` and therefore do not trip the
`_MAX_DELTA_T_WARN_MS` "alignment may be unreliable" warning — an
id-matched pair's timestamp delta is a delivery shift, not an
alignment-quality signal. Only fallback (non-id-matched) deltas count.
Offline runs (no `id` column → all pairs are fallback) behave
identically to before; the 14 original `test_alignment_engine` tests
are unchanged.

---

## 7. Ring buffer (live_accumulator)

Two caps, either trips first:
- `max_rows` (default 10 000) — keep N most recent rows per source.
- `max_age_ms` (default 600 000 ms = 10 minutes) — drop rows older than
  `newest_t - max_age_ms`. Milliseconds, matching §6.0's wire convention
  (renamed from `max_age_s` when the live pipeline switched to real Unix
  epoch millis timestamps).

Two `pd.DataFrame`s (twin, ecu) — NOT one combined DataFrame with a
`source` column. This parallels the offline two-`LoadResult` contract.

The buffer does NOT pre-populate all canonical columns (so
`columns_found` in `build_load_result` reflects only signals actually
seen). A later sample carrying a new signal widens the buffer via pandas
concat-union; earlier rows get `NaN`.

`snapshot()` returns a **copy** — the alignment engine may hold a
reference while the buffer keeps growing. Without the copy, an append
would mutate the snapshot. The `id` column is **retained** in the
returned copy (no longer stripped) so `alignment_engine.align()` can
pair twin and ECU rows by id — see §6.5. The `id` column is not a
canonical signal, so `build_load_result` keeps it out of
`columns_found`/`columns_missing`.

---

## 8. Shared helper: `build_load_result`

`data_loader.build_load_result(df, source_label, source_columns,
warnings)` is the shared tail of both load paths. The offline CSV
loader (`load_csv`) and the live accumulator (`LiveBuffer.build_results`)
both produce a clean DataFrame and hand it here to get a `LoadResult`
with consistent metadata. **Do not duplicate this logic** — if you add
a third load path, reuse `build_load_result`.

---

## 9. Testing conventions

### 9.1 pytest, 237 tests, ~4s

```bash
source ../venv/bin/activate
python -m pytest tests/ -q
```

The venv lives at `final_tool/venv` (i.e. `source venv/bin/activate` when
your cwd is `final_tool/`; the historical `../venv` path assumes a
different cwd).

### 9.2 Pure-Python tests vs Qt tests

- `test_data_loader.py`, `test_alignment_engine.py`,
  `test_statistics_engine.py`, `test_export_engine.py`,
  `test_live_schema.py`, `test_live_accumulator.py` — pure Python, no Qt.
- `test_comparison_worker.py`, `test_plot_manager.py`,
  `test_live_controller.py`, `test_back.py` — need `QApplication`
  (offscreen). Set `os.environ.setdefault("QT_QPA_PLATFORM",
  "offscreen")` before importing PyQt5.

### 9.3 conftest.py fixtures

- `twin_csv_path` / `ecu_csv_path` — on-disk 5-signal CSVs.
- `make_load_result` — in-memory `LoadResult` builder.
- `twin_result_five` / `ecu_result_five` / `ecu_result_three` —
  ready-made 5-signal / 3-signal fixtures.
- `make_aligned_data` — in-memory `AlignedData` factory.

### 9.4 Qt test pattern

Long-lived workers need explicit teardown or the QThread hangs the
process. Use the `controller_on_thread` / `window` fixtures (they
quit + wait the thread in their teardown). Do NOT construct a
`MainWindow` without a teardown plan in a test — the live-controller
thread will hang.

### 9.5 When you add a feature, add tests

Every new module has its own `test_live_*.py` file. The test count is a
point of pride — don't let it drop.

---

## 10. paho-mqtt specifics

### 10.1 v1 callback signatures

`mqtt_worker.py` does **NOT** pass `callback_api_version` to
`mqtt.Client()` — paho 2.x defaults to **v1 callback signatures**
(`_on_connect(client, userdata, flags, rc)`,
`_on_disconnect(client, userdata, rc)`) and emits a deprecation warning.
This is known and documented. If you upgrade to v2 signatures, pass
`callback_api_version=mqtt.CallbackAPIVersion.VERSION2` and adjust all
three callbacks' parameter lists.

### 10.2 `loop_start()`, not `loop_forever()`

`MqttWorker` uses `loop_start()` so paho's network loop runs on paho's
own background thread and the worker's Qt event loop keeps spinning —
that's what makes `publish_broker` / `disconnect_broker` slots
reachable (queued connections need a running event loop).

### 10.3 QoS 1 default

The UI's QoS combo defaults to index 1 ("1 — At least once"). This is
why the `id` dedup field exists.

---

## 11. pyqtgraph 0.14 compatibility

`plot_manager.py` works around a pyqtgraph 0.14 breaking change:
`getLegend()` was removed. We use `plot.plotItem.legend` instead, with a
`_clear_legend` compat helper. If you upgrade pyqtgraph, check the
legend API first.

SoC / SoH now have their own dedicated overlay plots (see §4.1a) — like
every signal's overlay plot, they're unconditional (no checkbox gates
them). Only the error plot is checkbox-gated, and that gating applies
equally to all five signals, including SoC/SoH.

---

## 12. Documentation conventions

### 12.1 Two layers

- `high_lvl_explaination.md` — architecture overview, module list,
  mermaid diagrams, per-module summaries with API tables. Read first.
- `docs/<module>.md` — per-module line-by-line explanation. One per
  source file. Read when you need to understand a specific module's
  internals.

### 12.2 Per-module doc format

```
# <Module> — Line-by-Line Code Explanation
## Module Purpose
## Required Knowledge
## <per-symbol sections in source order, with verbatim code + why-explanations>
## <reference tables: Fatal vs Non-Fatal, Constants, Signals/Slots, etc.>
```

When you add a module, write its `docs/<module>.md`. When you change a
module's public API or a non-trivial decision, update both the
per-module doc and `high_lvl_explaination.md`.

### 12.3 No emojis in docs or code

Unless the user explicitly asks for them.

---

## 13. Decisions others (including AI agents) must follow

1. **Edit `f.ui`, not `f.py`.** Recompile with `pyuic5 f.ui -o f.py`.
2. **Keep engines pure Python.** No Qt/paho imports in
   `data_loader`, `alignment_engine`, `statistics_engine`,
   `export_engine`, `live_schema`, `live_accumulator`.
3. **Reuse `build_load_result`** for any new load path. Don't duplicate
   the `columns_found`/`columns_missing`/`time_range` logic.
4. **Reuse the offline populate methods** for live results. No duplicate
   UI logic.
5. **Keep the `id` field and stringification, and keep `id` in the
   snapshot.** QoS 1 redelivery would double-count samples without the
   id/stringification, and `align()`'s id-aware branch (§6.5) needs the
   `id` column to survive `snapshot()` to pair shifted twin/ECU streams.
   Do NOT re-introduce the old `snapshot()` strip of `id`.
6. **Keep `decimals=4` on tolerance spinboxes.** The 2-decimal default
   was a bug.
7. **Long-lived workers stay alive.** Don't wire `finished → thread.quit`
   on `ComparisonWorker` or `LiveController`.
8. **`start` doesn't clear the buffer; `reset` does.** Don't merge them.
9. **No auto tab-jump in live mode.** Would be jarring at 500 ms.
10. **Add tests for every new feature.** Don't let the test count drop.
11. **Update docs when you change a public API or a non-trivial decision.**
    Both `high_lvl_explaination.md` and the per-module `docs/<module>.md`.
12. **No emojis in code or docs** unless explicitly requested.
13. **No comments in code** unless asked (per the repo's code style).
14. **Never commit unless the user explicitly asks.**
15. **Run `python -m pytest tests/ -q` before declaring a task done.**
    The full suite must pass and the test count must not drop (currently
    237; the old "202" figure was stale — the suite has grown).
16. **Live-mode wire timestamps are Unix epoch MILLISECONDS.** See §6.0.
    Every consumer that compares raw timestamp deltas
    (`live_accumulator.DEFAULT_MAX_AGE_MS`,
    `alignment_engine._MAX_DELTA_T_WARN_MS`, `back.py._LIVE_PLOT_WINDOW_MS`)
    is scaled to milliseconds — if you add another one, scale it too.
17. **Graphs-tab signal selection lives on the Graphs tab, not Config.**
    See §4.1a. Don't reintroduce a Config-tab "signals to include"
    control — that was removed as a UX bug (selection belonged next to
    the plots it drives).

---

## 14. Known gaps / future work

- No `requirements.txt` (venv has PyQt5 5.15.11, paho-mqtt 2.1.0,
  pyqtgraph 0.14.0, reportlab 5.0.0, pytest 9.1.1, openpyxl, numpy,
  pandas).
- `ui_guide.md` and `ui_explanation.md` are stale (predate the 7-tab
  redesign). `high_lvl_explaination.md` is the current source of truth.
- `theme_manager.py` is deferred (no dark/light toggle in the UI).
- Export doesn't auto-append file extension.
- `mqtt_worker_sender.py` / `mqtt_worker_receiver.py` demos predate
  `live_schema` (fixed ramp payloads, not the canonical JSON shape).
  Prefer `synthetic_mqtt_publisher.py` (combined, single-process) or
  `synthetic_mqtt_publisher_twin.py` + `synthetic_mqtt_publisher_ecu.py`
  (two independent processes, same `--seed` to stay correlated — closer
  to how the real Simulink/ECU sources will be wired up) for live-mode
  testing against the current schema.
- `alignment_engine._MAX_DELTA_T_WARN_MS`'s gap-warning threshold is
  tuned for the live pipeline's millisecond timestamps (§6.0) but is
  shared code with the offline CSV path, which stays in whatever unit
  the CSV uses — cosmetic warning text only (not fatal), but a future
  refactor could thread an explicit unit through `align()` if it becomes
  a real pain point.

---

## 15. Quick reference — commands

```bash
# Activate venv
source ../venv/bin/activate

# Run tests
python -m pytest tests/ -q

# Recompile UI from Qt Designer source
pyuic5 f.ui -o f.py

# Verify f.py has no hand-edits
pyuic5 f.ui -o /tmp/check.py && diff <(grep -v "^# -*-" /tmp/check.py) <(grep -v "^# -*-" f.py)

# Launch the tool
python back.py

# Quick import check
python -c "import f; import back; print('ok')"
```

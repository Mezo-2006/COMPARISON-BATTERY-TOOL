"""Event Detection engine — dictionary-vs-dictionary anomaly detection.

Compares one "real" (ECU / real battery) sample dict against one
"digital twin" (MATLAB) sample dict — the shape produced e.g. by
``live_schema`` for a single sample, or handed in directly by a caller
that isn't going through the CSV/MQTT pipeline at all — and raises
structured :class:`Alert` objects whenever a configurable rule is
violated.

Unlike :mod:`alignment_engine` / :mod:`statistics_engine` (which work on
a whole *aligned* run after the fact), this module is **stateful and
streaming**: call :meth:`EventDetector.update` once per incoming sample
pair and it maintains just enough per-signal history (previous value,
a short rolling window, freeze/jump counters) to catch conditions that
only make sense across consecutive updates — a spike, a freeze, an
oscillation, a permanent jump.

Detectors implemented (see the module's task list)
----------------------------------------------------
1. **Validation**       — required signals present in both dicts.
2. **Comparison**        — ``difference = real - twin`` for every common
   signal.
3. **Threshold**         — ``abs(difference) > tolerance`` → alert.
4. **Spike**              — single-step increase beyond a threshold.
5. **Sudden Drop**        — single-step decrease beyond a threshold.
6. **Communication Loss** — a source's own ``timestamp`` field stops
   advancing for longer than a configured timeout.
7. **Value Freeze**       — a signal reads the exact same value for N
   consecutive updates (checked on both real and twin streams).
8. **Oscillation**        — a signal alternates up/down/up/down over a
   rolling window.
9. **Timeout**            — the wall-clock gap between two calls to
   :meth:`update` exceeds the expected update interval.
10. **Out of Range**       — a value falls outside its engineering
    bounds.
11. **Unexpected Jump**    — a large single-step change that then
    *persists* near its new level (as opposed to a Spike/Drop, which
    says nothing about what happens afterwards).
12. **Alert Generation**   — every violated rule becomes an
    :class:`Alert` with timestamp/signal/event/severity/values/
    difference/threshold.
13. **Event Log**          — :class:`EventLog` accumulates every alert
    and exposes it as a :class:`pandas.DataFrame` for the GUI / export.

Pure Python — no Qt, no MQTT — so it is unit-testable in isolation and
reusable from both an offline dict-replay script and the live path.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical signals — kept in sync with the rest of the project.
# ---------------------------------------------------------------------------
SIGNALS: List[str] = ["voltage", "current", "temperature", "soc", "soh"]
REQUIRED_SIGNALS: List[str] = ["timestamp", "voltage", "current", "temperature", "soc"]
OPTIONAL_SIGNALS: List[str] = ["soh"]

# ---------------------------------------------------------------------------
# Severity levels
# ---------------------------------------------------------------------------
SEVERITY_LOW = "Low"
SEVERITY_MEDIUM = "Medium"
SEVERITY_HIGH = "High"

# ---------------------------------------------------------------------------
# Event type names
# ---------------------------------------------------------------------------
EVENT_MISSING_SIGNAL = "Missing Signal"
EVENT_THRESHOLD_EXCEEDED = "Threshold Exceeded"
EVENT_SPIKE = "Spike"
EVENT_SUDDEN_DROP = "Sudden Drop"
EVENT_COMMUNICATION_LOSS = "Communication Loss"
EVENT_VALUE_FREEZE = "Value Freeze"
EVENT_OSCILLATION = "Oscillation"
EVENT_TIMEOUT = "Timeout"
EVENT_OUT_OF_RANGE = "Out of Range"
EVENT_UNEXPECTED_JUMP = "Unexpected Jump"

SOURCE_REAL = "real"
SOURCE_TWIN = "twin"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class DetectorConfig:
    """Tunable rule thresholds. Every dict is keyed by canonical signal name."""

    # Task 3 — abs(real - twin) tolerance per signal (signal units).
    tolerances: Dict[str, float] = field(default_factory=lambda: {
        "voltage": 0.05, "current": 0.5, "temperature": 1.0,
        "soc": 2.0, "soh": 2.0,
    })

    # Task 4/5 — single-step increase/decrease that counts as abnormal.
    spike_threshold: Dict[str, float] = field(default_factory=lambda: {
        "voltage": 0.3, "current": 2.0, "temperature": 5.0,
        "soc": 10.0, "soh": 5.0,
    })
    drop_threshold: Dict[str, float] = field(default_factory=lambda: {
        "voltage": 0.3, "current": 2.0, "temperature": 5.0,
        "soc": 10.0, "soh": 5.0,
    })

    # Task 7 — consecutive identical updates before a Value Freeze fires.
    freeze_count: int = 5
    freeze_epsilon: float = 1e-9

    # Task 8 — rolling window + noise floor for oscillation detection.
    oscillation_window: int = 5
    oscillation_min_amplitude: Dict[str, float] = field(default_factory=lambda: {
        "voltage": 0.02, "current": 0.2, "temperature": 0.3,
        "soc": 0.5, "soh": 0.5,
    })

    # Task 6 — a source's timestamp field must advance at least this often.
    communication_timeout_s: float = 5.0

    # Task 9 — expected cadence between update() calls.
    expected_update_interval_s: float = 1.0
    timeout_tolerance_s: float = 0.5

    # Task 10 — physical/engineering bounds (min, max).
    out_of_range: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "voltage": (3.0, 4.2), "current": (0.0, 30.0),
        "temperature": (-10.0, 60.0), "soc": (0.0, 100.0), "soh": (0.0, 100.0),
    })

    # Task 11 — a step larger than this starts a "candidate" jump; it is
    # confirmed once the value stays within `jump_stability_band` of the
    # new level for `jump_persist_count` further updates.
    jump_threshold: Dict[str, float] = field(default_factory=lambda: {
        "voltage": 0.5, "current": 5.0, "temperature": 8.0,
        "soc": 15.0, "soh": 10.0,
    })
    jump_stability_band: Dict[str, float] = field(default_factory=lambda: {
        "voltage": 0.05, "current": 1.0, "temperature": 1.0,
        "soc": 3.0, "soh": 2.0,
    })
    jump_persist_count: int = 3


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class Alert:
    """One detected anomaly — matches the spec's Alert Generation shape."""

    timestamp: float
    signal: str
    event: str
    severity: str
    real_value: Optional[float]
    digital_twin_value: Optional[float]
    difference: Optional[float]
    threshold: Optional[float]
    source: Optional[str] = None  # "real" / "twin" — which stream triggered it

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "signal": self.signal,
            "event": self.event,
            "severity": self.severity,
            "real_value": self.real_value,
            "digital_twin_value": self.digital_twin_value,
            "difference": self.difference,
            "threshold": self.threshold,
        }


@dataclass
class ValidationResult:
    """Task 1 — which signals both dicts agree on / are missing."""

    real_present: bool
    twin_present: bool
    common_signals: List[str]
    missing_from_real: List[str]
    missing_from_twin: List[str]
    ignored_real: List[str]
    ignored_twin: List[str]


@dataclass
class EventLog:
    """Task 13 — append-only log of every alert raised so far."""

    alerts: List[Alert] = field(default_factory=list)

    def add(self, alert: Alert) -> None:
        self.alerts.append(alert)

    def add_many(self, alerts: List[Alert]) -> None:
        self.alerts.extend(alerts)

    def clear(self) -> None:
        self.alerts.clear()

    def __len__(self) -> int:
        return len(self.alerts)

    def to_dataframe(self) -> pd.DataFrame:
        """Full event log as a DataFrame — one row per alert."""
        if not self.alerts:
            return pd.DataFrame(columns=[
                "timestamp", "signal", "event", "severity",
                "real_value", "digital_twin_value", "difference", "threshold",
                "source",
            ])
        return pd.DataFrame([
            {
                "timestamp": a.timestamp,
                "signal": a.signal,
                "event": a.event,
                "severity": a.severity,
                "real_value": a.real_value,
                "digital_twin_value": a.digital_twin_value,
                "difference": a.difference,
                "threshold": a.threshold,
                "source": a.source,
            }
            for a in self.alerts
        ])


# ---------------------------------------------------------------------------
# Task 1 — dictionary validation (standalone, also used internally)
# ---------------------------------------------------------------------------
def validate_dictionaries(
    real_data: Optional[dict],
    digital_twin_data: Optional[dict],
) -> ValidationResult:
    """Check both dicts exist, find common/missing/ignored signal keys."""
    real_present = isinstance(real_data, dict) and bool(real_data)
    twin_present = isinstance(digital_twin_data, dict) and bool(digital_twin_data)

    real_keys = set(real_data.keys()) if real_present else set()
    twin_keys = set(digital_twin_data.keys()) if twin_present else set()

    known = set(SIGNALS)
    common_signals = [s for s in SIGNALS if s in real_keys and s in twin_keys]

    missing_from_real = [s for s in REQUIRED_SIGNALS if s not in real_keys]
    missing_from_twin = [s for s in REQUIRED_SIGNALS if s not in twin_keys]

    ignored_real = sorted(real_keys - known - {"timestamp", "id"})
    ignored_twin = sorted(twin_keys - known - {"timestamp", "id"})

    return ValidationResult(
        real_present=real_present,
        twin_present=twin_present,
        common_signals=common_signals,
        missing_from_real=missing_from_real,
        missing_from_twin=missing_from_twin,
        ignored_real=ignored_real,
        ignored_twin=ignored_twin,
    )


# ---------------------------------------------------------------------------
# Per-signal, per-source rolling state
# ---------------------------------------------------------------------------
class _SignalState:
    __slots__ = (
        "last_value", "recent", "freeze_streak", "oscillating",
        "jump_candidate", "jump_confirm_count", "jump_baseline",
    )

    def __init__(self, window: int):
        self.last_value: Optional[float] = None
        self.recent: Deque[float] = deque(maxlen=window)
        self.freeze_streak: int = 1
        self.oscillating: bool = False
        self.jump_candidate: Optional[float] = None
        self.jump_confirm_count: int = 0
        # The last *confirmed-stable* value a signal sat at — the reference
        # point new candidates are measured against. This is what lets a
        # transient spike-then-revert (up and straight back down) avoid
        # being misread as "jumped to a new permanent level": the revert
        # lands back within `jump_threshold` of the baseline, so it never
        # starts a new candidate.
        self.jump_baseline: Optional[float] = None


class _TimestampTracker:
    __slots__ = ("value", "changed_at", "alerted")

    def __init__(self):
        self.value: Optional[float] = None
        self.changed_at: Optional[float] = None
        self.alerted: bool = False


def _severity_for_ratio(ratio: float) -> str:
    """How far past the threshold, expressed as a severity band."""
    if ratio >= 3.0:
        return SEVERITY_HIGH
    if ratio >= 1.5:
        return SEVERITY_MEDIUM
    return SEVERITY_LOW


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------
class EventDetector:
    """Stateful comparator: feed sample dicts in, get :class:`Alert` s out.

    Usage::

        det = EventDetector()
        log = EventLog()
        for real, twin in stream:
            alerts = det.update(real, twin)
            log.add_many(alerts)
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.cfg = config or DetectorConfig()
        window = max(
            self.cfg.oscillation_window,
            self.cfg.jump_persist_count + 2,
        )
        self._state: Dict[str, Dict[str, _SignalState]] = {
            SOURCE_REAL: {s: _SignalState(window) for s in SIGNALS},
            SOURCE_TWIN: {s: _SignalState(window) for s in SIGNALS},
        }
        self._ts_tracker: Dict[str, _TimestampTracker] = {
            SOURCE_REAL: _TimestampTracker(),
            SOURCE_TWIN: _TimestampTracker(),
        }
        self._reported_missing: set = set()
        self._last_call_wall: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(
        self,
        real_data: Optional[dict],
        digital_twin_data: Optional[dict],
        now: Optional[float] = None,
    ) -> List[Alert]:
        """Compare one sample pair and return every alert it triggers."""
        now = time.time() if now is None else now
        alerts: List[Alert] = []

        validation = validate_dictionaries(real_data, digital_twin_data)
        sample_ts = _first_timestamp(real_data, digital_twin_data, now)

        alerts += self._validation_alerts(validation, sample_ts)
        alerts += self._timeout_alert(now, sample_ts)
        alerts += self._communication_loss_alerts(
            SOURCE_REAL, real_data, now, sample_ts,
        )
        alerts += self._communication_loss_alerts(
            SOURCE_TWIN, digital_twin_data, now, sample_ts,
        )

        for signal in validation.common_signals:
            real_v = _as_float(real_data.get(signal))
            twin_v = _as_float(digital_twin_data.get(signal))

            if real_v is not None and twin_v is not None:
                alerts += self._threshold_alert(signal, real_v, twin_v, sample_ts)

            if real_v is not None:
                alerts += self._out_of_range_alert(
                    signal, SOURCE_REAL, real_v, real_v, twin_v, sample_ts)
                alerts += self._per_sample_signal_alerts(
                    signal, SOURCE_REAL, real_v, real_v, twin_v, sample_ts)
            if twin_v is not None:
                alerts += self._out_of_range_alert(
                    signal, SOURCE_TWIN, twin_v, real_v, twin_v, sample_ts)
                alerts += self._per_sample_signal_alerts(
                    signal, SOURCE_TWIN, twin_v, real_v, twin_v, sample_ts)

        self._last_call_wall = now
        return alerts

    # ------------------------------------------------------------------
    # Task 1 — validation → alerts (edge-triggered so it doesn't spam)
    # ------------------------------------------------------------------
    def _validation_alerts(
        self, validation: ValidationResult, sample_ts: float,
    ) -> List[Alert]:
        alerts: List[Alert] = []
        missing = set(validation.missing_from_real) | set(validation.missing_from_twin)
        for signal in missing:
            key = signal
            if key not in self._reported_missing:
                self._reported_missing.add(key)
                alerts.append(Alert(
                    timestamp=sample_ts, signal=signal,
                    event=EVENT_MISSING_SIGNAL, severity=SEVERITY_LOW,
                    real_value=None, digital_twin_value=None,
                    difference=None, threshold=None,
                ))
        # Signal reappeared — allow it to be reported again if it vanishes later.
        for signal in list(self._reported_missing):
            if signal not in missing:
                self._reported_missing.discard(signal)
        return alerts

    # ------------------------------------------------------------------
    # Task 3 — threshold evaluation
    # ------------------------------------------------------------------
    def _threshold_alert(
        self, signal: str, real_v: float, twin_v: float, sample_ts: float,
    ) -> List[Alert]:
        tol = self.cfg.tolerances.get(signal)
        if tol is None:
            return []
        diff = real_v - twin_v
        if abs(diff) <= tol:
            return []
        ratio = abs(diff) / tol if tol > 0 else float("inf")
        return [Alert(
            timestamp=sample_ts, signal=signal,
            event=EVENT_THRESHOLD_EXCEEDED, severity=_severity_for_ratio(ratio),
            real_value=real_v, digital_twin_value=twin_v,
            difference=diff, threshold=tol,
        )]

    # ------------------------------------------------------------------
    # Task 10 — out of range (checked per source independently)
    # ------------------------------------------------------------------
    def _out_of_range_alert(
        self, signal: str, source: str, value: float,
        real_v: Optional[float], twin_v: Optional[float], sample_ts: float,
    ) -> List[Alert]:
        bounds = self.cfg.out_of_range.get(signal)
        if bounds is None:
            return []
        lo, hi = bounds
        if lo <= value <= hi:
            return []
        return [Alert(
            timestamp=sample_ts, signal=signal,
            event=EVENT_OUT_OF_RANGE, severity=SEVERITY_HIGH,
            real_value=real_v, digital_twin_value=twin_v,
            difference=value, threshold=hi if value > hi else lo,
            source=source,
        )]

    # ------------------------------------------------------------------
    # Tasks 4, 5, 7, 8, 11 — everything driven by a signal's own history
    # ------------------------------------------------------------------
    def _per_sample_signal_alerts(
        self, signal: str, source: str, value: float,
        real_v: Optional[float], twin_v: Optional[float], sample_ts: float,
    ) -> List[Alert]:
        alerts: List[Alert] = []
        state = self._state[source][signal]

        if state.last_value is not None:
            delta = value - state.last_value
            if state.jump_baseline is None:
                state.jump_baseline = state.last_value

            # --- Task 4/5: spike / sudden drop -------------------------
            spike_th = self.cfg.spike_threshold.get(signal)
            drop_th = self.cfg.drop_threshold.get(signal)
            if spike_th is not None and delta > spike_th:
                alerts.append(Alert(
                    timestamp=sample_ts, signal=signal,
                    event=EVENT_SPIKE, severity=SEVERITY_HIGH,
                    real_value=real_v, digital_twin_value=twin_v,
                    difference=delta, threshold=spike_th, source=source,
                ))
            elif drop_th is not None and -delta > drop_th:
                alerts.append(Alert(
                    timestamp=sample_ts, signal=signal,
                    event=EVENT_SUDDEN_DROP, severity=SEVERITY_HIGH,
                    real_value=real_v, digital_twin_value=twin_v,
                    difference=delta, threshold=drop_th, source=source,
                ))

            # --- Task 11: unexpected (persistent) jump ------------------
            # A candidate is measured against `jump_baseline` (the last
            # confirmed-stable level), not the immediately preceding
            # sample, so a spike that reverts straight back to baseline
            # never gets mistaken for a new permanent level.
            jump_th = self.cfg.jump_threshold.get(signal, float("inf"))
            band = self.cfg.jump_stability_band.get(signal, 0.0)
            if state.jump_candidate is None:
                if abs(value - state.jump_baseline) > jump_th:
                    state.jump_candidate = value
                    state.jump_confirm_count = 1
            else:
                if abs(value - state.jump_candidate) <= band:
                    state.jump_confirm_count += 1
                    if state.jump_confirm_count >= self.cfg.jump_persist_count:
                        if abs(state.jump_candidate - state.jump_baseline) > jump_th:
                            alerts.append(Alert(
                                timestamp=sample_ts, signal=signal,
                                event=EVENT_UNEXPECTED_JUMP, severity=SEVERITY_HIGH,
                                real_value=real_v, digital_twin_value=twin_v,
                                difference=state.jump_candidate - state.jump_baseline,
                                threshold=jump_th, source=source,
                            ))
                        state.jump_baseline = state.jump_candidate
                        state.jump_candidate = None
                        state.jump_confirm_count = 0
                elif abs(value - state.jump_baseline) > jump_th:
                    # Still far from baseline but drifted off the previous
                    # candidate — restart tracking from here.
                    state.jump_candidate = value
                    state.jump_confirm_count = 1
                else:
                    # Settled back near baseline — nothing to report.
                    state.jump_candidate = None
                    state.jump_confirm_count = 0

            # --- Task 7: value freeze ------------------------------------
            if abs(delta) <= self.cfg.freeze_epsilon:
                state.freeze_streak += 1
            else:
                state.freeze_streak = 1
            if state.freeze_streak == self.cfg.freeze_count:
                alerts.append(Alert(
                    timestamp=sample_ts, signal=signal,
                    event=EVENT_VALUE_FREEZE, severity=SEVERITY_MEDIUM,
                    real_value=real_v, digital_twin_value=twin_v,
                    difference=0.0, threshold=self.cfg.freeze_count, source=source,
                ))
        else:
            state.freeze_streak = 1

        # --- Task 8: oscillation ----------------------------------------
        state.recent.append(value)
        window = self.cfg.oscillation_window
        min_amp = self.cfg.oscillation_min_amplitude.get(signal, 0.0)
        if len(state.recent) >= window:
            recent = list(state.recent)[-window:]
            diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
            alternating = all(abs(d) >= min_amp for d in diffs) and all(
                diffs[i] * diffs[i + 1] < 0 for i in range(len(diffs) - 1)
            )
            if alternating and not state.oscillating:
                state.oscillating = True
                alerts.append(Alert(
                    timestamp=sample_ts, signal=signal,
                    event=EVENT_OSCILLATION, severity=SEVERITY_MEDIUM,
                    real_value=real_v, digital_twin_value=twin_v,
                    difference=max(abs(d) for d in diffs), threshold=min_amp,
                    source=source,
                ))
            elif not alternating:
                state.oscillating = False

        state.last_value = value
        return alerts

    # ------------------------------------------------------------------
    # Task 9 — timeout between update() calls
    # ------------------------------------------------------------------
    def _timeout_alert(self, now: float, sample_ts: float) -> List[Alert]:
        if self._last_call_wall is None:
            return []
        elapsed = now - self._last_call_wall
        expected = self.cfg.expected_update_interval_s + self.cfg.timeout_tolerance_s
        if elapsed <= expected:
            return []
        return [Alert(
            timestamp=sample_ts, signal="system",
            event=EVENT_TIMEOUT, severity=SEVERITY_MEDIUM,
            real_value=None, digital_twin_value=None,
            difference=elapsed, threshold=expected,
        )]

    # ------------------------------------------------------------------
    # Task 6 — communication loss (a source's timestamp field is stale)
    # ------------------------------------------------------------------
    def _communication_loss_alerts(
        self, source: str, data: Optional[dict], now: float, sample_ts: float,
    ) -> List[Alert]:
        tracker = self._ts_tracker[source]
        ts_val = _as_float(data.get("timestamp")) if isinstance(data, dict) else None
        if ts_val is None:
            return []

        if tracker.value is None or ts_val != tracker.value:
            tracker.value = ts_val
            tracker.changed_at = now
            tracker.alerted = False
            return []

        changed_at = tracker.changed_at if tracker.changed_at is not None else now
        stale_for = now - changed_at
        if stale_for > self.cfg.communication_timeout_s and not tracker.alerted:
            tracker.alerted = True
            return [Alert(
                timestamp=sample_ts, signal=source,
                event=EVENT_COMMUNICATION_LOSS, severity=SEVERITY_HIGH,
                real_value=None, digital_twin_value=None,
                difference=stale_for, threshold=self.cfg.communication_timeout_s,
                source=source,
            )]
        return []


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _as_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_timestamp(real_data: Optional[dict], twin_data: Optional[dict], fallback: float) -> float:
    for data in (real_data, twin_data):
        if isinstance(data, dict) and "timestamp" in data:
            ts = _as_float(data["timestamp"])
            if ts is not None:
                return ts
    return fallback

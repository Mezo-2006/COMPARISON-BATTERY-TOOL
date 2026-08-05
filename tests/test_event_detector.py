"""Tests for ``event_detector.py`` — dictionary-vs-dictionary event detection."""

from __future__ import annotations

import pytest

from event_detector import (
    DetectorConfig,
    EventDetector,
    EventLog,
    EVENT_COMMUNICATION_LOSS,
    EVENT_MISSING_SIGNAL,
    EVENT_OSCILLATION,
    EVENT_OUT_OF_RANGE,
    EVENT_SPIKE,
    EVENT_SUDDEN_DROP,
    EVENT_THRESHOLD_EXCEEDED,
    EVENT_TIMEOUT,
    EVENT_UNEXPECTED_JUMP,
    EVENT_VALUE_FREEZE,
    validate_dictionaries,
)


def _pair(real_overrides=None, twin_overrides=None, ts=15.25):
    real = {"timestamp": ts, "voltage": 3.72, "current": 5.3, "temperature": 26.1, "soc": 79.8}
    twin = {"timestamp": ts, "voltage": 3.70, "current": 5.2, "temperature": 26.0, "soc": 80.1}
    real.update(real_overrides or {})
    twin.update(twin_overrides or {})
    return real, twin


# ---------------------------------------------------------------------------
# Task 1 — validation
# ---------------------------------------------------------------------------
def test_validate_common_signals_all_present():
    real, twin = _pair()
    result = validate_dictionaries(real, twin)
    assert result.real_present and result.twin_present
    assert set(result.common_signals) == {"voltage", "current", "temperature", "soc"}
    assert result.missing_from_real == []
    assert result.missing_from_twin == []


def test_validate_reports_missing_required_signal():
    real, twin = _pair()
    del twin["current"]
    result = validate_dictionaries(real, twin)
    assert "current" in result.missing_from_twin
    assert "current" not in result.common_signals


def test_validate_ignores_unsupported_keys():
    real, twin = _pair(real_overrides={"pressure": 1.2})
    result = validate_dictionaries(real, twin)
    assert "pressure" in result.ignored_real
    assert "pressure" not in result.common_signals


def test_validate_handles_missing_dict():
    result = validate_dictionaries(None, {"timestamp": 1.0})
    assert result.real_present is False
    assert result.common_signals == []


def test_missing_signal_alert_is_edge_triggered():
    det = EventDetector()
    real, twin = _pair()
    del twin["current"]
    alerts1 = det.update(real, twin, now=0.0)
    assert any(a.event == EVENT_MISSING_SIGNAL and a.signal == "current" for a in alerts1)
    # Same missing signal on the next call must not re-fire.
    alerts2 = det.update(real, twin, now=1.0)
    assert not any(a.event == EVENT_MISSING_SIGNAL for a in alerts2)


# ---------------------------------------------------------------------------
# Task 2/3 — comparison + threshold evaluation
# ---------------------------------------------------------------------------
def test_threshold_alert_fires_when_difference_exceeds_tolerance():
    det = EventDetector()
    real, twin = _pair(real_overrides={"voltage": 3.90})  # 0.20 V gap > 0.05 default tol
    alerts = det.update(real, twin, now=0.0)
    hits = [a for a in alerts if a.event == EVENT_THRESHOLD_EXCEEDED and a.signal == "voltage"]
    assert len(hits) == 1
    assert hits[0].difference == pytest.approx(0.20)
    assert hits[0].threshold == pytest.approx(0.05)


def test_no_threshold_alert_within_tolerance():
    det = EventDetector()
    real, twin = _pair()  # voltage gap 0.02 V < 0.05 default tol
    alerts = det.update(real, twin, now=0.0)
    assert not any(a.event == EVENT_THRESHOLD_EXCEEDED for a in alerts)


# ---------------------------------------------------------------------------
# Task 4/5 — spike / sudden drop
# ---------------------------------------------------------------------------
def test_spike_detection_matches_spec_example():
    det = EventDetector()
    real, twin = _pair(real_overrides={"voltage": 3.72})
    det.update(real, twin, now=0.0)
    real2, twin2 = _pair(real_overrides={"voltage": 4.18}, ts=15.35)
    alerts = det.update(real2, twin2, now=1.0)
    spikes = [a for a in alerts if a.event == EVENT_SPIKE and a.signal == "voltage"]
    assert len(spikes) == 1
    assert spikes[0].severity == "High"
    assert spikes[0].source == "real"


def test_sudden_drop_matches_spec_example():
    cfg = DetectorConfig()
    det = EventDetector(cfg)
    real, twin = _pair(real_overrides={"current": 20.0})
    det.update(real, twin, now=0.0)
    real2, twin2 = _pair(real_overrides={"current": 5.0}, ts=15.35)
    alerts = det.update(real2, twin2, now=1.0)
    drops = [a for a in alerts if a.event == EVENT_SUDDEN_DROP and a.signal == "current"]
    assert len(drops) == 1


# ---------------------------------------------------------------------------
# Task 7 — value freeze
# ---------------------------------------------------------------------------
def test_value_freeze_after_consecutive_identical_updates():
    cfg = DetectorConfig(freeze_count=3)
    det = EventDetector(cfg)
    real, twin = _pair()
    for i in range(3):
        alerts = det.update(real, twin, now=float(i))
    freezes = [a for a in alerts if a.event == EVENT_VALUE_FREEZE and a.signal == "voltage"]
    # Both streams are held constant across calls, so both freeze.
    assert {a.source for a in freezes} == {"real", "twin"}


def test_value_freeze_does_not_fire_before_threshold():
    cfg = DetectorConfig(freeze_count=5)
    det = EventDetector(cfg)
    real, twin = _pair()
    alerts = []
    for i in range(3):
        alerts = det.update(real, twin, now=float(i))
    assert not any(a.event == EVENT_VALUE_FREEZE for a in alerts)


def test_value_freeze_resets_when_value_changes():
    cfg = DetectorConfig(freeze_count=3)
    det = EventDetector(cfg)
    real, twin = _pair()
    det.update(real, twin, now=0.0)
    real2, twin2 = _pair(real_overrides={"voltage": 3.75}, twin_overrides={"voltage": 3.80})
    det.update(real2, twin2, now=1.0)
    alerts = det.update(real2, twin2, now=2.0)
    real_voltage_freezes = [
        a for a in alerts
        if a.event == EVENT_VALUE_FREEZE and a.source == "real" and a.signal == "voltage"
    ]
    assert real_voltage_freezes == []


# ---------------------------------------------------------------------------
# Task 8 — oscillation
# ---------------------------------------------------------------------------
def test_oscillation_detected_on_alternating_pattern():
    cfg = DetectorConfig(oscillation_window=5)
    det = EventDetector(cfg)
    pattern = [3.70, 3.80, 3.70, 3.80, 3.70]
    alerts = []
    for i, v in enumerate(pattern):
        real, twin = _pair(real_overrides={"voltage": v})
        alerts = det.update(real, twin, now=float(i))
    assert any(a.event == EVENT_OSCILLATION and a.signal == "voltage" for a in alerts)


def test_no_oscillation_on_monotonic_signal():
    cfg = DetectorConfig(oscillation_window=5)
    det = EventDetector(cfg)
    alerts = []
    for i, v in enumerate([3.70, 3.71, 3.72, 3.73, 3.74]):
        real, twin = _pair(real_overrides={"voltage": v})
        alerts = det.update(real, twin, now=float(i))
    assert not any(a.event == EVENT_OSCILLATION for a in alerts)


# ---------------------------------------------------------------------------
# Task 9 — timeout between updates
# ---------------------------------------------------------------------------
def test_timeout_when_update_arrives_late():
    cfg = DetectorConfig(expected_update_interval_s=1.0, timeout_tolerance_s=0.2)
    det = EventDetector(cfg)
    real, twin = _pair()
    det.update(real, twin, now=0.0)
    alerts = det.update(real, twin, now=5.0)  # way later than expected
    assert any(a.event == EVENT_TIMEOUT for a in alerts)


def test_no_timeout_when_update_on_time():
    cfg = DetectorConfig(expected_update_interval_s=1.0, timeout_tolerance_s=0.2)
    det = EventDetector(cfg)
    real, twin = _pair()
    det.update(real, twin, now=0.0)
    alerts = det.update(real, twin, now=1.1)
    assert not any(a.event == EVENT_TIMEOUT for a in alerts)


# ---------------------------------------------------------------------------
# Task 10 — out of range
# ---------------------------------------------------------------------------
def test_out_of_range_detected_above_max():
    det = EventDetector()
    real, twin = _pair(real_overrides={"voltage": 4.45})  # default max is 4.2
    alerts = det.update(real, twin, now=0.0)
    hits = [a for a in alerts if a.event == EVENT_OUT_OF_RANGE and a.signal == "voltage"]
    assert len(hits) == 1
    assert hits[0].severity == "High"
    assert hits[0].source == "real"


def test_in_range_value_does_not_fire():
    det = EventDetector()
    real, twin = _pair()
    alerts = det.update(real, twin, now=0.0)
    assert not any(a.event == EVENT_OUT_OF_RANGE for a in alerts)


# ---------------------------------------------------------------------------
# Task 6 — communication loss (a source's timestamp field goes stale)
# ---------------------------------------------------------------------------
def test_communication_loss_when_timestamp_stalls():
    cfg = DetectorConfig(communication_timeout_s=2.0)
    det = EventDetector(cfg)
    real, twin = _pair(ts=15.25)
    det.update(real, twin, now=0.0)
    det.update(real, twin, now=1.0)  # timestamp unchanged, still under timeout
    alerts = det.update(real, twin, now=3.0)  # unchanged for > 2s now
    losses = [a for a in alerts if a.event == EVENT_COMMUNICATION_LOSS]
    assert len(losses) == 2  # both real and twin stalled on the same fixed timestamp
    assert {a.source for a in losses} == {"real", "twin"}


def test_no_communication_loss_when_timestamp_advances():
    cfg = DetectorConfig(communication_timeout_s=2.0)
    det = EventDetector(cfg)
    for i in range(5):
        real, twin = _pair(ts=15.0 + i)
        alerts = det.update(real, twin, now=float(i))
    assert not any(a.event == EVENT_COMMUNICATION_LOSS for a in alerts)


# ---------------------------------------------------------------------------
# Task 11 — unexpected (persistent) jump
# ---------------------------------------------------------------------------
def test_unexpected_jump_confirmed_after_persistence():
    cfg = DetectorConfig(jump_persist_count=3)
    det = EventDetector(cfg)
    real, twin = _pair(real_overrides={"soc": 80.0})
    det.update(real, twin, now=0.0)
    alerts = []
    for i, soc in enumerate([45.0, 45.2, 44.9, 45.1], start=1):
        real, twin = _pair(real_overrides={"soc": soc})
        alerts = det.update(real, twin, now=float(i))
        if any(a.event == EVENT_UNEXPECTED_JUMP for a in alerts):
            break
    assert any(a.event == EVENT_UNEXPECTED_JUMP and a.signal == "soc" for a in alerts)


def test_transient_spike_does_not_confirm_as_jump():
    """A one-off spike that immediately reverts should not become a Jump."""
    cfg = DetectorConfig(jump_persist_count=3)
    det = EventDetector(cfg)
    real, twin = _pair(real_overrides={"soc": 80.0})
    det.update(real, twin, now=0.0)
    values = [45.0, 80.0, 80.0, 80.0]  # jumps then immediately reverts
    all_alerts = []
    for i, soc in enumerate(values, start=1):
        real, twin = _pair(real_overrides={"soc": soc})
        all_alerts += det.update(real, twin, now=float(i))
    assert not any(a.event == EVENT_UNEXPECTED_JUMP for a in all_alerts)


# ---------------------------------------------------------------------------
# Task 12/13 — alert shape + event log
# ---------------------------------------------------------------------------
def test_alert_to_dict_matches_spec_shape():
    det = EventDetector()
    real, twin = _pair(real_overrides={"voltage": 3.72})
    det.update(real, twin, now=0.0)
    real2, twin2 = _pair(real_overrides={"voltage": 4.18}, ts=15.25)
    alerts = det.update(real2, twin2, now=1.0)
    spike = next(a for a in alerts if a.event == EVENT_SPIKE)
    d = spike.to_dict()
    assert set(d.keys()) == {
        "timestamp", "signal", "event", "severity",
        "real_value", "digital_twin_value", "difference", "threshold",
    }
    assert d["signal"] == "voltage"
    assert d["event"] == EVENT_SPIKE


def test_event_log_accumulates_and_exports_dataframe():
    det = EventDetector()
    log = EventLog()
    real, twin = _pair(real_overrides={"voltage": 3.90})
    alerts = det.update(real, twin, now=0.0)
    log.add_many(alerts)
    assert len(log) == len(alerts)
    df = log.to_dataframe()
    assert list(df.columns) == [
        "timestamp", "signal", "event", "severity",
        "real_value", "digital_twin_value", "difference", "threshold", "source",
    ]
    assert len(df) == len(alerts)


def test_event_log_empty_dataframe_has_expected_columns():
    log = EventLog()
    df = log.to_dataframe()
    assert len(df) == 0
    assert "event" in df.columns

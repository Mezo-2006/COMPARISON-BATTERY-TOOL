"""Unit tests for ``synthetic_data_generator.py``.

Pure Python + numpy: no Qt, no paho. Covers determinism (same seed ->
identical output), the twin/ecu divergence staying "small" (~0.1-scale),
signals staying in physically normal ranges, and that generated batches
are actually accepted by the real wire-format parser (``live_schema``).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import time

from synthetic_data_generator import (
    GeneratorConfig,
    SyntheticBatteryGenerator,
    batch_to_json_bytes,
    ID_MODE_SEQUENCE,
    ID_MODE_TIMESTAMP,
)
from live_schema import parse_message

TWIN_ROOT = "bms/twin/#"
ECU_ROOT = "bms/actual/#"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
class TestDeterminism:
    def test_same_seed_produces_identical_batches(self):
        gen_a = SyntheticBatteryGenerator(GeneratorConfig(seed=7))
        gen_b = SyntheticBatteryGenerator(GeneratorConfig(seed=7))

        twin_a, ecu_a = gen_a.batch(20)
        twin_b, ecu_b = gen_b.batch(20)

        assert twin_a == twin_b
        assert ecu_a == ecu_b

    def test_same_seed_produces_identical_json_bytes(self):
        gen_a = SyntheticBatteryGenerator(GeneratorConfig(seed=123))
        gen_b = SyntheticBatteryGenerator(GeneratorConfig(seed=123))

        twin_a, _ = gen_a.batch(10)
        twin_b, _ = gen_b.batch(10)

        assert batch_to_json_bytes(twin_a) == batch_to_json_bytes(twin_b)

    def test_default_id_starts_from_zero(self):
        gen = SyntheticBatteryGenerator(GeneratorConfig(seed=1))

        twin_payload, _ = gen.batch(1)
        assert twin_payload["samples"][0]["id"] == 0

    def test_different_seeds_diverge(self):
        gen_a = SyntheticBatteryGenerator(GeneratorConfig(seed=1))
        gen_b = SyntheticBatteryGenerator(GeneratorConfig(seed=2))

        twin_a, _ = gen_a.batch(20)
        twin_b, _ = gen_b.batch(20)

        assert twin_a != twin_b

    def test_generator_is_stateful_across_calls(self):
        """Consecutive ``.batch()`` calls continue the same trajectory, not reset it."""
        gen = SyntheticBatteryGenerator(GeneratorConfig(seed=42))
        first, _ = gen.batch(5)
        second, _ = gen.batch(5)

        assert first["samples"][-1]["id"] == 4
        assert second["samples"][0]["id"] == 5


# ---------------------------------------------------------------------------
# Twin/ECU divergence stays small
# ---------------------------------------------------------------------------
class TestDivergence:
    @pytest.mark.parametrize("signal,expected_scale", [
        ("voltage", 0.1),
        ("current", 0.1),
        ("temperature", 0.2),
        ("soc", 0.2),
        ("soh", 0.1),
    ])
    def test_mean_abs_divergence_is_small(self, signal, expected_scale):
        gen = SyntheticBatteryGenerator(GeneratorConfig(seed=99))
        twin_payload, ecu_payload = gen.batch(500)

        twin_vals = np.array([s[signal] for s in twin_payload["samples"]])
        ecu_vals = np.array([s[signal] for s in ecu_payload["samples"]])

        mean_abs_diff = float(np.mean(np.abs(ecu_vals - twin_vals)))
        # "small, around 0.1" per the project's own twin/ecu test fixtures.
        assert mean_abs_diff < expected_scale

    def test_divergence_is_not_zero(self):
        """The two streams must actually differ -- a generator that outputs
        identical twin/ecu values would silently defeat the point of a
        comparison tool."""
        gen = SyntheticBatteryGenerator(GeneratorConfig(seed=5))
        twin_payload, ecu_payload = gen.batch(50)

        twin_v = [s["voltage"] for s in twin_payload["samples"]]
        ecu_v = [s["voltage"] for s in ecu_payload["samples"]]
        assert twin_v != ecu_v


# ---------------------------------------------------------------------------
# Values stay in physically normal ranges
# ---------------------------------------------------------------------------
class TestRanges:
    def test_signals_stay_within_configured_bounds(self):
        cfg = GeneratorConfig(seed=17)
        gen = SyntheticBatteryGenerator(cfg)
        twin_payload, ecu_payload = gen.batch(2000)  # long run to stress the walk/noise

        for payload in (twin_payload, ecu_payload):
            for sample in payload["samples"]:
                for signal, (lo, hi) in cfg.bounds.items():
                    assert lo <= sample[signal] <= hi, (
                        f"{signal}={sample[signal]} outside [{lo}, {hi}]"
                    )

    def test_soc_and_soh_are_percentages(self):
        gen = SyntheticBatteryGenerator(GeneratorConfig(seed=3))
        twin_payload, _ = gen.batch(100)
        for sample in twin_payload["samples"]:
            assert 0.0 <= sample["soc"] <= 100.0
            assert 0.0 <= sample["soh"] <= 100.0


# ---------------------------------------------------------------------------
# Batches are accepted by the real live_schema parser
# ---------------------------------------------------------------------------
class TestSchemaCompliance:
    def test_generator_uses_id_as_timestamp_field(self):
        gen = SyntheticBatteryGenerator(GeneratorConfig(seed=42))
        twin_payload, ecu_payload = gen.batch(5)

        for payload in (twin_payload, ecu_payload):
            for sample in payload["samples"]:
                assert "id" in sample
                assert "timestamp" not in sample

    def test_twin_batch_parses_and_routes_as_twin(self):
        gen = SyntheticBatteryGenerator(GeneratorConfig(seed=42))
        twin_payload, _ = gen.batch(5)
        payload_bytes = batch_to_json_bytes(twin_payload)

        samples = parse_message("bms/twin/data", payload_bytes, TWIN_ROOT, ECU_ROOT)

        assert len(samples) == 5
        assert all(s.source_label == "twin" for s in samples)
        for s in samples:
            for key in ("timestamp", "voltage", "current", "temperature", "soc", "soh"):
                assert key in s.columns

    def test_ecu_batch_parses_and_routes_as_ecu(self):
        gen = SyntheticBatteryGenerator(GeneratorConfig(seed=42))
        _, ecu_payload = gen.batch(5)
        payload_bytes = batch_to_json_bytes(ecu_payload)

        samples = parse_message("bms/actual/data", payload_bytes, TWIN_ROOT, ECU_ROOT)

        assert len(samples) == 5
        assert all(s.source_label == "ecu" for s in samples)

    def test_sample_ids_are_unique_and_stable(self):
        """Each logical sample gets a distinct id so QoS-1 redelivery dedup works."""
        gen = SyntheticBatteryGenerator(GeneratorConfig(seed=42))
        twin_payload, ecu_payload = gen.batch(10)

        twin_bytes = batch_to_json_bytes(twin_payload)
        ecu_bytes = batch_to_json_bytes(ecu_payload)

        twin_samples = parse_message("bms/twin/data", twin_bytes, TWIN_ROOT, ECU_ROOT)
        ecu_samples = parse_message("bms/actual/data", ecu_bytes, TWIN_ROOT, ECU_ROOT)

        twin_ids = [s.id for s in twin_samples]
        ecu_ids = [s.id for s in ecu_samples]
        assert len(twin_ids) == len(set(twin_ids))
        # Twin and ecu samples for the "same" logical step share an id --
        # that's how a real publisher would let a consumer correlate them.
        assert twin_ids == ecu_ids

    def test_output_is_valid_json(self):
        gen = SyntheticBatteryGenerator(GeneratorConfig(seed=1))
        twin_payload, ecu_payload = gen.batch(3)
        json.loads(batch_to_json_bytes(twin_payload))
        json.loads(batch_to_json_bytes(ecu_payload))


# ---------------------------------------------------------------------------
# id_mode: "sequence" (default) vs "timestamp"
# ---------------------------------------------------------------------------
class TestIdMode:
    def test_default_id_mode_is_sequence(self):
        assert GeneratorConfig().id_mode == ID_MODE_SEQUENCE

    def test_sequence_mode_start_id_customizable(self):
        gen = SyntheticBatteryGenerator(
            GeneratorConfig(seed=1, id_mode=ID_MODE_SEQUENCE, start_id=100)
        )
        twin_payload, ecu_payload = gen.batch(3)
        ids = [s["id"] for s in twin_payload["samples"]]
        assert ids == [100, 101, 102]
        # twin and ecu share the same sequence id for a given logical step.
        assert [s["id"] for s in ecu_payload["samples"]] == ids

    def test_timestamp_mode_emits_timestamp_not_id(self):
        gen = SyntheticBatteryGenerator(
            GeneratorConfig(seed=1, id_mode=ID_MODE_TIMESTAMP)
        )
        twin_payload, ecu_payload = gen.batch(3)
        for payload in (twin_payload, ecu_payload):
            for sample in payload["samples"]:
                assert "timestamp" in sample
                assert "id" not in sample

    def test_timestamp_mode_values_are_unix_ms_scale(self):
        before_ms = time.time() * 1000.0
        gen = SyntheticBatteryGenerator(
            GeneratorConfig(seed=1, id_mode=ID_MODE_TIMESTAMP)
        )
        twin_payload, _ = gen.batch(1)
        after_ms = time.time() * 1000.0
        ts = twin_payload["samples"][0]["timestamp"]
        # Constructed "now", give or take test execution slack -- nowhere
        # near a small sequence counter's magnitude (0, 1, 2, ...).
        assert before_ms - 1000 <= ts <= after_ms + 1000

    def test_timestamp_mode_advances_by_dt_seconds_in_ms(self):
        gen = SyntheticBatteryGenerator(
            GeneratorConfig(seed=1, id_mode=ID_MODE_TIMESTAMP, dt=0.2)
        )
        twin_payload, _ = gen.batch(3)
        ts = [s["timestamp"] for s in twin_payload["samples"]]
        assert ts[1] - ts[0] == pytest.approx(200.0, abs=1e-6)
        assert ts[2] - ts[1] == pytest.approx(200.0, abs=1e-6)

    def test_timestamp_mode_ecu_offset_from_twin(self):
        gen = SyntheticBatteryGenerator(
            GeneratorConfig(seed=1, id_mode=ID_MODE_TIMESTAMP, ecu_time_offset=0.05)
        )
        twin_payload, ecu_payload = gen.batch(1)
        twin_ts = twin_payload["samples"][0]["timestamp"]
        ecu_ts = ecu_payload["samples"][0]["timestamp"]
        assert ecu_ts - twin_ts == pytest.approx(50.0, abs=1e-6)  # 0.05 s -> 50 ms

    def test_timestamp_mode_parses_with_timestamp_axis_kind(self):
        gen = SyntheticBatteryGenerator(
            GeneratorConfig(seed=1, id_mode=ID_MODE_TIMESTAMP)
        )
        twin_payload, _ = gen.batch(2)
        samples = parse_message(
            "bms/twin/data", batch_to_json_bytes(twin_payload), TWIN_ROOT, ECU_ROOT,
        )
        assert all(s.axis_kind == "timestamp" for s in samples)
        assert all(s.id is None for s in samples)  # no dedup id in this mode

    def test_sequence_mode_parses_with_sequence_axis_kind(self):
        gen = SyntheticBatteryGenerator(GeneratorConfig(seed=1))
        twin_payload, _ = gen.batch(2)
        samples = parse_message(
            "bms/twin/data", batch_to_json_bytes(twin_payload), TWIN_ROOT, ECU_ROOT,
        )
        assert all(s.axis_kind == "sequence" for s in samples)

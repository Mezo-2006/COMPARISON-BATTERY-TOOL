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

from synthetic_data_generator import (
    GeneratorConfig,
    SyntheticBatteryGenerator,
    batch_to_json_bytes,
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

        assert first["samples"][-1]["timestamp"] < second["samples"][0]["timestamp"]
        assert second["samples"][0]["id"] == 6


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
